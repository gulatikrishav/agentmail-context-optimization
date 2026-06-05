# Set-up
import os
import anthropic
import numpy as np
from dotenv import load_dotenv
from agentmail import AgentMail
from sentence_transformers import SentenceTransformer

#download pre-learned weights
load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')
client = AgentMail(api_key=os.getenv("AGENTMAIL_API_KEY"))

INBOX_ID = "krishav-9911@agentmail.to"

# Returns True if answer contains all required facts, False otherwise
def is_correct(answer):
    if "May agreement" not in answer or "indemnification" not in answer:
        return False
    return True

########################################
# Step 1: Read every thread and every message, store as list of dicts
threads = client.inboxes.threads.list(INBOX_ID).threads
thread_data = []

# Note: For future experimentation - our agent retrieves at the thread level (to start)
# So when we do our experiments we remove whole threads at a time
# Future direction: doing this at the message level for even finer precision
for thread in threads:
    thread_id = thread.thread_id
    # Get the full thread (the agentmail lightweight list doesn't have messages)
    full_thread = client.inboxes.threads.get(INBOX_ID, thread_id)
    body = ""
    has_attachment = False
    # Loop through every message in the thread
    for message in full_thread.messages:
        # Combine all message bodies into one string (for embedding and similarity score - later on)
        body += message.extracted_text or ""
        # Flag the thread if any message has an attachment
        if message.attachments is not None:
            has_attachment = True
    # Store the thread as a dictionary
    thread_data.append({
        "thread_id": thread_id,
        "subject": full_thread.subject,
        "body": body,
        "has_attachment": has_attachment
    })

############################
# Step 2: Vector Embeddings
# Combining the subject and the body to get one vector
thread_texts = [t["subject"] + " " + t["body"] for t in thread_data]
thread_embeddings = model.encode(thread_texts)

# Need to do the same for queries
query = "What is the approved Acme contract and what were John's final comments before signing?"
query_embedding = model.encode(query)

#######################
# Step 3: Ranking System & Top K (QUICK WIN)
# Compute dot product for cosine similarity (ranking system)
scores = np.dot(query_embedding, thread_embeddings.T)
ranked_indices = np.argsort(-scores)

# Top k scores
ranked_scores = [scores[i] for i in ranked_indices]

# Select threads dynamically: cut at the first drop exceeding 20% from the previous score
cut_point = len(ranked_scores)  # default: keep all

for i in range(len(ranked_scores) - 1):
    drop = (ranked_scores[i] - ranked_scores[i+1]) / ranked_scores[i]
    if drop > 0.20: # 20% is an arbitrarily chosen value — can be changed based on inbox size and score distribution
        cut_point = i + 1
        break

top_k_indices = ranked_indices[:cut_point]

# ######################
# Step 4: LLM Step
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# subject and body parsed together
context = ""
for idx in top_k_indices:
    t = thread_data[idx]
    context += f"Subject: {t['subject']}\n{t['body']}\n\n"

# context and query parsed together
content = f"Here are the relevant email threads:\n\n{context}\nQuestion: {query}\n\nAnswer based only on the emails above."

# API Call -
# NOTE: In the real implementation we'd replace the Claude call with whatever AgentMail's actual agent/search method is.
response = anthropic_client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": content}]
)

#######################
# NECESSITY TEST - same as step 4 but going through an ablated list
# Keeping track of the results in a list
necessity_results = []
# Loop through top_k and ablate one at a time
for current_index in top_k_indices:
    ablated = [i for i in top_k_indices if i != current_index]

    # Build context from ablated list
    ablated_context = ""
    for idx in ablated:
        t = thread_data[idx]
        ablated_context += f"Subject: {t['subject']}\n{t['body']}\n\n"

    # Context and query parsed together
    ablated_content = f"Here are the relevant email threads:\n\n{ablated_context}\nQuestion: {query}\n\nAnswer based only on the emails above."

    # Call LLM with ablated context
    ablated_response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": ablated_content}]
    )

    # Check if removing this thread broke the answer
    necessary = not is_correct(ablated_response.content[0].text)
    necessity_results.append({
        "subject": thread_data[current_index]['subject'],
        "necessary": necessary
    })

##############################
# SUFFICIENCY TEST - start with nothing, add back one thread at a time
sufficient = []
minimal_set = []  # default empty in case no sufficient set found

for current_index in top_k_indices:
    sufficient.append(current_index)

    # Build context from threads added so far
    sufficient_context = ""
    for idx in sufficient:
        t = thread_data[idx]
        sufficient_context += f"Subject: {t['subject']}\n{t['body']}\n\n"

    # Context and query parsed together
    sufficient_content = f"Here are the relevant email threads:\n\n{sufficient_context}\nQuestion: {query}\n\nAnswer based only on the emails above."

    # Call LLM with growing context
    sufficient_response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": sufficient_content}]
    )

    # Check if we have enough context to answer correctly
    if is_correct(sufficient_response.content[0].text):
        # Go through each index in sufficient and pull out just the subject name; store as a list.
        minimal_set = [thread_data[idx]['subject'] for idx in sufficient]
        break

###################
# Results SUMMARY
print("\n" + "="*60)
print("CAUSAL EVALUATION SUMMARY")
print("="*60)
print(f"\nQuery: {query}\n")

print("RETRIEVAL — all threads by similarity score:")
for rank, idx in enumerate(ranked_indices):
    in_top_k = "✓ selected" if idx in top_k_indices else "  excluded"
    print(f"  {in_top_k} | {thread_data[idx]['subject']} | score: {scores[idx]:.3f}")

print("\nNECESSITY TEST:")
for r in necessity_results:
    tag = "NECESSARY" if r['necessary'] else "not necessary"
    print(f"  {r['subject']} | {tag}")

print("\nSUFFICIENCY TEST:")
print(f"  Minimal sufficient set ({len(minimal_set)} threads):")
for s in minimal_set:
    print(f"    - {s}")

print("="*60)
noise = len(top_k_indices) - len(minimal_set)
print(f"\nIMPLICATION:")
print(f"  Retrieved {len(top_k_indices)} threads. Only {len(minimal_set)} were needed.")
print(f"  Context reduced by {int(noise/len(top_k_indices)*100)}% with no loss in answer quality.")
print(f"\nWITHOUT causal eval: {len(top_k_indices)} threads sent to LLM")
print(f"WITH causal eval:    {len(minimal_set)} threads sent to LLM")
print(f"Result: same answer, {int(noise/len(top_k_indices)*100)}% less context")
print("="*60)