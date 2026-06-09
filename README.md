# Causal Evaluation of Email Retrieval Pipelines

A research experiment that measures, after retrieval, which context actually mattered for the correct answer — and how much retrieved context was unnecessary.

## The Problem

Retrieval systems optimize for similarity — which emails *look* relevant to a query. But looking relevant and actually mattering are different things. The goal: find the minimal set of context that still produces the correct answer, and cut everything else. Less noise, lower cost, same output quality.

## How It Works

**Pipeline:**
1. Pull all threads from an AgentMail inbox via the API
2. Embed each thread and the query using a pre-trained sentence transformer
3. Rank threads by cosine similarity — select top-k dynamically by cutting where similarity scores fall off a cliff
4. Run the agent on the full retrieved context — establish a baseline
5. **Necessity test:** remove one thread at a time, re-run the agent. If the answer breaks — that thread was necessary. If nothing changes — it was noise.
6. **Sufficiency test:** start from no context, add threads back one at a time in similarity order until the answer is correct. That's the minimal sufficient set.

The necessity test identifies which threads individually mattered. The sufficiency test finds the smallest set that still produces the correct answer. Together they give you the bloat number — a metric you can keep driving down.

## Result

```
============================================================
CAUSAL EVALUATION SUMMARY
============================================================

Query: What is the approved Acme contract and what were John's final comments before signing?

RETRIEVAL — all threads by similarity score:
  ✓ selected | Re: Acme Agreement — Final Terms   | score: 0.728
  ✓ selected | Acme Contract Approval              | score: 0.725
  ✓ selected | Acme Contract — Draft for Review    | score: 0.718
  ✓ selected | Following up on our Acme call       | score: 0.696
    excluded  | Vendor Agreement — Approved by John | score: 0.473
    excluded  | Team Offsite — June                 | score: 0.164
    excluded  | Q2 Budget Planning                  | score: 0.100

NECESSITY TEST:
  Re: Acme Agreement — Final Terms  | NECESSARY
  Acme Contract Approval             | NECESSARY
  Acme Contract — Draft for Review   | not necessary
  Following up on our Acme call      | not necessary

SUFFICIENCY TEST:
  Minimal sufficient set (2 threads):
    - Re: Acme Agreement — Final Terms
    - Acme Contract Approval

============================================================
IMPLICATION:
  Retrieved 4 threads. Only 2 were needed.
  Context reduced by 50% with no loss in answer quality.

WITHOUT causal eval: 4 threads sent to LLM
WITH causal eval:    2 threads sent to LLM
Result: same answer, 50% less context
============================================================
```

## Three Takeaways

**1. Quick win: smarter top-k cutoff**

The 4 selected threads scored between 0.696 and 0.728 — tightly clustered. The next thread dropped to 0.473, a 32% fall. That cliff is the natural boundary between signal and noise. Cut dynamically where scores fall off instead of using a fixed k. No new infrastructure, no extra compute — just better pre-processing before the context window gets built.

**2. Offline quantification of context bloat**

This experiment can't run at inference time — it's too slow. That's not what it's for. Run it offline across hundreds of real queries: retrieve threads, run the agent, ablate one thread at a time, measure what breaks. The output: "X% of retrieved context was unnecessary on average across real production queries." That's a baseline most retrieval pipelines don't have — a number you can keep driving down.

**3. Future: lightweight classifier at inference time**

The offline tests generate labeled data as a side effect — (thread, query, necessary: true/false) pairs. Across hundreds of queries that's a training set. Train a lightweight classifier on it. At inference time, score each retrieved thread before it enters the context window. No ablation, no extra LLM calls — just a fast scoring step that cuts predicted noise before it hits the downstream system. The classifier doesn't need to be perfect. It needs to beat the current baseline of passing everything.

**Full pipeline with classifier:**
```
User query
→ retrieval returns candidate threads
→ lightweight classifier scores each (query, thread) pair
→ keep likely necessary threads
→ drop likely unnecessary threads
→ send smaller context bundle to the downstream agent
```

## Stack

- [AgentMail](https://agentmail.to) — email API for AI agents
- `sentence-transformers` — semantic embeddings (all-MiniLM-L6-v2)
- `anthropic` — Claude Haiku as the downstream agent
- `numpy` — cosine similarity and ranking

## Setup

```bash
pip install agentmail python-dotenv sentence-transformers numpy anthropic
```

Add to `.env`:
```
AGENTMAIL_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

```bash
python causal_eval.py
```
