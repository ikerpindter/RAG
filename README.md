# Hybrid RAG over SEC 10-K filings

Hybrid retrieval (dense + BM25 + RRF + rerank) over SEC 10-K filings, with per-claim citation verification and an eval pipeline that measured every architecture change.

[![CI](https://github.com/ikerpindter/RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/ikerpindter/RAG/actions/workflows/ci.yml)

Every architecture stage was frozen as a versioned baseline (Ragas metrics, judge and known issues documented in each file under [`evals/`](evals/)):

| Stage | Corpus | Questions | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|---|
| [Hybrid + citations](evals/baseline_hybrid.json) | 1 doc | 12 | 0.981 | 0.861 | 0.755 | 0.861 |
| [4x corpus](evals/baseline_full_corpus.json) | 4 docs | 20 | 0.810 | 0.809 | 0.480 | 0.613 |
| [+ query analysis](evals/baseline_smart_retrieval.json) | 4 docs | 20 | 0.921 | 0.875 | 0.545 | 0.738 |
| [+ reranker](evals/baseline_reranker.json) | 4 docs | 20 | 0.965 | 0.932 | 0.802 | 0.912 |

The drop in row 2 is the point: expanding from one filing to four (Lennar and D.R. Horton, FY2023 and FY2024) made the problem realistic — consecutive-year filings from the same company are textual near-twins, and retrieval quality collapsed until the pipeline learned to tell them apart. The first row also uses an easier, smaller gold set (12 questions, single document), so it is not directly comparable to the other three.

## What makes this different

- **The gold set is expert-written, not generated.** 20 questions with references extracted manually from the filings by a finance expert, including deliberate traps — e.g. the guarantor-subsidiary summarized figures ($32.6B "total revenues") that sit a few chunks away from the consolidated income statement ($35.4B) and would fool a careless reference.
- **The eval caught a real hallucination and the fix was retrieval, not prompting.** On the 4x corpus, one question produced invented revenue figures that scored 0.95 on answer relevancy while faithfulness exposed them at 0.00 ([baseline_full_corpus.json](evals/baseline_full_corpus.json), q16). The same question scores 1.00/1.00 on context metrics after the reranker — no prompt was touched to fix it.
- **Cross-year contamination was diagnosed and its fix measured.** "Lennar total revenues in FY2024" retrieved FY2023 chunks (context precision/recall 0.00/0.00); metadata filtering from query analysis took it to 1.00/1.00. The failure, the diagnosis and the fix are all in the versioned baselines.
- **Comparative questions rerank per company.** Reranking a merged candidate pool would let one company dominate the top-k and destroy the coverage guarantee that query decomposition exists to provide; instead, each per-company sub-retrieval is reranked independently (top-4 per company).
- **Citations are verifiable, and the sources section cannot be hallucinated.** The model only emits inline `[n]` markers; the sources list is rendered by code from retrieval metadata. `--verify` then checks each cited claim against its cited chunk with a cheap LLM verdict.

## Architecture

```
question
  │
  ├─ query analysis ──── {companies, fiscal_years, comparative}   (strict JSON,
  │                       validated against the corpus catalog; empty = no filter)
  ▼
metadata-filtered hybrid retrieval
  dense (cosine over normalized embeddings)  +  BM25 (rebuilt on the filtered subset)
  fused with Reciprocal Rank Fusion (k=60)
  [comparative: one sub-retrieval per company]
  ▼
rerank: RRF pool of 20 → top-5   (per company when comparative: pool 20 → top-4 each)
  ▼
generation with inline citations [n]   (extracts labeled with company + fiscal year)
  ▼
sources section built from retrieval metadata (not model output)
optional --verify: per-claim support check against the cited chunks
```

| Component | Implementation | Model (swappable in [src/config.py](src/config.py)) |
|---|---|---|
| Embeddings | OpenAI | `text-embedding-3-small` |
| Generation | OpenAI Responses API | `gpt-5.4-nano` |
| Query analysis | OpenAI, strict JSON + catalog validation | `gpt-5.4-nano` |
| Citation verification | OpenAI, binary verdict per claim | `gpt-5.4-nano` |
| Reranker | Cohere (`RERANKER_ENABLED` flag = control group) | `rerank-v4.0-pro` |
| Eval metrics | Ragas 0.4.3 (collections API) | judge: `gpt-5.4-nano` |
| Vector store | numpy, one index file per document, no infra | — |
| Keyword search | rank-bm25 (BM25Okapi) | — |

## Design decisions

- **10-K filings as corpus.** Long, public, verifiable documents whose consecutive-year editions are structurally near-identical — a retrieval problem that actually punishes sloppy pipelines, unlike toy corpora.
- **Evals before the reranker.** The reranker was deliberately built last, so it had numeric targets written in advance (three questions at 0.00 context metrics, precision at 0.545). Its +0.257 precision gain is measured against a frozen baseline, not assumed.
- **Manual gold set over synthetic generation.** Synthetic drafts were skipped entirely: references written by a domain expert are cheaper here (20 questions), catch subtleties like consolidated-vs-subsidiary figures, and make eval numbers trustworthy.
- **RRF for fusion.** Cosine similarities and BM25 scores are not comparable; RRF combines rankings using positions only, with the standard k=60, and needs no score normalization.
- **Re-ingest instead of migrating the index.** Chunking is deterministic, so rebuilding the whole 4-document index cost about $0.01 of embeddings — cheaper than writing and trusting migration code. One index file per document makes ingestion idempotent per document.
- **No silent fallbacks.** Malformed analyzer JSON, out-of-catalog values, missing API keys and persistent Cohere 429s all stop with explicit errors. A pipeline that silently degrades (e.g. skips reranking) contaminates every experiment that runs on top of it.

## Running it

Requirements: [uv](https://docs.astral.sh/uv/), an OpenAI API key, a Cohere API key (free trial tier is enough; 429s are retried with a visible warning).

```bash
uv sync
cp .env.example .env        # fill OPENAI_API_KEY and COHERE_API_KEY

# One-time ingestion: downloads the 4 filings from SEC EDGAR, chunks, embeds (~$0.01)
uv run python -m src.ingest

# Ask questions (fractions of a cent each)
uv run python -m src.query "What was the dollar value of Lennar's backlog at November 30, 2024?"
uv run python -m src.query "Compare Lennar and D.R. Horton total revenues in fiscal 2024." --debug
uv run python -m src.query "How many homes did Lennar deliver in fiscal year 2024?" --verify

# Eval harness over the expert gold set (~$0.10-0.15 per full run with the nano judge)
uv run python -m evals.harness
uv run python -m evals.harness --limit 2   # cheap smoke test

# Tests: deterministic, no API keys, no network (same command CI runs)
uv run pytest
```

## Honest limitations

- **The judge is a cheap model.** `gpt-5.4-nano` runs with `temperature` forced to 1.0 (reasoning models accept no other value), which produces a documented ±0.03 run-to-run variance on context metrics, plus occasional verdict errors — both recorded in the baseline metadata rather than averaged away.
- **Cross-company context_precision reads 0.00 and is a metric false negative.** The judge does not credit per-company contexts against combined references, even when answers are correct and supported. Limitation of the metric setup, not of retrieval; noted in [baseline_reranker.json](evals/baseline_reranker.json).
- **Financial tables are handled pragmatically.** Table rows are flattened to `cell | cell | cell` text at ingestion. There is no deep tabular understanding; questions that require reading a table usually work because the relevant row survives flattening, not because the system parses structure.
- **The corpus is 4 documents by design.** Small enough to iterate fast and re-ingest for cents, structured enough (2 companies × 2 fiscal years) to reproduce real retrieval failures. It is not a scale demonstration.
- **Phrasing sensitivity was observed.** Before the reranker, two phrasings of the same revenue question retrieved different chunks and one produced a wrong answer; the reranker fixed the measured case, but the underlying sensitivity of embedding retrieval remains.
