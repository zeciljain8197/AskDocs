# AskDocs

[![RAG Quality Gate](https://github.com/zeciljain8197/askdocs/actions/workflows/eval.yml/badge.svg)](https://github.com/zeciljain8197/askdocs/actions/workflows/eval.yml)
[![Lint & Test](https://github.com/zeciljain8197/askdocs/actions/workflows/ci.yml/badge.svg)](https://github.com/zeciljain8197/askdocs/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions over LangChain documentation. Combines BM25 sparse search with dense vector search, fuses them via Reciprocal Rank Fusion, and reranks results with a cross-encoder — all gated by an automated RAGAS evaluation pipeline on every commit.

---

## Architecture

```
 ┌──────────────────────────── INGESTION ──────────────────────────────────┐
 │  LangChain Docs  ──►  Recursive Chunker  ──►  sentence-transformers     │
 │                                                       │                 │
 │                                               Qdrant DB (dense)         │
 │                                               BM25 Index (sparse)       │
 └─────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌──────────────────────────── RETRIEVAL ──────────────────────────────────┐
 │                                                                         │
 │   Query ──► BM25 Search ──────────────────────┐                        │
 │         ──► Vector ANN  ──────────────────────┼──► RRF Fusion           │
 │                                               │        │               │
 │                                               └────────▼               │
 │                                               Cross-encoder Reranker   │
 │                                                        │               │
 │                                                     Top-K Chunks       │
 └─────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌──────────────────────────── GENERATION ─────────────────────────────────┐
 │  Top-K Chunks ──► Prompt Builder ──► Groq LLaMA 3.1 8B                 │
 │                                              │                          │
 │                                     Answer + Inline Citations           │
 └─────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌──────────────────────────── EVALUATION (CI) ────────────────────────────┐
 │  Golden Q&A Dataset ──► RAGAS 0.4 Scoring ──► Quality Gate              │
 │  Faithfulness · Answer Relevancy · Context Recall · Context Precision   │
 │  CI fails automatically if any metric drops below threshold             │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Results

Scored on a 10-sample golden Q&A dataset using RAGAS 0.4 with Groq LLaMA 3.1 8B as the judge LLM.

| Metric              | Score  | Threshold | Status |
|---------------------|--------|-----------|--------|
| Faithfulness        | 0.6549 | —         | tracked |
| Answer Relevancy    | 0.9524 | ≥ 0.75    | ✓ PASS  |
| Context Recall      | 0.8519 | ≥ 0.65    | ✓ PASS  |
| Context Precision   | 0.4951 | —         | tracked |

---

## Tech Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| Ingestion    | GitHub API (Git Trees), LangChain text splitters|
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` (local)|
| Vector store | Qdrant (embedded, no server required)           |
| Sparse search| rank-bm25                                       |
| Fusion       | Reciprocal Rank Fusion (RRF)                    |
| Reranking    | `cross-encoder/ms-marco-MiniLM-L-6-v2`          |
| LLM          | Groq API — LLaMA 3.1 8B Instant (free tier)     |
| Serving      | FastAPI + Uvicorn                               |
| Evaluation   | RAGAS 0.4 (Faithfulness, Relevancy, Recall)     |
| CI/CD        | GitHub Actions                                  |

---

## Quick Start

```bash
git clone https://github.com/zeciljain8197/askdocs
cd askdocs
pip install -e ".[dev]"
cp .env.example .env        # then add your free Groq API key
make ingest-quick           # builds a 30-page index for a smoke test
make serve                  # http://localhost:8000
```

Full index (all ~300 LangChain doc pages):
```bash
make ingest                 # ~10 min first run; cached on subsequent runs
```

---

## Setup Guide

### 1. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (no credit card, 500k tokens/day)
- CUDA GPU optional — embeddings and reranking run on CPU if unavailable

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and set:

```env
GROQ_API_KEY=your_groq_api_key_here
```

All other values have working defaults. See `.env.example` for the full reference.

### 4. Build the index

The index is built once and cached locally. Subsequent runs skip ingestion automatically.

```bash
make ingest          # full index (~300 pages, ~10 min)
make ingest-quick    # 30-page subset for rapid testing
```

This creates:
- `data/processed/bm25_index.pkl` — BM25 sparse index
- `data/processed/bm25_chunks.json` — chunk store
- `data/qdrant_db/` — Qdrant vector database

### 5. Run the API

```bash
make serve
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## API Reference

### `POST /ask`

Ask a question and receive a grounded answer with citations.

**Request**
```json
{
  "question": "How does LCEL work?",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "LCEL (LangChain Expression Language) is a declarative way to compose chains [1]. It provides a unified Runnable interface that supports streaming, async, and batch execution [2].",
  "citations": [
    {
      "index": 1,
      "chunk_id": "a3f9b21c4d01",
      "source": "https://python.langchain.com/docs/expression_language/",
      "title": "LangChain Expression Language",
      "excerpt": "LCEL makes it easy to build complex chains from basic components..."
    }
  ],
  "query": "How does LCEL work?",
  "model": "llama-3.1-8b-instant"
}
```

**cURL example**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a retriever in LangChain?"}'
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Docker

Requires the index to be built first (`make ingest`).

```bash
# Build and start
docker compose up --build

# API is available at http://localhost:8000
```

The container mounts `./data` as a volume — no data is baked into the image.

---

## Evaluation

Run the full RAGAS evaluation pipeline locally:

```bash
make eval
```

This runs all 10 golden Q&A samples through the full pipeline (retrieval → generation → scoring) and prints a metric report. It also writes `data/processed/eval_results.json` and exits with code 1 if any metric falls below its threshold (the same check run in CI).

The evaluation uses Groq's free tier as the judge LLM (`llama-3.1-8b-instant`). A full run takes ~80 minutes due to Groq's 6,000 TPM rate limit on the free tier.

---

## Development

```bash
make test      # pytest
make lint      # ruff check
make format    # ruff format
make ask       # interactive CLI
```

### Running tests

```bash
pytest tests/ -v
```

### Project structure

```
askdocs/
├── src/
│   ├── config.py                  # All settings loaded from .env
│   ├── models.py                  # Shared dataclasses and Pydantic models
│   ├── api.py                     # FastAPI serving layer
│   ├── ingestion/
│   │   ├── loader.py              # Scrape and parse LangChain docs
│   │   ├── chunker.py             # Recursive text splitter
│   │   ├── embedder.py            # sentence-transformers wrapper
│   │   └── vector_store.py        # Qdrant client wrapper
│   ├── retrieval/
│   │   ├── bm25_retriever.py      # BM25 sparse search
│   │   ├── hybrid_retriever.py    # RRF fusion + cross-encoder reranking
│   │   └── query_expansion.py     # Query expansion
│   ├── generation/
│   │   └── generator.py           # Groq LLM + citation parsing
│   └── evaluation/
│       └── evaluator.py           # RAGAS metrics + CI quality gate
├── scripts/
│   ├── ingest.py                  # Ingestion entry point
│   └── ask.py                     # Interactive Q&A CLI
├── tests/
│   └── test_pipeline.py
├── data/
│   └── processed/
│       └── golden_dataset.json    # Hand-curated Q&A evaluation set
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint + unit tests
│       └── eval.yml               # RAGAS quality gate
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## CI/CD

Two GitHub Actions workflows run on every push to `main` and every pull request:

**`ci.yml` — Lint & Test** (fast, ~2 min)
1. Ruff lint check
2. Ruff format check
3. pytest unit tests

**`eval.yml` — RAG Quality Gate** (slow, ~90 min)
1. Restore index from Actions cache (skips ingestion if unchanged)
2. Run full 10-sample RAGAS evaluation
3. Fail the build if Answer Relevancy < 0.75 or Context Recall < 0.65 (Faithfulness is tracked but not gated — llama-3.1-8b-instant has high judge variance)
4. Upload `eval_results.json` as a build artifact

To enable CI, add `GROQ_API_KEY` as a secret in your repository settings:  
**Settings → Secrets and variables → Actions → New repository secret**

---

## License

[MIT](LICENSE)
