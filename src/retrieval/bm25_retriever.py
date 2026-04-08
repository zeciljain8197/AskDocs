"""
BM25 sparse retrieval index.

BM25 (Best Match 25) is TF-IDF's more sophisticated cousin — it's been the
backbone of search engines (including Elasticsearch) for decades.

Strengths:
  - Exact keyword matching (catches things dense vectors miss)
  - No GPU / embedding model needed
  - Very fast on CPU

We persist the index + chunk list to disk so ingestion doesn't re-run
on every startup.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from loguru import logger
from rank_bm25 import BM25Okapi

from src.models import Chunk, ScoredChunk


INDEX_DIR = Path("data/processed")
BM25_PATH   = INDEX_DIR / "bm25_index.pkl"
CHUNKS_PATH = INDEX_DIR / "bm25_chunks.json"


# ── Tokenization ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercased, strips punctuation, keeps alphanumerics.
    Good enough for technical docs; swap with NLTK/spaCy if you need stemming.
    """
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9_\-\.]+\b", text)
    return [t for t in tokens if len(t) > 1]


# ── Build & persist ───────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    """
    Build a BM25 index from chunks and save to disk.
    """
    if not chunks:
        raise ValueError(
            "Cannot build BM25 index: 0 chunks loaded. "
            "Check loader logs above for the root cause."
        )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    tokenized = [_tokenize(c.content) for c in chunks]
    index = BM25Okapi(tokenized)

    # Save index
    with open(BM25_PATH, "wb") as f:
        pickle.dump(index, f)

    # Save chunk metadata (no embeddings — saves space)
    chunk_data = [
        {
            "chunk_id":   c.chunk_id,
            "content":    c.content,
            "source":     c.source,
            "title":      c.title,
            "char_start": c.char_start,
            "char_end":   c.char_end,
        }
        for c in chunks
    ]
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunk_data, f)

    logger.success(f"BM25 index built — {len(chunks)} documents")
    return index


def load_bm25_index() -> tuple[BM25Okapi, list[Chunk]]:
    """Load BM25 index and chunks from disk."""
    if not BM25_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "BM25 index not found. Run the ingestion pipeline first."
        )

    with open(BM25_PATH, "rb") as f:
        index = pickle.load(f)

    with open(CHUNKS_PATH) as f:
        raw = json.load(f)

    chunks = [
        Chunk(
            chunk_id=d["chunk_id"],
            content=d["content"],
            source=d["source"],
            title=d["title"],
            char_start=d.get("char_start", 0),
            char_end=d.get("char_end", 0),
        )
        for d in raw
    ]

    logger.info(f"Loaded BM25 index ({len(chunks)} chunks)")
    return index, chunks


# ── Search ────────────────────────────────────────────────────────────────────

_bm25_cache: tuple[BM25Okapi, list[Chunk]] | None = None


def bm25_search(query: str, top_k: int = 20) -> list[ScoredChunk]:
    """
    BM25 keyword search. Returns ScoredChunk list sorted by BM25 score desc.
    Index is lazily loaded and cached in memory.
    """
    global _bm25_cache
    if _bm25_cache is None:
        _bm25_cache = load_bm25_index()

    index, chunks = _bm25_cache
    tokens = _tokenize(query)
    scores = index.get_scores(tokens)

    # Pair (score, chunk) and sort descending
    paired = sorted(
        zip(scores, chunks), key=lambda x: x[0], reverse=True
    )[:top_k]

    return [
        ScoredChunk(chunk=chunk, score=float(score), retrieval_method="bm25")
        for score, chunk in paired
        if score > 0   # filter zero-score results
    ]
