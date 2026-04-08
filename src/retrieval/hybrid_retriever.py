"""
Hybrid retrieval pipeline:
  1. BM25 sparse retrieval   → top-K keyword matches
  2. Vector ANN retrieval    → top-K semantic matches
  3. Reciprocal Rank Fusion  → merge & deduplicate both lists
  4. Cross-encoder reranking → re-score the fused set with a more expensive model

Why RRF?
  RRF(d) = Σ 1 / (k + rank_i(d))
  It's robust — doesn't require score normalization across retrieval systems.
  k=60 is the standard constant (Cormack et al., 2009).

Why cross-encoder reranking?
  Bi-encoders (like our embedder) encode query and document independently —
  fast but coarse. A cross-encoder sees (query, document) together, enabling
  fine-grained relevance scoring at the cost of latency.
  We only rerank the top ~40 fused results, keeping it fast.
"""
from __future__ import annotations

from collections import defaultdict

from loguru import logger
from transformers import pipeline as hf_pipeline

from src.config import settings
from src.ingestion.embedder import embed_query
from src.ingestion.vector_store import vector_search
from src.models import Chunk, RerankedChunk, ScoredChunk
from src.retrieval.bm25_retriever import bm25_search


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

RRF_K = 60   # standard constant


def reciprocal_rank_fusion(
    *ranked_lists: list[ScoredChunk],
) -> list[ScoredChunk]:
    """
    Merge multiple ranked lists using RRF.
    Deduplicates by chunk_id, taking the highest-content chunk on collision.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map:  dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            cid = scored.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (RRF_K + rank)
            chunk_map[cid] = scored.chunk   # last write wins (same chunk anyway)

    merged = [
        ScoredChunk(
            chunk=chunk_map[cid],
            score=rrf_scores[cid],
            retrieval_method="rrf",
        )
        for cid in sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)
    ]
    return merged


# ── Cross-encoder reranker ────────────────────────────────────────────────────

_reranker_cache = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        _reranker_cache = hf_pipeline(
            "text-classification",
            model=RERANKER_MODEL,
            function_to_apply="sigmoid",
            batch_size=16,
        )
        logger.success("Cross-encoder loaded")
    return _reranker_cache


def rerank(query: str, candidates: list[ScoredChunk], top_k: int) -> list[RerankedChunk]:
    """
    Score each (query, chunk) pair with the cross-encoder.
    Returns top_k chunks sorted by rerank score descending.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [
        {"text": query, "text_pair": c.chunk.content[:512]}  # model max 512 tokens
        for c in candidates
    ]
    scores = reranker(pairs)   # list of {"label": ..., "score": float}

    reranked = sorted(
        [
            RerankedChunk(
                chunk=candidate.chunk,
                rerank_score=result["score"],
                original_rrf_score=candidate.score,
            )
            for candidate, result in zip(candidates, scores)
        ],
        key=lambda r: r.rerank_score,
        reverse=True,
    )

    return reranked[:top_k]


# ── Main entry point ──────────────────────────────────────────────────────────

def retrieve(
    query: str,
    bm25_k: int   = None,
    vector_k: int = None,
    rerank_k: int = None,
) -> list[RerankedChunk]:
    """
    Full hybrid retrieval pipeline for a query.

    Args:
        query:    User's natural language question.
        bm25_k:   Candidates from BM25 (default from settings).
        vector_k: Candidates from vector search (default from settings).
        rerank_k: Final chunks after reranking (default from settings).

    Returns:
        List of RerankedChunk, best first.
    """
    bm25_k   = bm25_k   or settings.bm25_top_k
    vector_k = vector_k or settings.vector_top_k
    rerank_k = rerank_k or settings.rerank_top_k

    # 1. Sparse retrieval
    bm25_results = bm25_search(query, top_k=bm25_k)
    logger.debug(f"BM25: {len(bm25_results)} results")

    # 2. Dense retrieval
    query_vec = embed_query(query)
    vector_results = vector_search(query_vec, top_k=vector_k)
    logger.debug(f"Vector: {len(vector_results)} results")

    # 3. Fuse
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    logger.debug(f"RRF fused: {len(fused)} unique chunks")

    # 4. Rerank
    reranked = rerank(query, candidates=fused, top_k=rerank_k)
    logger.debug(f"After rerank: {len(reranked)} final chunks")

    return reranked
