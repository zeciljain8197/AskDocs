"""
Shared data models for the entire pipeline.
Using plain dataclasses + pydantic where serialization matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel


# ── Ingestion ──────────────────────────────────────────────────────────────────


@dataclass
class Document:
    """A raw document before chunking."""

    content: str
    source: str  # URL or file path
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A single retrievable unit — the atom of the RAG system."""

    chunk_id: str  # deterministic: sha256(source + char_start)[:12]
    content: str
    source: str  # original document URL / path
    title: str  # parent document title
    char_start: int = 0
    char_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


# ── Retrieval ──────────────────────────────────────────────────────────────────


@dataclass
class ScoredChunk:
    """A chunk paired with its retrieval score (used before reranking)."""

    chunk: Chunk
    score: float
    retrieval_method: str = ""  # "bm25" | "vector" | "rrf"


@dataclass
class RerankedChunk:
    """A chunk after cross-encoder reranking."""

    chunk: Chunk
    rerank_score: float
    original_rrf_score: float = 0.0


# ── Generation ────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    index: int  # [1], [2], … as they appear in the answer
    chunk_id: str
    source: str
    title: str
    excerpt: str  # first 120 chars of the chunk


class AnswerWithCitations(BaseModel):
    answer: str
    citations: list[Citation]
    query: str
    model: str


# ── Evaluation ────────────────────────────────────────────────────────────────


class EvalSample(BaseModel):
    question: str
    ground_truth: str
    contexts: list[str] = []  # filled during eval run
    answer: str = ""  # filled during eval run
