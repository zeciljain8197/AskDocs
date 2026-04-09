"""
Unit tests — these run on every CI push and don't require a live LLM.
"""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_documents, chunk_stats
from src.models import Chunk, Document, ScoredChunk
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            content="LangChain Expression Language (LCEL) is a declarative way to compose LLM chains.\n\n"
            "It supports streaming, async execution, and parallel branching.\n\n"
            "You can build a chain with: chain = prompt | llm | parser",
            source="https://python.langchain.com/docs/expression_language",
            title="LCEL Overview",
        ),
        Document(
            content="Retrievers in LangChain implement the BaseRetriever interface.\n\n"
            "They expose a get_relevant_documents() method that returns a list of Documents.\n\n"
            "Vector stores can be converted to retrievers with .as_retriever().",
            source="https://python.langchain.com/docs/retrievers",
            title="Retrievers",
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents) -> list[Chunk]:
    return chunk_documents(sample_documents, chunk_size=200, chunk_overlap=20)


# ── Chunker tests ─────────────────────────────────────────────────────────────


class TestChunker:
    def test_produces_chunks(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        assert len(chunks) > 0

    def test_chunk_ids_are_deterministic(self, sample_documents):
        chunks_a = chunk_documents(sample_documents)
        chunks_b = chunk_documents(sample_documents)
        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert ids_a == ids_b

    def test_chunk_ids_are_unique(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_content_not_empty(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        for chunk in chunks:
            assert chunk.content.strip(), f"Empty chunk found: {chunk.chunk_id}"

    def test_source_preserved(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        sources = {c.source for c in chunks}
        expected = {d.source for d in sample_documents}
        assert sources == expected

    def test_chunk_stats_keys(self, sample_chunks):
        stats = chunk_stats(sample_chunks)
        for key in ("total", "min_len", "max_len", "avg_len", "unique_sources"):
            assert key in stats


# ── RRF tests ─────────────────────────────────────────────────────────────────


class TestRRF:
    def _make_scored_chunks(self, ids: list[str]) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=Chunk(chunk_id=cid, content=f"content {cid}", source="src", title="t"),
                score=1.0 / (i + 1),
                retrieval_method="test",
            )
            for i, cid in enumerate(ids)
        ]

    def test_deduplicates_results(self):
        list_a = self._make_scored_chunks(["a", "b", "c"])
        list_b = self._make_scored_chunks(["b", "c", "d"])
        fused = reciprocal_rank_fusion(list_a, list_b)
        ids = [r.chunk.chunk_id for r in fused]
        assert len(ids) == len(set(ids)), "RRF must deduplicate"
        assert set(ids) == {"a", "b", "c", "d"}

    def test_overlap_ranks_higher(self):
        list_a = self._make_scored_chunks(["shared", "only_a"])
        list_b = self._make_scored_chunks(["shared", "only_b"])
        fused = reciprocal_rank_fusion(list_a, list_b)
        # "shared" appears in both → should have highest RRF score
        assert fused[0].chunk.chunk_id == "shared"

    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_single_list_passthrough(self):
        list_a = self._make_scored_chunks(["x", "y", "z"])
        fused = reciprocal_rank_fusion(list_a)
        ids = [r.chunk.chunk_id for r in fused]
        assert set(ids) == {"x", "y", "z"}
