"""
Vector store — wraps Qdrant in local (on-disk) mode.

Why Qdrant?
  - Production-grade: used at Notion, Disney, and others
  - Local mode: no server, no Docker needed for dev
  - Same client API works against a hosted Qdrant Cloud instance later
  - Supports filtered search (useful for multi-tenant RAG)

Schema stored in each point payload:
  { chunk_id, content, source, title, char_start, char_end, metadata }
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config import settings
from src.models import Chunk, ScoredChunk


_client_cache: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Singleton Qdrant client (local on-disk mode)."""
    global _client_cache
    if _client_cache is None:
        Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
        _client_cache = QdrantClient(path=settings.qdrant_path)
    return _client_cache


def ensure_collection(recreate: bool = False) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if recreate and settings.qdrant_collection in existing:
        client.delete_collection(settings.qdrant_collection)
        existing = []

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")
    else:
        logger.info(f"Using existing collection: {settings.qdrant_collection}")


def upsert_chunks(chunks: list[Chunk], batch_size: int = 256) -> None:
    """
    Upsert embedded chunks into Qdrant.
    Chunks without embeddings are skipped with a warning.
    """
    client = get_client()
    valid = [c for c in chunks if c.embedding is not None]
    skipped = len(chunks) - len(valid)
    if skipped:
        logger.warning(f"Skipping {skipped} chunks without embeddings")

    points: list[PointStruct] = []
    for chunk in valid:
        points.append(
            PointStruct(
                id=int(chunk.chunk_id, 16) % (2**63),  # Qdrant needs uint64
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "title": chunk.title,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "metadata": chunk.metadata,
                },
            )
        )

    # Upsert in batches
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)
        logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    logger.success(f"Upserted {len(valid)} chunks into Qdrant")


def vector_search(
    query_vector: list[float],
    top_k: int = 20,
    source_filter: str | None = None,
) -> list[ScoredChunk]:
    """
    ANN vector search. Returns ScoredChunk list sorted by cosine similarity.
    Optionally filter by source URL prefix.
    """
    client = get_client()

    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    results = response.points

    scored: list[ScoredChunk] = []
    for r in results:
        p = r.payload
        chunk = Chunk(
            chunk_id=p["chunk_id"],
            content=p["content"],
            source=p["source"],
            title=p["title"],
            char_start=p.get("char_start", 0),
            char_end=p.get("char_end", 0),
            metadata=p.get("metadata", {}),
        )
        scored.append(ScoredChunk(chunk=chunk, score=r.score, retrieval_method="vector"))

    return scored


def collection_size() -> int:
    client = get_client()
    info = client.get_collection(settings.qdrant_collection)
    return info.points_count
