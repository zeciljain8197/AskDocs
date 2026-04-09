"""
Embedding module — wraps sentence-transformers for local, free embeddings.

Model: all-MiniLM-L6-v2
  - 384-dim vectors, very fast on CPU
  - Strong performance on semantic similarity benchmarks
  - ~90MB download, cached by HuggingFace after first use

Batching: processes chunks in configurable batch sizes to stay within RAM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models import Chunk


_model_cache: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model (singleton)."""
    global _model_cache
    if _model_cache is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model_cache = SentenceTransformer(settings.embedding_model)
        logger.success("Embedding model loaded")
    return _model_cache


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of strings.
    Returns shape (N, embedding_dim) float32 array.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # cosine sim == dot product when normalized
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> list[Chunk]:
    """
    Add embeddings to a list of Chunk objects in-place (returns same list).
    """
    logger.info(f"Embedding {len(chunks)} chunks …")
    texts = [c.content for c in chunks]
    vectors = embed_texts(texts, batch_size=batch_size)

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec.tolist()

    logger.success(f"Done embedding — shape: {vectors.shape}")
    return chunks


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Used at retrieval time."""
    vec = embed_texts([query], show_progress=False)[0]
    return vec.tolist()


def save_embeddings(chunks: list[Chunk], path: str | Path) -> None:
    """Persist chunk embeddings to .npy for fast re-loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array([c.embedding for c in chunks], dtype=np.float32)
    np.save(path, matrix)
    logger.info(f"Saved embeddings → {path}  shape={matrix.shape}")


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load saved embeddings."""
    return np.load(Path(path))
