"""
Chunking strategy: Recursive Character Text Splitter.

Why recursive?
  - Tries to split on paragraph breaks first (\\n\\n), then sentences (\\n),
    then words — preserving semantic units as much as possible.
  - Overlap ensures context isn't cut off at chunk boundaries.

Chunk IDs are deterministic (sha256 of source + char offset) so re-ingesting
the same docs produces the same IDs — essential for eval reproducibility.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.models import Chunk, Document


# ── Defaults (proven good for technical documentation) ────────────────────────
DEFAULT_CHUNK_SIZE    = 512    # tokens ≈ chars/4; sweet spot for retrieval
DEFAULT_CHUNK_OVERLAP = 64     # ~12% overlap to preserve cross-boundary context
DEFAULT_SEPARATORS    = ["\n\n", "\n", ". ", " ", ""]


def _chunk_id(source: str, char_start: int) -> str:
    raw = f"{source}:{char_start}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def chunk_documents(
    documents: list[Document],
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split a list of Documents into Chunks.

    Returns a flat list of Chunk objects sorted by (source, char_start).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Chunk] = []

    for doc in documents:
        # LangChain splitter returns list of strings + keeps track of offsets
        lc_docs = splitter.create_documents(
            texts=[doc.content],
            metadatas=[{"source": doc.source, "title": doc.title, **doc.metadata}],
        )

        char_cursor = 0
        for lc_doc in lc_docs:
            text = lc_doc.page_content
            # Find true char_start by scanning forward from cursor
            char_start = doc.content.find(text[:50], char_cursor)
            if char_start == -1:
                char_start = char_cursor   # fallback
            char_end = char_start + len(text)
            char_cursor = max(char_cursor, char_start)   # don't go backwards

            chunk = Chunk(
                chunk_id=_chunk_id(doc.source, char_start),
                content=text,
                source=doc.source,
                title=doc.title,
                char_start=char_start,
                char_end=char_end,
                metadata=lc_doc.metadata,
            )
            all_chunks.append(chunk)

    logger.success(
        f"Chunked {len(documents)} documents → {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Return basic statistics about the chunk set — useful for debugging."""
    if not chunks:
        return {"total": 0, "min_len": 0, "max_len": 0, "avg_len": 0, "unique_sources": 0}
    lengths = [len(c.content) for c in chunks]
    return {
        "total": len(chunks),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "unique_sources": len({c.source for c in chunks}),
    }