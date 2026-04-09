"""
Ingestion pipeline — run this once to build the index.

Steps:
  1. Scrape LangChain docs (cached to disk)
  2. Chunk documents
  3. Embed chunks (local model)
  4. Upsert into Qdrant (vector store)
  5. Build BM25 index
  6. Print stats

Usage:
    python -m scripts.ingest
    python -m scripts.ingest --max-pages 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
import time

from loguru import logger

from src.ingestion.chunker import chunk_documents, chunk_stats
from src.ingestion.embedder import embed_chunks
from src.ingestion.loader import load_langchain_docs
from src.ingestion.vector_store import ensure_collection, upsert_chunks, collection_size
from src.retrieval.bm25_retriever import build_bm25_index


def run(max_pages: int | None = None, recreate: bool = False) -> None:
    t0 = time.time()

    logger.info("═══ AskDocs Ingestion Pipeline ═══")

    # 1. Load
    logger.info("Step 1/5  Loading documents …")
    documents = load_langchain_docs(max_pages=max_pages)

    if not documents:
        logger.error("No documents loaded — aborting. Check loader output above.")
        sys.exit(1)

    # 2. Chunk
    logger.info("Step 2/5  Chunking …")
    chunks = chunk_documents(documents)
    stats = chunk_stats(chunks)
    logger.info(f"  Chunk stats: {stats}")

    # 3. Embed
    logger.info("Step 3/5  Embedding (this takes a few minutes on first run) …")
    chunks = embed_chunks(chunks)

    # 4. Vector store
    logger.info("Step 4/5  Upserting into Qdrant …")
    ensure_collection(recreate=recreate)
    upsert_chunks(chunks)
    logger.info(f"  Collection size: {collection_size()} vectors")

    # 5. BM25
    logger.info("Step 5/5  Building BM25 index …")
    build_bm25_index(chunks)

    elapsed = time.time() - t0
    logger.success(f"Ingestion complete in {elapsed:.1f}s")
    logger.info(f"  Documents: {len(documents)}")
    logger.info(f"  Chunks:    {len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AskDocs ingestion pipeline")
    parser.add_argument("--max-pages", type=int, default=None, help="Cap pages (for testing)")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection")
    args = parser.parse_args()

    run(max_pages=args.max_pages, recreate=args.recreate)
