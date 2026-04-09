"""
Interactive CLI — test the full pipeline from your terminal.

Usage:
    python -m scripts.ask
    python -m scripts.ask --hyde          # use HyDE query expansion
    python -m scripts.ask --expand        # use multi-query expansion
    python -m scripts.ask --debug         # show retrieval details
"""

from __future__ import annotations

import argparse

from loguru import logger

from src.generation.generator import generate_answer
from src.retrieval.hybrid_retriever import retrieve
from src.retrieval.query_expansion import expand_query, hyde_embed
from src.ingestion.embedder import embed_query
from src.ingestion.vector_store import vector_search
from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.retrieval.hybrid_retriever import rerank
from src.config import settings


def ask(query: str, use_hyde: bool = False, use_expand: bool = False, debug: bool = False):
    print(f"\n{'─' * 60}")
    print(f"Q: {query}")
    print(f"{'─' * 60}")

    # --- Retrieval with optional expansion ---
    if use_expand:
        queries = expand_query(query, n=3)
        print(f"[Expanded to {len(queries)} queries]")
        all_bm25, all_vec = [], []
        for q in queries:
            all_bm25 += bm25_search(q, top_k=settings.bm25_top_k)
            qvec = hyde_embed(q) if use_hyde else embed_query(q)
            all_vec += vector_search(qvec, top_k=settings.vector_top_k)
        fused = reciprocal_rank_fusion(all_bm25, all_vec)
        chunks = rerank(query, candidates=fused, top_k=settings.rerank_top_k)
    elif use_hyde:
        print("[Using HyDE embedding]")
        qvec = hyde_embed(query)
        bm25_r = bm25_search(query, top_k=settings.bm25_top_k)
        vec_r = vector_search(qvec, top_k=settings.vector_top_k)
        fused = reciprocal_rank_fusion(bm25_r, vec_r)
        chunks = rerank(query, candidates=fused, top_k=settings.rerank_top_k)
    else:
        chunks = retrieve(query)

    if debug:
        print(f"\n[Retrieved {len(chunks)} chunks after reranking]")
        for i, rc in enumerate(chunks, 1):
            print(f"  [{i}] score={rc.rerank_score:.4f}  {rc.chunk.title[:50]}")
            print(f"       {rc.chunk.content[:100].strip()} …")

    # --- Generation ---
    result = generate_answer(query, chunks)

    print(f"\nA: {result.answer}\n")

    if result.citations:
        print("Sources:")
        for c in result.citations:
            print(f"  [{c.index}] {c.title}")
            print(f"       {c.source}")


def main():
    parser = argparse.ArgumentParser(description="Ask AskDocs a question")
    parser.add_argument("--hyde", action="store_true", help="Use HyDE query expansion")
    parser.add_argument("--expand", action="store_true", help="Use multi-query expansion")
    parser.add_argument("--debug", action="store_true", help="Show retrieval debug info")
    args = parser.parse_args()

    print("AskDocs — LangChain documentation assistant")
    print("Type 'exit' or Ctrl+C to quit\n")

    while True:
        try:
            query = input("Ask: ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                break
            ask(query, use_hyde=args.hyde, use_expand=args.expand, debug=args.debug)
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as exc:
            logger.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
