"""
Query expansion via HyDE (Hypothetical Document Embeddings).

The problem with vanilla dense retrieval:
  Short queries like "LCEL streaming" don't embed the same way as the
  long, detailed documentation passages that answer them.

HyDE solution (Gao et al., 2022):
  1. Ask the LLM to generate a *hypothetical* answer to the query
  2. Embed the hypothetical answer instead of the raw query
  3. Use that richer vector to find real matching documents

Why it works:
  The hypothetical answer is written in the same "style" as documentation,
  so it lands closer to real docs in embedding space than a terse query would.

We also support multi-query expansion: generate N paraphrase variants,
retrieve for each, then union the results before RRF fusion.
"""
from __future__ import annotations

from groq import Groq
from loguru import logger

from src.config import settings
from src.ingestion.embedder import embed_texts

_groq: Groq | None = None


def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=settings.groq_api_key)
    return _groq


# ── HyDE ─────────────────────────────────────────────────────────────────────

HYDE_PROMPT = """Write a short, technical passage (3-5 sentences) from documentation 
that would directly answer this question. Be specific and use technical terminology.
Do not say "this document" or "this passage". Just write the content directly.

Question: {query}"""


def hyde_embed(query: str) -> list[float]:
    """
    Generate a hypothetical document for the query, then embed it.
    Falls back to embedding the original query on LLM failure.
    """
    try:
        client = _get_groq()
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            temperature=0.7,
            max_tokens=200,
        )
        hypothetical = resp.choices[0].message.content.strip()
        logger.debug(f"HyDE generated: {hypothetical[:80]} …")
        return embed_texts([hypothetical], show_progress=False)[0].tolist()
    except Exception as exc:
        logger.warning(f"HyDE failed ({exc}), falling back to raw query embedding")
        return embed_texts([query], show_progress=False)[0].tolist()


# ── Multi-query expansion ─────────────────────────────────────────────────────

MULTI_QUERY_PROMPT = """Generate {n} different phrasings of this question for document retrieval.
Each phrasing should use different vocabulary but ask for the same information.
Output only the questions, one per line, no numbering.

Question: {query}"""


def expand_query(query: str, n: int = 3) -> list[str]:
    """
    Generate N paraphrase variants of the query.
    Returns the original + N variants.
    """
    try:
        client = _get_groq()
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query, n=n)}],
            temperature=0.8,
            max_tokens=150,
        )
        variants = [
            line.strip()
            for line in resp.choices[0].message.content.strip().splitlines()
            if line.strip() and len(line.strip()) > 10
        ][:n]
        all_queries = [query] + variants
        logger.debug(f"Expanded query into {len(all_queries)} variants")
        return all_queries
    except Exception as exc:
        logger.warning(f"Query expansion failed ({exc}), using original query only")
        return [query]
