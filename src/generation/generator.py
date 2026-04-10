"""
Generation module — builds the prompt, calls LLM, parses citations.

LLM priority:
  1. Groq API (llama-3.1-8b-instant) — fast, free tier, requires GROQ_API_KEY
  2. Ollama (local) — fully offline fallback, requires `ollama serve` + a pulled model
  3. Stub — returns a clear error message so the rest of the pipeline still works

Citation format:
  Retrieved chunks are numbered [1], [2] … in the prompt.
  The LLM cites them inline; we parse and return structured Citation objects.
"""

from __future__ import annotations

import re

from loguru import logger

from src.config import settings
from src.models import AnswerWithCitations, Citation, RerankedChunk


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for LangChain documentation.
Answer the user's question using ONLY the provided context chunks.
Cite sources inline using [1], [2], etc. that correspond to the chunk numbers.
If the answer is not in the context, say "I don't have enough context to answer this."
Be concise and precise. Do not hallucinate."""


def _build_context_block(chunks: list[RerankedChunk]) -> str:
    lines: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Source: {rc.chunk.title} ({rc.chunk.source})\n{rc.chunk.content.strip()}"
        )
    return "\n\n---\n\n".join(lines)


def _build_user_message(query: str, context: str) -> str:
    return f"Context:\n{context}\n\n---\n\nQuestion: {query}"


# ── Citation parsing ──────────────────────────────────────────────────────────


def _parse_citations(answer: str, chunks: list[RerankedChunk]) -> list[Citation]:
    indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
    citations: list[Citation] = []
    for idx in sorted(indices):
        pos = idx - 1
        if 0 <= pos < len(chunks):
            rc = chunks[pos]
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=rc.chunk.chunk_id,
                    source=rc.chunk.source,
                    title=rc.chunk.title,
                    excerpt=rc.chunk.content[:120].strip() + " …",
                )
            )
    return citations


# ── Groq backend ──────────────────────────────────────────────────────────────


def _call_groq(system: str, user: str) -> str:
    import time
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if "rate_limit_exceeded" in str(exc) and attempt < 3:
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                logger.warning(f"Groq rate limit — retrying in {wait}s (attempt {attempt + 1}/3)")
                time.sleep(wait)
            else:
                raise


# ── Ollama backend (local fallback) ───────────────────────────────────────────

OLLAMA_MODEL = "llama3.2"  # pull with: ollama pull llama3.2


def _call_ollama(system: str, user: str) -> str:
    import requests as req

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = req.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ── Main entry point ──────────────────────────────────────────────────────────


def generate_answer(query: str, chunks: list[RerankedChunk]) -> AnswerWithCitations:
    """
    Generate a grounded answer with citations.
    Tries Groq first, falls back to Ollama if Groq fails.
    """
    if not chunks:
        return AnswerWithCitations(
            answer="I couldn't find relevant context to answer your question.",
            citations=[],
            query=query,
            model="none",
        )

    context = _build_context_block(chunks)
    user_msg = _build_user_message(query, context)
    logger.debug(f"Calling LLM with {len(chunks)} context chunks")

    answer_text = ""
    model_used = ""

    # Try Groq first
    if settings.groq_api_key:
        try:
            answer_text = _call_groq(SYSTEM_PROMPT, user_msg)
            model_used = settings.llm_model
            logger.debug(f"Groq answered ({len(answer_text)} chars)")
        except Exception as exc:
            logger.warning(f"Groq failed: {exc} — trying Ollama …")

    # Fallback: Ollama (local)
    if not answer_text:
        try:
            answer_text = _call_ollama(SYSTEM_PROMPT, user_msg)
            model_used = f"ollama/{OLLAMA_MODEL}"
            logger.debug(f"Ollama answered ({len(answer_text)} chars)")
        except Exception as exc:
            logger.error(f"Ollama also failed: {exc}")
            answer_text = (
                "⚠️  LLM unavailable. Check that GROQ_API_KEY is set in .env "
                "OR that Ollama is running (ollama serve) with llama3.2 pulled.\n\n"
                "Retrieved context chunks are shown above with --debug."
            )
            model_used = "unavailable"

    citations = _parse_citations(answer_text, chunks)
    logger.debug(f"Citations found: {len(citations)}")

    return AnswerWithCitations(
        answer=answer_text,
        citations=citations,
        query=query,
        model=model_used,
    )
