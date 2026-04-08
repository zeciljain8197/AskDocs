"""
FastAPI serving layer for the AskDocs RAG system.

Endpoints:
  POST /ask          — ask a question, get grounded answer with citations
  GET  /health       — health check
  GET  /docs         — auto-generated Swagger UI (from FastAPI)

Run with:
    uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generation.generator import generate_answer
from src.models import AnswerWithCitations
from src.retrieval.hybrid_retriever import retrieve

app = FastAPI(
    title="AskDocs",
    description="Domain-specific RAG over LangChain documentation",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5

    model_config = {"json_schema_extra": {"example": {"question": "What is LCEL?"}}}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerWithCitations)
def ask(request: AskRequest) -> AnswerWithCitations:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks = retrieve(request.question, rerank_k=request.top_k)
    result = generate_answer(request.question, chunks)
    return result
