"""
Central configuration — all settings loaded from .env once at import time.
Every module imports from here; nothing reads os.environ directly.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    groq_api_key: str = Field(default="", description="Groq API key")
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    max_tokens: int = 1024

    # RAGAS evaluation LLM.
    # llama-3.1-8b-instant: 500k tokens/day free — enough for a full 10-sample run.
    # llama-3.3-70b-versatile: only 100k tokens/day — exhausted after ~5 samples.
    # We use 8b-instant and prevent its JSON truncation issue by capping context
    # content via ragas_context_chars (keeps the statement list short).
    ragas_llm_model: str = "llama-3.1-8b-instant"
    ragas_rerank_top_k: int = 8  # chunks retrieved+generated with (more → better recall)
    ragas_context_limit: int = 5  # number of contexts passed to RAGAS scoring
    ragas_context_chars: int = 400  # max chars per context chunk sent to RAGAS
    ragas_inter_sample_delay: float = 1.0  # seconds between samples (TPM throttle)

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Qdrant
    qdrant_path: str = str(DATA_DIR / "qdrant_db")
    qdrant_collection: str = "askdocs"

    # Retrieval
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 5

    # Evaluation thresholds
    answer_relevancy_threshold: float = 0.75
    context_recall_threshold: float = 0.65


# Singleton — import `settings` everywhere
settings = Settings()
