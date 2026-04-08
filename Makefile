.PHONY: install ingest ingest-quick serve test eval lint format ask

install:
	pip install -e ".[dev]"

## Build the full index (~300 LangChain doc pages, takes ~10 min on first run)
ingest:
	python -m scripts.ingest

## Quick smoke-test index (30 pages only)
ingest-quick:
	python -m scripts.ingest --max-pages 30

## Start the FastAPI server (hot-reload)
serve:
	uvicorn src.api:app --reload --port 8000

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Run RAGAS evaluation + CI quality gate
eval:
	python -m src.evaluation.evaluator

## Lint with ruff
lint:
	ruff check src/ tests/ scripts/

## Auto-format with ruff
format:
	ruff format src/ tests/ scripts/

## Interactive Q&A CLI
ask:
	python -m scripts.ask
