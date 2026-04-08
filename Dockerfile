FROM python:3.11-slim

WORKDIR /app

# Build deps needed by some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

# Copy source
COPY src/ src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Data (indexes + vector DB) is mounted via docker-compose volume.
# Run `make ingest` on the host before `docker compose up`.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
