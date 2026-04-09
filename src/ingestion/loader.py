"""
Document loader — downloads LangChain docs directly from GitHub as Markdown.

Why GitHub instead of scraping the website?
  - LangChain's public docs site migrated to Mintlify (a JS SPA) and blocks scrapers.
  - The API reference sitemap URL changes with each version.
  - GitHub is the authoritative source, always available, and returns clean Markdown
    with zero HTML parsing needed.

Source: https://github.com/langchain-ai/langchain/tree/master/docs/docs
  ~400 .mdx / .md files covering concepts, how-tos, integrations, and API guides.

We use the GitHub Contents API (no auth needed for public repos, 60 req/hr limit)
with disk caching so repeated runs are instant and we stay well within rate limits.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests
from loguru import logger

from src.models import Document

# ── Constants ──────────────────────────────────────────────────────────────────

GITHUB_API = "https://api.github.com"
REPO = "langchain-ai/docs"
DOCS_PATH = "src/oss/python"  # Python conceptual docs only
SKIP_PATHS = ["integrations/", "migrate/", "releases/"]  # exclude integration stubs
BRANCH = "main"
CACHE_DIR = Path("data/raw/langchain_cache")
REQUEST_DELAY = 0.5  # stay well under 60 req/hr unauthenticated
MAX_PAGES = 300


# ── GitHub API helpers ────────────────────────────────────────────────────────


def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _fetch_json(url: str, session: requests.Session) -> dict | list | None:
    """GET a GitHub API URL, cache the JSON response to disk."""
    cache = _cache_path(url)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(REQUEST_DELAY)
        return data
    except Exception as exc:
        logger.warning(f"GitHub API fetch failed [{url}]: {exc}")
        return None


def _fetch_text(url: str, session: requests.Session) -> str | None:
    """GET a raw file URL, cache the text to disk."""
    cache = _cache_path(url)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["text"]

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        text = resp.text
        cache.write_text(json.dumps({"text": text}), encoding="utf-8")
        time.sleep(REQUEST_DELAY)
        return text
    except Exception as exc:
        logger.warning(f"Raw file fetch failed [{url}]: {exc}")
        return None


def _list_md_files(
    path: str,
    session: requests.Session,
    depth: int = 0,
) -> list[dict]:
    """
    Recursively list all .md / .mdx files under a GitHub repo path.
    Returns list of GitHub tree items: {name, path, download_url, ...}
    """
    if depth > 6:
        return []

    url = f"{GITHUB_API}/repos/{REPO}/contents/{path}?ref={BRANCH}"
    data = _fetch_json(url, session)
    if not data or not isinstance(data, list):
        return []

    files: list[dict] = []
    for item in data:
        item_path = item.get("path", "")
        # Skip integration stubs, migration notes, and changelogs
        if any(skip in item_path for skip in SKIP_PATHS):
            continue
        if item.get("type") == "file" and item["name"].endswith((".md", ".mdx")):
            files.append(item)
        elif item.get("type") == "dir":
            files.extend(_list_md_files(item["path"], session, depth + 1))

    return files


# ── Markdown cleaning ─────────────────────────────────────────────────────────


def _clean_markdown(text: str) -> str:
    """
    Strip MDX/JSX components and frontmatter that don't add retrieval value.
    Keeps all actual prose, code blocks, and headers.
    """
    lines: list[str] = []
    in_frontmatter = False

    for i, line in enumerate(text.splitlines()):
        # Strip YAML frontmatter (--- ... ---)
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if line.strip() == "---":
                in_frontmatter = False
            continue

        # Skip JSX import lines and component tags
        stripped = line.strip()
        if stripped.startswith("import ") and " from " in stripped:
            continue
        if stripped.startswith("<") and stripped.endswith(">") and "/" in stripped:
            continue

        lines.append(line)

    return "\n".join(lines).strip()


# ── Public API ────────────────────────────────────────────────────────────────


def load_langchain_docs(max_pages: int | None = MAX_PAGES) -> list[Document]:
    """
    Download LangChain docs Markdown files from GitHub and return Documents.
    Disk-cached — subsequent runs are near-instant.
    """
    import os

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    headers = {
        "User-Agent": "AskDocs-RAG-Project/0.1 (educational)",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
        logger.debug("Using GITHUB_TOKEN for authenticated API requests (5000 req/hr)")
    else:
        logger.debug("No GITHUB_TOKEN — using unauthenticated GitHub API (60 req/hr)")
    session.headers.update(headers)

    logger.info(f"Listing Markdown files in github.com/{REPO}/blob/{BRANCH}/{DOCS_PATH} …")
    files = _list_md_files(DOCS_PATH, session)

    if not files:
        logger.error(
            "No Markdown files found. GitHub API may be rate-limiting you "
            "(60 req/hr unauthenticated). Wait an hour or add a GITHUB_TOKEN to .env."
        )
        return []

    logger.info(f"Found {len(files)} Markdown files")

    if max_pages:
        files = files[:max_pages]

    documents: list[Document] = []
    for i, item in enumerate(files):
        raw_url = item.get("download_url")
        if not raw_url:
            continue

        text = _fetch_text(raw_url, session)
        if not text:
            continue

        content = _clean_markdown(text)
        if len(content.strip()) < 80:
            continue

        # Skip thin provider/platform stub pages
        skip_dirs = ["/integrations/providers/", "/integrations/platforms/"]
        if any(s in item["path"] for s in skip_dirs):
            continue

        # Title: first H1 heading, or filename
        title = (
            item["name"].replace(".mdx", "").replace(".md", "").replace("-", " ").replace("_", " ")
        )
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        # Source: GitHub permalink (useful for citations)
        source = f"https://github.com/{REPO}/blob/{BRANCH}/{item['path']}"

        documents.append(Document(content=content, source=source, title=title))

        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{len(files)} files loaded …")

    logger.success(f"Loaded {len(documents)} documents from LangChain GitHub docs")
    return documents


def load_from_directory(path: str | Path) -> list[Document]:
    """Load .txt / .md files from a local directory."""
    docs: list[Document] = []
    for fp in Path(path).rglob("*"):
        if fp.suffix not in {".txt", ".md"}:
            continue
        content = fp.read_text(encoding="utf-8", errors="ignore")
        if len(content.strip()) < 50:
            continue
        docs.append(Document(content=content, source=str(fp), title=fp.stem))
    logger.info(f"Loaded {len(docs)} documents from {path}")
    return docs
