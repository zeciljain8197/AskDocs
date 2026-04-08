"""
Debug script — confirms GitHub API access is working.
Usage: python -m scripts.debug_sitemap
"""
import requests

GITHUB_API = "https://api.github.com"
REPO       = "langchain-ai/docs"
DOCS_PATH  = "src/oss"
BRANCH     = "main"

session = requests.Session()
session.headers.update({
    "User-Agent": "AskDocs-RAG-Project/0.1 (educational)",
    "Accept":     "application/vnd.github+json",
})

url = f"{GITHUB_API}/repos/{REPO}/contents/{DOCS_PATH}?ref={BRANCH}"
print(f"Fetching: {url}\n")

resp = session.get(url, timeout=20)
print(f"Status:        {resp.status_code}")
print(f"Rate limit remaining: {resp.headers.get('x-ratelimit-remaining', 'unknown')}")
print(f"Rate limit reset:     {resp.headers.get('x-ratelimit-reset', 'unknown')}")

if resp.status_code == 200:
    items = resp.json()
    print(f"\nFound {len(items)} items in {DOCS_PATH}:")
    for item in items[:10]:
        print(f"  [{item['type']:4s}] {item['name']}")
    if len(items) > 10:
        print(f"  ... and {len(items)-10} more")
    print("\nGitHub API is working correctly.")
elif resp.status_code == 403:
    print("\nRate limited. Check x-ratelimit-reset (Unix timestamp) above.")
    print("Add a GITHUB_TOKEN to .env to get 5000 req/hr instead of 60.")
else:
    print(f"\nUnexpected response: {resp.text[:300]}")
