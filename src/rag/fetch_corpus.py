"""Download Wikipedia articles for the RAG corpus (Phase 5).

Fetches the full plain-text extract (prop=extracts&explaintext=true) of each
mathematician's Wikipedia article via the MediaWiki Action API -- not the REST
page/summary endpoint, which only returns the lead paragraph. The Action API
extract is the whole article body as clean prose: section headers as plain
text, no wiki markup, no citation markers. This gives retriever.py real
multi-paragraph documents to chunk and index instead of a one-line summary.

Writes one file per mathematician to data/corpus/ (e.g. gauss.txt), named via
the MATHEMATICIANS list below rather than auto-slugified from the title, so
names like "John von Neumann" get a predictable, glob-friendly filename.
"""

from pathlib import Path

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "rag-hallucination-detector/0.1 (research project; contact: nanaysonic@gmail.com)"
CORPUS_DIR = Path("data/corpus")
MATHEMATICIANS = [
    ("Carl Friedrich Gauss", "gauss"),
    ("Leonhard Euler", "euler"),
    ("Bernhard Riemann", "riemann"),
    ("John von Neumann", "von_neumann"),
    ("David Hilbert", "hilbert"),
]


def fetch_article_text(title: str) -> str:
    """Fetch the full plain-text extract for a Wikipedia article title."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "formatversion": 2,
    }
    resp = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    page = resp.json()["query"]["pages"][0]
    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title!r}")
    return page["extract"]


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for title, stem in MATHEMATICIANS:
        text = fetch_article_text(title)
        out_path = CORPUS_DIR / f"{stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"{stem}.txt: {len(text):,} characters")


if __name__ == "__main__":
    main()
