"""
sources/__main__.py

This file lets you run the `sources` package directly from the command line:

    python -m sources

When Python sees `python -m sources`, it looks for either:
  1. sources/__main__.py  ← this file (used when sources/ is a package directory)
  2. sources.py           ← if sources were a single file module

Without this file, `python -m sources` would fail with:
  "No module named sources.__main__; 'sources' is a package and cannot be directly executed"

We simply import and run the aggregator — the most useful entry point for the package.
"""

# Import the main aggregation function from aggregator.py (relative import within package)
from .aggregator import fetch_all_news
import logging

if __name__ == "__main__":
    # Set up basic logging so any logger.error() calls in the aggregator are visible
    logging.basicConfig(
        level=logging.ERROR,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 60)
    print("  AI Daily Digest — Aggregating all sources")
    print("=" * 60)

    articles = fetch_all_news()

    print("\n" + "=" * 60)
    print(f"  TOTAL ARTICLES COLLECTED: {len(articles)}")
    print("=" * 60 + "\n")

    # Group results by source for a readable summary
    by_source: dict[str, list] = {}
    for article in articles:
        source = article.get("source", "Unknown")
        by_source.setdefault(source, []).append(article)

    for source, items in by_source.items():
        print(f"--- {source} ({len(items)} items) ---")
        for i, item in enumerate(items, start=1):
            print(f"  {i}. {item['title']}")
            print(f"     {item['url']}")
        print()
