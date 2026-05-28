"""
sources/aggregator.py

The central hub that calls all news sources and merges their results.

This module's job is simple:
  1. Call each source's fetch function
  2. Combine all results into one flat list
  3. Handle any errors gracefully so one failing source doesn't crash everything

By keeping aggregation logic here (separate from the sources themselves),
each source module stays focused on one thing, and we can easily add or
remove sources just by editing this file.
"""

import logging  
from .hackernews import get_ai_stories
from .newsapi_source import get_ai_news
from .arxiv_source import get_arxiv_papers
from .rss_feeds import get_rss_articles

logger = logging.getLogger(__name__)

SOURCES = [
    ("Hacker News",  get_ai_stories),
    ("NewsAPI",      get_ai_news),
    ("ArXiv",        get_arxiv_papers),
    ("RSS Feeds",    get_rss_articles),
]


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

def fetch_all_news() -> list[dict]:
    """
    Calls all registered news sources and returns a combined list of articles.

    Error handling strategy:
        - Each source is called inside a try/except block.
        - If a source raises any exception (network error, API key missing,
          parsing failure, etc.), the error is logged and we move on.
        - This ensures a broken source never prevents the digest from running.

    Returns:
        A flat list of article dicts from all sources that succeeded.
        Each dict has at minimum: title, url, score, source.
        Some also include: description.
    """
    all_articles = []  # Accumulator: we'll extend this with each source's results

    for source_name, fetch_fn in SOURCES:
        try:
            print(f"\n[Aggregator] Fetching from: {source_name}")

            # Call the source's fetch function — each returns a list[dict]
            articles = fetch_fn()

            # `extend()` adds each item from `articles` individually into `all_articles`.
            # Using `append(articles)` would add the whole list as a single nested element,
            # which is NOT what we want.
            all_articles.extend(articles)

            print(f"[Aggregator] ✓ {source_name}: {len(articles)} items")

        except Exception as e:
            # `logger.error()` logs at ERROR level with the source name and exception.
            # `exc_info=True` appends the full stack trace to the log — very helpful
            # for debugging, but not as noisy as always printing tracebacks.
            logger.error(f"[Aggregator] ✗ Failed to fetch from {source_name}: {e}", exc_info=True)
            print(f"[Aggregator] ✗ {source_name} failed: {e} — skipping.")

    return all_articles


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure basic logging so logger.error() messages show up in the console.
    # format includes the log level and message for clarity.
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

    # Group and display results by source for clarity
    # We use a dict to bucket articles: { source_name: [articles] }
    by_source: dict[str, list] = {}
    for article in articles:
        source = article.get("source", "Unknown")
        # `setdefault` creates an empty list for a key if it doesn't exist yet,
        # then returns the list so we can append to it in one line.
        by_source.setdefault(source, []).append(article)

    for source, items in by_source.items():
        print(f"--- {source} ({len(items)} items) ---")
        for i, item in enumerate(items, start=1):
            print(f"  {i}. {item['title']}")
            print(f"     {item['url']}")
        print()
