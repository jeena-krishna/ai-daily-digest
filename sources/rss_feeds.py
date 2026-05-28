"""
sources/rss_feeds.py

Fetches recent posts from AI-focused blogs via their RSS/Atom feeds.

RSS (Really Simple Syndication) is a standard XML format that websites use
to publish a feed of their latest content. The `feedparser` library handles
parsing the XML and normalizing differences between RSS and Atom formats.

Library docs: https://feedparser.readthedocs.io/
"""

import feedparser  # Third-party library: pip install feedparser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many recent posts to retrieve per feed.
# Feeds often contain 20–50 entries; we take only the freshest few.
POSTS_PER_FEED = 5

# A dict mapping human-readable blog names to their RSS feed URLs.
# Keeping it as a dict makes it easy to add/remove feeds later.
RSS_FEEDS = {
    "OpenAI Blog":       "https://openai.com/blog/rss.xml",
    "Google AI Blog":    "https://blog.google/technology/ai/rss",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    # The Verge's dedicated AI section — reliable feed with broad industry coverage
    "The Verge AI":      "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    # TechCrunch AI category — consistent coverage of AI startups and research
    "TechCrunch AI":     "https://techcrunch.com/category/artificial-intelligence/feed",
}


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

def fetch_feed(blog_name: str, feed_url: str) -> list[dict]:
    """
    Parses a single RSS/Atom feed and returns the most recent posts.

    `feedparser.parse()` downloads and parses the feed in one call.
    It's resilient: it handles malformed XML, redirects, and both RSS and Atom formats.

    Args:
        blog_name: A human-readable label for the blog (used as the "source" field).
        feed_url:  The URL of the RSS or Atom feed.

    Returns:
        A list of dicts (up to POSTS_PER_FEED entries), or an empty list on error.
    """
    print(f"  [RSS] Fetching: {blog_name}...")

    # `feedparser.parse()` is the main entry point of the library.
    # It returns a FeedParserDict with attributes like:
    #   - feed.parse_result.entries  — list of post objects
    #   - feed.bozo                  — True if the feed had parse errors
    #   - feed.status                — HTTP status code (if available)
    parsed = feedparser.parse(feed_url)

    # `parsed.bozo` is True when feedparser encountered a malformed feed.
    # Many feeds are slightly broken but still parseable — we log a warning but continue.
    if parsed.bozo:
        # `parsed.bozo_exception` contains details about what went wrong
        print(f"  [RSS] WARNING: {blog_name} feed had parse issues: {parsed.bozo_exception}")

    posts = []

    # `parsed.entries` is the list of individual posts in the feed.
    # We slice it with [:POSTS_PER_FEED] to get only the most recent N entries.
    # Feeds are typically ordered newest-first, but this is not guaranteed.
    for entry in parsed.entries[:POSTS_PER_FEED]:
        # `entry.get("title", "")` safely retrieves the post title.
        # feedparser normalizes field names across RSS and Atom formats,
        # so "title" works for both.
        title = entry.get("title", "").strip()

        # `entry.get("link", "")` is the URL of the full blog post.
        url = entry.get("link", "").strip()

        # Skip entries that are missing essential fields
        if not title or not url:
            continue

        # Try to get a description/summary — the field name varies by feed format:
        #   - RSS 2.0 uses "summary"
        #   - Atom uses "summary" or "content"
        #   - Some feeds use "description"
        # feedparser normalizes these into "summary" when possible.
        description = entry.get("summary") or entry.get("description") or ""

        # Descriptions from RSS feeds often contain raw HTML tags (e.g., <p>, <a>).
        # For now we store the raw string; we can strip HTML later if needed.
        # Truncate to keep things manageable.
        if len(description) > 200:
            description = description[:200] + "..."

        posts.append({
            "title": title,
            "url": url,
            "score": 0,             # RSS feeds don't expose engagement metrics
            "source": blog_name,    # The human-readable name we defined in RSS_FEEDS
            "description": description,
        })

    print(f"  [RSS] Got {len(posts)} posts from {blog_name}.")
    return posts


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_rss_articles() -> list[dict]:
    """
    Iterates over all defined RSS feeds and returns a combined list of posts.

    Each failed individual feed is skipped with a warning — we don't let one
    broken feed stop us from collecting the rest.

    Returns a flat list of all collected dicts from all feeds.
    """
    print("[RSS Feeds] Fetching all blog feeds...")

    all_posts = []

    # `.items()` unpacks the dict into (key, value) pairs — here: (name, url)
    for blog_name, feed_url in RSS_FEEDS.items():
        try:
            posts = fetch_feed(blog_name, feed_url)
            # `extend` appends all items from `posts` into `all_posts`
            # (unlike `append`, which would add the list itself as a single element)
            all_posts.extend(posts)
        except Exception as e:
            # Catch-all: if fetching or parsing crashes entirely, log and move on.
            # This ensures one broken feed doesn't stop the others.
            print(f"  [RSS] ERROR fetching {blog_name}: {e}")

    print(f"[RSS Feeds] Total posts collected: {len(all_posts)}")
    return all_posts


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    articles = get_rss_articles()

    print("\n--- Recent Posts from AI Blogs (RSS) ---\n")
    for i, article in enumerate(articles, start=1):
        print(f"{i:>3}. [{article['source']}] {article['title']}")
        print(f"       {article['url']}")
        if article["description"]:
            print(f"       {article['description'][:100]}...")
        print()

    print(f"Total: {len(articles)} posts fetched across all feeds.")
