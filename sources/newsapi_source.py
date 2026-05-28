"""
sources/newsapi_source.py

Fetches recent AI-related news articles from NewsAPI.org.

NewsAPI is a service that aggregates headlines from thousands of publishers.
The free tier allows up to 100 requests/day and returns up to 100 articles per request.

API docs: https://newsapi.org/docs/endpoints/everything
"""

import os           # Built-in module to access environment variables
import requests     # Third-party library for making HTTP requests

# `load_dotenv()` reads the `.env` file in the project root and loads each
# KEY=VALUE pair as an environment variable, making it accessible via os.getenv().
# This way, secrets never need to be hard-coded in source files.
from dotenv import load_dotenv

# Load the .env file. Call this before any os.getenv() calls.
# If there is no .env file (e.g., in CI/CD), it silently does nothing —
# environment variables set externally (e.g., by the shell or Docker) still work.
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The NewsAPI base URL for the "everything" endpoint.
# This endpoint searches across all articles, not just top headlines.
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# The search query sent to NewsAPI.
# The OR operator broadens the search: an article matches if ANY of these terms appear.
SEARCH_QUERY = "artificial intelligence OR LLM OR OpenAI OR Anthropic OR machine learning"

# How many articles to request per API call.
# The free NewsAPI tier allows up to 100 per request.
PAGE_SIZE = 20


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_ai_news() -> list[dict]:
    """
    Calls the NewsAPI /v2/everything endpoint and returns a list of AI-related articles.

    Each returned dict has:
        - title       (str): Headline of the article
        - url         (str): Full link to the article
        - score       (int): Set to 0 — NewsAPI doesn't provide upvote/engagement scores
        - source      (str): The publication name (e.g., "TechCrunch", "Wired")
        - description (str): A short summary/excerpt of the article

    Returns an empty list if the API key is missing or the request fails.
    """

    # Retrieve the API key from environment variables.
    # This will be None if NEWSAPI_KEY is not set in .env or the shell environment.
    api_key = os.getenv("NEWSAPI_KEY")

    # Guard clause: if no API key is found, warn the user and return early.
    # Continuing without a key would just result in a 401 Unauthorized error anyway.
    if not api_key:
        print("[NewsAPI] WARNING: NEWSAPI_KEY not found in environment. Skipping.")
        return []

    # These are the query parameters appended to the URL as ?key=value&key=value...
    # The `requests` library handles URL-encoding them automatically.
    params = {
        "q": SEARCH_QUERY,          # The search query
        "language": "en",           # Only English articles
        "sortBy": "publishedAt",    # Most recent articles first
        "pageSize": PAGE_SIZE,      # Number of results to return
        "apiKey": api_key,          # Authentication key
    }

    print("[NewsAPI] Fetching articles...")

    # Make the GET request to the API with our parameters
    response = requests.get(NEWSAPI_URL, params=params)

    # Raise an exception if the server returned an error status code (4xx or 5xx)
    response.raise_for_status()

    # Parse the JSON response body into a Python dict.
    # NewsAPI wraps results in: { "status": "ok", "totalResults": N, "articles": [...] }
    data = response.json()

    # Safely get the "articles" list; default to empty list if key is missing
    raw_articles = data.get("articles", [])

    articles = []
    for article in raw_articles:
        # Skip articles where title or URL is missing — they're not usable
        title = article.get("title", "")
        url = article.get("url", "")
        if not title or not url:
            continue

        # `article.get("source", {})` retrieves the nested source dict.
        # `.get("name", "NewsAPI")` then retrieves the publication name from it,
        # defaulting to "NewsAPI" if neither the dict nor the name key exist.
        source_name = article.get("source", {}).get("name", "NewsAPI")

        # Description can be None in the API response — use empty string as fallback
        description = article.get("description") or ""

        articles.append({
            "title": title,
            "url": url,
            "score": 0,             # NewsAPI doesn't provide engagement metrics
            "source": source_name,
            "description": description,
        })

    print(f"[NewsAPI] Retrieved {len(articles)} articles.")
    return articles


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    articles = get_ai_news()

    print("\n--- AI News from NewsAPI ---\n")
    for i, article in enumerate(articles, start=1):
        print(f"{i:>3}. [{article['source']}] {article['title']}")
        print(f"       {article['url']}")
        if article["description"]:
            # Print just the first 100 chars of the description to keep output clean
            print(f"       {article['description'][:100]}...")
        print()

    print(f"Total: {len(articles)} articles fetched.")
