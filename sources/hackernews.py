"""
sources/hackernews.py

Fetches the top stories from Hacker News and filters them for AI-related content.

Hacker News provides a public Firebase REST API (no authentication needed):
  - Top story IDs:  https://hacker-news.firebaseio.com/v0/topstories.json
  - Single story:   https://hacker-news.firebaseio.com/v0/item/{id}.json

The API returns raw JSON, which Python's `requests` library makes easy to work with.
"""

import requests  
# The base URL for the Hacker News Firebase API.
# All endpoints are built by appending a path to this base.
HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"

# How many of the top stories to fetch from Hacker News.
# The API returns up to 500 top story IDs, but we only need the top 100.
TOP_N = 100

# A list of keywords we use to decide if a story is AI-related.
# We check whether any of these words appear in the story title (case-insensitive).
AI_KEYWORDS = [
    "ai",
    "llm",
    "gpt",
    "claude",
    "openai",
    "anthropic",
    "machine learning",
    "neural",
    "transformer",
    "agent",
    "deep learning",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fetch_top_story_ids() -> list[int]:
    """
    Fetches the list of top story IDs from Hacker News.

    Returns a Python list of integers, e.g. [38291234, 38291099, ...]
    The list is already ordered by "top" ranking (best first).

    We slice it to only take the first TOP_N IDs so we don't fetch
    thousands of individual stories unnecessarily.
    """
    # Build the full URL for the top stories endpoint
    url = f"{HN_BASE_URL}/topstories.json"

    # Make an HTTP GET request to the URL.
    # `requests.get()` sends the request and waits for the response.
    response = requests.get(url)

    # `.raise_for_status()` will throw an error if the HTTP status code
    # indicates a problem (e.g., 404 Not Found, 500 Server Error).
    # If status is 200 OK, it does nothing.
    response.raise_for_status()

    # `.json()` parses the response body (a JSON string) into a Python object.
    # For this endpoint, that's a list of integers (story IDs).
    all_ids = response.json()

    # Return only the first TOP_N IDs — no need to fetch all 500
    return all_ids[:TOP_N]


def fetch_story(story_id: int) -> dict | None:
    """
    Fetches the details for a single Hacker News story by its ID.

    Returns a dict with fields like: id, title, url, score, by, time, type
    Returns None if the request fails or the response is empty.

    Example response from the API:
    {
        "id": 38291234,
        "title": "OpenAI releases GPT-5",
        "url": "https://openai.com/...",
        "score": 842,
        "by": "someuser",
        "time": 1716820000,
        "type": "story"
    }
    """
    # Build the URL for this specific story's item endpoint
    url = f"{HN_BASE_URL}/item/{story_id}.json"

    response = requests.get(url)
    response.raise_for_status()

    # Parse the JSON response into a Python dict
    story = response.json()

    # Occasionally the API returns `null` (None in Python) for deleted/removed items.
    # We return None so the caller knows to skip this story.
    return story


def is_ai_related(title: str) -> bool:
    """
    Returns True if the story title contains any of our AI keywords.

    We convert the title to lowercase before checking so that
    "GPT-4" matches the keyword "gpt", and "Machine Learning" matches "machine learning".

    Args:
        title: The title string of a Hacker News story.
    """
    # Lowercase once so all keyword checks are case-insensitive
    title_lower = title.lower()

    # `any()` returns True as soon as one keyword is found — it short-circuits,
    # meaning it stops checking after the first match (efficient).
    return any(keyword in title_lower for keyword in AI_KEYWORDS)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_ai_stories() -> list[dict]:
    """
    Fetches the top 100 Hacker News stories, filters for AI-related ones,
    and returns them sorted by score (highest first).

    Each returned dict has:
        - title  (str):  The story headline
        - url    (str):  Link to the article (falls back to HN discussion page if no URL)
        - score  (int):  Upvote count on Hacker News
        - source (str):  Always "Hacker News" — useful when combining multiple sources later

    Returns:
        A list of dicts, sorted descending by score.
    """
    print("Fetching top story IDs from Hacker News...")
    story_ids = fetch_top_story_ids()
    print(f"  → Got {len(story_ids)} story IDs. Fetching details...")

    ai_stories = []  # We'll collect matching stories here

    for i, story_id in enumerate(story_ids):
        # Fetch the full details for each story one by one.
        # (A future improvement: fetch these in parallel using threading or asyncio.)
        story = fetch_story(story_id)

        # Skip stories that came back as None (deleted/missing items)
        if story is None:
            continue

        # Some HN items are "Ask HN", "Show HN", or job posts — these may not
        # have a title. We skip anything without a title.
        title = story.get("title", "")
        if not title:
            continue

        # Check if this story's title contains any AI-related keywords
        if is_ai_related(title):
            # Build the URL: prefer the external article link, but fall back
            # to the HN discussion page (useful for "Ask HN" posts that have no URL).
            url = story.get("url") or f"https://news.ycombinator.com/item?id={story_id}"

            # `story.get("score", 0)` safely retrieves the score,
            # defaulting to 0 if the field is missing.
            score = story.get("score", 0)

            # Append a clean, normalized dict — consistent structure makes it
            # easy to combine results from different sources later.
            ai_stories.append({
                "title": title,
                "url": url,
                "score": score,
                "source": "Hacker News",
            })

        # Progress indicator so we can see the script is working
        # `end="\r"` overwrites the same line each time (a simple progress ticker)
        print(f"  → Processed {i + 1}/{len(story_ids)} stories...", end="\r")

    # Move to a new line after the progress ticker finishes
    print()

    # Sort stories by score, highest first.
    # `key=lambda s: s["score"]` tells Python how to extract the value to sort by.
    # `reverse=True` means descending order (highest score first).
    ai_stories.sort(key=lambda s: s["score"], reverse=True)

    print(f"  → Found {len(ai_stories)} AI-related stories.")
    return ai_stories


# ---------------------------------------------------------------------------
# Test / manual run block
# ---------------------------------------------------------------------------

# This block only runs when you execute this file directly:
#   python sources/hackernews.py
#
# It does NOT run when another file imports this module, which means
# importing `get_ai_stories` elsewhere won't trigger this test code.
if __name__ == "__main__":
    stories = get_ai_stories()

    print("\n--- AI Stories from Hacker News ---\n")

    # Enumerate gives us (index, item) pairs starting at 1
    for i, story in enumerate(stories, start=1):
        print(f"{i:>3}. [{story['score']:>4} pts] {story['title']}")
        print(f"       {story['url']}")
        print()

    print(f"Total: {len(stories)} AI-related stories found in top {TOP_N} HN posts.")
