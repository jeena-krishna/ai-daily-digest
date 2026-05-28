"""
sources/arxiv_source.py

Fetches the most recent academic papers from ArXiv in AI and ML categories.

ArXiv is a free, open-access repository of scientific papers ("preprints") —
papers that are publicly shared before or alongside peer review.

The `arxiv` Python library wraps ArXiv's official API, handling pagination,
rate limiting, and response parsing automatically.

Library docs: https://lukasschwab.me/arxiv.py/index.html
ArXiv API docs: https://info.arxiv.org/help/api/index.html
"""

import arxiv  # Third-party library: pip install arxiv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many recent papers to retrieve per category
MAX_RESULTS = 20

# ArXiv category codes we want to search.
# cs.AI = Computer Science - Artificial Intelligence
# cs.LG = Computer Science - Machine Learning
# The pipe "|" between them is ArXiv's OR operator, so we get papers from either category.
CATEGORIES = "cat:cs.AI OR cat:cs.LG"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_arxiv_papers() -> list[dict]:
    """
    Fetches the most recent papers from ArXiv's cs.AI and cs.LG categories.

    Uses the `arxiv` library to build and send the API query, then returns
    a normalized list of dicts compatible with our other news sources.

    Each returned dict has:
        - title       (str): The paper title
        - url         (str): Direct link to the paper's abstract page on arxiv.org
        - score       (int): Set to 0 — ArXiv doesn't have upvotes
        - source      (str): Always "ArXiv"
        - description (str): The paper's abstract, truncated to 200 characters

    Returns an empty list if the request fails.
    """
    print("[ArXiv] Fetching recent papers...")

    # `arxiv.Search` describes what we want to query — it does NOT fetch anything yet.
    #   query       — a search string (ArXiv uses Lucene-style query syntax)
    #   max_results — caps how many papers to return
    #   sort_by     — arxiv.SortCriterion.SubmittedDate sorts newest-first
    search = arxiv.Search(
        query=CATEGORIES,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # Most recently submitted first
    )

    # As of arxiv library v4.0, results must be fetched through a Client instance.
    # `arxiv.Client()` manages HTTP connections, rate limiting, and retries.
    #
    # delay_seconds=3 — waits 3 seconds between API requests. ArXiv's terms of service
    #   ask that automated clients wait at least 3 seconds between requests to avoid
    #   overloading their servers. Skipping this risks getting rate-limited or banned.
    # num_retries=3   — if a request fails (e.g., a transient network error or timeout),
    #   the client will automatically retry up to 3 times before raising an exception.
    client = arxiv.Client(delay_seconds=3, num_retries=3)

    papers = []

    # `client.results(search)` returns a lazy generator of `arxiv.Result` objects.
    # Iterating it fetches pages from the API on demand.
    for paper in client.results(search):
        # `paper.title` is the full paper title (may contain LaTeX notation)
        title = paper.title

        # `paper.entry_id` is the canonical ArXiv URL for this paper, e.g.:
        #   https://arxiv.org/abs/2405.12345
        # This links to the abstract page, which is more reader-friendly than the PDF.
        url = paper.entry_id

        # `paper.summary` is the abstract — often several paragraphs long.
        # We truncate it to 200 characters and add "..." to signal it's cut off.
        # The `or ""` handles the rare case where summary is None.
        abstract = paper.summary or ""
        description = abstract[:200] + "..." if len(abstract) > 200 else abstract

        papers.append({
            "title": title,
            "url": url,
            "score": 0,         # ArXiv has no upvote or engagement metric
            "source": "ArXiv",
            "description": description,
        })

    print(f"[ArXiv] Retrieved {len(papers)} papers.")
    return papers


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    papers = get_arxiv_papers()

    print("\n--- Recent AI/ML Papers from ArXiv ---\n")
    for i, paper in enumerate(papers, start=1):
        print(f"{i:>3}. {paper['title']}")
        print(f"       {paper['url']}")
        # Print description with indentation for readability
        if paper["description"]:
            print(f"       Abstract: {paper['description']}")
        print()

    print(f"Total: {len(papers)} papers fetched.")
