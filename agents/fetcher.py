import time
from sources.aggregator import fetch_all_news
from memory.article_history import deduplicate_from_history

class FetcherAgent:
    """
    FetcherAgent fetches raw articles from Hacker News, NewsAPI, arXiv, and RSS Feeds.
    It then filters out any articles that have already been sent in previous newsletters.
    """
    def __init__(self):
        pass

    def run(self) -> dict:
        """
        Executes the article fetching and deduplication pipeline.
        
        Returns:
            A dictionary containing:
                "raw_articles": A list of all fetched articles.
                "total_fetched": Total raw articles fetched.
                "duplicates_removed": Count of articles removed.
                "unique_articles": A list of unique, deduplicated articles.
        """
        print("[FetcherAgent] Ingesting articles from sources...")
        start_time = time.time()
        
        # 1. Fetch raw articles from all configured sources
        try:
            raw_articles = fetch_all_news()
        except Exception as e:
            print(f"[FetcherAgent] Error during news aggregation: {e}")
            raise e
            
        total_fetched = len(raw_articles)
        print(f"[FetcherAgent] Ingested {total_fetched} raw articles.")
        
        # 2. Run deduplication using JSON history
        print("[FetcherAgent] Deduplicating articles using JSON history file...")
        unique_articles = deduplicate_from_history(raw_articles)
        
        duplicates_removed = total_fetched - len(unique_articles)
        duration = time.time() - start_time
        
        print(f"[FetcherAgent] ✓ Completed in {duration:.2f}s.")
        print(f"[FetcherAgent] Summary: Ingested={total_fetched}, Duplicates={duplicates_removed}, Unique={len(unique_articles)} remaining.")
        
        return {
            "raw_articles": raw_articles,
            "total_fetched": total_fetched,
            "duplicates_removed": duplicates_removed,
            "unique_articles": unique_articles
        }
