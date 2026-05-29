import os
import json
from datetime import datetime

# Path to the persistent JSON database
HISTORY_PATH = os.path.join("data", "article_history.json")

def load_history() -> dict:
    """
    Reads the JSON file and returns a dictionary of articles keyed by URL.
    Returns an empty dict if the file does not exist or is empty.
    """
    if not os.path.exists(HISTORY_PATH):
        return {}
    
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception as e:
        print(f"[History] WARNING: Failed to load history: {e}")
        return {}

def save_history(history: dict) -> None:
    """
    Writes the history dictionary back to the JSON file.
    """
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[History] ERROR: Failed to save history to {HISTORY_PATH}: {e}")

def is_already_sent(article: dict, history: dict) -> bool:
    """
    Checks if an article's URL exists in the history.
    """
    url = article.get("url")
    if not url:
        return False
    return url in history

def add_to_history(articles: list[dict], history: dict) -> None:
    """
    Adds a list of articles to the history dict with the current date,
    and then writes the updated history dict back to the JSON file.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    for art in articles:
        url = art.get("url")
        if not url:
            continue
        history[url] = {
            "title": art.get("title", ""),
            "date": today_str,
            "source": art.get("source", "")
        }
    save_history(history)

def deduplicate_from_history(articles: list[dict]) -> list[dict]:
    """
    Loads history, filters out already-sent articles, and returns unique ones.
    """
    history = load_history()
    unique_articles = []
    for art in articles:
        if not is_already_sent(art, history):
            unique_articles.append(art)
    return unique_articles
