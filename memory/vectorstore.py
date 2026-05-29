import os
import chromadb
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize ChromaDB persistent client pointing to ./digest_memory
client = chromadb.PersistentClient(path="./digest_memory")

# Create or retrieve the collection with Cosine Distance configuration
# Cosine distance = 1.0 - cosine_similarity.
# In cosine space, identical vectors have distance 0.0, completely orthogonal have 1.0.
collection = client.get_or_create_collection(
    name="daily_digests",
    metadata={"hnsw:space": "cosine"}
)

def store_articles(date_str: str, articles: list[dict]) -> None:
    """
    Stores articles in ChromaDB.
    Each article's title + description is stored as the document text.
    The article URL is used as the unique ID.
    
    Args:
        date_str: The YYYY-MM-DD date when these articles were processed.
        articles: A list of article dictionaries.
    """
    if not articles:
        return

    ids = []
    documents = []
    metadatas = []

    for art in articles:
        url = art.get("url")
        if not url:
            continue  # URL is the unique ID, so we skip any articles without one

        title = art.get("title", "").strip()
        description = art.get("description") or ""
        doc_text = f"{title}\n{description}".strip()

        # Deduplicate within this insert batch to avoid multiple upserts of same URL
        if url not in ids:
            ids.append(url)
            documents.append(doc_text)
            metadatas.append({
                "date": date_str,
                "source": art.get("source", ""),
                "url": url,
                "title": title
            })

    if ids:
        # Use upsert to overwrite if the URL already exists (avoiding duplicates)
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[Vectorstore] Saved {len(ids)} articles to memory collection.")

def is_duplicate(article: dict, threshold: float = 0.85) -> bool:
    """
    Checks if a similar article exists in memory.
    
    Args:
        article: The article dictionary to check.
        threshold: Cosine similarity threshold (0.0 to 1.0).
                   Similarity is computed as: 1.0 - cosine_distance.
                   Default is 0.85.
                   
    Returns:
        True if a matching article is found with similarity >= threshold, False otherwise.
    """
    title = article.get("title", "").strip()
    description = article.get("description") or ""
    doc_text = f"{title}\n{description}".strip()

    # If the database is completely empty, it has no duplicates
    if collection.count() == 0:
        return False

    try:
        results = collection.query(
            query_texts=[doc_text],
            n_results=1
        )
    except Exception as e:
        print(f"[Vectorstore] Error querying vector store: {e}")
        return False

    if not results or "distances" not in results or not results["distances"] or not results["distances"][0]:
        return False

    # Extract cosine distance
    distance = results["distances"][0][0]
    
    # Cosine distance = 1.0 - cosine_similarity
    # So, similarity = 1.0 - distance
    similarity = 1.0 - distance

    # If similarity is above the threshold, mark it as duplicate
    if similarity >= threshold:
        matched_title = results["metadatas"][0][0].get("title", "")
        print(f"[Vectorstore] Duplicate detected: similarity={similarity:.4f} for '{title}' vs '{matched_title}'")
        return True

    return False

def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """
    Filters out articles that have already been stored in memory.
    
    Args:
        articles: A list of raw articles.
        
    Returns:
        A list of articles that are not duplicates.
    """
    unique_articles = []
    for art in articles:
        if not is_duplicate(art):
            unique_articles.append(art)
    return unique_articles

def search_past(query: str, n_results: int = 10) -> dict:
    """
    Performs a semantic search on past articles.
    
    Args:
        query: The search query.
        n_results: Maximum number of results to return.
        
    Returns:
        The standard ChromaDB query result dictionary.
    """
    total_count = collection.count()
    if total_count == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    return collection.query(
        query_texts=[query],
        n_results=min(n_results, total_count)
    )

def ask_about_news(question: str) -> str:
    """
    Performs full Retrieval-Augmented Generation (RAG):
    Searches past articles, formats context, and sends a prompt to Gemini for answering.
    
    Args:
        question: The user's query about past news.
        
    Returns:
        A synthesized textual response from Google Gemini.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    # Search past articles
    search_results = search_past(question, n_results=5)
    
    context_blocks = []
    if search_results and "documents" in search_results and search_results["documents"] and search_results["documents"][0]:
        docs = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]
        for doc, meta in zip(docs, metadatas):
            date = meta.get("date", "Unknown")
            source = meta.get("source", "Unknown")
            title = meta.get("title", "Unknown")
            url = meta.get("url", "")
            context_blocks.append(
                f"Date: {date}\n"
                f"Source: {source}\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet:\n{doc}"
            )

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant past articles found in memory."

    # Call Gemini client
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a helpful research assistant for the AI Daily Digest newsletter.
You have access to the following context containing past articles curated by the newsletter:

---
{context}
---

USER QUESTION:
{question}

INSTRUCTIONS:
1. Synthesize an answer based on the provided context.
2. Be objective, concise, and professional.
3. If the context does not provide sufficient info to answer the question, state that clearly but summarize any partially relevant articles from the context first, and then supplement with general knowledge if needed.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Daily Digest — Vector Store & RAG Demonstration")
    print("=" * 60)

    # Define 3 dummy articles
    dummy_articles = [
        {
            "title": "Google launches Gemini 2.5 Flash model",
            "url": "https://blog.google/gemini-2.5-flash",
            "source": "Google Blog",
            "description": "Google introduces Gemini 2.5 Flash, a high-performance, cost-effective model designed for high-frequency tasks and large-scale applications."
        },
        {
            "title": "ChromaDB reaches version 1.0 with persistent storage",
            "url": "https://chromadb.org/news/v1",
            "source": "ChromaDB Blog",
            "description": "ChromaDB has officially launched version 1.0, featuring robust persistent clients, cosine similarity space, and better indexing."
        },
        {
            "title": "OpenAI releases new GPT-4o model with voice features",
            "url": "https://openai.com/gpt-4o",
            "source": "OpenAI Blog",
            "description": "OpenAI announces GPT-4o, a flagship model capable of real-time audio, visual, and text processing."
        }
    ]

    print("\n[Demo 1] Storing dummy articles...")
    store_articles("2026-05-28", dummy_articles)
    print(f"Total articles in collection: {collection.count()}")

    # Define duplicates to test
    dup_article_1 = {
        "title": "Google launches Gemini 2.5 Flash model",
        "url": "https://blog.google/gemini-2.5-flash",
        "source": "Google Blog",
        "description": "Google introduces Gemini 2.5 Flash, a high-performance, cost-effective model designed for high-frequency tasks and large-scale applications."
    }

    dup_article_2 = {
        "title": "Google Releases New Gemini 2.5 Flash",
        "url": "https://news.ycombinator.com/item?id=gemini-flash-HN",
        "source": "Hacker News",
        "description": "Google's new Gemini 2.5 Flash model is out. It is optimized for speed, low cost, and supports a massive context window."
    }

    new_article = {
        "title": "Apple announces Apple Intelligence AI features for iOS",
        "url": "https://apple.com/newsroom/apple-intelligence",
        "source": "Apple Newsroom",
        "description": "Apple Intelligence brings powerful generative models to iPhone, iPad, and Mac, integrated deeply into system applications."
    }

    print("\n[Demo 2] Checking for duplicates...")
    print(f"Checking exact duplicate (same URL): {is_duplicate(dup_article_1)} (Expected: True)")
    print(f"Checking semantic duplicate (different URL, similar description): {is_duplicate(dup_article_2)} (Expected: True)")
    print(f"Checking new unrelated article: {is_duplicate(new_article)} (Expected: False)")

    print("\n[Demo 3] Testing deduplicate_articles function...")
    test_list = [dup_article_1, dup_article_2, new_article]
    deduped = deduplicate_articles(test_list)
    print(f"Original article count: {len(test_list)}")
    print(f"Deduplicated article count: {len(deduped)}")
    for i, art in enumerate(deduped, start=1):
        print(f"  {i}. {art['title']} ({art['url']})")

    print("\n[Demo 4] Searching vector store for 'fast models'...")
    search_results = search_past("fast models", n_results=2)
    if search_results and "documents" in search_results and search_results["documents"][0]:
        for i, (doc, meta) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0])):
            print(f"Match {i+1}:")
            print(f"  Title:  {meta.get('title')}")
            print(f"  Source: {meta.get('source')}")
            print(f"  URL:    {meta.get('url')}")
            print(f"  Doc:    {doc}")

    print("\n[Demo 5] Running RAG: Asking Gemini about what Google launched...")
    try:
        rag_answer = ask_about_news("What did Google release recently according to past news?")
        print(f"RAG Response:\n{rag_answer}")
    except Exception as e:
        print(f"RAG failed with error: {e}")
