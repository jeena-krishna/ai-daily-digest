import sys
import os
import argparse
from dotenv import load_dotenv

# Add project root to sys.path for safety
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.vectorstore import collection, search_past
from memory.article_history import load_history

# ANSI formatting codes
BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

def check_empty_database():
    """
    Checks if ChromaDB is empty. If it is, prints a friendly warning and exits.
    """
    if collection.count() == 0:
        print(f"{RED}No articles stored yet. Run python cli.py digest first to build your knowledge base.{RESET}")
        sys.exit(0)

def handle_ask(question: str):
    """
    Performs full RAG: searches ChromaDB for relevant articles,
    queries Gemini with the context, and prints answer with citations.
    """
    check_empty_database()
    
    # Load GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(f"{RED}Error: GEMINI_API_KEY environment variable is not set.{RESET}")
        sys.exit(1)

    print(f"{BLUE}{BOLD}Question:{RESET} {question}")
    print(f"{CYAN}Searching database and synthesizing answer with Gemini...{RESET}\n")

    # 1. Search past articles (grab top 5)
    results = search_past(question, n_results=5)
    
    context_blocks = []
    citations = []
    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        for doc, meta in zip(docs, metadatas):
            date_val = meta.get("date", "Unknown Date")
            source = meta.get("source", "Unknown Source")
            title = meta.get("title", "Unknown Title")
            url = meta.get("url", "")
            
            context_blocks.append(
                f"Date: {date_val}\nSource: {source}\nTitle: {title}\nURL: {url}\nContent:\n{doc}"
            )
            citations.append({
                "title": title,
                "date": date_val,
                "url": url,
                "source": source
            })

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant articles found in history."

    # 2. Call Gemini
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f"""You are a helpful research assistant for the AI Daily Digest newsletter.
Answer the user's question based on the following context containing past curated articles:

---
{context}
---

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide a concise, clear, and technically accurate answer based on the context.
2. If the context does not contain enough information, state that clearly but summarize any relevant facts from the context first, and then supplement with general knowledge if helpful.
3. Keep the tone helpful, professional, and developer-oriented.
"""
        # Try primary model and fallback model if quota exceeded
        from agent.curator import GEMINI_MODELS
        answer = None
        for i, model_name in enumerate(GEMINI_MODELS):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                answer = response.text
                break
            except Exception as model_err:
                err_str = str(model_err)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_429 and i < len(GEMINI_MODELS) - 1:
                    print(f"{YELLOW}[CLI] Model {model_name} quota exhausted, falling back...{RESET}")
                    continue
                else:
                    raise model_err

        if not answer:
            raise ValueError("No response generated from models.")
            
    except Exception as e:
        print(f"{RED}Error communicating with Gemini: {e}{RESET}")
        sys.exit(1)

    # 3. Print answer
    print(f"{GREEN}{BOLD}Answer:{RESET}")
    print(answer)
    print("\n" + "-" * 60)

    # 4. Print Citations
    if citations:
        print(f"\n{BLUE}{BOLD}Source Citations:{RESET}")
        for idx, cite in enumerate(citations, start=1):
            print(f"  [{idx}] {BOLD}{cite['title']}{RESET}")
            print(f"      Date:   {cite['date']}")
            print(f"      Source: {cite['source']}")
            print(f"      URL:    {CYAN}{cite['url']}{RESET}")
    else:
        print(f"\n{YELLOW}No citations available (no matching articles found in memory).{RESET}")

def handle_search(query: str):
    """
    Performs pure vector search over historical digests in ChromaDB.
    """
    check_empty_database()

    print(f"{BLUE}{BOLD}Searching database for:{RESET} {query}\n")

    results = search_past(query, n_results=10)

    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for idx, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), start=1):
            similarity = 1.0 - dist
            title = meta.get("title", "Untitled")
            date_val = meta.get("date", "Unknown Date")
            source = meta.get("source", "Unknown Source")
            url = meta.get("url", "")

            print(f"{GREEN}{BOLD}{idx}. {title}{RESET} (Similarity: {similarity:.4f})")
            print(f"   Date:   {date_val} | Source: {source}")
            print(f"   URL:    {CYAN}{url}{RESET}")
            print(f"   Snippet: {doc[:150].replace('\n', ' ')}...")
            print()
    else:
        print(f"{YELLOW}No matching articles found.{RESET}")

def handle_history():
    """
    Displays repository statistics reading from both JSON and ChromaDB.
    """
    # 1. Read ChromaDB
    chroma_count = collection.count()

    # 2. Read JSON History
    history = load_history()
    json_count = len(history)

    if json_count == 0 and chroma_count == 0:
        print(f"{YELLOW}No history found in database or JSON file. Run python cli.py digest first.{RESET}")
        return

    # Process JSON history metrics
    dates = []
    sources = {}
    for url, info in history.items():
        dt = info.get("date")
        if dt:
            dates.append(dt)
        src = info.get("source", "Unknown")
        sources[src] = sources.get(src, 0) + 1

    date_range = "None"
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        if min_date == max_date:
            date_range = min_date
        else:
            date_range = f"{min_date} to {max_date}"

    # Print pretty summary card
    print(f"{BLUE}{BOLD}=" * 55)
    print(f"  🤖 AI Daily Digest — System Database History")
    print(f"=" * 55 + f"{RESET}")
    print(f"  {BOLD}Total Curated Articles (JSON History):{RESET} {GREEN}{json_count}{RESET}")
    print(f"  {BOLD}Total Indexed Vectors (ChromaDB):{RESET}      {GREEN}{chroma_count}{RESET}")
    print(f"  {BOLD}Date Range of Curated Articles:{RESET}      {CYAN}{date_range}{RESET}")
    print()
    print(f"  {BOLD}Breakdown by Source (from JSON):{RESET}")
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / json_count) * 100 if json_count > 0 else 0
        bar = "█" * int(percentage / 5)
        print(f"    - {src:<18} {GREEN}{count:>4}{RESET} ({percentage:>5.1f}%)  {BLUE}{bar}{RESET}")
    print(f"{BLUE}{BOLD}=" * 55 + f"{RESET}")

def handle_digest():
    """
    Runs the full aggregator, curation, and email dispatch pipeline.
    """
    from main import main as run_pipeline
    print(f"{BLUE}{BOLD}Triggering master pipeline (fetch → curate → email)...{RESET}\n")
    run_pipeline()

def handle_topics():
    """
    Analyzes historical titles and uses Gemini to synthesize top trends/topics.
    """
    check_empty_database()

    history = load_history()
    titles = [info.get("title", "") for info in history.values() if info.get("title")]

    if not titles:
        print(f"{YELLOW}No titles found in history file to analyze.{RESET}")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(f"{RED}Error: GEMINI_API_KEY environment variable is not set.{RESET}")
        sys.exit(1)

    print(f"{BLUE}{BOLD}Analyzing {len(titles)} historical article titles using Gemini...{RESET}\n")

    # Format list and query Gemini
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        titles_text = "\n".join(f"- {t}" for t in titles)
        prompt = f"""You are a research analyst for the AI Daily Digest.
Below is a list of article titles that were curated and sent in past digests:

{titles_text}

INSTRUCTIONS:
Analyze these titles and identify the top 5-7 most common technical topics or trends.
For each topic/trend, provide:
1. A clear topic name/headline.
2. A short description (2-3 sentences) summarizing what these articles reveal about the topic (e.g., capability jumps, tool expansions, research patterns).
3. A count or mention of which key articles fall under it.

Use standard markdown formatting for the output. Keep it concise, professional, and developer-oriented. Do not write a preamble or conclusion, start directly with the list of topics.
"""
        from agent.curator import GEMINI_MODELS
        response_text = None
        for i, model_name in enumerate(GEMINI_MODELS):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                response_text = response.text
                break
            except Exception as model_err:
                err_str = str(model_err)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_429 and i < len(GEMINI_MODELS) - 1:
                    print(f"{YELLOW}[CLI] Model {model_name} quota exhausted, falling back...{RESET}")
                    continue
                else:
                    raise model_err

        if not response_text:
            raise ValueError("No response generated from models.")

        print(f"{GREEN}{BOLD}Top Historical Topics & Trends:{RESET}\n")
        print(response_text)
        
    except Exception as e:
        print(f"{RED}Error communicating with Gemini: {e}{RESET}")
        sys.exit(1)

def main_cli():
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="CLI tool for the AI Daily Digest to search and query historical digests."
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Subcommands"
    )
    subparsers.required = True

    # Subcommand: ask
    parser_ask = subparsers.add_parser("ask", help="Ask questions about past digests using RAG.")
    parser_ask.add_argument("question", type=str, help="The question to ask.")

    # Subcommand: search
    parser_search = subparsers.add_parser("search", help="Perform vector search over past articles.")
    parser_search.add_argument("query", type=str, help="The search query.")

    # Subcommand: history
    subparsers.add_parser("history", help="Show statistics and date range of historical digests.")

    # Subcommand: digest
    subparsers.add_parser("digest", help="Run the full master pipeline (fetch -> curate -> email).")

    # Subcommand: topics
    subparsers.add_parser("topics", help="Analyze and display major trends/topics from past digests.")

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if args.command == "ask":
        handle_ask(args.question)
    elif args.command == "search":
        handle_search(args.query)
    elif args.command == "history":
        handle_history()
    elif args.command == "digest":
        handle_digest()
    elif args.command == "topics":
        handle_topics()

if __name__ == "__main__":
    main_cli()
