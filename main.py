import sys
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Import components from our codebase
from sources.aggregator import fetch_all_news
from agent.curator import curate_digest
from email_sender import send_digest_email
from memory.vectorstore import store_articles
from memory.article_history import deduplicate_from_history, add_to_history, load_history

def main():
    # Load environment variables from .env if present
    load_dotenv()

    print("=" * 60)
    print(f"  AI Daily Digest Master Pipeline — Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Fetch News
    # -------------------------------------------------------------------------
    print("\n[Step 1/3] Fetching articles from all sources...")
    start_fetch = time.time()
    try:
        articles = fetch_all_news()
    except Exception as e:
        print(f"[Main] ✗ Error: Fetching failed with exception: {e}")
        sys.exit(1)
        
    duration_fetch = time.time() - start_fetch
    print(f"[Main] ✓ Fetching completed in {duration_fetch:.2f}s.")
    print(f"[Main] Found total of {len(articles)} raw articles.")

    # Deduplicate articles using our JSON file memory (works on stateless GitHub Actions)
    print("[Main] Deduplicating articles using JSON history file...")
    original_count = len(articles)
    articles = deduplicate_from_history(articles)
    filtered_count = original_count - len(articles)
    print(f"[Main] Filtered {filtered_count} duplicate articles. {len(articles)} unique articles remaining.")

    if not articles:
        print("[Main] All articles filtered as duplicates. Exiting pipeline.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # STEP 2: Curate with Gemini
    # -------------------------------------------------------------------------
    print("\n[Step 2/3] Curating digest using Google Gemini...")
    start_curation = time.time()
    try:
        html_digest, curated_articles = curate_digest(articles)
    except Exception as e:
        print(f"[Main] ✗ Error: Curation failed: {e}")
        sys.exit(1)

    duration_curation = time.time() - start_curation
    print(f"[Main] ✓ Curation completed in {duration_curation:.2f}s.")
    print(f"[Main] Generated HTML digest: {len(html_digest):,} characters.")

    # -------------------------------------------------------------------------
    # SAVE COPY LOCALLY
    # -------------------------------------------------------------------------
    # Always save a local copy of the digest.
    output_path = "digest_preview.html"
    try:
        # Wrap the clean HTML snippet returned by curator in a standard html shell for local viewing
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Daily Digest Preview</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 680px; margin: 40px auto;
            padding: 0 20px; color: #1a1a1a; line-height: 1.6; }}
    h1   {{ font-size: 1.6rem; border-bottom: 2px solid #1a1a1a; padding-bottom: 8px; }}
    h2   {{ font-size: 1.2rem; color: #333; margin-top: 2rem; }}
    h3   {{ font-size: 1rem; margin-bottom: 4px; }}
    a    {{ color: #0066cc; }}
    hr   {{ border: none; border-top: 1px solid #ddd; margin: 1.5rem 0; }}
    li   {{ margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
{html_digest}
</body>
</html>""")
        print(f"[Main] Saved local preview copy to: {output_path}")
    except Exception as e:
        print(f"[Main] WARNING: Failed to save local HTML preview file. Error: {e}")

    # -------------------------------------------------------------------------
    # STEP 3: Email Delivery
    # -------------------------------------------------------------------------
    print("\n[Step 3/3] Sending digest email...")
    start_email = time.time()
    
    # Check if a custom recipient is specified in the environment
    recipient = os.getenv("DIGEST_RECIPIENT")
    if recipient:
        print(f"[Main] Using DIGEST_RECIPIENT environment variable: {recipient}")
    else:
        print("[Main] DIGEST_RECIPIENT environment variable not found. Defaulting to sender's address.")

    # We send the html_digest content as the body.
    # Note: send_digest_email wraps this in its MIME block and sends it.
    email_success = send_digest_email(html_digest, recipient_email=recipient)
    duration_email = time.time() - start_email
    
    if email_success:
        print(f"[Main] ✓ Email sent successfully in {duration_email:.2f}s.")
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Store in persistent JSON history (primary storage, version controlled)
        try:
            print(f"[Main] Recording today's {len(curated_articles)} curated articles in JSON history...")
            history = load_history()
            add_to_history(curated_articles, history)
            print("[Main] ✓ Recorded in JSON history successfully.")
        except Exception as e:
            print(f"[Main] WARNING: Failed to record curated articles in JSON history: {e}")

        # 2. Store in local vector store memory (for semantic search & RAG queries)
        try:
            print(f"[Main] Storing today's {len(curated_articles)} curated articles in vector store memory...")
            store_articles(today_str, curated_articles)
            print("[Main] ✓ Vector store memory updated successfully.")
        except Exception as e:
            print(f"[Main] WARNING: Failed to store curated articles in vector store: {e}")
    else:
        print(f"[Main] ✗ Error: Email delivery failed (took {duration_email:.2f}s). Local copy is preserved at '{output_path}'.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Pipeline Summary
    # -------------------------------------------------------------------------
    total_duration = duration_fetch + duration_curation + duration_email
    print("\n" + "=" * 60)
    print("  PIPELINE RUN SUMMARY")
    print("=" * 60)
    print(f"  - Fetching step:   {duration_fetch:.2f}s")
    print(f"  - Curation step:   {duration_curation:.2f}s")
    print(f"  - Email step:      {duration_email:.2f}s")
    print(f"  - Total Duration:  {total_duration:.2f}s")
    print(f"  - Status:          Success")
    print("=" * 60)

if __name__ == "__main__":
    main()
