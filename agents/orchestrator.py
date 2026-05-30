import sys
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Import components
from agents.fetcher import FetcherAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent

from email_sender import send_digest_email
from memory.vectorstore import store_articles
from memory.article_history import load_history, add_to_history

class OrchestratorAgent:
    """
    OrchestratorAgent coordinates the entire AI Daily Digest pipeline:
    FetcherAgent -> AnalystAgent -> WriterAgent -> Email Sender -> Database updates.
    It implements step-level timing, logs progress, and manages robust fallbacks.
    """
    def __init__(self):
        # Ensure environment variables are loaded
        load_dotenv()

    def run(self) -> str:
        """
        Executes the master digest pipeline.
        
        Returns:
            The HTML string of the newsletter.
        """
        print("=" * 60)
        print(f"  AI Daily Digest Master Pipeline — Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # -------------------------------------------------------------------------
        # STEP 1: Fetch and Deduplicate Articles
        # -------------------------------------------------------------------------
        print("\n[Step 1/5] Fetching and deduplicating articles...")
        start_fetch = time.time()
        fetcher = FetcherAgent()
        try:
            fetch_result = fetcher.run()
            unique_articles = fetch_result["unique_articles"]
        except Exception as e:
            print(f"[Orchestrator] ✗ Error: Fetching failed with exception: {e}")
            sys.exit(1)
            
        duration_fetch = time.time() - start_fetch
        print(f"[Orchestrator] ✓ Fetching completed in {duration_fetch:.2f}s.")

        if not unique_articles:
            print("[Orchestrator] All articles filtered as duplicates. Exiting pipeline.")
            sys.exit(0)

        # -------------------------------------------------------------------------
        # STEP 2: Editorial Selection (Analyst)
        # -------------------------------------------------------------------------
        print("\n[Step 2/5] Running Analyst selection...")
        start_analyst = time.time()
        analyst = AnalystAgent()
        
        top_articles = []
        story_of_day_index = 0
        analyst_failed = False
        
        try:
            analysis_result = analyst.run(unique_articles)
            top_articles = analysis_result["top_articles"]
            story_of_day_index = analysis_result["story_of_day_index"]
        except Exception as e:
            print(f"[Orchestrator] ✗ Analyst selection failed: {e}")
            print("[Orchestrator] Falling back to top 10 articles by score descending.")
            top_articles = sorted(unique_articles, key=lambda a: a.get("score", 0), reverse=True)[:10]
            story_of_day_index = 0
            analyst_failed = True
            
        duration_analyst = time.time() - start_analyst
        print(f"[Orchestrator] ✓ Analyst step completed in {duration_analyst:.2f}s.")

        # -------------------------------------------------------------------------
        # STEP 3: Newsletter Generation (Writer)
        # -------------------------------------------------------------------------
        print("\n[Step 3/5] Running Writer newsletter generation...")
        start_writer = time.time()
        writer = WriterAgent()
        
        html_digest = ""
        writer_failed = False
        
        if not analyst_failed:
            try:
                html_digest = writer.run(top_articles)
            except Exception as e:
                print(f"[Orchestrator] ✗ Writer generation failed: {e}")
                writer_failed = True
        else:
            # If Analyst failed, we don't try to call Writer to avoid compounding LLM issues
            print("[Orchestrator] Analyst fallback active. Skipping Writer LLM call to save quota/prevent failures.")
            writer_failed = True
            
        if writer_failed:
            print("[Orchestrator] Generating basic HTML fallback digest...")
            today_str = datetime.now().strftime("%B %d, %Y")
            html_digest = f"""<h1>AI Daily Digest — {today_str}</h1>
<p style="color: #d9534f; font-weight: bold; font-style: italic;">Note: AI curation was unavailable today. Here are the top stories by community score.</p>
<hr>
<h2>Top Stories</h2>
<ul>
"""
            for art in top_articles:
                title = art.get("title", "Untitled").strip()
                url = art.get("url", "#").strip()
                source = art.get("source", "Unknown").strip()
                score = art.get("score", 0)
                score_info = f" (Score: {score})" if score else ""
                html_digest += f'  <li style="margin-bottom: 12px;">\n    <strong><a href="{url}">{title}</a></strong><br>\n    <span style="color: #555; font-size: 0.9rem;">Source: {source}{score_info}</span>\n  </li>\n'
            
            html_digest += """</ul>
<hr>
<p><em>Curated by AI Daily Digest · Fallback Curation System</em></p>
"""

        duration_writer = time.time() - start_writer
        print(f"[Orchestrator] ✓ Writer step completed in {duration_writer:.2f}s.")
        print(f"[Orchestrator] Final HTML digest: {len(html_digest):,} characters.")

        # Save copy locally for convenience
        output_path = "digest_preview.html"
        try:
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
            print(f"[Orchestrator] Saved local preview copy to: {output_path}")
        except Exception as e:
            print(f"[Orchestrator] WARNING: Failed to save local HTML preview file. Error: {e}")

        # -------------------------------------------------------------------------
        # STEP 4: Email Delivery
        # -------------------------------------------------------------------------
        print("\n[Step 4/5] Sending digest email...")
        start_email = time.time()
        
        recipient = os.getenv("DIGEST_RECIPIENT")
        if recipient:
            print(f"[Orchestrator] Using DIGEST_RECIPIENT environment variable: {recipient}")
        else:
            print("[Orchestrator] DIGEST_RECIPIENT environment variable not found. Defaulting to sender's address.")

        email_success = send_digest_email(html_digest, recipient_email=recipient)
        duration_email = time.time() - start_email
        
        if not email_success:
            print(f"[Orchestrator] ✗ Error: Email delivery failed (took {duration_email:.2f}s).")
            sys.exit(1)
            
        print(f"[Orchestrator] ✓ Email sent successfully in {duration_email:.2f}s.")

        # -------------------------------------------------------------------------
        # STEP 5: Updating Memory/History
        # -------------------------------------------------------------------------
        print("\n[Step 5/5] Updating JSON and Vector DB histories...")
        start_store = time.time()
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Update JSON history (primary storage)
        try:
            history = load_history()
            add_to_history(top_articles, history)
            print("[Orchestrator] ✓ JSON history successfully updated.")
        except Exception as e:
            print(f"[Orchestrator] WARNING: Failed to update JSON history: {e}")

        # 2. Update Vector DB memory
        try:
            store_articles(today_str, top_articles)
            print("[Orchestrator] ✓ Vector store memory successfully updated.")
        except Exception as e:
            print(f"[Orchestrator] WARNING: Failed to update vector store: {e}")
            
        duration_store = time.time() - start_store

        # -------------------------------------------------------------------------
        # Run Summary
        # -------------------------------------------------------------------------
        total_duration = duration_fetch + duration_analyst + duration_writer + duration_email + duration_store
        print("\n" + "=" * 60)
        print("  PIPELINE RUN SUMMARY")
        print("=" * 60)
        print(f"  - Fetching step:   {duration_fetch:.2f}s")
        print(f"  - Analyst step:    {duration_analyst:.2f}s")
        print(f"  - Writer step:     {duration_writer:.2f}s")
        print(f"  - Email step:      {duration_email:.2f}s")
        print(f"  - Storing step:    {duration_store:.2f}s")
        print(f"  - Total Duration:  {total_duration:.2f}s")
        print(f"  - Status:          Success")
        print("=" * 60)

        return html_digest
