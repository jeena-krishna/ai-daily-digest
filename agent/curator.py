"""
agent/curator.py

The editorial brain of the AI Daily Digest.

This module takes raw articles from sources/aggregator.py and runs them through
a TWO-STEP Gemini pipeline to produce a clean HTML email digest.

Why two calls instead of one?
  Asking a single LLM call to both reason AND format reliably is hard — the model
  mixes reasoning text into the output, uses inconsistent delimiters, or ignores
  formatting instructions under a long context. Separating the jobs eliminates
  all parsing ambiguity:

  Call 1 — ANALYSIS (structured JSON):
    Input:  All ~60 raw articles.
    Task:   Deduplicate, filter, rank. Return ONLY a JSON array of top-10 indices.
    Output: Parsed with json.loads() — zero regex, zero ambiguity.

  Call 2 — WRITING (pure HTML):
    Input:  Only the 10 articles selected by Call 1.
    Task:   Write the newsletter. No reasoning, no JSON, just HTML.
    Output: Used directly — no tag extraction needed.
"""

import os
import re           # Built-in: regex used for light HTML cleanup
import json         # Built-in: parse Call 1's JSON response reliably
import ast          # Built-in: fallback parser if json.loads() fails on near-valid JSON
import textwrap     # Built-in: dedent() strips indentation from triple-quoted strings
import time         # Built-in: sleep for retrying API calls

from google import genai             # pip install google-genai
from google.genai import types       # GenerateContentConfig lives here

from dotenv import load_dotenv       # pip install python-dotenv

# Load .env file so GEMINI_API_KEY is available via os.getenv()
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The Gemini model to use. gemini-2.0-flash is fast, cost-effective, and
# handles large context windows — ideal for batch digest processing.
GEMINI_MODEL = "gemini-2.5-flash"

# Maximum number of articles to send to the LLM in one prompt.
# Sending all 70+ raw articles risks exceeding the prompt size we want,
# and most beyond ~60 will be duplicates or low-signal anyway.
MAX_ARTICLES_TO_SEND = 60


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_articles_for_prompt(articles: list[dict]) -> str:
    """
    Converts the list of article dicts into a compact numbered text block
    suitable for embedding in a prompt.

    We include only the fields the LLM needs to make editorial decisions:
    index, source, title, URL, and description. We omit `score` because
    not all sources have scores, and the LLM should judge importance
    independently rather than just sorting by score.

    Args:
        articles: List of article dicts from fetch_all_news().

    Returns:
        A multi-line string like:
            [1] SOURCE: Hacker News
                TITLE: OpenAI releases GPT-5
                URL: https://...
                DESC: A short description...
    """
    lines = []
    # Limit how many articles we send to avoid an overly long prompt
    for i, article in enumerate(articles[:MAX_ARTICLES_TO_SEND], start=1):
        title = article.get("title", "").strip()
        url = article.get("url", "").strip()
        source = article.get("source", "Unknown").strip()
        # Description might be missing or empty — default to "No description."
        desc = (article.get("description") or "No description.").strip()
        # Truncate description to 150 chars to keep the prompt tight
        if len(desc) > 150:
            desc = desc[:150] + "..."

        lines.append(
            f"[{i}] SOURCE: {source}\n"
            f"    TITLE: {title}\n"
            f"    URL:   {url}\n"
            f"    DESC:  {desc}"
        )

    return "\n\n".join(lines)


def _build_analysis_prompt(articles_text: str, article_count: int) -> str:
    """
    Builds the Call 1 prompt: asks Gemini to analyse all articles and return
    a ranked JSON array of the top 10 indices. No HTML, no prose, no metadata.

    Separating analysis from writing means:
      - This call can reason freely without worrying about output format
      - The output is machine-readable JSON — parsed with json.loads(), not regex
      - If this call fails or returns bad JSON, we fall back gracefully

    Args:
        articles_text:  Formatted article list from _format_articles_for_prompt().
        article_count:  How many articles are in the list (used in the prompt).

    Returns:
        The analysis prompt string.
    """
    return textwrap.dedent(f"""
        You are a senior technical editor for an AI newsletter aimed at developers
        and ML researchers. Your job is EDITORIAL SELECTION ONLY — no writing yet.

        Below are {article_count} articles collected today from Hacker News, NewsAPI,
        ArXiv, and AI blogs. Articles are numbered starting at 1.

        ---
        {articles_text}
        ---

        YOUR TASK:
        Select the top 10 articles for a developer-focused AI digest.

        Apply this judgment in order:
        1. FILTER false positives — articles that mention AI only tangentially
           (e.g., a business article that says "leverages AI" once) should be removed.
        2. DEDUPLICATE — if the same story appears from multiple sources, keep only
           the best one (prefer the primary source: company blog > research paper
           > trade press > generic news).
        3. RANK by importance for developers building AI products today:
           - Model releases / significant capability changes  (highest priority)
           - Safety, alignment, or policy developments
           - New tools, APIs, SDKs, or frameworks that ship now
           - Research with clear near-term practical implications
           - Industry news affecting AI development or deployment
           - Interesting early-stage research                  (lowest priority)
        4. PICK one "Story of the Day" from your top 10.

        OUTPUT FORMAT — respond with ONLY valid JSON. No markdown backticks, no
        explanation, no preamble. Start your response with [ and end with ].

        The JSON must be an array of exactly 10 objects, ordered best-first
        (index 0 = most important = Story of the Day):

        [
          {{"index": 1, "score": 9}},
          {{"index": 5, "score": 8}}
        ]

        CRITICAL FORMATTING RULES — violating any of these will break the parser:
          - Return ONLY a JSON array. No text before or after.
          - No markdown backticks. No ```json. Just raw JSON starting with [.
          - Use double quotes for ALL keys and string values.
          - No trailing commas after the last item in any array or object.
          - The array must have EXACTLY 10 objects. No more, no less.
          - Each object must have EXACTLY these 2 keys: "index", "score".
          - "index" is the 1-based article number from the list above (integer).
          - "score" is 1-10 importance rating (integer).
          - First object in the array is the Story of the Day.
        Respond with ONLY the JSON array. Nothing else.
    """).strip()


def _build_writing_prompt(selected_articles_text: str, trends: list[dict] = None, story_of_day_index: int = 0) -> str:
    """
    Builds the Call 2 prompt: asks Gemini to write the HTML newsletter from
    only the pre-selected top 10 articles. No analysis, no JSON — pure HTML.

    Because the input is already curated, Gemini's entire attention goes to
    writing quality. This produces consistently better prose than when the
    model is simultaneously trying to rank AND write.

    Args:
        selected_articles_text: Formatted text of only the top 10 articles.
        trends:                  Unused now (trends are identified in Call 2 directly).
        story_of_day_index:      0-based position in selected_articles_text that
                                 is the Story of the Day (usually 0, meaning first).

    Returns:
        The writing prompt string.
    """
    return textwrap.dedent(f"""
        You are writing the HTML body of a daily AI newsletter for developers and
        ML researchers. The editorial selection has already been done for you.
        Your ONLY job is to write clean, engaging HTML — no analysis, no JSON.

        ARTICLES TO COVER (in priority order, best first):
        Article {story_of_day_index + 1} is the Story of the Day.

        ---
        {selected_articles_text}
        ---

        YOUR TASKS:
        1. Write a compelling, custom headline for each article (do not just copy the source headline if it is vague).
        2. Write a 2-3 sentence summary/explanation for each article answering: "So what can a developer/ML researcher do with this today?"
        3. Identify 2-4 major trends or patterns across these 10 articles and list them.

        OUTPUT FORMAT:
        Respond with ONLY HTML. No preamble, no explanation, no markdown.
        Do not write ```html or ``` anywhere.
        Start your response directly with <h1> and end with the last closing tag.

        REQUIRED HTML STRUCTURE (follow exactly):

        <h1>AI Daily Digest — [Today's Date, e.g. May 28, 2025]</h1>
        <p><em>[One sentence: what kind of day was it in AI? Be specific, not generic.]</em></p>

        <h2>⭐ Story of the Day</h2>
        <h3>[Your own clear headline — rewrite if original is vague or clickbait-y]</h3>
        <p>[2-3 sentences. Open with WHY this matters to a developer TODAY, not WHAT
           happened. No adjectives like "groundbreaking" or "exciting" — show impact
           concretely. What can someone build or do differently because of this?]</p>
        <p><a href="[URL]">[Source Name] →</a></p>

        <hr>
        <h2>Top Stories</h2>

        [Repeat the block below for each of the remaining 9 articles:]
        <div>
          <h3>[Clear, specific headline]</h3>
          <p>[2-3 sentences: what changed, why a developer should care, what the
             practical implication is. Prefer facts and numbers over vague claims.]</p>
          <p><a href="[URL]">[Source Name] →</a></p>
        </div>
        <hr>

        <h2>Trends & Patterns</h2>
        <ul>
          [One <li> per trend: <strong>Label:</strong> 1-2 sentence explanation.]
        </ul>

        <hr>
        <p><em>Curated by AI Daily Digest · Sources: Hacker News, NewsAPI, ArXiv,
           OpenAI Blog, Google AI Blog, Hugging Face Blog, The Verge AI, TechCrunch AI</em></p>

        WRITING RULES (non-negotiable):
          - Developer audience: use technical terms freely (LoRA, RLHF, inference,
            fine-tuning, tokenizer, embedding, quantization, etc.)
          - Banned words as standalone adjectives: groundbreaking, revolutionary,
            game-changing, exciting, amazing, powerful, incredible, remarkable
          - Every story must answer: "So what can I do with this?"
          - Research papers: lead with practical implication, not abstract finding
          - Tool releases: name what you can build, not just that it exists
          - Use specific numbers when available ("40% faster", "1M token context")
          - If context is missing, say so ("Details are sparse, but...")
    """).strip()


# ---------------------------------------------------------------------------
# Gemini API interaction
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, temperature: float = 0.4, max_tokens: int = 4096) -> str:
    """
    Sends a prompt to Gemini and returns the raw text response.

    This is a thin wrapper around the google-genai SDK. Both Call 1 and Call 2
    use this function — they differ only in the prompt and temperature they pass.

    Args:
        prompt:      The full prompt string to send.
        temperature: Controls randomness. Lower = more deterministic.
                     Call 1 (JSON) uses 0.1 for consistency.
                     Call 2 (HTML) uses 0.4 for readable, varied prose.
        max_tokens:  Upper bound on response length.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.

    Returns:
        The model's response as a plain string.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key from "
            "https://aistudio.google.com/app/apikey"
        )

    # The Client object manages the HTTP connection to the Gemini API.
    # Constructing it here (rather than at module level) is intentional:
    # it avoids keeping a persistent connection open between the two calls.
    client = genai.Client(api_key=api_key)

    print(f"[Curator] → Gemini ({GEMINI_MODEL}): {len(prompt):,} chars, temp={temperature}")

    max_retries = 3
    for attempt in range(max_retries + 1):  # 0 (initial call), 1, 2, 3 (retries)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str
            
            # If we've exhausted all retries, or the error is not transient, raise it
            if attempt == max_retries or not is_transient:
                raise e
            
            code = "503" if ("503" in err_str or "UNAVAILABLE" in err_str) else "429"
            print(f"[Curator] Gemini returned {code}, retrying in 30s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(30)


# ---------------------------------------------------------------------------
# Call 1 helpers: JSON analysis
# ---------------------------------------------------------------------------

def _parse_analysis_json(raw_json: str) -> tuple[list[dict], list[dict]]:
    """
    Parses Call 1's JSON response into (ranked_items, trends).

    Uses a multi-layer recovery strategy so transient Gemini formatting
    quirks don't kill the pipeline:
      1. Strip markdown fences (``` ... ```).
      2. Extract the substring from the first '[' to the last ']' — this
         discards any preamble or postamble text around the array.
      3. Remove trailing commas before } or ] (LLM-specific JSON bug).
      4. Try json.loads() — the strict, fast path.
      5. If that fails, try ast.literal_eval() — handles single-quoted strings
         and some other near-valid Python dict syntax.
      6. If both fail, raise so the caller can activate the score-based fallback.

    The new schema from _build_analysis_prompt() is a flat 10-element array:
      [{"index": N, "score": N, "headline": "...", "reason": "...", "trend": "..."}]
    Story of the Day is always position 0 (best-first ordering).
    No separate trends object is appended.

    Args:
        raw_json: The raw string returned by Call 1.

    Returns:
        A tuple of:
          - ranked_items: list of 10 dicts with keys index, score, headline, reason, trend
          - trends:       list of trend dicts derived from the "trend" field of each item
                          (deduplicated, formatted for the writing prompt)

    Raises:
        ValueError: If parsing fails after all recovery attempts.
    """
    text = raw_json.strip()

    # --- Debug: show raw Call 1 output so we can see what Gemini actually sent ---
    # Print the first 500 chars — enough to see the structure without flooding the log.
    preview = text[:500] + ("..." if len(text) > 500 else "")
    print(f"[Curator] Call 1 raw response (first 500 chars):\n{'-' * 40}\n{preview}\n{'-' * 40}")

    # --- Cleanup step 1: strip markdown code fences ---
    # Gemini sometimes wraps JSON in ```json ... ``` despite being told not to.
    # re.sub with re.IGNORECASE handles ```JSON, ```Json, etc.
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\n?```$", "", text).strip()

    # --- Cleanup step 2: extract substring from first '[' to last ']' ---
    # This removes any preamble ("Here is the JSON:") or postamble ("Let me know...").
    # Using rfind (right-find) for ']' ensures we grab the outermost closing bracket,
    # not an inner one from a nested structure.
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    if first_bracket != -1 and (last_bracket == -1 or last_bracket < first_bracket):
        text = text + "]"
        last_bracket = text.rfind("]")
        print("[Curator] Appended missing closing bracket to truncated JSON")

    if first_bracket == -1 or last_bracket == -1 or last_bracket <= first_bracket:
        raise ValueError("No JSON array brackets found in Call 1 response.")

    if first_bracket > 0:
        print(f"[Curator] Trimming {first_bracket} chars of preamble before '['.")

    text = text[first_bracket : last_bracket + 1]  # slice is exclusive on the right

    # --- Cleanup step 3: remove trailing commas before ] or } ---
    # Standard JSON forbids trailing commas; LLMs emit them constantly.
    # Run it twice to handle nested cases like [{"a": 1,},] in one pass.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # --- Parse attempt 1: json.loads() ---
    # The strict, canonical JSON parser. Fastest path.
    data = None
    parse_error = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        parse_error = e
        print(f"[Curator] json.loads() failed: {e}")

    # --- Parse attempt 2: ast.literal_eval() ---
    # ast.literal_eval() is Python's safe expression evaluator. It accepts
    # Python dict/list syntax, which is close to JSON but allows:
    #   - Single-quoted strings ('value' instead of "value")
    #   - True/False/None (Python booleans) instead of true/false/null
    # It CANNOT execute arbitrary code — it only handles literals.
    if data is None:
        try:
            print("[Curator] Trying ast.literal_eval() as fallback...")
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError) as e:
            # Both parsers failed — raise with the original json error for clarity
            raise ValueError(
                f"Could not parse Call 1 response as JSON or Python literal. "
                f"json error: {parse_error} | ast error: {e}"
            )

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    # Filter to only objects that have the required 'index' key
    # (guards against the model sneaking in extra commentary objects)
    ranked_items = [item for item in data if isinstance(item, dict) and "index" in item]

    if not ranked_items:
        raise ValueError("No valid article objects found in Call 1 JSON response.")

    # We no longer derive trends here, as trends are identified in Call 2 directly.
    return ranked_items, []


def _select_top_articles(
    all_articles: list[dict],
    ranked_items: list[dict],
) -> tuple[list[dict], int]:
    """
    Uses the ranked indices from Call 1 to pull the actual article dicts.

    Gemini returns 1-based indices (matching the numbered list in the prompt).
    We convert them to 0-based list indices and look up the article.

    Args:
        all_articles:  The full list of article dicts (up to MAX_ARTICLES_TO_SEND).
        ranked_items:  Parsed JSON items from _parse_analysis_json().

    Returns:
        A tuple of:
          - selected: list of article dicts in ranked order (best first)
          - story_of_day_pos: 0-based position of the Story of the Day in `selected`
    """
    selected = []
    # Story of the Day is position 0 in the ranked list (best-first ordering).
    # Gemini is instructed to put the most important story first, so we use index 0.
    story_of_day_pos = 0

    for i, item in enumerate(ranked_items):
        # Convert from 1-based (prompt) to 0-based (Python list)
        idx = item.get("index", 0) - 1

        # Guard against out-of-range indices — the model might hallucinate a number
        # beyond the article list length.
        if 0 <= idx < len(all_articles):
            article = all_articles[idx].copy()  # copy so we don't mutate the original
            selected.append(article)
        else:
            print(f"[Curator] WARNING: Ignoring out-of-range index {idx + 1} from Call 1.")

    return selected, story_of_day_pos


# ---------------------------------------------------------------------------
# Call 2 helpers: HTML cleanup
# ---------------------------------------------------------------------------

def _clean_html_response(raw_html: str) -> str:
    """
    Light cleanup of Call 2's HTML response.

    Call 2 is told to output only HTML starting with <h1>, so cleanup is
    minimal compared to the old single-call approach. We just handle the
    common case where the model adds a markdown fence or a brief sentence
    before the first tag.

    Args:
        raw_html: The raw string returned by Call 2.

    Returns:
        Cleaned HTML string.
    """
    html = raw_html.strip()

    # Strip markdown code fences (```html ... ``` or ``` ... ```)
    html = re.sub(r"^```[a-z]*\n?", "", html, flags=re.IGNORECASE).strip()
    html = re.sub(r"\n?```$", "", html).strip()

    # If the model added a sentence before <h1>, trim everything before it.
    # e.g., "Here is the HTML digest:\n<h1>..."
    h1_start = html.lower().find("<h1")
    if h1_start > 0:
        print(f"[Curator] Trimming {h1_start} chars of preamble before <h1>.")
        html = html[h1_start:]

    return html.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def curate_digest(articles: list[dict]) -> tuple[str, list[dict]]:
    """
    Main entry point: orchestrates the two-call curation pipeline.

    Call 1 — Analysis:
      Sends up to MAX_ARTICLES_TO_SEND articles to Gemini and asks for a JSON
      ranking of the top 10 plus trend notes. Parsed with json.loads() — no regex.
      If JSON parsing fails, we fall back to using the first 10 articles as-is.

    Call 2 — Writing:
      Sends only the 10 selected articles and asks for pure HTML output.
      The response is used directly after light cleanup — no tag extraction.

    Args:
        articles: List of article dicts from sources/aggregator.py.

    Returns:
        A clean HTML string ready to be embedded in an email.

    Raises:
        ValueError: If GEMINI_API_KEY is missing or no articles are provided.
    """
    if not articles:
        raise ValueError("No articles provided to curate. Run the aggregator first.")

    # Limit the pool we send to Gemini
    pool = articles[:MAX_ARTICLES_TO_SEND]
    print(f"[Curator] Starting two-call curation pipeline ({len(pool)} articles)...")

    # Format the article pool into a numbered text block
    articles_text = _format_articles_for_prompt(pool)

    # -----------------------------------------------------------------------
    # CALL 1: Analysis — get ranked JSON of top 10 indices
    # -----------------------------------------------------------------------
    print("\n[Curator] ── Call 1/2: Analysis (JSON ranking) ──")
    analysis_prompt = _build_analysis_prompt(articles_text, len(pool))

    # Use a low temperature for Call 1: we want consistent, deterministic ranking
    # decisions, not creative variation. 0.1 keeps the model focused.
    raw_analysis = _call_gemini(analysis_prompt, temperature=0.1, max_tokens=8192)
    print(f"[Curator] Call 1 response: {len(raw_analysis):,} chars")

    # Parse the JSON response and select the actual article dicts
    selected_articles: list[dict]
    trends: list[dict]
    story_of_day_pos: int

    try:
        ranked_items, trends = _parse_analysis_json(raw_analysis)

        if not ranked_items:
            raise ValueError("Call 1 returned an empty ranking list.")

        selected_articles, story_of_day_pos = _select_top_articles(pool, ranked_items)
        print(f"[Curator] ✓ Call 1 selected {len(selected_articles)} articles, "
              f"Story of Day at position {story_of_day_pos + 1}.")

        # Log each selection
        for i, art in enumerate(selected_articles):
            marker = " ⭐" if i == story_of_day_pos else ""
            print(f"  {i+1:>2}.{marker} [{art['source']}] {art['title'][:60]}")

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Fallback: Call 1 failed to return parseable JSON. This can happen if
        # the model ignores instructions and adds prose around the JSON.
        # Rather than crash, we fall back to the top 10 articles by score.
        print(f"[Curator] ✗ Call 1 JSON parsing failed: {e}")
        print("[Curator] Falling back to top-10 by score for Call 2.")
        # Sort a copy of the pool by score descending so we at least pick the
        # highest-signal articles. `article.get("score", 0)` safely defaults to 0
        # for sources (ArXiv, RSS) that don't have an engagement score.
        selected_articles = sorted(pool, key=lambda a: a.get("score", 0), reverse=True)[:10]
        trends = []
        story_of_day_pos = 0

    # -----------------------------------------------------------------------
    # CALL 2: Writing — produce pure HTML from the curated selection
    # -----------------------------------------------------------------------
    print("\n[Curator] ── Call 2/2: Writing (HTML newsletter) ──")

    # Format only the selected articles for the writing prompt
    selected_text = _format_articles_for_prompt(selected_articles)
    writing_prompt = _build_writing_prompt(selected_text, trends, story_of_day_pos)

    # Use a slightly higher temperature for writing so the prose doesn't feel robotic
    raw_html = _call_gemini(writing_prompt, temperature=0.4, max_tokens=8192)
    print(f"[Curator] Call 2 response: {len(raw_html):,} chars")

    # Light cleanup — no tag extraction needed; model outputs HTML directly
    html_digest = _clean_html_response(raw_html)

    print(f"\n[Curator] ✓ Digest ready. Final HTML: {len(html_digest):,} characters.")
    return html_digest, selected_articles


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When run directly: `python agent/curator.py`
    # Or as a module:    `python -m agent.curator`
    #
    # This imports fetch_all_news from our sources package.
    # We use an absolute import here because this __main__ block is only run
    # from the project root, where `sources` is on the Python path.
    import sys
    import os

    # Ensure the project root is on sys.path so `from sources...` works when
    # running this file directly (e.g., `python agent/curator.py` from project root)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from sources.aggregator import fetch_all_news

    import logging
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("  AI Daily Digest — Full Pipeline Test")
    print("=" * 60)

    # --- Step 1: Fetch all news ---
    print("\n[1/2] Fetching articles from all sources...")
    articles = fetch_all_news()
    print(f"      Total articles fetched: {len(articles)}")

    if not articles:
        print("ERROR: No articles fetched. Check your sources and network connection.")
        sys.exit(1)

    # --- Step 2: Curate with Gemini ---
    print("\n[2/2] Running Gemini curation...")
    try:
        html_digest, selected_articles = curate_digest(articles)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- Output ---
    print("\n" + "=" * 60)
    print("  FINAL HTML DIGEST")
    print("=" * 60 + "\n")
    print(html_digest)

    # Optionally save to a file for easy browser preview
    output_path = "digest_preview.html"
    with open(output_path, "w", encoding="utf-8") as f:
        # Wrap in a minimal HTML shell so it renders properly in a browser
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
    print(f"\n[Curator] Preview saved to: {output_path}")
    print(f"          Open in browser: file://{os.path.abspath(output_path)}")
