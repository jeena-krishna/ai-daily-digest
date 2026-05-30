import json
import re
import ast
import textwrap
from agents.gemini_utils import call_gemini_with_fallback

# Max articles to process to keep prompt sizing stable
MAX_ARTICLES_TO_SEND = 60

class AnalystAgent:
    """
    AnalystAgent reviews all unique raw articles, formats them into a prompt,
    calls Gemini to rank and score the top 10 relevant items, and parses the JSON response.
    """
    def __init__(self):
        pass

    def _format_articles_for_prompt(self, articles: list[dict]) -> str:
        """
        Formats article objects into a compact text block for the prompt.
        """
        lines = []
        for i, article in enumerate(articles[:MAX_ARTICLES_TO_SEND], start=1):
            title = article.get("title", "").strip()
            url = article.get("url", "").strip()
            source = article.get("source", "Unknown").strip()
            desc = (article.get("description") or "No description.").strip()
            if len(desc) > 150:
                desc = desc[:150] + "..."

            lines.append(
                f"[{i}] SOURCE: {source}\n"
                f"    TITLE: {title}\n"
                f"    URL:   {url}\n"
                f"    DESC:  {desc}"
            )
        return "\n\n".join(lines)

    def _build_analysis_prompt(self, articles_text: str, article_count: int) -> str:
        """
        Builds the Call 1 ranking prompt text.
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

    def _parse_analysis_json(self, raw_json: str) -> list[dict]:
        """
        Parses Call 1's JSON response using the multi-layer recovery strategy.
        """
        text = raw_json.strip()

        preview = text[:500] + ("..." if len(text) > 500 else "")
        print(f"[AnalystAgent] Call 1 raw response (first 500 chars):\n{'-' * 40}\n{preview}\n{'-' * 40}")

        # 1. Strip markdown code fences
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\n?```$", "", text).strip()

        # 2. Extract substring from first '[' to last ']'
        first_bracket = text.find("[")
        last_bracket = text.rfind("]")

        if first_bracket != -1 and (last_bracket == -1 or last_bracket < first_bracket):
            text = text + "]"
            last_bracket = text.rfind("]")
            print("[AnalystAgent] Appended missing closing bracket to truncated JSON")

        if first_bracket == -1 or last_bracket == -1 or last_bracket <= first_bracket:
            raise ValueError("No JSON array brackets found in Call 1 response.")

        text = text[first_bracket : last_bracket + 1]

        # 3. Remove trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # 4. Strict json.loads()
        data = None
        parse_error = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            parse_error = e

        # 5. ast.literal_eval fallback
        if data is None:
            try:
                print("[AnalystAgent] Trying ast.literal_eval() as fallback...")
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Could not parse Call 1 response as JSON or Python literal. "
                    f"json error: {parse_error} | ast error: {e}"
                )

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

        ranked_items = [item for item in data if isinstance(item, dict) and "index" in item]
        if not ranked_items:
            raise ValueError("No valid article objects found in Call 1 JSON response.")

        return ranked_items

    def _select_top_articles(self, all_articles: list[dict], ranked_items: list[dict]) -> tuple[list[dict], int]:
        """
        Uses the ranked indices from Call 1 to pull the actual article dictionaries.
        """
        selected = []
        story_of_day_pos = 0

        for idx, item in enumerate(ranked_items):
            # Convert 1-based index to 0-based list index
            raw_idx = item.get("index", 0) - 1
            if 0 <= raw_idx < len(all_articles):
                article = all_articles[raw_idx].copy()
                # Store the model rating directly on the dictionary for sorting fallback uses
                article["model_score"] = item.get("score", 0)
                selected.append(article)
            else:
                print(f"[AnalystAgent] WARNING: Ignoring out-of-range index {raw_idx + 1} from Call 1.")

        return selected, story_of_day_pos

    def run(self, articles: list[dict]) -> dict:
        """
        Performs the analysis step.
        
        Args:
            articles: A list of unique raw articles.
            
        Returns:
            A dictionary containing rankings and scores.
        """
        if not articles:
            raise ValueError("No articles provided to analyze.")
            
        pool = articles[:MAX_ARTICLES_TO_SEND]
        print(f"[AnalystAgent] Starting article selection (pool size: {len(pool)})...")
        
        articles_text = self._format_articles_for_prompt(pool)
        prompt = self._build_analysis_prompt(articles_text, len(pool))
        
        # Call Gemini (Call 1 uses temperature 0.1 for maximum determinism)
        raw_analysis = call_gemini_with_fallback(prompt, temperature=0.1, max_tokens=8192)
        
        # Parse JSON output
        ranked_items = self._parse_analysis_json(raw_analysis)
        
        # Match indices back to actual article dictionaries
        top_articles, story_of_day_index = self._select_top_articles(pool, ranked_items)
        scores = [item.get("score", 0) for item in ranked_items]
        
        print(f"[AnalystAgent] ✓ Successfully selected {len(top_articles)} top stories.")
        for idx, art in enumerate(top_articles):
            marker = " ⭐" if idx == story_of_day_index else ""
            print(f"  {idx+1:>2}.{marker} [{art['source']}] {art['title'][:65]}")
            
        return {
            "top_articles": top_articles,
            "scores": scores,
            "trends": [],
            "story_of_day_index": story_of_day_index
        }
