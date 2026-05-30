import re
import textwrap
from datetime import date
from agents.gemini_utils import call_gemini_with_fallback

class WriterAgent:
    """
    WriterAgent is responsible for taking the pre-selected top 10 articles,
    formatting them for Call 2, prompting Gemini to generate the HTML newsletter,
    and performing light cleanup on the output.
    """
    def __init__(self):
        pass

    def _format_articles_for_prompt(self, articles: list[dict]) -> str:
        """
        Converts the list of article dicts into a compact numbered text block.
        """
        lines = []
        for i, article in enumerate(articles, start=1):
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

    def _build_writing_prompt(self, selected_articles_text: str, story_of_day_index: int = 0) -> str:
        """
        Builds the Call 2 prompt: asks Gemini to write the HTML newsletter from
        only the pre-selected top 10 articles.
        """
        today_str = date.today().strftime('%B %d, %Y')
        return textwrap.dedent(f"""
            Today's date is {today_str}. Use this EXACT date in the heading — do not guess or use any other date.

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

            <h1>AI Daily Digest — {today_str}</h1>
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

    def _clean_html_response(self, raw_html: str) -> str:
        """
        Light cleanup of the HTML response.
        """
        html = raw_html.strip()

        # Strip markdown code fences (```html ... ``` or ``` ... ```)
        html = re.sub(r"^```[a-z]*\n?", "", html, flags=re.IGNORECASE).strip()
        html = re.sub(r"\n?```$", "", html).strip()

        # Trim preamble sentences if any exist before the actual <h1> tag.
        h1_start = html.lower().find("<h1")
        if h1_start > 0:
            print(f"[WriterAgent] Trimming {h1_start} chars of preamble before <h1>.")
            html = html[h1_start:]

        return html.strip()

    def run(self, top_articles: list[dict], trends: list = None) -> str:
        """
        Generates the HTML newsletter body.

        Args:
            top_articles: The list of 10 pre-selected and ranked articles.
            trends: Optional trend list (unused directly, as Gemini identifies trends in Call 2).

        Returns:
            Clean HTML string of the digest.
        """
        if not top_articles:
            raise ValueError("No top articles provided to WriterAgent.")

        print(f"[WriterAgent] Generating newsletter for {len(top_articles)} selected articles...")
        
        selected_text = self._format_articles_for_prompt(top_articles)
        
        # Story of the Day is always at position 0 in the ranked list.
        writing_prompt = self._build_writing_prompt(selected_text, story_of_day_index=0)
        
        # Use temperature 0.4 for creative yet technical prose style.
        raw_html = call_gemini_with_fallback(writing_prompt, temperature=0.4, max_tokens=8192)
        
        html_digest = self._clean_html_response(raw_html)
        print(f"[WriterAgent] ✓ Newsletter generated. Size: {len(html_digest):,} chars.")
        
        return html_digest
