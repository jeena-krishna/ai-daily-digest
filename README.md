# 🤖 AI Daily Digest

An automated AI-powered news aggregator and newsletter curator. It polls developer and machine learning channels, selects and summarizes the most relevant stories using Google Gemini, and emails a beautifully formatted HTML digest directly to your inbox.

---

## 🌟 Features

*   **Multi-Source Aggregation**: Fetches articles, announcements, and research papers from:
    *   **Hacker News**: Retrieves top and new stories, filtered dynamically by AI/ML keyword relevance.
    *   **NewsAPI**: Queries recent headlines from hundreds of mainstream and tech publications.
    *   **arXiv**: Pulls new academic preprints on computer science and machine learning.
    *   **Custom RSS Feeds**: Parses specialized RSS feeds (e.g., Hugging Face blog, OpenAI announcements).
*   **Intelligent Two-Stage Curation**:
    *   **Stage 1 (Scoring)**: Gemini 2.5 Flash reviews all collected raw headlines and outputs structured JSON containing the top 10 articles based on relevance, trendiness, and technical importance.
    *   **Stage 2 (Writing)**: Gemini writes a clean, standalone, responsive HTML newsletter summarizing the top 10 articles.
*   **Clean & Responsive HTML Design**: Renders a premium readable layout that works seamlessly across desktop and mobile email clients.
*   **Reliable SMTP SSL Delivery**: Utilizes Gmail's SMTP servers over port 465 via secure SSL connections.
*   **Fully Automated via GitHub Actions**: Runs on a daily cron schedule every morning and can be manually triggered at any time.

---

## 🛠️ Tech Stack

*   **Runtime**: Python 3.11
*   **AI Integration**: Google GenAI SDK (`google-genai` using Gemini 2.5 Flash)
*   **Libraries**:
    *   `feedparser` (RSS processing)
    *   `requests` (APIs communication)
    *   `python-dotenv` (local environment management)
*   **Deployment & Automation**: GitHub Actions

---

## 📂 Project Structure

```text
├── .github/
│   └── workflows/
│       └── daily-digest.yml   # GitHub Actions cron & manual trigger workflow
├── agent/
│   ├── __init__.py
│   └── curator.py             # Dual-call curation logic with Gemini
├── sources/
│   ├── __init__.py            # Package indicator
│   ├── __main__.py            # CLI entry for fetching/testing sources
│   ├── aggregator.py          # Merges and filters articles from all sources
│   ├── arxiv_source.py        # Pulls preprints from arXiv API
│   ├── hackernews.py          # HN story index fetcher with keyword filtering
│   ├── newsapi_source.py      # Connects to NewsAPI endpoints
│   └── rss_feeds.py           # Parses custom RSS feed subscriptions
├── .env.example               # Template environment configuration file
├── .gitignore                 # Standard Python gitignore rules
├── digest_preview.html        # Local copy of the last generated HTML newsletter
├── email_sender.py            # Gmail SMTP SSL delivery module
├── main.py                    # Master pipeline orchestrator
└── requirements.txt           # Python application dependencies
```

---

## 🚀 Local Setup Guide

Follow these steps to run the pipeline on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-daily-digest.git
cd ai-daily-digest
```

### 2. Configure a Virtual Environment
Create and activate a virtual environment to isolate project packages:
```bash
# Create environment
python3 -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Copy `.env.example` to a new file named `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in the required keys:
*   `GEMINI_API_KEY`: Get a free developer key at [Google AI Studio](https://aistudio.google.com/).
*   `NEWSAPI_KEY`: Optional but recommended. Get a free API key at [NewsAPI.org](https://newsapi.org/).
*   `GMAIL_ADDRESS`: The Gmail address used to dispatch emails.
*   `GMAIL_APP_PASSWORD`: An App Password generated for your Gmail account. *Do not use your main password*. Create one in [Google Account Settings](https://myaccount.google.com/apppasswords) with 2FA enabled.
*   `DIGEST_RECIPIENT`: (Optional) The email address that should receive the digest. Defaults to your `GMAIL_ADDRESS` if omitted.

### 5. Run the Pipeline
To run the full pipeline (Fetch → Curate → Save Preview → Send Email):
```bash
python main.py
```
*Note: A local copy of the newsletter is saved as `digest_preview.html` for you to inspect in your browser.*

---

## 🤖 GitHub Actions Setup Guide

Deploy the pipeline to GitHub Actions for automated daily curation:

1.  **Add Secrets**: In your GitHub Repository, go to **Settings** > **Secrets and variables** > **Actions** and click **New repository secret**. Add:
    *   `GEMINI_API_KEY`
    *   `NEWSAPI_KEY`
    *   `GMAIL_ADDRESS`
    *   `GMAIL_APP_PASSWORD`
    *   `DIGEST_RECIPIENT` (Optional: specify where to deliver the digest)
2.  **Enable Actions**: Navigate to the **Actions** tab of your repository. If prompted, enable workflows.
3.  **Run Manually**: Select the **Daily AI Digest** workflow from the sidebar and click **Run workflow** to trigger a manual curation run immediately.
4.  **Automatic Cron Schedule**: The workflow is configured to run automatically every day at **11:00 UTC (7:00 AM EST)**. You can change this trigger schedule by modifying the `cron` expression in [daily-digest.yml](file:///.github/workflows/daily-digest.yml).

---

## 🔌 Developer Guide: Adding a New News Source

You can expand the aggregator with new sources (e.g. specialized API feeds, other forums) by writing a module in the `sources` package:

### Step 1: Create a Source Module
Create a new file in `sources/` (for example, `sources/custom_source.py`):

```python
import requests

def get_custom_news() -> list[dict]:
    """
    Fetches articles from a custom endpoint.
    
    Returns:
        A list of article dictionaries with standard keys.
    """
    articles = []
    try:
        response = requests.get("https://api.example.com/ai-stories")
        if response.status_code == 200:
            data = response.json()
            for item in data.get("results", []):
                articles.append({
                    "title": item.get("headline"),
                    "url": item.get("link"),
                    "source": "Custom Source Name",
                    "score": item.get("votes", 0),          # Optional score or popularity measure
                    "description": item.get("summary", "")  # Optional brief snippet
                })
    except Exception as e:
        print(f"Error fetching from custom source: {e}")
        
    return articles
```

### Step 2: Register in the Aggregator
Open [sources/aggregator.py](file:///sources/aggregator.py), import your function, and add it to the `SOURCES` list:

```python
# 1. Import your new function
from .custom_source import get_custom_news

# 2. Register it in the SOURCES list
SOURCES = [
    ("Hacker News",   get_ai_stories),
    ("NewsAPI",       get_ai_news),
    ("ArXiv",         get_arxiv_papers),
    ("RSS Feeds",     get_rss_articles),
    ("My Custom App", get_custom_news),  # <--- Added new source
]
```

The aggregator handles the execution automatically, shielding the pipeline from failures inside individual sources.

---

## 🤝 Credits

Developed as a modern agentic AI workspace project leveraging Google Gemini and automated GitHub workflows.
