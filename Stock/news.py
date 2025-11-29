import feedparser
import re
import logging

logging.basicConfig(level=logging.DEBUG)

def fetch_news_rss(ticker: str, max_items: int = 8) -> list[str]:
    """
    Fetch latest financial news headlines using Yahoo Finance RSS.
    Includes safe fallbacks to avoid crashes.
    """
    ticker = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    try:
        feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})

        if not hasattr(feed, "entries") or not feed.entries:
            return ["No recent news available for this stock"]

        entries = feed.entries[:max_items]

        titles = []
        for entry in entries:
            title = entry.get("title", "").strip()

            # filter spam sources
            if title and not re.search(r"^(PRNewswire|Business Wire)", title):
                titles.append(title)

        if not titles:
            return ["News available but filtered"]

        return titles

    except Exception as e:
        logging.debug(f"RSS news failed for {ticker}: {e}")
        return ["News service temporarily unavailable"]
