import logging
from datetime import datetime, timedelta

import requests

from app.config import settings

logger = logging.getLogger(__name__)

SERP_API_URL = "https://serpapi.com/search"


def search_patents(query: str, max_pages: int = 10, years_back: int = 3) -> list[dict]:
    """Search Google Patents via SerpAPI, returning up to max_pages * 10 results.
    Filters to patents published within the last `years_back` years."""
    all_results = []
    cutoff_date = datetime.now() - timedelta(days=years_back * 365)

    for page in range(max_pages):
        params = {
            "engine": "google_patents",
            "q": query,
            "api_key": settings.SERP_API_KEY,
            "num": 10,
            "start": page * 10,
        }

        try:
            response = requests.get(SERP_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"SerpAPI request failed on page {page}: {e}")
            break

        if "error" in data:
            logger.warning(f"SerpAPI error: {data['error']}")
            break

        results = data.get("organic_results", [])
        if not results:
            break

        for r in results:
            # Filter by publication date (last N years)
            pub_date_str = r.get("publication_date") or r.get("filing_date")
            if pub_date_str:
                try:
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
                    if pub_date < cutoff_date:
                        continue
                except ValueError:
                    pass  # Keep patents with unparseable dates

            patent = {
                "title": r.get("title", ""),
                "publication_number": r.get("publication_number", ""),
                "patent_link": r.get("patent_link", ""),
                "priority_date": r.get("priority_date"),
                "filing_date": r.get("filing_date"),
                "publication_date": r.get("publication_date"),
                "inventor": r.get("inventor", ""),
                "assignee": r.get("assignee", ""),
                "snippet": r.get("snippet", ""),
                "pdf_url": r.get("pdf", ""),
                "search_query": query,
            }
            all_results.append(patent)

    logger.info(f"Found {len(all_results)} patents for query: {query}")
    return all_results
