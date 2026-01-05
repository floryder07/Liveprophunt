#!/usr/bin/env python3
"""
Scraper template (server-side) to collect match data from a listing page and
produce normalized JSON for downstream display.

IMPORTANT: Only run this against sites you have permission to scrape. Review
the target site's Terms of Service and robots.txt. This template uses
requests + BeautifulSoup and is intended for personal/testing use.

Usage:
  - Edit LISTING_URL and CSS SELECTORS to match the target page structure.
  - Run: python3 scripts/scraper_template.py
  - The script writes output to `output/matches.json`.

Notes:
  - Add delays (rate limiting) to avoid hammering the server.
  - Consider running this on a server you control, and expose the JSON via a simple HTTP endpoint
    (Flask/FastAPI) for Streamlit to consume.
"""
import json
import time
import os
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# === Configuration (edit to match the target site structure) ===
LISTING_URL = "https://example.com/table-tennis/live"  # replace with actual listing or feed URL
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "matches.json")

# CSS selectors for a listing page. These are placeholders and must be adapted.
MATCH_WRAPPER_SEL = ".match-item"       # placeholder
TEAM_A_SEL = ".team-a .name"           # placeholder
TEAM_B_SEL = ".team-b .name"           # placeholder
SCORE_A_SEL = ".team-a .score"         # placeholder
SCORE_B_SEL = ".team-b .score"         # placeholder
START_TIME_SEL = ".start-time"         # placeholder
INPLAY_FLAG_SEL = ".inplay"            # placeholder (presence indicates in-play)

# Rate limiting
REQUEST_DELAY_SECONDS = 1.0

# HTTP headers (polite default)
HEADERS = {
    "User-Agent": "TableTennisLiveBot/1.0 (+https://yourdomain.example) Python requests"
}


def fetch_page(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    resp = requests.get(url, headers=headers or HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_listing_html(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    matches = []
    for idx, m in enumerate(soup.select(MATCH_WRAPPER_SEL)):
        try:
            team_a = m.select_one(TEAM_A_SEL).get_text(strip=True) if m.select_one(TEAM_A_SEL) else ""
            team_b = m.select_one(TEAM_B_SEL).get_text(strip=True) if m.select_one(TEAM_B_SEL) else ""
            score_a = m.select_one(SCORE_A_SEL).get_text(strip=True) if m.select_one(SCORE_A_SEL) else None
            score_b = m.select_one(SCORE_B_SEL).get_text(strip=True) if m.select_one(SCORE_B_SEL) else None
            start_raw = m.select_one(START_TIME_SEL).get_text(strip=True) if m.select_one(START_TIME_SEL) else None
            start_iso = None
            if start_raw:
                try:
                    start_dt = datetime.fromisoformat(start_raw)
                    start_iso = start_dt.isoformat()
                except Exception:
                    start_iso = start_raw
            inplay = bool(m.select_one(INPLAY_FLAG_SEL))

            rec = {
                "id": f"scraped-{idx}",
                "title": f"{team_a} vs {team_b}",
                "teams": [team_a, team_b],
                "score": {"a": score_a, "b": score_b} if (score_a or score_b) else None,
                "start": start_iso,
                "inplay": inplay,
                "source_url": LISTING_URL,
                "scraped_at": datetime.utcnow().isoformat() + "Z",
            }
            matches.append(rec)
        except Exception as e:
            print("Skipping a match wrapper due to error:", e)
    return matches


def save_matches(matches: List[Dict]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(matches, fh, indent=2, ensure_ascii=False)
    print(f"Saved {len(matches)} matches to {OUTPUT_FILE}")


def main():
    print("Fetching listing:", LISTING_URL)
    html = fetch_page(LISTING_URL)
    print("Parsing listing HTML...")
    matches = parse_listing_html(html)
    save_matches(matches)


if __name__ == "__main__":
    main()
