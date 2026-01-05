#!/usr/bin/env python3
# list_draftkings_before_11_pst.py
"""
List events that include DraftKings and start before 11:00 Pacific time today.
Requires THEODDS_API_KEY set in environment.
"""

import os
import requests
from datetime import datetime, date, time, timezone
from zoneinfo import ZoneInfo  # Python 3.9+
from dateutil import parser as date_parser  # pip install python-dateutil

API_KEY = os.environ.get("THEODDS_API_KEY")
if not API_KEY:
    raise SystemExit("Set THEODDS_API_KEY environment variable (get a key at https://the-odds-api.com/)")

TARGET_TZ = ZoneInfo("America/Los_Angeles")  # Pacific time (handles PST/PDT automatically)

def local_cutoff_dt():
    now = datetime.now(TARGET_TZ)
    today = now.date()
    cutoff = datetime.combine(today, time(hour=11, minute=0, second=0), tzinfo=TARGET_TZ)
    return cutoff

def fetch_sports(api_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

def fetch_odds_for_sport(api_key: str, sport_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=us,uk,eu,au&markets=h2h&oddsFormat=decimal&apiKey={api_key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def has_draftkings(bookmakers):
    for bm in bookmakers or []:
        key = (bm.get("key") or "").lower()
        title = (bm.get("title") or "").lower()
        if "draftk" in key or "draftk" in title:
            return True
    return False

def main():
    cutoff = local_cutoff_dt()
    print(f"Cutoff (Pacific): {cutoff.isoformat()}")
    try:
        sports = fetch_sports(API_KEY)
    except Exception as e:
        raise SystemExit(f"Failed to fetch sports list: {e}")

    results = []
    # iterate sports (you can limit to a few keys if you want)
    for s in sports:
        sport_key = s.get("key")
        try:
            events = fetch_odds_for_sport(API_KEY, sport_key)
        except Exception:
            # ignore errors per sport to avoid premature exit
            continue
        for ev in events:
            if not has_draftkings(ev.get("bookmakers", [])):
                continue
            ct = ev.get("commence_time") or ev.get("start")
            if not ct:
                continue
            try:
                dt = date_parser.isoparse(ct)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_local = dt.astimezone(TARGET_TZ)
            except Exception:
                continue
            if dt_local < cutoff:
                results.append({
                    "sport": s.get("title"),
                    "sport_key": sport_key,
                    "event": ev.get("title") or " / ".join(ev.get("teams", [])),
                    "start_pacific": dt_local.isoformat(),
                    "draftkings_bookmakers": [bm for bm in (ev.get("bookmakers") or []) if ("draftk" in ((bm.get("key") or "").lower() + (bm.get("title") or "").lower()))]
                })

    if not results:
        print("No DraftKings events start before 11:00 Pacific today (or API returned no data).")
    else:
        print(f"Found {len(results)} events starting before 11:00 PT:")
        for r in results:
            print(f"- [{r['sport']}] {r['event']} — starts {r['start_pacific']} — DK entries: {len(r['draftkings_bookmakers'])}")

if __name__ == "__main__":
    main()
