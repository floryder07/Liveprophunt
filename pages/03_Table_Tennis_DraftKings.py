"""
Table Tennis — DraftKings live viewer (uses an odds API)

Usage:
- In the sidebar choose provider "TheOddsAPI (template)" and paste your API key.
- Enter sport key (default 'table_tennis' but your provider may use a different key).
- Or choose "Custom JSON URL" and paste a full endpoint URL that returns events JSON that includes bookmakers with id/name 'draftkings'.

Dependencies:
pip install requests streamlit-autorefresh pandas

Notes:
- This page filters the returned bookmakers for DraftKings only.
- Live/in-play detection is attempted via available fields (commence_time vs now); your API may provide an explicit inplay flag.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import os

# Ensure repo root is on sys.path so 'app' package can be imported from pages/
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st
import requests

# Guarded set_page_config
try:
    st.set_page_config(page_title="Table Tennis — DraftKings Live", page_icon="🏓", layout="wide")
except Exception:
    pass

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sidebar controls
st.sidebar.header("DraftKings Table Tennis (Odds API)")

provider = st.sidebar.selectbox(
    "API Provider",
    options=["TheOddsAPI (template)", "Custom JSON URL / Other provider"],
    index=0,
)

# Template for TheOddsAPI (https://the-odds-api.com) — user must obtain api_key there
api_key = ""
sport_key = "table_tennis"  # default; provider may vary
regions = "us"  # regions param (adjust for provider)
markets = "h2h"  # head-to-head / match odds
odds_format = "decimal"  # decimal/american etc.

if provider == "TheOddsAPI (template)":
    api_key = st.sidebar.text_input("API key for TheOddsAPI (or leave empty to use THEODDS_API_KEY env var)", value="", type="password")
    if not api_key:
        api_key = os.environ.get("THEODDS_API_KEY", "")
    sport_key = st.sidebar.text_input("Sport key (provider-specific)", value="table_tennis")
    regions = st.sidebar.text_input("Regions (comma-separated)", value="us")
    markets = st.sidebar.text_input("Markets (comma-separated)", value="h2h")
    odds_format = st.sidebar.selectbox("Odds format", options=["decimal", "american"], index=0)
    st.sidebar.markdown(
        "The default template builds a URL like:\n\n"
        "`https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions={regions}&markets={markets}&oddsFormat={odds_format}&bookmakers=draftkings&apiKey={api_key}`\n\n"
        "If you don't have an API key, sign up at the provider (free tier may be available)."
    )
else:
    custom_url = st.sidebar.text_input("Full JSON endpoint URL (include any API key in the URL or headers)", value="")
    custom_auth_header = st.sidebar.text_input("Optional Authorization header (Bearer ...)", value="", type="password")

# Display options
only_inplay = st.sidebar.checkbox("Show only in-play matches", value=True)
auto_refresh_enabled = st.sidebar.checkbox("Enable auto-refresh (client-side)", value=True)
auto_refresh_interval = st.sidebar.number_input("Refresh interval (seconds)", min_value=5, value=15, step=5)

# Helper: build the default TheOddsAPI URL
def build_theodds_url(sport: str, regions: str, markets: str, oddsfmt: str, key: str) -> str:
    base = "https://api.the-odds-api.com/v4/sports"
    return f"{base}/{sport}/odds/?regions={regions}&markets={markets}&oddsFormat={oddsfmt}&bookmakers=draftkings&apiKey={key}"

# Fetching with caching
@st.cache_data(ttl=10)
def fetch_odds_from_url(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 10) -> List[Dict[str, Any]]:
    logger.info("Fetching odds from URL: %s", url)
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    # Some providers wrap payload in top-level keys
    # Try common keys
    if isinstance(data, dict):
        for key in ("data", "events", "results", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # fallback: wrap dict
        return [data]
    raise ValueError("Unexpected JSON shape from odds endpoint")

def extract_draftkings_bookmaker(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Given one event dict, find the bookmaker entry that identifies DraftKings.
    Bookmaker identity varies by provider; we look for 'draftkings' in id/name (case-insensitive).
    """
    bks = event.get("bookmakers") or event.get("sites") or event.get("markets", [])
    if not isinstance(bks, list):
        return None
    for b in bks:
        # common keys: 'key', 'id', 'title', 'name'
        for k in ("key", "id", "title", "name"):
            val = (b.get(k) or "").lower() if isinstance(b.get(k), str) else ""
            if "draftk" in val:
                return b
    return None

def get_event_start_time(event: Dict[str, Any]) -> Optional[datetime]:
    # common keys: commence_time, start_time, date, scheduled
    for key in ("commence_time", "start_time", "date", "scheduled"):
        ts = event.get(key)
        if not ts:
            continue
        # TheOddsAPI uses ISO timestamps
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            try:
                # fallback parse numeric unix timestamp
                return datetime.fromtimestamp(float(ts), tz=timezone.utc)
            except Exception:
                continue
    return None

def is_inplay_event(event: Dict[str, Any]) -> bool:
    # Look for an explicit inplay/live flag
    if event.get("inplay") or event.get("live") or event.get("isLive"):
        return True
    # Some providers include status fields
    status = event.get("status") or event.get("state") or ""
    if isinstance(status, str) and "inplay" in status.lower():
        return True
    # If commence_time exists and is in the past, we may consider it started (not necessarily inplay)
    start = get_event_start_time(event)
    if start:
        now = datetime.now(timezone.utc)
        # started less than 24 hours ago -> may be live; configurable threshold
        if start <= now:
            return True
    return False

def format_odds_from_bookmaker(bk: Dict[str, Any]) -> List[str]:
    """
    Extract readable odds lines from a DraftKings bookmaker entry.
    Different providers call markets/outcomes differently.
    Returns list of strings like "Player A: 1.45"
    """
    lines: List[str] = []
    # TheOddsAPI shape: bookmakers[].markets[].outcomes -> {name, price}
    markets = bk.get("markets") or bk.get("markets", [])
    if isinstance(markets, dict):
        # sometimes markets keyed by name
        markets = [markets]
    for m in markets or []:
        outcomes = m.get("outcomes") or m.get("selections") or []
        for o in outcomes:
            nm = o.get("name") or o.get("label") or o.get("runner") or ""
            price = o.get("price") or o.get("odds") or o.get("price_decimal") or o.get("price_usd") or o.get("prob")
            lines.append(f"{nm}: {price}")
    # providers may have 'outcomes' at top-level
    if not lines:
        outcomes = bk.get("outcomes") or []
        for o in outcomes:
            nm = o.get("name") or ""
            price = o.get("price") or o.get("odds") or None
            lines.append(f"{nm}: {price}")
    return lines

# Build the request and fetch
def fetch_and_filter_draftkings(only_inplay_flag: bool = True) -> List[Dict[str, Any]]:
    # Build URL & headers
    headers = {}
    if provider == "TheOddsAPI (template)":
        if not api_key:
            raise ValueError("Please enter your API key for the chosen provider in the sidebar.")
        url = build_theodds_url(sport_key, regions, markets, odds_format, api_key)
    else:
        if not custom_url:
            raise ValueError("Please enter a custom JSON URL in the sidebar.")
        url = custom_url
        if custom_auth_header:
            headers["Authorization"] = custom_auth_header

    data = fetch_odds_from_url(url, headers=headers)
    results: List[Dict[str, Any]] = []
    for ev in data:
        try:
            # identify DraftKings bookmaker
            bk = extract_draftkings_bookmaker(ev)
            if not bk:
                continue  # no DraftKings odds for this event
            # determine inplay/live
            inplay = is_inplay_event(ev)
            if only_inplay_flag and not inplay:
                continue
            start = get_event_start_time(ev)
            # prepare a normalized record
            teams = ev.get("teams") or ev.get("runners") or ev.get("competitors") or []
            teams_norm = [t if isinstance(t, str) else (t.get("name") if isinstance(t, dict) else str(t)) for t in teams]
            rec = {
                "commence_time": start.isoformat() if start else None,
                "teams": teams_norm,
                "sport": ev.get("sport_key") or ev.get("sport") or None,
                "title": ev.get("title") or ev.get("name") or " / ".join(teams_norm),
                "inplay": inplay,
                "bookmaker": bk,
                "odds_lines": format_odds_from_bookmaker(bk),
                "raw_event": ev,
            }
            results.append(rec)
        except Exception:
            logger.exception("Skipping malformed event: %s", ev)
    # Sort results by commence_time (soonest first). Events with no start go last.
    def sort_key(r):
        try:
            return r["commence_time"] or ""
        except Exception:
            return ""
    results.sort(key=sort_key)
    return results

# Auto-refresh client-side (if available)
if auto_refresh_enabled:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=auto_refresh_interval * 1000, key="dk-autorefresh")
    except Exception:
        # fallback: we'll still refresh when the page re-runs or when user clicks Refresh
        pass

# Attempt fetch and display; handle errors gracefully
st.title("🏓 Table Tennis — DraftKings Live Matches")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Refresh now"):
        st.experimental_rerun()

# Fetch data and show spinner
try:
    with st.spinner("Fetching DraftKings odds..."):
        events = fetch_and_filter_draftkings(only_inplay_flag=only_inplay)
except Exception as e:
    st.error(f"Could not fetch odds: {e}")
    st.stop()
if not events:
    st.info("No DraftKings table-tennis matches found (matching filters). Try disabling 'Show only in-play' or check your API key/provider.")
else:
    # Build a summary table
    rows = []
    for ev in events:
        start = ev.get("commence_time") or ""
        teams = ev.get("teams") or []
        title = ev.get("title") or " / ".join(teams)
        lines = ev.get("odds_lines") or []
        short_lines = "; ".join(lines[:2]) if lines else ""
        rows.append({"title": title, "start": start, "inplay": ev.get("inplay", False), "odds_preview": short_lines})

    import pandas as pd
    df = pd.DataFrame(rows)
    # show table
    st.subheader("Live / Available DraftKings Matches")
    st.dataframe(df)

    # Expanded view for each event
    st.subheader("Matches")
    for ev in events:
        title = ev.get("title") or "Match"
        start = ev.get("commence_time") or ""
        inplay = ev.get("inplay")
        with st.expander(f"{title}  —  {'IN PLAY' if inplay else 'Upcoming'}  —  {start}", expanded=False):
            st.json(ev["raw_event"])  # raw JSON for debugging; remove or replace if undesirable
            st.markdown("**DraftKings lines**")
            for line in ev.get("odds_lines", []):
                st.write(line)
            # If you want to show split outcomes in columns:
            outcomes_markets = ev["bookmaker"].get("markets", [])
            if outcomes_markets:
                for m in outcomes_markets:
                    st.markdown(f"**Market: {m.get('key') or m.get('market_key') or m.get('key','')}**")
                    cols = st.columns(len(m.get("outcomes", []) or []))
                    for i, oc in enumerate(m.get("outcomes", []) or []):
                        with cols[i]:
                            st.metric(label=oc.get("name") or oc.get("label") or "", value=str(oc.get("price") or oc.get("odds") or ""))
            st.write("---")

st.caption(
    "Notes: This page fetches odds from an external provider and filters for DraftKings bookmaker entries. "
    "Make sure you have permission to use DraftKings data and a valid API key for the odds provider."
)
