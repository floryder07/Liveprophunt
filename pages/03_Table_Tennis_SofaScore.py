"""
Table Tennis — SofaScore-style feed viewer (Streamlit page)

This page consumes normalized match JSON (from a scraper or any JSON endpoint)
and displays live / available table-tennis matches. It does NOT scrape SofaScore
itself; it expects a JSON feed shaped like the example below.

Example expected record:
{
  "id": "match-1",
  "title": "Ma Long vs Fan Zhendong",
  "teams": ["Ma Long", "Fan Zhendong"],
  "start": "2026-01-05T12:00:00Z",
  "inplay": true,
  "score": {"a": "9", "b": "7"}
}

Place scraper output at output/matches.json or provide a raw gist/URL in the sidebar.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import logging
import requests
import streamlit as st

# Ensure repo root on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Guarded page config
try:
    st.set_page_config(page_title="Table Tennis — SofaScore Feed", page_icon="🏓", layout="wide")
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sidebar controls
st.sidebar.header("SofaScore-style feed")

json_url = st.sidebar.text_input(
    "JSON feed URL (raw gist, hosted file, or http endpoint). Leave empty to use local output/matches.json",
    value="",
)

# Default to local scraper output if present
if not json_url:
    local_default = Path("output/matches.json")
    if local_default.exists():
        json_url = f"file://{local_default.resolve()}"

only_inplay = st.sidebar.checkbox("Show only in-play matches", value=True)
auto_refresh = st.sidebar.checkbox("Enable auto-refresh (client-side)", value=True)
refresh_interval = st.sidebar.number_input("Refresh interval (seconds)", min_value=5, value=10, step=5)
cards_per_row = st.sidebar.selectbox("Cards per row", options=[1, 2, 3], index=2)

# enable client-side autorefresh if library present
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_interval * 1000, key="sofa-autorefresh")
    except Exception:
        # fallback: user can click Refresh now
        pass

def fetch_json_feed(url: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - HTTP(S) endpoints via requests
      - file:// local JSON paths
    """
    if not url:
        return []
    if url.startswith("file://"):
        path = url[len("file://"):]
        import json
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

    # normalize wrapper shapes
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "matches", "events", "items"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return [data]
    raise ValueError("Unexpected JSON shape from feed")

def normalize_match(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure minimal keys:
      id, title, teams (list[str]), start (string), inplay (bool), score (optional)
    """
    title = rec.get("title") or " / ".join(rec.get("teams", [])) or rec.get("id", "match")
    teams = rec.get("teams") or rec.get("competitors") or rec.get("runners") or []
    # flatten team objects if necessary
    norm_teams = []
    for t in teams:
        if isinstance(t, str):
            norm_teams.append(t)
        elif isinstance(t, dict):
            norm_teams.append(t.get("name") or t.get("title") or str(t))
        else:
            norm_teams.append(str(t))

    start = rec.get("start")
    start_display = start
    if isinstance(start, str):
        try:
            dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            start_display = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            start_display = start

    inplay = bool(rec.get("inplay") or rec.get("live") or ("status" in rec and "inplay" in str(rec.get("status")).lower()))
    score = rec.get("score") or rec.get("scores") or None

    return {
        "id": rec.get("id"),
        "title": title,
        "teams": norm_teams,
        "start": start_display,
        "inplay": inplay,
        "score": score,
        "raw": rec,
    }

# Page UI
st.title("🏓 Table Tennis — SofaScore Feed")

col_controls, _ = st.columns([3, 1])
with col_controls:
    if st.button("Refresh now"):
        st.experimental_rerun()

if not json_url:
    st.warning("No JSON feed configured (sidebar). Provide a raw gist URL or run a scraper that writes output/matches.json.")
    st.stop()

try:
    with st.spinner("Loading matches..."):
        raw = fetch_json_feed(json_url)
except Exception as e:
    st.error(f"Could not load feed: {e}")
    st.stop()

matches = [normalize_match(m) for m in raw or []]
if only_inplay:
    matches = [m for m in matches if m["inplay"]]

if not matches:
    st.info("No matches found (matching filters).")
else:
    # Summary table
    import pandas as pd
    rows = []
    for m in matches:
        teams_str = " vs ".join(m["teams"]) if m["teams"] else m["title"]
        score = ""
        if m["score"]:
            a = m["score"].get("a") or m["score"].get("home") or ""
            b = m["score"].get("b") or m["score"].get("away") or ""
            score = f"{a} - {b}"
        rows.append({"match": teams_str, "start": m["start"], "inplay": m["inplay"], "score": score})
    df = pd.DataFrame(rows)
    st.subheader("Matches")
    st.dataframe(df)

    # Cards
    st.subheader("Match Details")
    cols = st.columns(cards_per_row)
    for idx, m in enumerate(matches):
        c = cols[idx % cards_per_row]
        with c:
            st.markdown(f"### {m['title']}")
            if m["teams"]:
                st.write("Teams:", " / ".join(m["teams"]))
            if m["score"]:
                a = m["score"].get("a") or m["score"].get("home") or ""
                b = m["score"].get("b") or m["score"].get("away") or ""
                st.metric(label="Score", value=f"{a} — {b}")
            st.write("Start:", m["start"] or "—")
            st.write("In play:", "Yes" if m["inplay"] else "No")
            with st.expander("Raw data"):
                st.json(m["raw"])

st.caption("This page consumes a normalized JSON feed (derived from SofaScore-like pages). Do not scrape websites without permission; use licensed APIs for production.")
