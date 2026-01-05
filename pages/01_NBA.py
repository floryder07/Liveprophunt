import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'app' package can be imported from pages/
repo_root = Path(__file__).resolve().parent.parent  # pages/ -> repo root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import Streamlit and call set_page_config BEFORE any other Streamlit usage/imports
import streamlit as st

# Try to set page config; if Streamlit already set it (multi-page import), ignore the error
try:
    st.set_page_config(page_title="NBA Player Prop Tracker", page_icon="🏀", layout="wide")
except Exception:
    # Streamlit may already have initialized page config during multi-page import.
    # Ignore and continue — this prevents startup crashes.
    pass

# Now safe to import other libraries and local modules that do not do UI at import time
import logging
from typing import Any, Dict, List, Optional
from io import StringIO

import pandas as pd
import requests

from app.utils import parse_players  # ensure app/utils.py does NOT import streamlit at module import

# --- Logging --------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Sidebar: settings / data source -----------------------------------------
st.sidebar.header("Data & Settings")

data_source = st.sidebar.selectbox(
    "Live data source",
    options=["Mock (no API)", "CSV upload", "CSV / Google Sheet URL", "REST API (JSON)"],
    index=0,
)

# CSV upload
csv_file = None
if data_source == "CSV upload":
    csv_file = st.sidebar.file_uploader("Upload CSV (columns: name,stat_label,live_value,delta,target,pace)", type=["csv"])

# CSV URL / Google Sheet (public CSV export link)
csv_url = ""
if data_source == "CSV / Google Sheet URL":
    csv_url = st.sidebar.text_input(
        "CSV URL (public) or Google Sheets 'export?format=csv' URL",
        value="",
    )

# REST API
api_url = ""
api_token = ""
if data_source == "REST API (JSON)":
    api_url = st.sidebar.text_input("API endpoint that returns JSON list of player dicts", value="")
    api_token = st.sidebar.text_input("Bearer token (optional)", value="", type="password")
    st.sidebar.markdown(
        "Expected JSON shape: list of objects with keys: "
        "`name`, `stat_label`, `live_value`, `delta`, `target`, `pace` (or compatible keys)."
    )

# Display options
num_cols = st.sidebar.selectbox("Columns", options=[1, 2, 3], index=1)
show_pace = st.sidebar.checkbox("Show pace", value=True)
show_target = st.sidebar.checkbox("Show target", value=True)

# Auto-refresh options
st.sidebar.markdown("Auto-refresh:")
auto_refresh_enabled = st.sidebar.checkbox("Enable auto-refresh", value=False)
auto_refresh_interval = st.sidebar.number_input("Interval (seconds, min 5)", min_value=5, value=15, step=5)

# --- Helpers: data fetching --------------------------------------------------
@st.cache_data(ttl=10)
def fetch_csv_from_url(url: str) -> pd.DataFrame:
    """Fetch CSV from a URL and return DataFrame. Raises for non-2xx responses."""
    logger.info("Fetching CSV from URL: %s", url)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


@st.cache_data(ttl=10)
def fetch_json_from_api(url: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch JSON list of player dicts from a REST endpoint."""
    logger.info("Fetching JSON from API: %s", url)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        # If the API nests the list under a key like "players"
        for key in ("players", "data", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Not a list: try wrapping single-player dict
        return [data]
    if isinstance(data, list):
        return data
    # Unknown shape: raise
    raise ValueError("API returned unexpected JSON shape (not list/dict).")


@st.cache_data(ttl=10)
def fetch_live_players(source: str, csv_file, csv_url: str, api_url: str, api_token: str) -> List[Dict[str, Any]]:
    """
    Centralized fetch for live player data.
    Returns list of player dicts with keys matching parse_players output.
    """
    logger.info("Fetching live players from source: %s", source)
    if source == "Mock (no API)":
        # Return empty to fallback to sidebar defaults
        return []
    if source == "CSV upload":
        if csv_file is None:
            return []
        df = pd.read_csv(csv_file)
        return df.to_dict(orient="records")
    if source == "CSV / Google Sheet URL":
        if not csv_url:
            return []
        df = fetch_csv_from_url(csv_url)
        return df.to_dict(orient="records")
    if source == "REST API (JSON)":
        if not api_url:
            return []
        return fetch_json_from_api(api_url, token=api_token)
    return []


def to_player_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw dict (from CSV/JSON) into the internal player dict shape."""
    name = raw.get("name") or raw.get("player") or raw.get("player_name") or "Unknown"
    stat_label = raw.get("stat_label") or raw.get("stat") or raw.get("statLabel") or "Stat"
    # live_value could be numeric or string
    live_value_raw = raw.get("live_value") or raw.get("liveValue") or raw.get("value") or ""
    try:
        live_value = int(live_value_raw)
    except Exception:
        try:
            live_value = float(live_value_raw)
        except Exception:
            live_value = live_value_raw or None
    delta = raw.get("delta") or raw.get("difference") or ""
    target_raw = raw.get("target") or raw.get("line") or ""
    try:
        target = float(target_raw) if target_raw != "" else None
    except Exception:
        target = None
    pace_raw = raw.get("pace") or ""
    try:
        pace = float(pace_raw) if pace_raw != "" else None
    except Exception:
        pace = None

    return {
        "name": str(name),
        "stat_label": str(stat_label),
        "live_value": live_value,
        "delta": str(delta),
        "target": target,
        "pace": pace,
    }


# --- Default players (if using Mock or editing in UI) ------------------------
DEFAULT_PLAYERS = [
    {"name": "LeBron James", "stat_label": "Points", "live_value": 22, "delta": "+2.5 vs Line", "target": 19.5, "pace": 28.1},
    {"name": "Kevin Durant", "stat_label": "Rebounds", "live_value": 6, "delta": "-1.5 vs Line", "target": 7.5, "pace": 6.8},
]


# --- Parse/edit players (text area) -----------------------------------------
st.sidebar.markdown("Edit players (one per line):")
players_text_default = "\n".join(
    f"{p['name']}|{p['stat_label']}|{p['live_value']}|{p['delta']}|{p['target']}|{p['pace']}" for p in DEFAULT_PLAYERS
)
players_text = st.sidebar.text_area(
    "Format: name|stat_label|live_value|delta|target|pace",
    value=players_text_default,
    height=140,
)

# Parse players_text into player dicts
user_players = parse_players(players_text)

# Fetch live players from the selected data source
try:
    raw_live = fetch_live_players(data_source, csv_file, csv_url, api_url, api_token)
except Exception as e:
    st.sidebar.error(f"Error fetching live data: {e}")
    raw_live = []

# Normalize live players into same shape
live_players_map: Dict[str, Dict[str, Any]] = {}
for r in raw_live:
    try:
        rec = to_player_record(r)
        live_players_map[rec["name"].lower()] = rec
    except Exception:
        logger.exception("Skipping malformed record: %s", r)

# Merge live values into user_players by name match (case-insensitive)
merged_players: List[Dict[str, Any]] = []
for p in user_players:
    name_key = p["name"].lower()
    live = live_players_map.get(name_key, {})
    merged = dict(p)  # copy
    # override fields if present in live data
    if live.get("live_value") not in (None, ""):
        merged["live_value"] = live["live_value"]
    if live.get("delta"):
        merged["delta"] = live["delta"]
    if live.get("target") not in (None, ""):
        merged["target"] = live["target"]
    if live.get("pace") not in (None, ""):
        merged["pace"] = live["pace"]
    merged_players.append(merged)

# If no players found in UI, fallback to defaults (useful for demo)
players = merged_players or DEFAULT_PLAYERS

# --- Auto-refresh (attempt to use streamlit-autorefresh if available) -------
if auto_refresh_enabled:
    try:
        from streamlit_autorefresh import st_autorefresh

        # st_autorefresh expects milliseconds
        st_autorefresh(interval=auto_refresh_interval * 1000, key="autorefresh")
    except Exception:
        # streamlit-autorefresh not installed: show a hint and rely on manual refresh button
        st.sidebar.info("Install 'streamlit-autorefresh' to enable true client-side auto-refresh.")


# --- UI helpers ---------------------------------------------------------------
def render_player_card(player: Dict[str, Any], show_pace: bool = True, show_target: bool = True) -> None:
    """Render a player card into the current Streamlit column context."""
    st.subheader(player["name"])

    metric_label = f"Live {player.get('stat_label', 'Stat')}"
    # The delta string may include text like "+2.5 vs Line" — try to extract numeric prefix for clearer coloring
    delta_display = player.get("delta", "")
    # Attempt to coerce delta to numeric if it starts with +/-
    delta_numeric = None
    if isinstance(delta_display, (int, float)):
        delta_numeric = delta_display
    elif isinstance(delta_display, str) and delta_display.strip():
        # Extract leading signed number
        import re

        m = re.match(r"^([+-]?\d+(\.\d+)?)", delta_display.strip())
        if m:
            try:
                delta_numeric = float(m.group(1))
            except Exception:
                delta_numeric = None

    # Use Streamlit's metric (it will color positive deltas green, negative red)
    if delta_numeric is not None:
        st.metric(label=metric_label, value=str(player.get("live_value", "—")), delta=f"{delta_numeric}")
    else:
        # Keep delta as string if not numeric
        st.metric(label=metric_label, value=str(player.get("live_value", "—")), delta=str(delta_display))

    # Build caption with target/pace and a status emoji vs target
    caption_parts = []
    target = player.get("target")
    live_value = player.get("live_value")
    if show_target and target is not None:
        # Status relative to target
        status = ""
        try:
            if isinstance(live_value, (int, float)) and isinstance(target, (int, float)):
                if live_value >= target:
                    status = "✅"  # favorable
                else:
                    status = "⚠️"  # below
        except Exception:
            status = ""
        caption_parts.append(f"Target: {target} {status}")

    if show_pace and player.get("pace") not in (None, ""):
        caption_parts.append(f"Pace: {player.get('pace')}")

    if caption_parts:
        st.caption(" | ".join(caption_parts))


# --- Main layout -------------------------------------------------------------
st.title("🏀 NBA Player Prop Tracker")

col_refresh, _ = st.columns([1, 6])
with col_refresh:
    if st.button("Refresh now"):
        st.experimental_rerun()

if not players:
    st.info("No players configured. Use the sidebar to add players or select a data source.")
else:
    cols = st.columns(num_cols)
    for idx, player in enumerate(players):
        col = cols[idx % num_cols]
        with col:
            render_player_card(player, show_pace=show_pace, show_target=show_target)

# Footer
st.write("---")
st.caption(
    "Notes: \n"
    "- Use the sidebar to select a data source (Mock/CSV/URL/REST).\n"
    "- For Google Sheets, use the sheet's CSV export URL (export?format=csv).\n"
    "- To enable client-side auto-refresh install: pip install streamlit-autorefresh"
)
