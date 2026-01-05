# NBA Player Prop Tracker — Top-5 "Should Bet" by historical consistency
# Overwrites pages/01_NBA.py with TheOddsAPI + nba_api integration and graceful fallbacks.
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
from typing import Any, Dict, List, Optional, Tuple
from io import StringIO
import os
import math
import time
import difflib

import pandas as pd
import requests

from app.utils import parse_players  # keep dependency lightweight (no streamlit at import)

# Attempt to import nba_api (optional). If not present we'll fallback to heuristics.
try:
    from nba_api.stats.static import players as nba_players_static
    from nba_api.stats.endpoints import playergamelog
    NBA_API_AVAILABLE = True
except Exception:
    NBA_API_AVAILABLE = False

# components import for reload fallback
try:
    import streamlit.components.v1 as components
except Exception:
    components = None

# --- Logging --------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- small helper to safely rerun / reload the page ------------------------
def safe_rerun() -> None:
    """
    Try to perform a Streamlit rerun. If not available or it fails, fall back to a client-side reload (JS).
    """
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            logger.exception("st.experimental_rerun failed; falling back to client reload.")
    # Fallback: use components.html to trigger a browser reload
    try:
        if components is not None:
            components.html("<script>window.location.reload()</script>", height=0)
            return
    except Exception:
        logger.exception("components.html reload fallback failed.")
    try:
        st.info("Please refresh the page in your browser to reload the app.")
    except Exception:
        pass


# --- Sidebar: settings / data source -----------------------------------------
st.sidebar.header("Data & Settings")

data_source = st.sidebar.selectbox(
    "Live data source",
    options=["Mock (no API)", "CSV upload", "CSV / Google Sheet URL", "REST API (JSON)", "TheOddsAPI (live props)"],
    index=0,
)

# CSV upload
csv_file = None
if data_source == "CSV upload":
    csv_file = st.sidebar.file_uploader("Upload CSV (columns: name,stat_label,live_value,delta,target,pace,team,money)", type=["csv"])

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
        "`name`, `stat_label`, `live_value`, `delta`, `target`, `pace`, optionally `team` and `money`."
    )

# TheOddsAPI option
st.sidebar.markdown("TheOddsAPI (player props)")
theodds_mode = False
THEODDS_API_KEY = os.environ.get("THEODDS_API_KEY", "")
if data_source == "TheOddsAPI (live props)":
    theodds_mode = True
    THEODDS_API_KEY = st.sidebar.text_input("THEODDS_API_KEY (or set env var)", value=THEODDS_API_KEY or "", type="password")
    st.sidebar.markdown("The page will fetch available player props (points/rebounds/assists) and compute hit-rates using nba_api (if installed).")

# Display options
num_cols = st.sidebar.selectbox("Columns", options=[1, 2, 3], index=1)
show_pace = st.sidebar.checkbox("Show pace", value=True)
show_target = st.sidebar.checkbox("Show target", value=True)

# Top-safe/consistency picks options
st.sidebar.markdown("Top picks / consistency")
enable_top_consistency = st.sidebar.checkbox("Show Top N players to bet on (by consistency)", value=True)
top_n = st.sidebar.number_input("Top N", min_value=1, max_value=20, value=5, step=1)
recent_games = st.sidebar.number_input("Recent games to analyze (last N)", min_value=3, max_value=50, value=20, step=1)

# Auto-refresh options
st.sidebar.markdown("Auto-refresh:")
auto_refresh_enabled = st.sidebar.checkbox("Enable auto-refresh", value=False)
auto_refresh_interval = st.sidebar.number_input("Interval (seconds, min 5)", min_value=5, value=15, step=5)

# --- Helpers: data fetching --------------------------------------------------
@st.cache_data(ttl=10)
def fetch_csv_from_url(url: str) -> pd.DataFrame:
    logger.info("Fetching CSV from URL: %s", url)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


@st.cache_data(ttl=10)
def fetch_json_from_api(url: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    logger.info("Fetching JSON from API: %s", url)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        for key in ("players", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("API returned unexpected JSON shape (not list/dict).")


@st.cache_data(ttl=30)
def fetch_live_players(source: str, csv_file, csv_url: str, api_url: str, api_token: str) -> List[Dict[str, Any]]:
    """
    Centralized fetch for live player data (from CSV, URL, REST).
    If TheOddsAPI is selected, this returns structured props from TheOddsAPI instead.
    """
    logger.info("Fetching live players from source: %s", source)
    if source == "Mock (no API)":
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
    if source == "TheOddsAPI (live props)":
        if not THEODDS_API_KEY:
            return []
        try:
            return fetch_props_from_theodds(THEODDS_API_KEY)
        except Exception as e:
            logger.exception("Failed to fetch from TheOddsAPI: %s", e)
            return []
    return []


# --- TheOddsAPI helper -----------------------------------------------------
def _coerce_point(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        # try parse numbers inside strings like "19.5"
        try:
            import re
            m = re.search(r"(-?\d+(\.\d+)?)", str(value))
            if m:
                return float(m.group(1))
        except Exception:
            pass
    return None


@st.cache_data(ttl=20)
def fetch_props_from_theodds(api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch player props from TheOddsAPI v4 and return a list of prop dicts:
      {name, stat_label, target, book, odds, source_raw}
    """
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json() or []
    props: List[Dict[str, Any]] = []
    market_map = {
        "player_points": "Points",
        "player_rebounds": "Rebounds",
        "player_assists": "Assists",
    }
    # Iterate games -> bookmakers -> markets -> outcomes
    for game in data:
        for bm in game.get("bookmakers", []) or []:
            book = bm.get("title") or bm.get("key") or ""
            for market in bm.get("markets", []) or []:
                mkey = market.get("key") or ""
                stat_label = market_map.get(mkey)
                if not stat_label:
                    continue
                for outcome in market.get("outcomes", []) or []:
                    name = outcome.get("name") or outcome.get("player") or outcome.get("participant") or ""
                    point = _coerce_point(outcome.get("point") or outcome.get("line") or outcome.get("price") or outcome.get("value"))
                    odds = outcome.get("price") or outcome.get("odds") or outcome.get("priceDecimal") or None
                    if not name:
                        continue
                    props.append({
                        "name": name,
                        "stat_label": stat_label,
                        "target": point,
                        "book": book,
                        "money": odds,
                        "raw": {"game": game.get("id") or game.get("sport_key"), "market": market, "outcome": outcome},
                    })
    # Deduplicate: for the same player+stat take first seen (could be improved)
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in props:
        key = (p["name"].strip().lower(), p["stat_label"])
        if key not in deduped:
            deduped[key] = p
    return list(deduped.values())


# --- Player normalization ---------------------------------------------------
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
    target_raw = raw.get("target") or raw.get("line") or raw.get("point") or ""
    try:
        target = float(target_raw) if target_raw != "" else None
    except Exception:
        target = None
    pace_raw = raw.get("pace") or ""
    try:
        pace = float(pace_raw) if pace_raw != "" else None
    except Exception:
        pace = None

    # Optional fields: team and money (odds)
    team = raw.get("team") or raw.get("team_name") or raw.get("teamName") or raw.get("side") or None
    money = raw.get("money") or raw.get("moneyline") or raw.get("odds") or raw.get("price") or raw.get("moneyline_price") or raw.get("moneyline_odds") or None

    return {
        "name": str(name),
        "stat_label": str(stat_label),
        "live_value": live_value,
        "delta": str(delta),
        "target": target,
        "pace": pace,
        "team": team,
        "money": money,
    }


# --- Default players (if using Mock or editing in UI) ------------------------
DEFAULT_PLAYERS = [
    {"name": "LeBron James", "stat_label": "Points", "live_value": 22, "delta": "+2.5 vs Line", "target": 19.5, "pace": 28.1, "team": "LAL", "money": "-135"},
    {"name": "Kevin Durant", "stat_label": "Rebounds", "live_value": 6, "delta": "-1.5 vs Line", "target": 7.5, "pace": 6.8, "team": "PHX", "money": "+120"},
]


# --- Parse/edit players (text area) -----------------------------------------
st.sidebar.markdown("Edit players (one per line):")
players_text_default = "\n".join(
    f"{p['name']}|{p['stat_label']}|{p['live_value']}|{p['delta']}|{p['target']}|{p['pace']}|{p.get('team','')}|{p.get('money','')}" for p in DEFAULT_PLAYERS
)
players_text = st.sidebar.text_area(
    "Format: name|stat_label|live_value|delta|target|pace|team|money",
    value=players_text_default,
    height=160,
)

# Parse players_text into player dicts
user_players = parse_players(players_text)

# Fetch live players / props from the selected data source
try:
    raw_live = fetch_live_players(data_source, csv_file, csv_url, api_url, api_token)
except Exception as e:
    st.sidebar.error(f"Error fetching live data: {e}")
    raw_live = []

# If TheOddsAPI mode returned props (list of {name,stat_label,target,book,money})
# convert those into player records so they merge as usual.
normalized_live: List[Dict[str, Any]] = []
for r in raw_live:
    try:
        # If this looks like a TheOddsAPI prop (has 'stat_label' and 'target'), pass through
        if isinstance(r, dict) and "stat_label" in r and "target" in r:
            normalized_live.append(to_player_record(r))
        elif isinstance(r, dict):
            normalized_live.append(to_player_record(r))
        else:
            # unknown shape: attempt to skip
            continue
    except Exception:
        logger.exception("Skipping malformed record: %s", r)

# Normalize live players into same shape keyed by name (and team if available)
live_players_map: Dict[str, Dict[str, Any]] = {}
for rec in normalized_live:
    try:
        key = rec["name"].strip().lower()
        team = rec.get("team")
        if team:
            key = f"{key}|{str(team).strip().lower()}"
        live_players_map[key] = rec
    except Exception:
        logger.exception("Skipping malformed record during map build: %s", rec)

# Merge live values into user_players by name match (case-insensitive)
merged_players: List[Dict[str, Any]] = []
for p in user_players:
    name_key = p["name"].strip().lower()
    live = live_players_map.get(name_key, {})
    if not live:
        for k, v in live_players_map.items():
            if k.split("|")[0] == name_key:
                live = v
                break
    merged = dict(p)
    if live.get("live_value") not in (None, ""):
        merged["live_value"] = live["live_value"]
    if live.get("delta"):
        merged["delta"] = live["delta"]
    if live.get("target") not in (None, ""):
        merged["target"] = live["target"]
    if live.get("pace") not in (None, ""):
        merged["pace"] = live["pace"]
    if live.get("team"):
        merged["team"] = live["team"]
    if live.get("money"):
        merged["money"] = live["money"]
    merged_players.append(merged)

# If TheOddsAPI is used, gather props list (normalized_live) for automatic population
players_from_props: List[Dict[str, Any]] = normalized_live if data_source == "TheOddsAPI (live props)" else []

# Final players list:
# - If TheOddsAPI selected, prefer props (so you don't need to manually type names)
# - otherwise prefer merged_players (manual/sidebar) then props then defaults
if data_source == "TheOddsAPI (live props)":
    players: List[Dict[str, Any]] = players_from_props or merged_players or DEFAULT_PLAYERS
else:
    players: List[Dict[str, Any]] = merged_players or players_from_props or DEFAULT_PLAYERS

# --- Auto-refresh (attempt to use streamlit-autorefresh if available) -------
if auto_refresh_enabled:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=auto_refresh_interval * 1000, key="autorefresh")
    except Exception:
        st.sidebar.info("Install 'streamlit-autorefresh' to enable true client-side auto-refresh.")


# --- Historical / consistency helpers (nba_api) ----------------------------
def _find_player_id_by_name(name: str) -> Optional[int]:
    """
    Try to find the NBA player ID for a given full name using nba_api's static players.
    Falls back to fuzzy matching when exact match not found.
    """
    if not NBA_API_AVAILABLE:
        return None
    try:
        candidates = nba_players_static.find_players_by_full_name(name)
        if candidates:
            return candidates[0]["id"]
        # Fuzzy search against all players (slower but helpful)
        all_players = nba_players_static.get_active_players() + nba_players_static.get_players()
        names = [p["full_name"] for p in all_players]
        match = difflib.get_close_matches(name, names, n=1, cutoff=0.7)
        if match:
            for p in all_players:
                if p["full_name"] == match[0]:
                    return p.get("id")
    except Exception:
        logger.exception("nba_api lookup failed for name: %s", name)
    return None


@st.cache_data(ttl=60)
def get_recent_stat_series_for_player(player_id: int, stat_abbrev: str, season: Optional[str], last_n: int) -> List[float]:
    """
    Use nba_api to fetch last_n game totals for a stat (PTS, REB/TRB, AST).
    Returns a list of numeric stat values (may be empty).
    """
    if not NBA_API_AVAILABLE or not player_id:
        return []
    try:
        # playergamelog supports last_n_games param in some versions; fall back if not available
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, last_n_games=last_n)
        df = gl.get_data_frames()[0]
        # normalize column name mapping
        col = stat_abbrev
        if col not in df.columns:
            # try TRB for rebounds
            alt = {"REB": "TRB", "PTS": "PTS", "AST": "AST"}.get(stat_abbrev, stat_abbrev)
            col = alt if alt in df.columns else None
        if col and col in df.columns:
            series = []
            for v in df[col].tolist():
                try:
                    series.append(float(v))
                except Exception:
                    continue
            return series
    except Exception:
        logger.exception("Failed to fetch gamelog for player id %s", player_id)
    return []


def compute_hit_rate_from_series(series: List[float], target: float) -> Optional[float]:
    if not series or target is None:
        return None
    try:
        hits = sum(1 for v in series if v >= target)
        return hits / len(series)
    except Exception:
        return None


# Map stat label to nba_api columns
STAT_ABBREV_MAP = {
    "Points": "PTS",
    "Rebounds": "TRB",  # prefer TRB if present
    "Assists": "AST",
}


# --- Consistency scoring & ranking (fallbacks) -----------------------------
def compute_consistency_score(player: Dict[str, Any], last_n: int = 20) -> float:
    """
    Compute a consistency score where higher is better.
    Primary method:
      - If nba_api available and we can compute a hit_rate, use hit_rate * 100 (0..100)
      - If no history, fallback to a heuristic:
          * already above target -> base 80 + diff*10
          * close below target -> base 60 - gap*10
          * otherwise fallback lower (40)
    """
    # attempt nba_api hit rate
    name = player.get("name") or ""
    target = player.get("target")
    stat_label = player.get("stat_label") or ""
    if NBA_API_AVAILABLE and target is not None:
        pid = _find_player_id_by_name(name)
        if pid:
            abbrev = STAT_ABBREV_MAP.get(stat_label, None)
            if abbrev:
                series = get_recent_stat_series_for_player(pid, abbrev, season=None, last_n=last_n)
                hit_rate = compute_hit_rate_from_series(series, target)
                if hit_rate is not None:
                    # score = hit_rate scaled (0..100) + small bonus for recent mean relative to target
                    mean_recent = (sum(series) / len(series)) if series else 0.0
                    bonus = max(0.0, min(5.0, (mean_recent - target)))  # small bump
                    return float(hit_rate * 100.0 + bonus)
    # Fallback heuristic using live snapshot
    try:
        lv = player.get("live_value")
        if isinstance(lv, (int, float)) and isinstance(target, (int, float)):
            diff = lv - target
            if diff >= 0:
                return min(150.0, 80.0 + diff * 10.0)
            else:
                return max(0.0, 60.0 - abs(diff) * 10.0)
    except Exception:
        pass
    return 30.0


def get_top_consistent(players_list: List[Dict[str, Any]], n: int = 5, last_n: int = 20) -> List[Dict[str, Any]]:
    scored = []
    for p in players_list:
        try:
            score = compute_consistency_score(p, last_n=last_n)
        except Exception:
            score = 0.0
        p_copy = dict(p)
        p_copy["_consistency_score"] = score
        scored.append(p_copy)
    scored_sorted = sorted(scored, key=lambda x: x["_consistency_score"], reverse=True)
    return scored_sorted[:n]


# --- UI helpers ---------------------------------------------------------------
def render_player_card(player: Dict[str, Any], show_pace: bool = True, show_target: bool = True) -> None:
    """Render a player card into the current Streamlit column context."""
    st.subheader(player.get("name", "Unknown"))

    metric_label = f"Live {player.get('stat_label', 'Stat')}"
    delta_display = player.get("delta", "")
    delta_numeric = None
    if isinstance(delta_display, (int, float)):
        delta_numeric = delta_display
    elif isinstance(delta_display, str) and delta_display.strip():
        import re
        m = re.match(r"^([+-]?\d+(\.\d+)?)", delta_display.strip())
        if m:
            try:
                delta_numeric = float(m.group(1))
            except Exception:
                delta_numeric = None

    if delta_numeric is not None:
        st.metric(label=metric_label, value=str(player.get("live_value", "—")), delta=f"{delta_numeric}")
    else:
        st.metric(label=metric_label, value=str(player.get("live_value", "—")), delta=str(delta_display))

    caption_parts = []
    target = player.get("target")
    live_value = player.get("live_value")
    if show_target and target is not None:
        status = ""
        try:
            if isinstance(live_value, (int, float)) and isinstance(target, (int, float)):
                status = "✅" if live_value >= target else "⚠️"
        except Exception:
            status = ""
        caption_parts.append(f"Target: {target} {status}")

    if show_pace and player.get("pace") not in (None, ""):
        caption_parts.append(f"Pace: {player.get('pace')}")

    team = player.get("team")
    money = player.get("money")
    if team:
        caption_parts.append(f"Team: {team}")
    if money:
        caption_parts.append(f"Money: {money}")

    if caption_parts:
        st.caption(" | ".join(caption_parts))


# --- Main layout -------------------------------------------------------------
st.title("🏀 NBA Player Prop Tracker")

col_refresh, _ = st.columns([1, 6])
with col_refresh:
    if st.button("Refresh now"):
        safe_rerun()

# If TheOddsAPI selected but no key, show message
if data_source == "TheOddsAPI (live props)" and not THEODDS_API_KEY:
    st.warning("Selected TheOddsAPI but no API key provided. Set THEODDS_API_KEY in the sidebar or environment to fetch live props.")

# Top consistent picks (if enabled)
if enable_top_consistency:
    st.subheader(f"Top {int(top_n)} players to bet on — Ranked by recent consistency")
    top_consistent = get_top_consistent(players, int(top_n), last_n=int(recent_games))
    if not top_consistent:
        st.info("No players available to rank.")
    else:
        rows = []
        for i, p in enumerate(top_consistent, start=1):
            # try to compute a hit_rate summary if possible
            hit_rate_display = ""
            if NBA_API_AVAILABLE:
                pid = _find_player_id_by_name(p.get("name") or "")
                if pid and p.get("target") is not None:
                    abbrev = STAT_ABBREV_MAP.get(p.get("stat_label"), None)
                    if abbrev:
                        series = get_recent_stat_series_for_player(pid, abbrev, season=None, last_n=int(recent_games))
                        hr = compute_hit_rate_from_series(series, p.get("target"))
                        if hr is not None:
                            hit_rate_display = f"{hr:.2%}"
            rows.append({
                "rank": i,
                "name": p.get("name"),
                "team": p.get("team") or "",
                "stat": p.get("stat_label"),
                "live": p.get("live_value"),
                "target": p.get("target"),
                "pace": p.get("pace"),
                "delta": p.get("delta"),
                "money": p.get("money") or "",
                "consistency_score": round(p.get("_consistency_score", 0.0), 2),
                "hit_rate": hit_rate_display,
            })
        df_top = pd.DataFrame(rows)
        # display as table
        st.table(df_top)

        with st.expander("Show detailed cards for Top picks"):
            cols = st.columns(min(3, max(1, len(top_consistent))))
            for idx, pl in enumerate(top_consistent):
                c = cols[idx % len(cols)]
                with c:
                    st.markdown(f"#### #{idx+1} — {pl.get('name')}")
                    render_player_card(pl, show_pace=show_pace, show_target=show_target)
                    st.write(f"Consistency score: {round(pl.get('_consistency_score', 0.0),2)}")
                    if NBA_API_AVAILABLE and pl.get("target") is not None:
                        pid = _find_player_id_by_name(pl.get("name") or "")
                        if pid:
                            abbrev = STAT_ABBREV_MAP.get(pl.get("stat_label"), None)
                            if abbrev:
                                series = get_recent_stat_series_for_player(pid, abbrev, season=None, last_n=int(recent_games))
                                if series:
                                    st.write(f"Recent {len(series)} games mean: {sum(series)/len(series):.2f}")
                                    st.write(f"Recent values: {series[:min(8,len(series))]}")

# Main grid of player cards
if not players:
    st.info("No players configured. Use the sidebar to add players or select a data source.")
else:
    cols = st.columns(num_cols)
    for idx, player in enumerate(players):
        col = cols[idx % num_cols]
        with col:
            render_player_card(player, show_pace=show_pace, show_target=show_target)

# Footer / notes
st.write("---")
st.caption(
    "Notes: \n"
    "- Top players are ranked by a 'consistency score'. If nba_api is installed the page will compute a hit-rate from recent games (recommended). Otherwise a live-snapshot heuristic is used.\n"
    "- To use live sportsbook props install a TheOddsAPI key and select 'TheOddsAPI (live props)' as data source in the sidebar.\n"
    "- If you want more accurate modeling I can tweak the scoring (e.g., weight recent games, home/away adjustments, minutes played filters, or combine implied probability from moneylines).\n"
)
