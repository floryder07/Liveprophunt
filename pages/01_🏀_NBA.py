import logging
from typing import Dict, List, Optional

import streamlit as st

# Page configuration
st.set_page_config(title="NBA Player Prop Tracker", layout="wide")
logging.basicConfig(level=logging.INFO)


# --- Configuration / Mock data ------------------------------------------------
DEFAULT_PLAYERS: List[Dict] = [
    {
        "name": "LeBron James",
        "stat_label": "Points",
        "live_value": 22,
        "delta": "+2.5 vs Line",
        "target": 19.5,
        "pace": 28.1,
        "player_id": None,
    },
    {
        "name": "Kevin Durant",
        "stat_label": "Rebounds",
        "live_value": 6,
        "delta": "-1.5 vs Line",
        "target": 7.5,
        "pace": 6.8,
        "player_id": None,
    },
]


# --- Data fetching (replace with real API) -----------------------------------
@st.cache_data(ttl=10)  # cache for 10s to avoid spamming an API
def fetch_player_stats(player_id: Optional[int], name: str) -> Dict:
    """
    Placeholder fetch function. Replace this with a call to your live-data API.
    Return dict must contain keys: live_value, delta, target, pace (or compatible).
    """
    logging.info("Fetching stats for %s (id=%s)", name, player_id)
    # Mock return (in real use, call the API and return parsed values)
    # Example response shape:
    return {
        "live_value": None,  # None means use the static value defined in the player dict
        "delta": None,
        "target": None,
        "pace": None,
    }


# --- UI helpers ---------------------------------------------------------------
def render_player_card(player: Dict, show_pace: bool = True, show_target: bool = True) -> None:
    """
    Render a single player card to the current Streamlit column context.
    """
    # Merge live API values (if any) with the static/default player dict
    api = fetch_player_stats(player.get("player_id"), player["name"])

    live_value = api.get("live_value") or player.get("live_value", "—")
    delta = api.get("delta") or player.get("delta", "")
    target = api.get("target") or player.get("target", "")
    pace = api.get("pace") or player.get("pace", "")

    st.subheader(player["name"])
    # Show the main metric (stat label + live value)
    metric_label = f"Live {player.get('stat_label', 'Stat')}"
    st.metric(label=metric_label, value=str(live_value), delta=str(delta))
    # Small caption for extra info
    caption_parts = []
    if show_target and target != "":
        caption_parts.append(f"Target: {target}")
    if show_pace and pace != "":
        caption_parts.append(f"Pace: {pace}")
    if caption_parts:
        st.caption(" | ".join(caption_parts))


# --- Main page ---------------------------------------------------------------
st.title("🏀 NBA Player Prop Tracker")

# Sidebar controls
st.sidebar.header("Settings")
num_cols = st.sidebar.selectbox("Columns", options=[1, 2, 3], index=1)
show_pace = st.sidebar.checkbox("Show pace", value=True)
show_target = st.sidebar.checkbox("Show target", value=True)
use_mock_data = st.sidebar.checkbox("Use mock/static data (no API)", value=True)

# Allow user to add/edit the list of players quickly (simple text area)
st.sidebar.markdown("Edit players (one per line):")
players_text = st.sidebar.text_area(
    "Format: name|stat_label|live_value|delta|target|pace",
    value="\n".join(
        f"{p['name']}|{p['stat_label']}|{p['live_value']}|{p['delta']}|{p['target']}|{p['pace']}"
        for p in DEFAULT_PLAYERS
    ),
    height=120,
)

# Parse players_text into player dicts (very forgiving)
def parse_players(text: str) -> List[Dict]:
    players: List[Dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        # Fill missing parts with defaults
        while len(parts) < 6:
            parts.append("")
        name, stat_label, live_value, delta, target, pace = parts[:6]
        try:
            live_value_val = int(live_value) if live_value != "" else None
        except ValueError:
            try:
                live_value_val = float(live_value)
            except Exception:
                live_value_val = live_value or None
        players.append(
            {
                "name": name or "Unknown",
                "stat_label": stat_label or "Stat",
                "live_value": live_value_val,
                "delta": delta,
                "target": float(target) if target else None,
                "pace": float(pace) if pace else None,
                "player_id": None,
            }
        )
    return players


players = parse_players(players_text)

# Refresh controls
col_refresh, _ = st.columns([1, 4])
with col_refresh:
    if st.button("Refresh"):
        st.experimental_rerun()

# Layout players across columns
if not players:
    st.info("No players configured. Use the sidebar to add players.")
else:
    cols = st.columns(num_cols)
    # Distribute players into columns round-robin style
    for idx, player in enumerate(players):
        col = cols[idx % num_cols]
        with col:
            render_player_card(player, show_pace=show_pace, show_target=show_target)

# Footer / hints
st.write("---")
st.caption("Tip: Replace fetch_player_stats with your live-data API. Use caching to avoid API rate limits.")
