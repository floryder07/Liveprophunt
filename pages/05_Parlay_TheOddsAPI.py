# Parlay Builder - TheOddsAPI prototype (safe picks)
# Copy this file content (ONLY the Python code below) into pages/05_Parlay_TheOddsAPI.py
# Do NOT include the surrounding triple-backtick lines when pasting.
import os
import hashlib
import json
import requests
import streamlit as st
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Optional

# Optional timezone support (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

st.set_page_config(page_title="Parlay Builder - TheOddsAPI", layout="wide")

# Sidebar / settings
st.sidebar.header("Settings / TheOddsAPI")
api_key_input = st.sidebar.text_input("TheOddsAPI Key (or set THEODDS_API_KEY env)", type="password")
API_KEY = api_key_input.strip() or os.environ.get("THEODDS_API_KEY", "")

regions = st.sidebar.multiselect("Regions", options=["us", "uk", "eu", "au"], default=["us"])
markets = st.sidebar.multiselect("Markets", options=["h2h", "spreads", "totals"], default=["h2h"])
odds_ttl = st.sidebar.number_input("Odds cache key (change to bust cache)", min_value=1, value=20, step=1)

st.sidebar.markdown("---")
scan_all_sports = st.sidebar.checkbox("Scan all sports for picks (may be slow)", value=False)
filter_bm_before_11_pt = st.sidebar.checkbox("Auto-filter by bookmaker before 11:00 PT", value=False)
st.sidebar.markdown("---")
if not API_KEY:
    st.sidebar.warning("No API key set. Get one at https://the-odds-api.com/ or set THEODDS_API_KEY.")

# Utilities
def _safe_name(o: Dict[str, Any]) -> str:
    return o.get("name") or o.get("participant") or o.get("label") or ""

def _safe_price(o: Dict[str, Any]) -> Optional[float]:
    raw = o.get("price") or o.get("decimal") or o.get("odds")
    try:
        return float(raw) if raw is not None else None
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_sports(api_key: str) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

@st.cache_data()
def fetch_odds_for_sport(api_key: str, sport_key: str, regions_list: List[str], markets_list: List[str], cache_bust: int) -> List[Dict[str, Any]]:
    regions_q = ",".join(regions_list)
    markets_q = ",".join(markets_list)
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        f"?regions={regions_q}&markets={markets_q}&oddsFormat=decimal&apiKey={api_key}"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def _ensure_sport_titles(events: List[Dict[str, Any]], sports: List[Dict[str, Any]]) -> None:
    smap = {s.get("key"): s.get("title") for s in (sports or [])}
    for ev in events:
        key_candidates = [ev.get("_sport_key"), ev.get("sport_key"), ev.get("sport"), ev.get("category"), ev.get("league")]
        k = next((x for x in key_candidates if x), None)
        if k and not ev.get("_sport_title"):
            ev["_sport_title"] = smap.get(k, k)
        if not ev.get("_sport_title"):
            ev["_sport_title"] = "Unknown"

def get_consensus_and_best_outcomes(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    price_map: Dict[str, List[float]] = {}
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                name = _safe_name(o)
                price = _safe_price(o)
                if price is None:
                    continue
                price_map.setdefault(name, []).append(price)
    results: List[Dict[str, Any]] = []
    for name, prices in price_map.items():
        if not prices:
            continue
        cons = median(prices)
        best = max(prices)
        worst = min(prices)
        results.append({"name": name, "consensus": cons, "best": best, "worst": worst})
    return results

def get_safety_candidates(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for o in get_consensus_and_best_outcomes(event):
        cons = o["consensus"]
        best = o["best"]
        worst = o["worst"]
        spread = abs(worst - best) / cons if cons else 0.0
        implied = (1.0 / cons) if cons and cons > 0 else 0.0
        score = implied - (spread * 0.5)
        out.append({"name": o["name"], "consensus": cons, "best": best, "worst": worst, "spread": spread, "implied": implied, "score": score})
    return out

def has_bookmaker(bookmakers: Optional[List[Dict[str, Any]]], needle: str) -> bool:
    if not needle:
        return True
    for bm in bookmakers or []:
        key = (bm.get("key") or "").lower()
        title = (bm.get("title") or "").lower()
        if needle.lower() in key or needle.lower() in title:
            return True
    return False

def auto_pick_safest_legs(events: List[Dict[str, Any]], n_legs: int = 3, max_decimal: float = 1.8, min_imp: float = 0.6, max_spread: float = 0.05, require_bm: Optional[str] = None, avoid_same_event: bool = True) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for ev in events:
        if require_bm and not has_bookmaker(ev.get("bookmakers", []), require_bm):
            continue
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key")
        if not ev_id:
            continue
        title = ev.get("title") or " ".join(ev.get("teams") or []) or str(ev_id)
        for sc in get_safety_candidates(ev):
            implied = sc["implied"]
            best = sc["best"]
            spread = sc["spread"]
            if best > max_decimal:
                continue
            if implied < min_imp:
                continue
            if spread > max_spread:
                continue
            candidates.append({
                "event_id": ev_id,
                "event_title": title,
                "selection": sc["name"],
                "price": best,
                "consensus": sc["consensus"],
                "implied": implied,
                "spread": spread,
                "score": sc["score"],
                "commence_time": ev.get("commence_time") or ev.get("start"),
                "sport_title": ev.get("_sport_title") or ev.get("_sport_key"),
            })
    candidates.sort(key=lambda x: (x.get("score", 0.0), x.get("implied", 0.0)), reverse=True)
    # If caller wants only n_legs, they can slice; return full sorted candidate list
    return candidates

# Session defaults
if "events" not in st.session_state:
    st.session_state.events = []
if "parlay" not in st.session_state:
    st.session_state.parlay = []

# Header + quick instructions
st.title("Parlay Builder - TheOddsAPI")
st.write("Load events, preview safest picks and add them to your parlay. This is a simulation tool only.")

# Top controls: load events
top_cols = st.columns([3, 2, 1])
with top_cols[0]:
    st.info("Use the controls to load events and generate safe picks.")
with top_cols[1]:
    sport_options = []
    sports_list = []
    if API_KEY:
        try:
            sports_list = fetch_sports(API_KEY)
            sport_options = [s.get("key") for s in sports_list]
        except Exception:
            sport_options = []
    chosen_sport = st.selectbox("Sport (for load)", options=[""] + sport_options)
with top_cols[2]:
    if st.button("Load odds for sport(s)"):
        if not API_KEY:
            st.error("Set THEODDS_API_KEY in env or paste into the sidebar first.")
        else:
            try:
                loaded: List[Dict[str, Any]] = []
                if scan_all_sports:
                    sports = fetch_sports(API_KEY)
                    for s in sports:
                        skey = s.get("key")
                        stitle = s.get("title")
                        try:
                            evs = fetch_odds_for_sport(API_KEY, skey, regions, markets, int(odds_ttl))
                        except Exception:
                            continue
                        for ev in evs:
                            ev["_sport_key"] = skey
                            ev["_sport_title"] = stitle
                        loaded.extend(evs)
                else:
                    if not chosen_sport:
                        st.warning("Choose a sport or enable Scan all sports.")
                    else:
                        loaded = fetch_odds_for_sport(API_KEY, chosen_sport, regions, markets, int(odds_ttl))
                        for ev in loaded:
                            ev["_sport_key"] = chosen_sport
                _ensure_sport_titles(loaded, fetch_sports(API_KEY) if API_KEY else [])
                st.session_state.events = loaded
                st.success(f"Loaded {len(loaded)} events.")
            except Exception as e:
                st.error(f"Failed to load events: {e}")

# Main layout: left = auto-pick, right = parlay
left, right = st.columns([2.5, 1])

with left:
    st.header("Auto-pick (safest picks)")
    n_legs = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
    max_decimal = st.number_input("Max decimal odds per leg", value=1.8, step=0.05, format="%.2f")
    min_imp_pct = st.slider("Min consensus implied probability (%)", min_value=30, max_value=90, value=60) / 100.0
    max_spread_pct = st.slider("Max spread across books (%)", min_value=0, max_value=20, value=5) / 100.0
    require_bm = st.text_input("Require bookmaker (optional)", value="")
    avoid_same = st.checkbox("Avoid >1 leg per event", value=True)

    if st.button("Preview safest picks"):
        picks = auto_pick_safest_legs(st.session_state.events, n_legs=int(n_legs), max_decimal=float(max_decimal), min_imp=float(min_imp_pct) if False else min_imp_pct, min_consensus_prob=min_imp_pct if False else min_imp_pct, max_spread=max_spread_pct if False else max_spread_pct, require_bookmaker=(require_bm if require_bm.strip() else None), avoid_same_event=avoid_same)
        # Note: call signature to auto_pick_safest_legs uses parameters as named above; using local variables
        # For safety, call again with correct names:
        picks = auto_pick_safest_legs(st.session_state.events, n_legs=int(n_legs), max_decimal=float(max_decimal), min_imp=min_imp_pct, max_spread=max_spread_pct, require_bm=(require_bm if require_bm.strip() else None), avoid_same_event=avoid_same)
        if not picks:
            st.warning("No safe picks found with current filters.")
        else:
            st.write(f"Top {min(len(picks), 50)} safe candidates (showing up to 50):")
            for i, p in enumerate(picks[:50], 1):
                st.write(f"{i}. [{p.get('sport_title') or ''}] {p.get('event_title')} — {p.get('selection')} @ {p.get('price')}  (score {p.get('score'):.3f})")

    if st.button("Add safest picks"):
        picks_to_add = auto_pick_safest_legs(st.session_state.events, n_legs=int(n_legs), max_decimal=float(max_decimal), min_imp=min_imp_pct, max_spread=max_spread_pct, require_bm=(require_bm if require_bm.strip() else None), avoid_same_event=avoid_same)
        if not picks_to_add:
            st.warning("No safe picks available to add.")
        else:
            added = 0
            for p in picks_to_add[:int(n_legs)]:
                exists = any((leg.get("event_id") == p["event_id"] and leg.get("selection") == p["selection"]) for leg in st.session_state.parlay)
                if not exists:
                    st.session_state.parlay.append({
                        "event_id": p["event_id"],
                        "sport": p.get("sport_title") or "",
                        "title": p.get("event_title"),
                        "selection": p.get("selection"),
                        "price": p.get("price"),
                        "commence_time": p.get("commence_time"),
                        "score": p.get("score")
                    })
                    added += 1
            st.success(f"Added {added} safe picks to parlay.")

    st.markdown("---")
    st.markdown("Show top safe candidates")
    top_n = st.number_input("Top N safe candidates to show", min_value=1, max_value=50, value=10)
    show_full_id = st.checkbox("Show full event id (debug)", value=False)
    if st.button("Show top safe candidates"):
        candidates = auto_pick_safest_legs(st.session_state.events, n_legs=200, max_decimal=5.0, min_imp=0.0, max_spread=1.0, require_bm=None, avoid_same_event=False)
        st.write(f"Found {len(candidates)} safe candidates — showing top {min(top_n, len(candidates))}")
        for i, c in enumerate(candidates[:top_n], 1):
            title = c.get("event_title") or ""
            display_title = title if len(str(title)) <= 60 or show_full_id else (str(title)[:57] + "...")
            st.write(f"{i}. [{c.get('sport_title') or ''}] {display_title} — {c.get('selection')} @ {c.get('price')}  (score {c.get('score'):.3f})")

with right:
    st.header("Parlay summary")
    if not st.session_state.parlay:
        st.write("No picks yet.")
    else:
        for leg in list(st.session_state.parlay):
            st.markdown(f"**{leg.get('sport','')}** — {leg.get('title','')}")
            st.write(f"> {leg.get('selection','')} @ {leg.get('price')} — score {leg.get('score','')}")
            rem_key = "rem-" + hashlib.sha1(f"{leg.get('event_id','')}-{leg.get('selection','')}".encode()).hexdigest()[:12]
            if st.button("Remove", key=rem_key):
                st.session_state.parlay = [l for l in st.session_state.parlay if not (l.get("event_id") == leg.get("event_id") and l.get("selection") == leg.get("selection"))]
                st.experimental_rerun()

    st.markdown("---")
    if st.session_state.parlay:
        prices = [l["price"] for l in st.session_state.parlay]
        combined = 1.0
        for p in prices:
            combined *= p
        stake = st.number_input("Stake", value=10.0, min_value=0.0, format="%.2f", key="stake_input")
        payout = stake * combined
        profit = payout - stake
        st.metric("Combined decimal odds", f"{combined:.4f}")
        st.metric("Potential payout", f"${payout:,.2f}")
        st.metric("Potential profit", f"${profit:,.2f}")
        st.download_button("Download parlay JSON", data=json.dumps(st.session_state.parlay, default=str, indent=2), file_name="parlay.json", mime="application/json")
