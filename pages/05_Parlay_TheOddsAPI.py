"""
Parlay Builder — TheOddsAPI prototype (Streamlit)

Merged with an "auto-pick" feature that selects legs based on market value
(best bookmaker price vs market consensus) and explains why each pick was chosen.

Usage:
- Provide THEODDS_API_KEY via env or paste in the sidebar.
- Load sport odds, then use "Auto-pick by market value" to preview or add picks.
- This is a recommendation/simulation tool only.
"""
from typing import Any, Dict, List, Optional
import os
import math
import requests
import streamlit as st
from datetime import datetime
from statistics import median

st.set_page_config(page_title="Parlay Builder — TheOddsAPI", page_icon="🎯", layout="wide")
st.title("🎯 Parlay Builder — TheOddsAPI prototype")

# ---- Sidebar: API key and options ----
st.sidebar.header("Settings / TheOddsAPI")
api_key_input = st.sidebar.text_input("TheOddsAPI Key (or leave blank to use THEODDS_API_KEY env)", type="password")
API_KEY = api_key_input.strip() or os.environ.get("THEODDS_API_KEY", "")

regions = st.sidebar.multiselect("Regions (comma-select)", options=["us", "uk", "eu", "au"], default=["us"])
markets = st.sidebar.multiselect("Markets", options=["h2h", "spreads", "totals"], default=["h2h"])
bookmaker_filter = st.sidebar.text_input("Preferred bookmaker key/title (optional)", value="")
odds_ttl = st.sidebar.number_input("Odds cache key (change to bust cache)", min_value=1, value=20, step=1)
refresh = st.sidebar.button("Reload sports list / clear cache")

if not API_KEY:
    st.sidebar.warning("No API key set. Get one at https://the-odds-api.com/ and put it in THEODDS_API_KEY or paste here.")

# ---- Helpers ----
def decimal_to_american(d: float) -> str:
    if d <= 1:
        return "N/A"
    if d >= 2.0:
        return f"+{round((d - 1) * 100)}"
    try:
        return str(round(-100 / (d - 1)))
    except Exception:
        return "N/A"

def prod(xs: List[float]) -> float:
    p = 1.0
    for x in xs:
        p *= x
    return p

@st.cache_data(ttl=300)
def fetch_sports(api_key: str) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

# Note: include odds_ttl as a parameter so changing it busts the cache.
@st.cache_data()
def fetch_odds_for_sport(api_key: str, sport_key: str, regions_list: List[str], markets_list: List[str], cache_bust: int) -> List[Dict[str, Any]]:
    regions_q = ",".join(regions_list)
    markets_q = ",".join(markets_list)
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions={regions_q}&markets={markets_q}&oddsFormat=decimal&apiKey={api_key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def extract_outcomes(event: Dict[str, Any], bookmaker_filter: Optional[str]=None) -> List[Dict[str, Any]]:
    outcomes = []
    for bm in event.get("bookmakers", []):
        if bookmaker_filter and bookmaker_filter.strip():
            if bm.get("key") != bookmaker_filter and bm.get("title") != bookmaker_filter:
                continue
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                price = o.get("price") or o.get("decimal") or o.get("odds")
                try:
                    price = float(price)
                except Exception:
                    price = None
                if price:
                    outcomes.append({
                        "name": o.get("name") or o.get("participant") or o.get("label"),
                        "price": price,
                        "market": m.get("key"),
                        "bookmaker": bm.get("key"),
                    })
    return outcomes

# --- Auto-pick utilities (value-based heuristic) ---
def get_consensus_and_best_outcomes(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    For an event, return outcomes with consensus (median across all sources)
    and best_price (max across bookmakers).
    """
    price_map = {}  # name -> list[float]
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                name = o.get("name") or o.get("participant") or o.get("label")
                try:
                    price = float(o.get("price"))
                except Exception:
                    continue
                price_map.setdefault(name, []).append(price)

    results = []
    for name, prices in price_map.items():
        if not prices:
            continue
        cons = median(prices)
        best = max(prices)
        results.append({"name": name, "consensus": cons, "best_price": best})
    return results

def score_outcome_value(consensus: float, best_price: float) -> float:
    if consensus <= 0:
        return 0.0
    return (best_price / consensus) - 1.0

def auto_pick_legs_by_value(events: List[Dict,], n_legs: int = 3, min_value: float = 0.02, avoid_same_event: bool = True) -> List[Dict[str, Any]]:
    """
    Returns a list of picks:
      { event_id, event_title, selection, price, consensus, value, implied_consensus, reason }
    """
    candidates = []
    for ev in events:
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("title")
        title = ev.get("title") or " vs ".join(ev.get("teams", [])) or str(ev_id)
        outs = get_consensus_and_best_outcomes(ev)
        for o in outs:
            cons = o["consensus"]
            best = o["best_price"]
            value = score_outcome_value(cons, best)
            implied_consensus = (1.0 / cons) if cons > 0 else None
            implied_best = (1.0 / best) if best > 0 else None
            favorite_flag = cons < 2.0 if cons else False
            candidates.append({
                "event_id": ev_id,
                "event_title": title,
                "selection": o["name"],
                "price": best,
                "consensus": cons,
                "value": value,
                "implied_consensus": implied_consensus,
                "implied_best": implied_best,
                "favorite": favorite_flag
            })

    # sort by value desc, then by best price desc
    candidates.sort(key=lambda c: (c["value"], c["price"]), reverse=True)

    selected = []
    used_events = set()
    for c in candidates:
        if len(selected) >= n_legs:
            break
        if c["value"] < min_value:
            continue
        if avoid_same_event and c["event_id"] in used_events:
            continue
        reason_parts = []
        reason_parts.append(f"Best price {c['price']:.2f} vs consensus {c['consensus']:.2f} ({c['value']*100:.1f}% uplift)")
        if c["favorite"]:
            reason_parts.append("Market consensus marks this as a favorite.")
        else:
            reason_parts.append("Market consensus marks this as an underdog.")
        if c["implied_best"] and c["implied_consensus"]:
            reason_parts.append(f"Implied best prob {c['implied_best']*100:.1f}% vs consensus {c['implied_consensus']*100:.1f}%")
        reason = " — ".join(reason_parts)
        selected.append({
            "event_id": c["event_id"],
            "event_title": c["event_title"],
            "selection": c["selection"],
            "price": c["price"],
            "consensus": c["consensus"],
            "value": c["value"],
            "implied_consensus": c["implied_consensus"],
            "reason": reason
        })
        used_events.add(c["event_id"])
    return selected

def kelly_fraction(p: float, decimal_odds: float, f: float = 0.25) -> float:
    """
    Conservative Kelly fraction suggestion using implied prob p as model.
    Returns fraction of bankroll to stake.
    """
    b = decimal_odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    numerator = (p * (b + 1) - 1.0)
    denom = b
    if denom == 0:
        return 0.0
    raw = numerator / denom
    return max(0.0, f * raw)

# ---- UI: sports selector ----
sports: List[Dict[str, Any]] = []
if API_KEY:
    try:
        sports = fetch_sports(API_KEY)
    except Exception as e:
        st.sidebar.error(f"Could not fetch sports list: {e}")

sport_options = {s.get("key"): s.get("title") for s in sports}
selected_sport = st.sidebar.selectbox("Sport (select)", options=[""] + list(sport_options.keys()), format_func=lambda k: sport_options.get(k, " — choose sport — "))

# ---- Load events/odds ----
load_odds_btn = st.sidebar.button("Load odds for sport")
events: List[Dict[str, Any]] = []

if selected_sport:
    try:
        # include odds_ttl as cache-busting argument
        events = fetch_odds_for_sport(API_KEY, selected_sport, regions, markets, int(odds_ttl))
    except Exception as e:
        st.sidebar.error(f"Could not load odds: {e}")
        events = []

# Sample fallback
if not events:
    st.info("No live feed loaded — using sample events.")
    events = [
        {
            "id": "ev-1",
            "title": "Ma Long vs Fan Zhendong",
            "commence_time": "2026-01-05T12:00:00Z",
            "teams": ["Ma Long", "Fan Zhendong"],
            "bookmakers": [{"key":"draftkings","markets":[{"key":"h2h","outcomes":[{"name":"Ma Long","price":1.45},{"name":"Fan Zhendong","price":2.10}]}]}]
        },
        {
            "id": "ev-2",
            "title": "Player A vs Player B",
            "commence_time": "2026-01-05T13:30:00Z",
            "teams": ["Player A", "Player B"],
            "bookmakers": [{"key":"draftkings","markets":[{"key":"h2h","outcomes":[{"name":"Player A","price":1.80},{"name":"Player B","price":2.00}]}]}]
        }
    ]

# ---- Parlay session state ----
if "parlay" not in st.session_state:
    st.session_state.parlay = []

def add_leg(ev_id: str, ev_title: str, sel_name: str, price: float):
    for leg in st.session_state.parlay:
        if leg["event_id"] == ev_id:
            st.warning("You already added a leg from this event. Remove it first to add another selection.")
            return
    st.session_state.parlay.append({"event_id": ev_id, "title": ev_title, "selection": sel_name, "price": price})
    st.experimental_rerun()

def remove_leg(i: int):
    st.session_state.parlay.pop(i)
    st.experimental_rerun()

# ---- Main layout ----
left, right = st.columns([2, 1])

with left:
    st.header("Available events & outcomes")
    for ev in events:
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("title")
        title = ev.get("title") or " vs ".join(ev.get("teams", [])) or str(ev_id)
        outs = extract_outcomes(ev, bookmaker_filter=bookmaker_filter)
        if not outs:
            with st.expander(f"{title} (no odds)"):
                st.write("No usable odds/outcomes found for this event.")
            continue
        with st.expander(title):
            for o in outs:
                cols = st.columns([4, 1])
                cols[0].write(f"{o['name']}  —  {o['price']} (market: {o['market']} | bm: {o['bookmaker']})")
                if cols[1].button("Add", key=f"add-{ev_id}-{o['name']}"):
                    add_leg(ev_id, title, o["name"], o["price"])

with right:
    st.header("Parlay builder")

    # --- Auto-pick controls ---
    st.subheader("Auto-pick by market value")
    auto_n = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
    auto_min_pct = st.slider("Minimum uplift vs consensus (%)", min_value=0, max_value=50, value=2)
    auto_min_value = auto_min_pct / 100.0
    avoid_same_event = st.checkbox("Avoid >1 leg per event", value=True)
    preview_only = st.checkbox("Preview only (don't auto-add)", value=True)
    if st.button("Auto-pick by market value"):
        picks = auto_pick_legs_by_value(events, n_legs=auto_n, min_value=auto_min_value, avoid_same_event=avoid_same_event)
        if not picks:
            st.warning("No picks met the criteria.")
        else:
            st.success(f"Found {len(picks)} pick(s). {'Previewing — not added.' if preview_only else 'Added to parlay.'}")
            for p in picks:
                with st.expander(f"{p['event_title']} — {p['selection']} @ {p['price']}"):
                    st.write("Reason:", p["reason"])
                    st.write(f"Consensus: {p['consensus']:.2f} | Value uplift: {p['value']*100:.2f}%")
                    if p.get("implied_consensus"):
                        kf = kelly_fraction(p["implied_consensus"], p["price"], f=0.25)
                        st.write(f"Suggested Kelly fraction (conservative 25% Kelly): {kf*100:.2f}% of bankroll (theoretical).")
                    if not preview_only:
                        add_leg(p['event_id'], p['event_title'], p['selection'], p['price'])

    # ---- Current parlay UI ----
    legs = st.session_state.parlay
    if not legs:
        st.info("Add legs from the left column or use Auto-pick.")
    else:
        for idx, leg in enumerate(legs):
            c = st.columns([3, 1])
            c[0].markdown(f"**{leg['title']}** — {leg['selection']} @ {leg['price']}")
            if c[1].button("Remove", key=f"rem-{idx}"):
                remove_leg(idx)

    st.markdown("---")
    stake = st.number_input("Stake", value=10.0, min_value=0.0, format="%.2f")
    if legs:
        prices = [l["price"] for l in legs]
        combined = prod(prices)
        payout = stake * combined
        profit = payout - stake
        st.metric("Combined decimal odds", f"{combined:.4f}")
        st.metric("Combined American odds", decimal_to_american(combined))
        st.metric("Potential payout", f"${payout:,.2f}")
        st.metric("Potential profit", f"${profit:,.2f}")
        if len(legs) < 2:
            st.warning("Parlay typically requires 2+ legs.")
        else:
            if st.button("Simulate parlay"):
                st.success(f"Simulated: stake ${stake:.2f} → payout ${payout:,.2f} (profit ${profit:,.2f})")

st.caption("Prototype uses TheOddsAPI. Auto-picks are heuristic suggestions (value vs market consensus). This is a simulation tool only — do not auto-place real bets without user confirmation and proper licensing.")
