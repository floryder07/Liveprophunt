"""
Parlay Builder — TheOddsAPI prototype (Streamlit)

- Reads THEODDS_API_KEY from env or an obscured sidebar input.
- Lets you pick a sport (auto-fetched), load odds for that sport, filter by bookmaker,
  add outcomes to a parlay, and compute combined decimal/american odds and payout.
- Uses simple caching to avoid hammering the API (ttl for odds is configurable).
"""
from typing import Any, Dict, List, Optional
import os
import math
import requests
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Parlay Builder — TheOddsAPI", page_icon="🎯", layout="wide")
st.title("🎯 Parlay Builder — TheOddsAPI prototype")

# ---- Sidebar: API key and options ----
st.sidebar.header("Settings / TheOddsAPI")
api_key_input = st.sidebar.text_input("TheOddsAPI Key (or leave blank to use THEODDS_API_KEY env)", type="password")
API_KEY = api_key_input.strip() or os.environ.get("THEODDS_API_KEY", "")

regions = st.sidebar.multiselect("Regions (comma-select)", options=["us", "uk", "eu", "au"], default=["us"])
markets = st.sidebar.multiselect("Markets", options=["h2h", "spreads", "totals"], default=["h2h"])
bookmaker_filter = st.sidebar.text_input("Preferred bookmaker key/title (optional)", value="")
odds_ttl = st.sidebar.number_input("Odds cache TTL (seconds)", min_value=5, value=20, step=5)
refresh = st.sidebar.button("Reload sports list / clear cache")

if not API_KEY:
    st.sidebar.warning("No API key set. Get one at https://the-odds-api.com/ and put it in THEODDS_API_KEY or paste here.")

# ---- Helpers ----
def decimal_to_american(d: float) -> str:
    if d <= 1:
        return "N/A"
    if d >= 2.0:
        return f"+{round((d - 1) * 100)}"
    return str(round(-100 / (d - 1)))

@st.cache_data(ttl=300)
def fetch_sports(api_key: str) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=odds_ttl)
def fetch_odds_for_sport(api_key: str, sport_key: str, regions_list: List[str], markets_list: List[str]) -> List[Dict[str, Any]]:
    regions_q = ",".join(regions_list)
    markets_q = ",".join(markets_list)
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions={regions_q}&markets={markets_q}&oddsFormat=decimal&apiKey={api_key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def extract_outcomes(event: Dict[str, Any], bookmaker_filter: Optional[str]=None) -> List[Dict[str, Any]]:
    outcomes = []
    # TheOddsAPI shape: event['bookmakers'][...]['markets'][...]['outcomes']
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

def prod(xs: List[float]) -> float:
    p = 1.0
    for x in xs:
        p *= x
    return p

# ---- UI: sports selector ----
sports = []
if API_KEY:
    try:
        sports = fetch_sports(API_KEY)
    except Exception as e:
        st.sidebar.error(f"Could not fetch sports list: {e}")

if not sports:
    st.sidebar.info("No sports loaded (or API key missing). You can still use the sample events below.")
sport_options = {s.get("key"): s.get("title") for s in sports}
selected_sport = st.sidebar.selectbox("Sport (select)", options=[""] + list(sport_options.keys()), format_func=lambda k: sport_options.get(k, " — choose sport — "))

# ---- Load events/odds ----
load_odds_btn = st.sidebar.button("Load odds for sport")
events: List[Dict[str, Any]] = []

if selected_sport:
    try:
        events = fetch_odds_for_sport(API_KEY, selected_sport, regions, markets)
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
    # prevent duplicate event
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
    legs = st.session_state.parlay
    if not legs:
        st.info("Add legs from the left column.")
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

st.caption("Prototype uses TheOddsAPI. This is a simulation tool — not connected to any bookmaker placement API.")
