# Parlay Builder — TheOddsAPI (overwrite with this file)
#
# The complete page: safe-picks, "Preview/Add safest picks", "Show top safe candidates",
# display toggles for teams/players, best bookmaker, and event id.
# Paste this entire file content into pages/05_Parlay_TheOddsAPI.py (overwrite).
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

st.set_page_config(page_title="Parlay Builder — TheOddsAPI", layout="wide")

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

# ---- Utilities ----
def _safe_name(o: Dict[str, Any]) -> str:
    return (o.get("name") or o.get("participant") or o.get("label") or "").strip()

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
                if price is None or not name:
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

def get_best_bookmaker_for_selection(event: Dict[str, Any], selection_name: str) -> Optional[Dict[str, Any]]:
    best_price = None
    best_bm = None
    best_market = None
    for bm in event.get("bookmakers", []):
        bm_title = bm.get("title") or bm.get("key")
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                name = _safe_name(o)
                price = _safe_price(o)
                if name == selection_name and price is not None:
                    if (best_price is None) or (price > best_price):
                        best_price = price
                        best_bm = bm_title
                        best_market = m.get("key") or m.get("market")
    if best_price is None:
        return None
    return {"bookmaker": best_bm or "", "market": best_market or "", "price": best_price}

def has_bookmaker(bookmakers: Optional[List[Dict[str, Any]]], needle: str) -> bool:
    if not needle:
        return True
    for bm in bookmakers or []:
        key = (bm.get("key") or "").lower()
        title = (bm.get("title") or "").lower()
        if needle.lower() in key or needle.lower() in title:
            return True
    return False

def auto_pick_safest_legs(
    events: List[Dict[str, Any]],
    n_legs: int = 3,
    max_decimal: float = 1.8,
    min_imp: float = 0.6,
    max_spread: float = 0.05,
    require_bm: Optional[str] = None,
    avoid_same_event: bool = True,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for ev in events:
        if require_bm and not has_bookmaker(ev.get("bookmakers", []), require_bm):
            continue
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key")
        if not ev_id:
            continue
        teams = ev.get("teams") or []
        league = ev.get("sport_title") or ev.get("_sport_title") or ev.get("sport") or ""
        title = ev.get("title") or " / ".join(teams) or str(ev_id)
        seen = set()
        for o in get_consensus_and_best_outcomes(ev):
            cons = o.get("consensus") or 0.0
            best = o.get("best") or 0.0
            worst = o.get("worst") or best
            spread = abs(worst - best) / cons if cons else 0.0
            implied = (1.0 / cons) if cons and cons > 0 else 0.0
            if best > max_decimal:
                continue
            if implied < min_imp:
                continue
            if spread > max_spread:
                continue
            selection = o.get("name")
            if avoid_same_event and selection in seen:
                continue
            seen.add(selection)
            bm_info = get_best_bookmaker_for_selection(ev, selection) or {}
            pool.append({
                "event_id": ev_id,
                "league": league,
                "teams": teams,
                "event_title": title,
                "selection": selection,
                "price": float(best),
                "consensus": float(cons),
                "implied": float(implied),
                "spread": float(spread),
                "safety_score": float(implied - (spread * 0.5)),
                "commence_time": ev.get("commence_time") or ev.get("start"),
                "best_bookmaker": bm_info.get("bookmaker", ""),
                "best_market": bm_info.get("market", ""),
                "best_book_price": bm_info.get("price", None),
            })
    pool.sort(key=lambda x: (x.get("safety_score", 0.0), x.get("implied", 0.0)), reverse=True)
    return pool

# ---- Session state defaults ----
if "events" not in st.session_state:
    st.session_state.events = []
if "parlay" not in st.session_state:
    st.session_state.parlay = []

# ---- UI ----
st.title("Parlay Builder — TheOddsAPI")
st.write("Preview and add conservative (safe) picks. Shows teams/players and the best bookmaker where available.")

# Top controls (load)
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    st.info("Load events then use the Auto-pick controls.")
with col2:
    sport_options = []
    sport_titles = {}
    sports_list: List[Dict[str, Any]] = []
    if API_KEY:
        try:
            sports_list = fetch_sports(API_KEY)
            sport_options = [s.get("key") for s in sports_list]
            sport_titles = {s.get("key"): s.get("title") for s in sports_list}
        except Exception:
            sport_options = []
            sport_titles = {}
    chosen_sport = st.selectbox("Sport (for load)", options=[""] + sport_options, format_func=lambda k: sport_titles.get(k, "— choose sport —"))
with col3:
    if st.button("Load odds for sport(s)"):
        if not API_KEY:
            st.error("Set THEODDS_API_KEY first.")
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
                        evs = fetch_odds_for_sport(API_KEY, chosen_sport, regions, markets, int(odds_ttl))
                        for ev in evs:
                            ev["_sport_key"] = chosen_sport
                            ev["_sport_title"] = sport_titles.get(chosen_sport, chosen_sport)
                        loaded = evs
                _ensure_sport_titles(loaded, sports_list if sports_list else (fetch_sports(API_KEY) if API_KEY else []))
                st.session_state.events = loaded
                st.success(f"Loaded {len(loaded)} events.")
            except Exception as e:
                st.error(f"Could not load events: {e}")

# Layout: left auto-pick / right parlay
left_col, right_col = st.columns([2.5, 1])

with left_col:
    st.header("Auto-pick (safe)")
    n_legs = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
    max_decimal = st.number_input("Max decimal odds per leg", value=1.8, step=0.05, format="%.2f")
    min_imp_pct = st.slider("Min consensus implied probability (%)", min_value=30, max_value=90, value=60) / 100.0
    max_spread_pct = st.slider("Max spread across books (%)", min_value=0, max_value=20, value=5) / 100.0
    require_bm = st.text_input("Require bookmaker (optional)", value="")
    avoid_same = st.checkbox("Avoid >1 leg per event", value=True)

    # Display toggles
    show_teams = st.checkbox("Show teams/players in lists", value=True)
    show_bookmaker = st.checkbox("Show best bookmaker", value=True)
    show_event_id = st.checkbox("Show event id (debug)", value=False)

    if st.button("Preview safest picks"):
        pool = auto_pick_safest_legs(
            st.session_state.events,
            n_legs=int(n_legs),
            max_decimal=float(max_decimal),
            min_imp=float(min_imp_pct),
            max_spread=float(max_spread_pct),
            require_bm=(require_bm if require_bm.strip() else None),
            avoid_same_event=avoid_same,
        )
        if not pool:
            st.warning("No safe picks found with current filters.")
        else:
            st.write(f"Top {min(len(pool), 50)} safe candidates (showing up to 50):")
            for i, p in enumerate(pool[:50], 1):
                teams_text = " / ".join(p.get("teams") or []) if (p.get("teams") and show_teams) else ""
                bm_text = ""
                if show_bookmaker and p.get("best_bookmaker") and (p.get("best_book_price") is not None):
                    bm_text = f" — {p.get('best_bookmaker')} @ {float(p.get('best_book_price')):.2f}"
                id_text = f" [{p.get('event_id')}]" if show_event_id else ""
                st.write(f"{i}. [{p.get('league')}] {p.get('event_title')}{id_text} {teams_text} — {p.get('selection')} @ {p.get('price'):.2f}{bm_text} (score {p.get('safety_score'):.3f})")

    if st.button("Add safest picks"):
        pool = auto_pick_safest_legs(
            st.session_state.events,
            n_legs=int(n_legs),
            max_decimal=float(max_decimal),
            min_imp=float(min_imp_pct),
            max_spread=float(max_spread_pct),
            require_bm=(require_bm if require_bm.strip() else None),
            avoid_same_event=avoid_same,
        )
        if not pool:
            st.warning("No safe picks to add.")
        else:
            added = 0
            for p in pool[:int(n_legs)]:
                exists = any((leg.get("event_id") == p["event_id"] and leg.get("selection") == p["selection"]) for leg in st.session_state.parlay)
                if not exists:
                    st.session_state.parlay.append({
                        "event_id": p["event_id"],
                        "league": p.get("league"),
                        "teams": p.get("teams"),
                        "title": p.get("event_title"),
                        "selection": p.get("selection"),
                        "price": p.get("price"),
                        "commence_time": p.get("commence_time"),
                        "safety_score": p.get("safety_score"),
                        "best_bookmaker": p.get("best_bookmaker"),
                        "best_book_price": p.get("best_book_price"),
                    })
                    added += 1
            st.success(f"Added {added} safe picks to parlay.")

    st.markdown("---")
    st.markdown("Show top safe candidates")
    top_n = st.number_input("Top N safe candidates to show", min_value=1, max_value=100, value=10)
    if st.button("Show top safe candidates"):
        candidates = auto_pick_safest_legs(
            st.session_state.events,
            n_legs=200,
            max_decimal=5.0,
            min_imp=0.0,
            max_spread=1.0,
            require_bm=(require_bm if require_bm.strip() else None),
            avoid_same_event=False,
        )
        st.write(f"Found {len(candidates)} safe candidates — showing top {min(top_n, len(candidates))}")
        for i, c in enumerate(candidates[:top_n], 1):
            teams_text = " / ".join(c.get("teams") or []) if (c.get("teams") and show_teams) else ""
            bm_text = ""
            if show_bookmaker and c.get("best_bookmaker") and (c.get("best_book_price") is not None):
                bm_text = f" — {c.get('best_bookmaker')} @ {float(c.get('best_book_price')):.2f}"
            id_text = f" [{c.get('event_id')}]" if show_event_id else ""
            st.write(f"{i}. [{c.get('league')}] {c.get('event_title')}{id_text} {teams_text} — {c.get('selection')} @ {c.get('price'):.2f}{bm_text} (score {c.get('safety_score'):.3f})")

with right_col:
    st.header("Parlay summary")
    if not st.session_state.parlay:
        st.write("No picks yet.")
    else:
        for leg in list(st.session_state.parlay):
            teams_line = ""
            if leg.get("teams"):
                teams_line = " — " + " / ".join(leg.get("teams"))
            bm_text = ""
            if leg.get("best_bookmaker") and (leg.get("best_book_price") is not None):
                bm_text = f" — {leg.get('best_bookmaker')} @ {float(leg.get('best_book_price')):.2f}"
            st.markdown(f"**{leg.get('league','')}** — {leg.get('title','')}{teams_line}")
            st.write(f"> {leg.get('selection','')} @ {leg.get('price')} — score {leg.get('safety_score','')}{bm_text}")
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
