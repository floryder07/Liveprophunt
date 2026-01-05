"""
Parlay Builder — TheOddsAPI prototype

Final merged file:
- All helper functions defined before UI (no NameError).
- auto_pick_legs_by_value and auto_pick_safest_legs included.
- Ensures _sport_title is set for every loaded event.
- Improved layout (Events / Auto-pick / Debug tabs) and persistent Parlay summary.
- Debug panel included to inspect loaded events and sport titles.
- Color-coded safety badges (green/orange/red) for picks.
- "Open on DraftKings" search links for each pick and "open all" links.
- No automatic bet placement (only opens DraftKings search pages).
- Stable widget keys for Add/Remove buttons.
"""
from typing import Any, Dict, List, Optional
import os
import hashlib
import json
from urllib.parse import quote_plus
import requests
import streamlit as st
from datetime import datetime, time, timezone
from statistics import median

# zoneinfo for timezone handling (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

st.set_page_config(page_title="Parlay Builder — TheOddsAPI", page_icon="🎯", layout="wide")

# ---- Sidebar (minimal controls) ----
st.sidebar.header("Settings / TheOddsAPI")
api_key_input = st.sidebar.text_input(
    "TheOddsAPI Key (or leave blank to use THEODDS_API_KEY env)", type="password"
)
API_KEY = api_key_input.strip() or os.environ.get("THEODDS_API_KEY", "")

regions = st.sidebar.multiselect("Regions (comma-select)", options=["us", "uk", "eu", "au"], default=["us"])
markets = st.sidebar.multiselect("Markets", options=["h2h", "spreads", "totals"], default=["h2h"])
odds_ttl = st.sidebar.number_input("Odds cache key (change to bust cache)", min_value=1, value=20, step=1)

st.sidebar.markdown("---")
scan_all_sports = st.sidebar.checkbox("Scan all sports for picks (may be slow / rate-limited)", value=False)
filter_dk_before_11_pt = st.sidebar.checkbox("Auto-filter: DraftKings before 11:00 PT", value=False)
st.sidebar.markdown("---")
if not API_KEY:
    st.sidebar.warning("No API key set. Get one at https://the-odds-api.com/ and put it in THEODDS_API_KEY or paste here.")

# ---- Utilities & data functions ----
def _safe_name(o: Dict[str, Any]) -> str:
    return o.get("name") or o.get("participant") or o.get("label") or ""

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
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions={regions_q}&markets={markets_q}&oddsFormat=decimal&apiKey={api_key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def extract_outcomes(event: Dict[str, Any], bookmaker_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    for bm in event.get("bookmakers", []):
        if bookmaker_filter and bookmaker_filter.strip():
            if bm.get("key") != bookmaker_filter and bm.get("title") != bookmaker_filter:
                continue
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                name = _safe_name(o)
                price = o.get("price") or o.get("decimal") or o.get("odds")
                try:
                    price = float(price)
                except Exception:
                    price = None
                if price and name:
                    outcomes.append({
                        "name": name,
                        "price": price,
                        "market": m.get("key"),
                        "bookmaker": bm.get("key"),
                    })
    return outcomes

def get_consensus_and_best_outcomes(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    price_map: Dict[str, List[float]] = {}
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                name = _safe_name(o)
                try:
                    price = float(o.get("price"))
                except Exception:
                    continue
                price_map.setdefault(name, []).append(price)

    results: List[Dict[str, Any]] = []
    for name, prices in price_map.items():
        if not prices:
            continue
        cons = median(prices)
        best = max(prices)
        worst = min(prices)
        results.append({"name": name, "consensus": cons, "best_price": best, "worst_price": worst})
    return results

def score_outcome_value(consensus: float, best_price: float) -> float:
    if consensus <= 0:
        return 0.0
    return (best_price / consensus) - 1.0

def get_reason_for_selection(event: Dict[str, Any], selection_name: str) -> str:
    outs = get_consensus_and_best_outcomes(event)
    for o in outs:
        if o["name"] == selection_name:
            cons = o["consensus"]
            best = o["best_price"]
            if cons and best:
                uplift = (best / cons - 1.0) * 100.0
                implied_cons = (1.0 / cons) * 100.0 if cons else None
                reason = f"Best price {best:.2f} vs consensus {cons:.2f} ({uplift:.1f}% uplift)"
                if implied_cons:
                    reason += f" — implied {implied_cons:.1f}%"
                return reason
    return "No reason available"

def get_outcome_safety_metrics(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for o in get_consensus_and_best_outcomes(event):
        cons = o["consensus"]
        best = o["best_price"]
        worst = o["worst_price"]
        spread_pct = abs(worst - best) / cons if cons else 0.0
        implied_consensus = (1.0 / cons) if cons > 0 else None
        metrics.append({
            "name": o["name"],
            "consensus": cons,
            "best": best,
            "worst": worst,
            "spread_pct": spread_pct,
            "implied_consensus": implied_consensus,
        })
    return metrics

def has_draftkings(bookmakers: Optional[List[Dict[str, Any]]]) -> bool:
    for bm in bookmakers or []:
        key = (bm.get("key") or "").lower()
        title = (bm.get("title") or "").lower()
        if "draftk" in key or "draftk" in title:
            return True
    return False

def color_for_prob(p: Optional[float]) -> str:
    if p is None:
        return "#999999"
    if p >= 0.65:
        return "#2ECC71"
    if p >= 0.5:
        return "#F1C40F"
    return "#E67E22"

# --- Auto-pick functions (both must exist before UI) ---
def auto_pick_legs_by_value(
    events: List[Dict[str, Any]],
    n_legs: int = 3,
    min_value: float = 0.02,
    avoid_same_event: bool = True,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for ev in events:
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("title")
        title = ev.get("title") or " vs ".join(ev.get("teams", [])) or str(ev_id)
        # compute consensus and best prices for outcomes
        price_map: Dict[str, List[float]] = {}
        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                for o in m.get("outcomes", []):
                    name = _safe_name(o)
                    try:
                        price = float(o.get("price"))
                    except Exception:
                        continue
                    price_map.setdefault(name, []).append(price)

        for name, prices in price_map.items():
            if not prices:
                continue
            cons = median(prices)
            best = max(prices)
            value = (best / cons - 1.0) if cons else 0.0
            implied_consensus = (1.0 / cons) if cons > 0 else None
            favorite_flag = cons < 2.0 if cons else False
            candidates.append({
                "event_id": ev_id,
                "event_title": title,
                "selection": name,
                "price": best,
                "consensus": cons,
                "value": value,
                "implied_consensus": implied_consensus,
                "favorite": favorite_flag,
                "sport_key": ev.get("_sport_key"),
                "sport_title": ev.get("_sport_title"),
                "commence_time": ev.get("commence_time") or ev.get("start"),
            })

    # sort by value desc then price desc
    candidates.sort(key=lambda c: (c["value"], c["price"]), reverse=True)

    selected: List[Dict[str, Any]] = []
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
        reason_parts.append("Favorite" if c["favorite"] else "Underdog")
        if c["implied_consensus"]:
            reason_parts.append(f"Implied {c['implied_consensus']*100:.1f}%")
        reason = " — ".join(reason_parts)
        selected.append({
            "event_id": c["event_id"],
            "sport_key": c.get("sport_key"),
            "sport_title": c.get("sport_title"),
            "event_title": c["event_title"],
            "selection": c["selection"],
            "price": c["price"],
            "consensus": c["consensus"],
            "value": c["value"],
            "implied_consensus": c["implied_consensus"],
            "reason": reason,
            "commence_time": c.get("commence_time"),
        })
        used_events.add(c["event_id"])
    return selected

def auto_pick_safest_legs(
    events: List[Dict[str, Any]],
    n_legs: int = 3,
    max_decimal_odds: float = 1.8,
    min_consensus_prob: float = 0.6,
    max_spread_pct: float = 0.05,
    require_bookmaker: Optional[str] = "draftkings",
    avoid_same_event: bool = True,
) -> List[Dict[str, Any]]:
    """
    Conservative picks: prefer outcomes with high consensus (implied prob),
    low inter-book spread, and odds <= max_decimal_odds.
    """
    candidates: List[Dict[str, Any]] = []
    for ev in events:
        if require_bookmaker:
            found = False
            for bm in ev.get("bookmakers", []):
                if (bm.get("key") or "").lower() == require_bookmaker.lower() or (bm.get("title") or "").lower().find(require_bookmaker.lower()) != -1:
                    found = True
                    break
            if not found:
                continue

        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("title")
        title = ev.get("title") or " vs ".join(ev.get("teams", [])) or str(ev_id)
        metrics = get_outcome_safety_metrics(ev)
        for m in metrics:
            cons = m.get("consensus", 0)
            implied = m.get("implied_consensus") or 0.0
            best = m.get("best") or 0.0
            spread = m.get("spread_pct") or 0.0
            if best > max_decimal_odds:
                continue
            if implied < min_consensus_prob:
                continue
            if spread > max_spread_pct:
                continue
            safety_score = implied - (spread * 0.5)
            candidates.append({
                "event_id": ev_id,
                "event_title": title,
                "selection": m["name"],
                "price": best,
                "consensus": cons,
                "implied_consensus": implied,
                "spread_pct": spread,
                "safety_score": safety_score,
                "sport_key": ev.get("_sport_key"),
                "sport_title": ev.get("_sport_title"),
                "commence_time": ev.get("commence_time") or ev.get("start"),
            })

    # sort by safety_score then implied_consensus
    candidates.sort(key=lambda c: (c["safety_score"], c["implied_consensus"]), reverse=True)

    selected: List[Dict[str, Any]] = []
    used_events = set()
    for c in candidates:
        if len(selected) >= n_legs:
            break
        if avoid_same_event and c["event_id"] in used_events:
            continue
        reason = (
            f"Consensus {c['consensus']:.2f} (implied {c['implied_consensus']*100:.1f}%), "
            f"spread {c['spread_pct']*100:.2f}% — conservative pick"
        )
        selected.append({
            "event_id": c["event_id"],
            "sport_key": c.get("sport_key"),
            "sport_title": c.get("sport_title"),
            "event_title": c["event_title"],
            "selection": c["selection"],
            "price": c["price"],
            "reason": reason,
            "safety_score": c["safety_score"],
            "commence_time": c.get("commence_time"),
        })
        used_events.add(c["event_id"])
    return selected

# --- New helper: ensure sport titles for events (fixes missing sport names) ---
def _ensure_sport_titles(ev_list: List[Dict[str, Any]], sports_list: List[Dict[str, Any]]) -> None:
    """
    Ensure each event in ev_list has a readable _sport_title.
    Uses sports_list (from fetch_sports) to map keys to titles and checks common event fields.
    """
    smap = {s.get("key"): s.get("title") for s in (sports_list or [])}
    for ev in ev_list:
        candidate_keys = [
            ev.get("_sport_key"),
            ev.get("sport_key"),
            ev.get("sport"),
            ev.get("category"),
            ev.get("league"),
        ]
        sport_key = next((k for k in candidate_keys if k), None)
        if sport_key and not ev.get("_sport_title"):
            ev["_sport_title"] = smap.get(sport_key, sport_key)
        if not ev.get("_sport_title"):
            ev["_sport_title"] = ev.get("_sport_title") or "Unknown"

# ---- DraftKings link + safety helpers ----
def build_draftkings_search_url(selection: str, event_title: str) -> str:
    """
    Build a DraftKings sportsbook search URL for the selection + event.
    This opens DraftKings search with the query prefilled.
    """
    query = f"{selection} {event_title}"
    encoded = quote_plus(query)
    return f"https://sportsbook.draftkings.com/search?query={encoded}"

# Safety thresholds (tweak these as you like)
SAFE_PROB_THRESHOLD = 0.60    # implied prob >= 60% -> safe (green)
MIDDLE_PROB_THRESHOLD = 0.45  # implied prob >= 45% -> middle (orange)

def safety_level_for_leg(leg: dict) -> str:
    """
    Return 'safe' | 'middle' | 'danger' based on safety_score or implied prob (1/price).
    """
    # prefer explicit safety_score if present (auto-pick safest adds it)
    if leg.get("safety_score") is not None:
        try:
            s = float(leg["safety_score"])
            if s >= 0.25:
                return "safe"
            if s >= 0.05:
                return "middle"
            return "danger"
        except Exception:
            pass

    # fallback: implied probability from price
    try:
        p = 1.0 / float(leg.get("price", 0))
    except Exception:
        p = 0.0
    if p >= SAFE_PROB_THRESHOLD:
        return "safe"
    if p >= MIDDLE_PROB_THRESHOLD:
        return "middle"
    return "danger"

def color_for_level(level: str) -> str:
    return {
        "safe": "#2ECC71",    # green
        "middle": "#F39C12",  # orange
        "danger": "#E74C3C",  # red
    }.get(level, "#999999")

# ---- Session state defaults ----
if "events" not in st.session_state:
    st.session_state.events = []
if "parlay" not in st.session_state:
    st.session_state.parlay = []
if "events_page" not in st.session_state:
    st.session_state.events_page = 1

# ---- Header ----
st.markdown("# 🎯 Parlay Builder — TheOddsAPI prototype")
st.markdown("Browse events on the left, review and manage your parlay on the right. Use Debug to inspect loaded feed.")

# ---- Top controls row (Load / Sport selector) ----
top_cols = st.columns([3, 2, 1])
with top_cols[0]:
    st.info("Events browser / Auto-pick in left tabs; Parlay summary stays on the right.")
with top_cols[1]:
    # fetch sports list to populate sport selector and title mapping
    sport_options = []
    sport_titles = {}
    sports_list = []
    if API_KEY:
        try:
            sports_list = fetch_sports(API_KEY)
            sport_options = [s.get("key") for s in sports_list]
            sport_titles = {s.get("key"): s.get("title") for s in sports_list}
        except Exception:
            sport_options = []
            sport_titles = {}
    chosen_sport_for_load = st.selectbox("Sport (for load)", options=[""] + sport_options, format_func=lambda k: sport_titles.get(k, " — choose sport — "))
with top_cols[2]:
    if st.button("Load odds for sport(s)"):
        try:
            loaded: List[Dict[str, Any]] = []
            if scan_all_sports:
                for skey, stitle in sport_titles.items():
                    try:
                        evs = fetch_odds_for_sport(API_KEY, skey, regions, markets, int(odds_ttl))
                    except Exception:
                        continue
                    for ev in evs:
                        ev["_sport_key"] = skey
                        ev["_sport_title"] = stitle
                    loaded.extend(evs)
            else:
                if not chosen_sport_for_load:
                    st.warning("Select a sport to load, or enable Scan all sports in the sidebar.")
                else:
                    evs = fetch_odds_for_sport(API_KEY, chosen_sport_for_load, regions, markets, int(odds_ttl))
                    stitle = sport_titles.get(chosen_sport_for_load, chosen_sport_for_load)
                    for ev in evs:
                        ev["_sport_key"] = chosen_sport_for_load
                        ev["_sport_title"] = stitle
                    loaded = evs

            # ensure sport titles exist for all events (use sports_list mapping & fallbacks)
            _ensure_sport_titles(loaded, sports_list)
            st.session_state.events = loaded
            st.session_state.events_page = 1
            st.success(f"Loaded {len(loaded)} events.")
        except Exception as e:
            st.error(f"Could not load odds: {e}")

# ---- Main layout: left tabs + right parlay summary ----
left_col, right_col = st.columns([2.5, 1])

with left_col:
    tabs = st.tabs(["Events", "Auto-pick", "Debug"])
    # ---------- Events tab ----------
    with tabs[0]:
        st.subheader("Events browser")

        # Search & filters
        search_text = st.text_input("Search (event title, team, selection)", value="")
        sport_filter = st.selectbox(
            "Filter by sport",
            options=["All"] + sorted(list({ev.get("_sport_title") or ev.get("_sport_key") or "Unknown" for ev in st.session_state.events}))
        )
        sort_by = st.selectbox("Sort by", options=["Start time", "Odds value (best uplift)", "Sport", "Title"])

        # Pagination controls
        page_size = st.number_input("Events per page", min_value=2, max_value=50, value=8, step=1)
        total_events = len(st.session_state.events)
        total_pages = max(1, (total_events + page_size - 1) // page_size)
        page = st.session_state.events_page
        pg_cols = st.columns([1, 1, 2])
        if pg_cols[0].button("Prev"):
            st.session_state.events_page = max(1, page - 1)
        if pg_cols[1].button("Next"):
            st.session_state.events_page = min(total_pages, page + 1)
        pg_cols[2].markdown(f"Page {st.session_state.events_page} / {total_pages} — {total_events} events")

        # Filter & search logic
        filtered = []
        for ev in st.session_state.events:
            sport_title = ev.get("_sport_title") or ev.get("_sport_key") or ""
            if sport_filter and sport_filter != "All" and sport_title != sport_filter:
                continue
            title = ev.get("title") or " "
            teams_text = " ".join(ev.get("teams") or [])
            outcome_names = []
            for bm in ev.get("bookmakers", []):
                for m in bm.get("markets", []):
                    for o in m.get("outcomes", []):
                        outcome_names.append(_safe_name(o))
            hay = " ".join([title, teams_text, " ".join(outcome_names)]).lower()
            if search_text and search_text.strip().lower() not in hay:
                continue
            filtered.append(ev)

        # sort
        def event_value(ev):
            uplifts = []
            for o in get_consensus_and_best_outcomes(ev):
                cons = o.get("consensus") or 0
                best = o.get("best_price") or 0
                if cons:
                    uplifts.append((best / cons) - 1.0)
            return max(uplifts) if uplifts else 0.0

        if sort_by == "Start time":
            filtered.sort(key=lambda e: e.get("commence_time") or e.get("start") or "")
        elif sort_by == "Odds value (best uplift)":
            filtered.sort(key=lambda e: event_value(e), reverse=True)
        elif sort_by == "Sport":
            filtered.sort(key=lambda e: e.get("_sport_title") or e.get("_sport_key") or "")
        elif sort_by == "Title":
            filtered.sort(key=lambda e: e.get("title") or "")

        # Pagination slice
        start = (st.session_state.events_page - 1) * page_size
        end = start + page_size
        page_events = filtered[start:end]

        if not page_events:
            st.info("No events on this page. Try a different filter or load more events.")
        else:
            for ev in page_events:
                ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("title")
                sport_title = ev.get("_sport_title") or ev.get("_sport_key") or "—"
                title = ev.get("title") or " — "
                commence = ev.get("commence_time") or ev.get("start") or ""
                header = f"{sport_title} — {title}"
                with st.expander(header):
                    st.markdown(f"*starts:* {commence}")
                    # show outcomes grouped by market/bookmaker
                    for bm in ev.get("bookmakers", []):
                        bm_key = bm.get("key") or bm.get("title") or "bookmaker"
                        for m in bm.get("markets", []):
                            mkey = m.get("key") or "market"
                            for o in m.get("outcomes", []):
                                name = _safe_name(o)
                                price = o.get("price") or o.get("decimal") or o.get("odds")
                                try:
                                    price = float(price)
                                except Exception:
                                    price = None
                                if not price or not name:
                                    continue
                                row_cols = st.columns([6, 1])
                                row_cols[0].write(f"{name}  —  {price}  (market: {mkey} | bm: {bm_key})")
                                raw_key = f"{ev_id}|{name}|{mkey}|{bm_key}"
                                add_key = "add-" + hashlib.sha1(raw_key.encode()).hexdigest()[:12]
                                if row_cols[1].button("Add", key=add_key):
                                    reason = get_reason_for_selection(ev, name)
                                    add_leg = {
                                        "event_id": ev_id,
                                        "sport": sport_title,
                                        "title": title,
                                        "selection": name,
                                        "price": price,
                                        "commence_time": commence,
                                        "reason": reason
                                    }
                                    exists = any((leg.get("event_id") == ev_id and leg.get("selection") == name) for leg in st.session_state.parlay)
                                    if exists:
                                        st.warning("This selection is already in the parlay.")
                                    else:
                                        st.session_state.parlay.append(add_leg)

    # ---------- Auto-pick tab ----------
    with tabs[1]:
        st.subheader("Auto-pick (value & safety)")
        st.write("Configure auto-pick options and preview picks before adding.")
        # value pick controls
        st.markdown("**Value-based picks**")
        auto_n = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
        auto_min_pct = st.slider("Minimum uplift vs consensus (%)", min_value=0, max_value=50, value=2)
        auto_min_value = auto_min_pct / 100.0
        avoid_same_event = st.checkbox("Avoid >1 leg per event", value=True)
        preview_only = st.checkbox("Preview only (don't auto-add)", value=True)

        # safety pick controls
        st.markdown("**Safety picks (conservative)**")
        max_decimal_odds = st.number_input("Max decimal odds per leg", value=1.8, step=0.05, format="%.2f")
        min_consensus_prob = st.slider("Minimum consensus implied probability (%)", min_value=30, max_value=90, value=60) / 100.0
        max_spread_pct = st.slider("Max spread across books (%)", min_value=0, max_value=20, value=5) / 100.0
        require_dk = st.checkbox("Require DraftKings for safety picks", value=True)

        def _auto_pick_source_events():
            s_events = st.session_state.events[:]
            if filter_dk_before_11_pt and ZoneInfo is not None:
                tz = ZoneInfo("America/Los_Angeles")
                now = datetime.now(tz)
                cutoff = datetime.combine(now.date(), time(hour=11, minute=0), tzinfo=tz)
                fe = []
                for ev in s_events:
                    if not has_draftkings(ev.get("bookmakers", [])):
                        continue
                    ct = ev.get("commence_time") or ev.get("start")
                    try:
                        from dateutil import parser as date_parser  # type: ignore
                        dt = date_parser.isoparse(ct) if ct else None
                    except Exception:
                        try:
                            dt = datetime.fromisoformat(ct.replace("Z", "+00:00")) if ct else None
                            if dt and dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        except Exception:
                            dt = None
                    if not dt:
                        continue
                    if dt.astimezone(tz) < cutoff:
                        fe.append(ev)
                s_events = fe
            return s_events

        col_ap = st.columns([1, 1, 2])
        if col_ap[0].button("Preview value picks"):
            picks = auto_pick_legs_by_value(_auto_pick_source_events(), n_legs=int(auto_n), min_value=auto_min_value, avoid_same_event=avoid_same_event)
            if not picks:
                st.warning("No value picks found.")
            else:
                for p in picks:
                    st.markdown(f"**{p.get('sport_title') or p.get('sport_key') or ''}** — {p['event_title']}  \n"
                                f"> **{p['selection']}**  @ {p['price']}  — {p['reason']}")
        if col_ap[1].button("Add value picks"):
            picks = auto_pick_legs_by_value(_auto_pick_source_events(), n_legs=int(auto_n), min_value=auto_min_value, avoid_same_event=avoid_same_event)
            if not picks:
                st.warning("No value picks found.")
            else:
                added = 0
                for p in picks:
                    exists = any((leg.get("event_id") == p["event_id"] and leg.get("selection") == p["selection"]) for leg in st.session_state.parlay)
                    if not exists:
                        st.session_state.parlay.append({
                            "event_id": p["event_id"],
                            "sport": p.get("sport_title") or p.get("sport_key") or "",
                            "title": p["event_title"],
                            "selection": p["selection"],
                            "price": p["price"],
                            "commence_time": p.get("commence_time"),
                            "reason": p.get("reason")
                        })
                        added += 1
                st.success(f"Added {added} picks to parlay.")

        if col_ap[2].button("Preview safest picks"):
            picks = auto_pick_safest_legs(
                _auto_pick_source_events(),
                n_legs=int(auto_n),
                max_decimal_odds=float(max_decimal_odds),
                min_consensus_prob=float(min_consensus_prob),
                max_spread_pct=float(max_spread_pct),
                require_bookmaker="draftkings" if require_dk else None,
                avoid_same_event=avoid_same_event
            )
            if not picks:
                st.warning("No safe picks found.")
            else:
                for p in picks:
                    st.markdown(f"**{p.get('sport_title') or ''}** — {p['event_title']}  \n"
                                f"> **{p['selection']}**  @ {p['price']}  — {p['reason']}")
        if st.button("Add safest picks"):
            picks = auto_pick_safest_legs(
                _auto_pick_source_events(),
                n_legs=int(auto_n),
                max_decimal_odds=float(max_decimal_odds),
                min_consensus_prob=float(min_consensus_prob),
                max_spread_pct=float(max_spread_pct),
                require_bookmaker="draftkings" if require_dk else None,
                avoid_same_event=avoid_same_event
            )
            if not picks:
                st.warning("No safe picks found.")
            else:
                added = 0
                for p in picks:
                    exists = any((leg.get("event_id") == p["event_id"] and leg.get("selection") == p["selection"]) for leg in st.session_state.parlay)
                    if not exists:
                        st.session_state.parlay.append({
                            "event_id": p["event_id"],
                            "sport": p.get("sport_title") or "",
                            "title": p["event_title"],
                            "selection": p["selection"],
                            "price": p["price"],
                            "commence_time": p.get("commence_time"),
                            "reason": p.get("reason")
                        })
                        added += 1
                st.success(f"Added {added} safe picks to parlay.")

    # ---------- Debug tab ----------
    with tabs[2]:
        with st.expander("⚠️ Debug (raw events & session)", expanded=False):
            st.write("Events loaded:", len(st.session_state.events))
            if st.session_state.events:
                sample = st.session_state.events[0].copy()
                keys_to_show = ["_sport_key", "_sport_title", "id", "title", "commence_time", "bookmakers"]
                trimmed = {k: sample.get(k) for k in keys_to_show if k in sample}
                st.json(trimmed)
            st.write("Unique sport titles loaded:", sorted(list({ev.get("_sport_title") or ev.get("_sport_key") or "Unknown" for ev in st.session_state.events})))
            st.write("Session parlay:", len(st.session_state.parlay))
            st.json(st.session_state.parlay)

with right_col:
    st.header("Parlay summary")

    # Picked summary: color badges + DraftKings links
    def render_parlay_table_with_links():
        picked_html = """
        <style>
          .pill {display:inline-block;padding:6px 10px;border-radius:999px;color:#fff;font-size:12px;margin-right:6px;}
          table.picks {width:100%;border-collapse:collapse;}
          table.picks th, table.picks td {padding:6px;border-bottom:1px solid #eee;text-align:left;font-size:13px;vertical-align:middle}
          .dk-link {background:#111;color:#fff;padding:6px 8px;border-radius:6px;text-decoration:none}
        </style>
        """
        picked_html += "<table class='picks'><thead><tr><th>Safety</th><th>Sport</th><th>Event</th><th>Selection</th><th>Start</th><th>Price</th><th>Open on DK</th></tr></thead><tbody>"
        for i, leg in enumerate(st.session_state.parlay):
            level = safety_level_for_leg(leg)
            color = color_for_level(level)
            sport = leg.get("sport") or ""
            start = leg.get("commence_time") or ""
            title = leg.get("title") or ""
            selection = leg.get("selection") or ""
            price = leg.get("price") or ""
            dk_url = build_draftkings_search_url(selection, title)
            picked_html += (
                f"<tr>"
                f"<td><span class='pill' style='background:{color}'>{level.title()}</span></td>"
                f"<td>{sport}</td>"
                f"<td>{title}</td>"
                f"<td><strong>{selection}</strong></td>"
                f"<td>{start}</td>"
                f"<td>{price}</td>"
                f"<td><a class='dk-link' href='{dk_url}' target='_blank' rel='noreferrer noopener'>Open on DraftKings</a></td>"
                f"</tr>"
            )
        if not st.session_state.parlay:
            picked_html += "<tr><td colspan='7'><em>No picks yet</em></td></tr>"
        picked_html += "</tbody></table>"
        st.markdown(picked_html, unsafe_allow_html=True)

    render_parlay_table_with_links()

    # Provide "open all" links (one tab per pick)
    if st.session_state.parlay:
        st.markdown("**Open all picks on DraftKings** (one tab per pick):")
        for leg in st.session_state.parlay:
            dk_url = build_draftkings_search_url(leg.get("selection",""), leg.get("title",""))
            st.markdown(f"- [{leg.get('selection')} — {leg.get('title')}]({dk_url})", unsafe_allow_html=True)

    # Parlay controls
    if st.button("Clear parlay"):
        st.session_state.parlay = []
        st.success("Parlay cleared.")

    # Show detailed list with remove buttons
    st.subheader("Manage picks")
    if not st.session_state.parlay:
        st.info("No picks to manage.")
    else:
        for idx, leg in enumerate(list(st.session_state.parlay)):
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{leg.get('sport','')}** — {leg.get('title','')}  \n> **{leg.get('selection','')}**  @ {leg.get('price')}")
            rem_key = "rem-" + hashlib.sha1(f"{idx}|{leg.get('event_id','')}|{leg.get('selection','')}".encode()).hexdigest()[:12]
            if cols[1].button("Remove", key=rem_key):
                st.session_state.parlay.pop(idx)

    # Parlay metrics & export
    st.markdown("---")
    st.subheader("Parlay metrics")
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
        # Export JSON
        if st.button("Export parlay JSON"):
            st.download_button("Download JSON", data=json.dumps(st.session_state.parlay, default=str, indent=2), file_name="parlay.json", mime="application/json")
    else:
        st.write("Add some legs to see metrics.")

st.caption("Prototype uses TheOddsAPI. Auto-picks are heuristic suggestions (value vs market consensus). This is a simulation tool only — do not auto-place real bets without user confirmation and proper licensing.")
