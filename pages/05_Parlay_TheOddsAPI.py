```python name=parlay_builder.py
# Parlay Builder — TheOddsAPI prototype (safe-only, show top safe candidates)
#
# Copy this file content exactly (do NOT include the surrounding triple-backtick lines)
# into pages/05_Parlay_TheOddsAPI.py (or parlay_builder.py) to replace the broken page.
# After saving, run:
#   python -m py_compile pages/05_Parlay_TheOddsAPI.py
# to confirm syntax, then restart Streamlit.

from typing import Any, Dict, List, Optional
import os
import hashlib
import json
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
filter_bm_before_11_pt = st.sidebar.checkbox("Auto-filter by bookmaker before 11:00 PT", value=False)
st.sidebar.markdown("---")
if not API_KEY:
    st.sidebar.warning("No API key set. Get one at https://the-odds-api.com/ and put it in THEODDS_API_KEY or paste here.")

# ---- Utilities & data functions ----
def _safe_name(o: Dict[str, Any]) -> str:
    return o.get("name") or o.get("participant") or o.get("label") or ""

def _safe_price_from_outcome(o: Dict[str, Any]) -> Optional[float]:
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
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions={regions_q}&markets={markets_q}&oddsFormat=decimal&apiKey={api_key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def get_consensus_and_best_outcomes(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    price_map: Dict[str, List[float]] = {}
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                name = _safe_name(o)
                price = _safe_price_from_outcome(o)
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
        results.append({"name": name, "consensus": cons, "best_price": best, "worst_price": worst})
    return results

def get_outcome_safety_metrics(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for o in get_consensus_and_best_outcomes(event):
        cons = o["consensus"]
        best = o["best_price"]
        worst = o["worst_price"]
        spread_pct = abs(worst - best) / cons if cons else 0.0
        implied_consensus = (1.0 / cons) if cons and cons > 0 else 0.0
        metrics.append({
            "name": o["name"],
            "consensus": cons,
            "best": best,
            "worst": worst,
            "spread_pct": spread_pct,
            "implied_consensus": implied_consensus,
        })
    return metrics

def has_bookmaker(bookmakers: Optional[List[Dict[str, Any]]], needle: str) -> bool:
    for bm in bookmakers or []:
        key = (bm.get("key") or "").lower()
        title = (bm.get("title") or "").lower()
        if needle.lower() in key or needle.lower() in title:
            return True
    return False

# --- Auto-pick functions ---
def auto_pick_safest_legs(
    events: List[Dict[str, Any]],
    n_legs: int = 3,
    max_decimal_odds: float = 1.8,
    min_consensus_prob: float = 0.6,
    max_spread_pct: float = 0.05,
    require_bookmaker: Optional[str] = None,
    avoid_same_event: bool = True,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for ev in events:
        if require_bookmaker:
            if not has_bookmaker(ev.get("bookmakers", []), require_bookmaker):
                continue
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or None
        if not ev_id:
            continue
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

    candidates.sort(key=lambda c: (c.get("safety_score", 0.0), c.get("implied_consensus", 0.0)), reverse=True)
    # Return the full candidate list (caller will slice)
    return candidates

# --- Helper: ensure sport titles for events ---
def _ensure_sport_titles(ev_list: List[Dict[str, Any]], sports_list: List[Dict[str, Any]]) -> None:
    smap = {s.get("key"): s.get("title") for s in (sports_list or [])}
    for ev in ev_list:
        candidate_keys = [ev.get("_sport_key"), ev.get("sport_key"), ev.get("sport"), ev.get("category"), ev.get("league")]
        sport_key = next((k for k in candidate_keys if k), None)
        if sport_key and not ev.get("_sport_title"):
            ev["_sport_title"] = smap.get(sport_key, sport_key)
        if not ev.get("_sport_title"):
            ev["_sport_title"] = ev.get("_sport_title") or "Unknown"

# ---- Session state defaults ----
if "events" not in st.session_state:
    st.session_state.events = []
if "parlay" not in st.session_state:
    st.session_state.parlay = []
if "events_page" not in st.session_state:
    st.session_state.events_page = 1

# ---- Header ----
st.markdown("# 🎯 Parlay Builder — TheOddsAPI prototype (safe picks only)")
st.markdown("Browse events on the left, preview auto-picks (value & safety), and manage your parlay on the right.")

# ---- Top controls row (Load / Sport selector) ----
top_cols = st.columns([3, 2, 1])
with top_cols[0]:
    st.info("Events browser / Auto-pick in left tabs; Parlay summary stays on the right.")
with top_cols[1]:
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

    # Events tab
    with tabs[0]:
        st.subheader("Events browser")
        search_text = st.text_input("Search (event title, team, selection)", value="")
        sport_filter = st.selectbox("Filter by sport", options=["All"] + sorted(list({ev.get("_sport_title") or ev.get("_sport_key") or "Unknown" for ev in st.session_state.events})))
        sort_by = st.selectbox("Sort by", options=["Start time", "Odds value (best uplift)", "Sport", "Title"])
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

        def event_value(ev: Dict[str, Any]) -> float:
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
                    for bm in ev.get("bookmakers", []):
                        bm_key = bm.get("key") or bm.get("title") or "bookmaker"
                        for m in bm.get("markets", []):
                            mkey = m.get("key") or "market"
                            for o in m.get("outcomes", []):
                                name = _safe_name(o)
                                price = _safe_price_from_outcome(o)
                                if price is None or not name:
                                    continue
                                row_cols = st.columns([6, 1])
                                row_cols[0].write(f"{name}  —  {price}  (market: {mkey} | bm: {bm_key})")
                                raw_key = f"{ev_id}|{name}|{mkey}|{bm_key}"
                                add_key = "add-" + hashlib.sha1(raw_key.encode()).hexdigest()[:12]
                                if row_cols[1].button("Add", key=add_key):
                                    add_leg = {
                                        "event_id": ev_id,
                                        "sport": sport_title,
                                        "title": title,
                                        "selection": name,
                                        "price": price,
                                        "commence_time": commence,
                                        "reason": "manual add",
                                    }
                                    exists = any((leg.get("event_id") == ev_id and leg.get("selection") == name) for leg in st.session_state.parlay)
                                    if exists:
                                        st.warning("This selection is already in the parlay.")
                                    else:
                                        st.session_state.parlay.append(add_leg)

    # Auto-pick tab
    with tabs[1]:
        st.subheader("Auto-pick (value & safety)")
        st.write("Configure auto-pick options and preview picks before adding.")
        auto_n = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
        auto_min_pct = st.slider("Minimum uplift vs consensus (%)", min_value=0, max_value=50, value=2)
        auto_min_value = auto_min_pct / 100.0
        avoid_same_event = st.checkbox("Avoid >1 leg per event", value=True)
        preview_only = st.checkbox("Preview only (don't auto-add)", value=True)

        st.markdown("**Safety picks (conservative)**")
        max_decimal_odds = st.number_input("Max decimal odds per leg", value=1.8, step=0.05, format="%.2f")
        min_consensus_prob = st.slider("Minimum consensus implied probability (%)", min_value=30, max_value=90, value=60) / 100.0
        max_spread_pct = st.slider("Max spread across books (%)", min_value=0, max_value=20, value=5) / 100.0
        require_bm = st.text_input("Require bookmaker (leave blank for any)", value="")

        def _auto_pick_source_events() -> List[Dict[str, Any]]:
            s_events = st.session_state.events[:]
            if filter_bm_before_11_pt and ZoneInfo is not None:
                tz = ZoneInfo("America/Los_Angeles")
                now = datetime.now(tz)
                cutoff = datetime.combine(now.date(), time(hour=11, minute=0), tzinfo=tz)
                fe = []
                for ev in s_events:
                    if require_bm and require_bm.strip():
                        if not has_bookmaker(ev.get("bookmakers", []), require_bm):
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
        if col_ap[0].button("Preview safest picks"):
            picks = auto_pick_safest_legs(_auto_pick_source_events(), n_legs=int(auto_n), max_decimal_odds=float(max_decimal_odds), min_consensus_prob=float(min_consensus_prob), max_spread_pct=float(max_spread_pct), require_bookmaker=(require_bm if require_bm.strip() else None), avoid_same_event=avoid_same_event)
            if not picks:
                st.warning("No safe picks found with current filters.")
            else:
                for p in picks[:20]:
                    st.write(f"{p.get('sport_title') or ''} — {p.get('event_title')}\n  {p.get('selection')} @ {p.get('price')} — safety {p.get('safety_score')}")

        # Show top N safe candidates (improved display, ID toggle, auto-load helper)
        st.markdown("---")
        st.markdown("### Show top safe candidates")
        top_n = st.number_input("Top N safe candidates to show", min_value=1, max_value=50, value=10)
        show_full_id = st.checkbox("Show full event id (debug)", value=False)

        # Auto-load events when API key present AND scan_all_sports is checked (manual confirmation)
        if API_KEY and scan_all_sports:
            if st.button("Auto-load all sports now (API key present & Scan all sports enabled)"):
                with st.spinner("Loading events from all sports (may be slow)..."):
                    try:
                        sports_list = fetch_sports(API_KEY)
                        loaded: List[Dict[str, Any]] = []
                        for s in sports_list:
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
                        _ensure_sport_titles(loaded, sports_list)
                        st.session_state.events = loaded
                        st.success(f"Loaded {len(loaded)} events (scan all sports).")
                    except Exception as e:
                        st.error(f"Could not load events: {e}")

        if st.button("Show top safe candidates"):
            candidates = auto_pick_safest_legs(
                _auto_pick_source_events(),
                n_legs=200,
                max_decimal_odds=5.0,
                min_consensus_prob=0.0,
                max_spread_pct=1.0,
                require_bookmaker=(require_bm if require_bm.strip() else None),
                avoid_same_event=False,
            )
            count = len(candidates)
            shown = min(top_n, count)
            st.write(f"Found {count} safe candidates — showing top {shown}")

            hdr_cols = st.columns([0.5, 2, 4, 3, 1.2, 1.2, 1])
            hdr_cols[0].write("#")
            hdr_cols[1].write("Sport")
            hdr_cols[2].write("Event")
            hdr_cols[3].write("Selection / Start")
            hdr_cols[4].write("Implied %")
            hdr_cols[5].write("Spread %")
            hdr_cols[6].write("Score")

            for i, c in enumerate(candidates[:top_n]):
                c_sport = c.get("sport_title") or c.get("sport_key") or ""
                c_event = c.get("event_title") or ""
                display_title = c_event if len(str(c_event)) <= 40 or show_full_id else (str(c_event)[:37] + "...")
                c_sel = c.get("selection") or ""
                c_price = float(c.get("price") or 0.0)
                c_implied = (c.get("implied_consensus") or 0.0) * 100.0
                c_spread = (c.get("spread_pct") or 0.0) * 100.0
                c_score = c.get("safety_score") or 0.0
                commence = c.get("commence_time") or ""

                row_cols = st.columns([0.5, 2, 4, 3, 1.2, 1.2, 1])
                row_cols[0].write(f"{i+1}")
                row_cols[1].write(c_sport)
                row_cols[2].markdown(f"{display_title}")
                row_cols[3].markdown(f"**{c_sel}**  @ {c_price:.2f}\n\n{commence}")
                row_cols[4].write(f"{c_implied:.1f}%")
                row_cols[5].write(f"{c_spread:.2f}%")
                row_cols[6].write(f"{c_score:.3f}")

                add_key = "addsafe-" + hashlib.sha1(f"{c.get('event_id','')}|{c_sel}".encode()).hexdigest()[:12]
                if row_cols[6].button("Add", key=add_key):
                    exists = any((leg.get("event_id") == c.get("event_id") and leg.get("selection") == c_sel) for leg in st.session_state.parlay)
                    if exists:
                        st.warning("Selection already in parlay.")
                    else:
                        st.session_state.parlay.append({
                            "event_id": c.get("event_id"),
                            "sport": c_sport,
                            "title": c_event,
                            "selection": c_sel,
                            "price": c_price,
                            "commence_time": c.get("commence_time"),
                            "reason": f"Auto-safe (score {c_score:.3f})",
                        })
                        st.success("Added to parlay.")

    # Debug tab
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

    def render_parlay_table() -> None:
        rows = []
        for leg in st.session_state.parlay:
            level = "safe" if (leg.get("safety_score") and leg.get("safety_score") >= 0.25) else ("middle" if (1.0 / (leg.get("price") or 999)) >= 0.45 else "danger")
            rows.append([level.title(), leg.get("sport") or "", leg.get("title") or "", leg.get("selection") or "", leg.get("commence_time") or "", leg.get("price")])
        if not rows:
            st.write("*No picks yet*")
            return
        st.table({"Safety": [r[0] for r in rows], "Sport": [r[1] for r in rows], "Event": [r[2] for r in rows], "Selection": [r[3] for r in rows], "Start": [r[4] for r in rows], "Price": [r[5] for r in rows]})

    render_parlay_table()

    if st.button("Clear parlay"):
        st.session_state.parlay = []
        st.success("Parlay cleared.")

    st.subheader("Manage picks")
    if not st.session_state.parlay:
        st.info("No picks to manage.")
    else:
        for leg in list(st.session_state.parlay):
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{leg.get('sport','')}** — {leg.get('title','')}  \n> **{leg.get('selection','')}**  @ {leg.get('price')}")
            unique_key_src = f"{leg.get('event_id','')}-{leg.get('selection','')}"
            rem_key = "rem-" + hashlib.sha1(unique_key_src.encode()).hexdigest()[:12]
            if cols[1].button("Remove", key=rem_key):
                st.session_state.parlay = [l for l in st.session_state.parlay if not (l.get("event_id") == leg.get("event_id") and l.get("selection") == leg.get("selection"))]
                st.experimental_rerun()

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
        st.download_button("Download parlay JSON", data=json.dumps(st.session_state.parlay, default=str, indent=2), file_name="parlay.json", mime="application/json")
    else:
        st.write("Add some legs to see metrics.")

st.caption("Prototype uses TheOddsAPI. Auto-picks are heuristic suggestions (value vs market consensus). This is a simulation tool only — do not auto-place real bets without user confirmation and proper licensing.")
```
