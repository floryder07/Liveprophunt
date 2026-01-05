"""
Parlay Builder — TheOddsAPI prototype (Streamlit)

Improved layout & navigation:
- Main area split: left (Events + Auto-pick tabs), right (persistent Parlay summary)
- Search, sport filter, sorting, pagination for event browsing
- Auto-pick controls centralized in Auto-pick tab
- Debug tab (collapsed) to inspect raw events & session state
- Stable widget keys (hashed) to avoid duplicates
- Keeps conservative defaults (preview-only enabled)
"""
from typing import Any, Dict, List, Optional
import os
import hashlib
import json
import requests
import streamlit as st
from datetime import datetime, time, timezone
from statistics import median

# zoneinfo for timezone handling
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

# Page config
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

# ---- Session state defaults ----
if "events" not in st.session_state:
    st.session_state.events = []  # loaded events
if "parlay" not in st.session_state:
    st.session_state.parlay = []
if "events_page" not in st.session_state:
    st.session_state.events_page = 1

# ---- Header ----
st.markdown("# 🎯 Parlay Builder — TheOddsAPI prototype")
st.markdown("Use the left tab to browse events and the right column to review your parlay. Auto-pick in the Auto-pick tab.")

# ---- Top controls row (Load / Sport selector) ----
top_cols = st.columns([3, 2, 1])
with top_cols[0]:
    st.info("Browse events: search, filter, and add outcomes to the parlay.")
with top_cols[1]:
    sport_options = []
    if API_KEY:
        try:
            sports_list = fetch_sports(API_KEY)
            sport_options = [s.get("key") for s in sports_list]
            sport_titles = {s.get("key"): s.get("title") for s in sports_list}
        except Exception:
            sport_options = []
            sport_titles = {}
    else:
        sport_titles = {}
    chosen_sport_for_load = st.selectbox("Sport (for load)", options=[""] + sport_options, format_func=lambda k: sport_titles.get(k, " — choose sport — "))
with top_cols[2]:
    if st.button("Load odds for sport(s)"):
        try:
            loaded = []
            if scan_all_sports:
                for skey, stitle in sport_titles.items():
                    try:
                        evs = fetch_odds_for_sport(API_KEY, skey, regions, markets, int(odds_ttl))
                        for ev in evs:
                            ev["_sport_key"] = skey
                            ev["_sport_title"] = stitle
                        loaded.extend(evs)
                    except Exception:
                        continue
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
        sport_filter = st.selectbox("Filter by sport", options=["All"] + sorted(list({ev.get("_sport_title") or ev.get("_sport_key") or "" for ev in st.session_state.events})))
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
            # sport filter
            sport_title = ev.get("_sport_title") or ev.get("_sport_key") or ""
            if sport_filter and sport_filter != "All" and sport_title != sport_filter:
                continue
            # search
            title = ev.get("title") or " "
            teams_text = " ".join(ev.get("teams") or [])
            # collect outcome names for search
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
            # compute top uplift across outcomes for quick sort
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
                                # stable hashed key for Add button
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
                                    # prevent duplicates same-event+selection
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

        # helper to build filtered_events for auto-pick (respect DK before 11 PT if set)
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

        # Auto-pick actions
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
                st.json(st.session_state.events[0])
            st.write("Session parlay:", len(st.session_state.parlay))
            st.json(st.session_state.parlay)

with right_col:
    st.header("Parlay summary")

    # Picked summary (HTML table)
    def render_parlay_table():
        picked_html = """
        <style>
          .pill {display:inline-block;padding:4px 8px;border-radius:999px;color:#fff;font-size:12px;margin-right:6px;}
          table.picks {width:100%;border-collapse:collapse;}
          table.picks th, table.picks td {padding:6px;border-bottom:1px solid #eee;text-align:left;font-size:13px;}
        </style>
        """
        picked_html += "<table class='picks'><thead><tr><th>Sport</th><th>Event</th><th>Selection</th><th>Start</th><th>Price</th><th>Reason</th><th></th></tr></thead><tbody>"
        for i, leg in enumerate(st.session_state.parlay):
            prob = None
            try:
                prob = 1.0 / float(leg.get("price", 0))
            except Exception:
                prob = None
            color = color_for_prob(prob)
            sport = leg.get("sport") or ""
            start = leg.get("commence_time") or ""
            # include remove button cell (we'll render a placeholder; actual Remove is below)
            picked_html += (
                f"<tr>"
                f"<td><span class='pill' style='background:{color}'>{sport}</span></td>"
                f"<td>{leg.get('title','')}</td>"
                f"<td><strong>{leg.get('selection','')}</strong></td>"
                f"<td>{start}</td>"
                f"<td>{leg.get('price')}</td>"
                f"<td>{leg.get('reason','')}</td>"
                f"<td> </td>"
                f"</tr>"
            )
        if not st.session_state.parlay:
            picked_html += "<tr><td colspan='7'><em>No picks yet</em></td></tr>"
        picked_html += "</tbody></table>"
        st.markdown(picked_html, unsafe_allow_html=True)

    render_parlay_table()

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
                st.experimental_rerun()

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
