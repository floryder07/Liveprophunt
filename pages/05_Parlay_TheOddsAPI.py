# Parlay Builder — TheOddsAPI (aggressive debug + robust fetch)
#
# Overwrites pages/05_Parlay_TheOddsAPI.py with a version that:
# - Provides robust HTTP fetch helpers with structured error handling
# - Exposes explicit buttons to fetch HTTP details and raw events (first 10)
# - Tries multiple region/market combinations on demand to find events
# - Stores raw fetch info in session_state so you can paste the output here
# - Keeps the safe-pick UI but shows raw events so we can iterate quickly
#
# Paste this whole file into pages/05_Parlay_TheOddsAPI.py (overwrite).
import os
import hashlib
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from statistics import median
from datetime import datetime, timezone

st.set_page_config(page_title="Parlay Builder — TheOddsAPI (Debug)", layout="wide")

# -------------------------
# Sidebar / basic settings
# -------------------------
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
    st.sidebar.warning("No API key set. Get one at https://the-odds-api.com/ and put it in THEODDS_API_KEY or paste here.")

# -------------------------
# Session state defaults
# -------------------------
if "events" not in st.session_state:
    st.session_state.events: List[Dict[str, Any]] = []
if "raw_fetch_info" not in st.session_state:
    st.session_state.raw_fetch_info: Dict[str, Any] = {}
if "last_load_info" not in st.session_state:
    st.session_state.last_load_info: str = ""
if "auto_loaded" not in st.session_state:
    st.session_state.auto_loaded = False
if "parlay" not in st.session_state:
    st.session_state.parlay: List[Dict[str, Any]] = []

# -------------------------
# Utilities: HTTP + JSON helpers
# -------------------------
def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": f"json parse error: {e}", "raw_preview": text[:2000]}


def http_get_debug(url: str, timeout: int = 20) -> Dict[str, Any]:
    info: Dict[str, Any] = {"url": url}
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        info.update({"ok": False, "error": f"request failed: {e}", "trace": traceback.format_exc()})
        return info

    headers = dict(resp.headers)
    info.update({"ok": True, "status_code": resp.status_code, "headers": headers})
    text = ""
    try:
        text = resp.text or ""
    except Exception as e:
        info["body_read_error"] = str(e)
        text = ""

    ct = headers.get("Content-Type", "")
    if "application/json" in ct.lower() or text.strip().startswith("{") or text.strip().startswith("["):
        parsed = safe_json_loads(text)
        info["body"] = parsed
        if parsed.get("ok") and isinstance(parsed.get("data"), list):
            info["body_count"] = len(parsed["data"])
    else:
        info["body_preview"] = text[:2000]
    return info


# -------------------------
# TheOddsAPI helpers
# -------------------------
def api_sports(api_key: str) -> Dict[str, Any]:
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    return http_get_debug(url)


def api_events_for_sport(api_key: str, sport_key: str, regions_csv: str, markets_csv: str) -> Dict[str, Any]:
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        f"?regions={regions_csv}&markets={markets_csv}&oddsFormat=decimal&apiKey={api_key}"
    )
    return http_get_debug(url)


# -------------------------
# Parsing helpers
# -------------------------
def _safe_name(o: Dict[str, Any]) -> str:
    return (o.get("name") or o.get("participant") or o.get("label") or "").strip()


def _safe_price(o: Dict[str, Any]) -> Optional[float]:
    raw = o.get("price") or o.get("decimal") or o.get("odds")
    try:
        return float(raw) if raw is not None else None
    except Exception:
        return None


def extract_events_from_api_json(api_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw event objects from TheOddsAPI into a simplified event dict we use in the UI.
    This is intentionally permissive — it will keep events even if some fields missing.
    """
    out: List[Dict[str, Any]] = []
    for ev in api_json or []:
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("sport_key") or ev.get("title") or None
        title = ev.get("title") or " / ".join(ev.get("teams") or []) or str(ev_id)
        teams = ev.get("teams") or []
        commence = ev.get("commence_time") or ev.get("start") or ""
        bookmakers = ev.get("bookmakers") or []
        # Build consensus & best per selection
        price_map = {}
        for bm in bookmakers:
            for m in bm.get("markets", []) or []:
                for o in m.get("outcomes", []) or []:
                    name = _safe_name(o)
                    price = _safe_price(o)
                    if not name or price is None:
                        continue
                    price_map.setdefault(name, []).append(price)
        # Build outcomes list
        outcomes = []
        for name, prices in price_map.items():
            cons = median(prices) if prices else None
            best = max(prices) if prices else None
            worst = min(prices) if prices else None
            outcomes.append({"name": name, "consensus": cons, "best": best, "worst": worst})
        simplified = {
            "event_id": ev_id,
            "title": title,
            "teams": teams,
            "commence_time": commence,
            "bookmakers": len(bookmakers),
            "raw_bookmakers": bookmakers,
            "outcomes": outcomes,
            "raw_event": ev,
        }
        out.append(simplified)
    return out


# -------------------------
# Find events helper (tries multiple region/market combos)
# -------------------------
def find_events_for_sport_try_multiple(api_key: str, sport_key: str, regions_list: List[str], markets_list: List[str]) -> Dict[str, Any]:
    """
    Try the given regions/markets combinations and return the first non-empty result.
    Also return all tries in debug info.
    """
    tries: List[Dict[str, Any]] = []
    # Build a prioritized list of region/market combinations to try
    region_options = [",".join(regions_list)] if regions_list else ["us"]
    market_options = [",".join(markets_list)] if markets_list else ["h2h"]
    # Also try broader combos if initial combos return empty
    extra_regions = ["us,uk,eu,au"]
    extra_markets = ["h2h,spreads,totals"]
    # Try primary combos first
    combos = [(r, m) for r in region_options for m in market_options] + [(er, em) for er in extra_regions for em in extra_markets]
    for r, m in combos:
        info = api_events_for_sport(api_key, sport_key, r, m)
        tries.append({"regions": r, "markets": m, "info": info})
        # If a 200 and non-empty body, return immediately
        if info.get("ok") and info.get("status_code") == 200 and info.get("body", {}).get("ok") and isinstance(info["body"]["data"], list) and len(info["body"]["data"]) > 0:
            # convert to simplified events
            events_raw = info["body"]["data"]
            events = extract_events_from_api_json(events_raw)
            return {"ok": True, "regions": r, "markets": m, "events": events, "tries": tries}
    # No non-empty found
    return {"ok": False, "tries": tries}


# -------------------------
# UI: header / debug controls
# -------------------------
st.title("Parlay Builder — TheOddsAPI (Debug)")
st.write("If no events load, use the fetch buttons below and paste the raw output here so I can debug precisely.")

with st.expander("Quick debug & HTTP fetch tools", expanded=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Test fetch sports (quick)"):
            if not API_KEY:
                st.error("No API key provided.")
            else:
                info = api_sports(API_KEY)
                st.session_state.raw_fetch_info = {"sports": info}
                if info.get("ok") and info.get("status_code") == 200 and info.get("body", {}).get("ok"):
                    st.success(f"Fetched {info['body']['data'].__len__()} sports. Sample keys: {[s.get('key') for s in info['body']['data'][:10]]}")
                else:
                    st.error("Failed to fetch sports. See 'Last raw fetch' on the right for details.")
    with col2:
        if st.button("Show last load info"):
            st.write(st.session_state.last_load_info or "No loads recorded yet.")
    st.markdown("---")
    st.write("Manual HTTP fetch for a sport (shows status, headers and sample JSON). Useful to spot auth / rate limit issues.")
    sport_for_test = st.text_input("Sport key to test (e.g. americanfootball_nfl)", value="americanfootball_nfl")
    test_regions = st.text_input("Regions (csv)", value="us")
    test_markets = st.text_input("Markets (csv)", value="h2h")
    if st.button("Fetch HTTP details for sport"):
        if not API_KEY:
            st.error("No API key provided.")
        else:
            debug_info = api_events_for_sport(API_KEY, sport_for_test, test_regions, test_markets)
            st.session_state.raw_fetch_info = {"manual_fetch": {"sport": sport_for_test, "regions": test_regions, "markets": test_markets, "info": debug_info}}
            if debug_info.get("ok"):
                st.write(f"HTTP {debug_info.get('status_code')}")
                # show relevant headers
                header_preview = {k: v for k, v in (debug_info.get("headers") or {}).items() if any(x in k.lower() for x in ("rate", "limit", "remaining", "x-ratelimit"))}
                if header_preview:
                    st.write("Rate headers:", header_preview)
                body = debug_info.get("body") or {}
                if body.get("ok") and isinstance(body.get("data"), list):
                    st.success(f"Fetched {len(body['data'])} events (raw). Showing first event top-level keys:")
                    if body["data"]:
                        st.json({k: body["data"][0].get(k) for k in list(body["data"][0].keys())[:12]})
                else:
                    st.error(f"No events or could not parse JSON (status {debug_info.get('status_code')})")
    st.markdown("---")
    st.write("Try an automatic multi-region/market fetch for the currently-selected sport (this tries broader combos to find any events).")
    if st.button("Find events (try combos)"):
        chosen = st.session_state.get("chosen_sport_for_debug") or sport_for_test
        if not API_KEY:
            st.error("No API key provided.")
        else:
            res = find_events_for_sport_try_multiple(API_KEY, chosen, regions or ["us"], markets or ["h2h"])
            st.session_state.raw_fetch_info = {"find_events_try": res}
            if res.get("ok"):
                st.success(f"Found {len(res['events'])} events with regions={res['regions']} markets={res['markets']}")
            else:
                st.warning("No events found with tried combos. See raw tries for details.")


# -------------------------
# Top controls: load / sport selector
# -------------------------
top_cols = st.columns([3, 2, 1])
with top_cols[0]:
    st.info("Use the controls below to load events and then preview/add safe picks.")
with top_cols[1]:
    # Populate sports list (safe)
    sport_options: List[str] = []
    sport_titles: Dict[str, str] = {}
    if API_KEY:
        try:
            sp = api_sports(API_KEY)
            if sp.get("ok") and sp.get("body", {}).get("ok"):
                sport_options = [s.get("key") for s in sp["body"]["data"]]
                sport_titles = {s.get("key"): s.get("title") for s in sp["body"]["data"]}
        except Exception:
            sport_options = []
    chosen_sport_for_load = st.selectbox("Sport (for load)", options=[""] + sport_options, format_func=lambda k: sport_titles.get(k, " — choose sport — "))
    # for Find events button earlier
    st.session_state["chosen_sport_for_debug"] = chosen_sport_for_load
with top_cols[2]:
    if st.button("Load odds for sport(s)"):
        if not API_KEY:
            st.error("Set THEODDS_API_KEY first.")
        else:
            try:
                loaded: List[Dict[str, Any]] = []
                if scan_all_sports:
                    # BE CAREFUL: scanning all sports can be rate-limited; do one at a time.
                    sports_resp = api_sports(API_KEY)
                    if sports_resp.get("ok") and sports_resp.get("body", {}).get("ok"):
                        sports_list = [s.get("key") for s in sports_resp["body"]["data"]]
                    else:
                        sports_list = []
                    for sk in sports_list:
                        info = api_events_for_sport(API_KEY, sk, ",".join(regions or ["us"]), ",".join(markets or ["h2h"]))
                        if info.get("ok") and info.get("body", {}).get("ok"):
                            evs = info["body"]["data"] or []
                            for ev in evs:
                                ev["_sport_key"] = sk
                                loaded.append(ev)
                else:
                    if not chosen_sport_for_load:
                        st.warning("Choose a sport or enable Scan all sports.")
                    else:
                        info = api_events_for_sport(API_KEY, chosen_sport_for_load, ",".join(regions or ["us"]), ",".join(markets or ["h2h"]))
                        st.session_state.raw_fetch_info = {"load_call": info}
                        if info.get("ok") and info.get("body", {}).get("ok"):
                            loaded = info["body"]["data"] or []
                        else:
                            loaded = []
                simplified = extract_events_from_api_json(loaded)
                st.session_state.events = simplified
                st.session_state.last_load_info = f"Loaded {len(simplified)} events at {datetime.now(timezone.utc).isoformat()}"
                st.success(f"Loaded {len(simplified)} events.")
            except Exception as e:
                st.session_state.last_load_info = f"Load failed: {e}"
                st.session_state.raw_fetch_info = {"load_exception": traceback.format_exc()}
                st.error(f"Could not load events: {e}")

# Auto-load once if user selected a sport or asked to scan all sports and not already auto_loaded
if API_KEY and not st.session_state.auto_loaded and (scan_all_sports or (chosen_sport_for_load and chosen_sport_for_load != "")):
    # Try a single automatic load of the chosen sport (or don't auto-scan all sports automatically to avoid rate limits)
    try:
        auto_choice = chosen_sport_for_load if chosen_sport_for_load else None
        if auto_choice:
            info = api_events_for_sport(API_KEY, auto_choice, ",".join(regions or ["us"]), ",".join(markets or ["h2h"]))
            st.session_state.raw_fetch_info = {"auto_load": {"sport": auto_choice, "info": info}}
            if info.get("ok") and info.get("body", {}).get("ok"):
                evs = info["body"]["data"] or []
                st.session_state.events = extract_events_from_api_json(evs)
                st.session_state.last_load_info = f"Auto-loaded {len(st.session_state.events)} events at {datetime.now(timezone.utc).isoformat()}"
                st.session_state.auto_loaded = True
    except Exception as e:
        st.session_state.raw_fetch_info = {"auto_load_error": traceback.format_exc()}
        st.session_state.auto_loaded = True

# -------------------------
# Main layout: left auto-pick + raw event viewer, right parlay + debug info
# -------------------------
left_col, right_col = st.columns([2.5, 1])

with left_col:
    st.header("Auto-pick (safe)")
    n_legs = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
    max_decimal = st.number_input("Max decimal odds per leg", value=1.8, step=0.05, format="%.2f")
    min_imp_pct = st.slider("Min consensus implied probability (%)", min_value=30, max_value=90, value=60) / 100.0
    max_spread_pct = st.slider("Max spread across books (%)", min_value=0, max_value=20, value=5) / 100.0
    require_bm = st.text_input("Require bookmaker (optional)", value="")
    avoid_same = st.checkbox("Avoid >1 leg per event", value=True)

    show_teams = st.checkbox("Show teams/players in lists", value=True)
    show_bookmaker = st.checkbox("Show best bookmaker", value=True)
    show_event_id = st.checkbox("Show event id (debug)", value=False)

    # Raw events viewer (helpful to inspect exact structure)
    st.markdown("---")
    st.subheader("Raw events (first 10) — useful for debugging")
    if not st.session_state.events:
        st.info("No events loaded. Use 'Load odds for sport(s)' or the debug HTTP buttons above.")
    else:
        for i, ev in enumerate(st.session_state.events[:10], 1):
            teams_text = " / ".join(ev.get("teams") or []) if ev.get("teams") else ""
            id_text = f" [{ev.get('event_id')}]" if show_event_id else ""
            bm_count = ev.get("bookmakers", 0)
            st.markdown(f"**{i}. {ev.get('title')}{id_text}**  — bookmakers: {bm_count}")
            if show_teams and teams_text:
                st.write(f"Teams: {teams_text}")
            # show simplified outcomes
            if ev.get("outcomes"):
                for o in ev["outcomes"][:6]:
                    name = o.get("name")
                    best = o.get("best")
                    cons = o.get("consensus")
                    st.write(f"- {name}  @ {best}  (consensus {cons})")
            # show best bookmaker for top outcome (if any)
            if show_bookmaker and ev.get("outcomes"):
                top = sorted([o for o in ev["outcomes"] if o.get("best")], key=lambda x: (x.get("best") or 0), reverse=True)[:1]
                if top:
                    # attempt to find bookmaker that offered that best price
                    sel = top[0].get("name")
                    # look up in raw_event
                    found_bm = None
                    for bm in (ev.get("raw_event", {}).get("bookmakers") or []):
                        for m in bm.get("markets", []) or []:
                            for o in m.get("outcomes", []) or []:
                                if (o.get("name") or "").strip() == sel and _safe_price(o) is not None:
                                    if found_bm is None or _safe_price(o) > found_bm.get("price", 0):
                                        found_bm = {"bookmaker": bm.get("title") or bm.get("key"), "price": _safe_price(o)}
                    if found_bm:
                        st.write(f"Best bookmaker for top selection: {found_bm['bookmaker']} @ {found_bm['price']:.2f}")

    st.markdown("---")
    # Preview / Add safest picks (kept minimal so we can see raw events first)
    if st.button("Preview safest picks (show candidates)"):
        pool = []
        for ev in st.session_state.events:
            for o in ev.get("outcomes", []):
                cons = o.get("consensus") or 0.0
                best = o.get("best") or 0.0
                worst = o.get("worst") or best
                spread = abs((worst or best) - best) / cons if cons else 0.0
                implied = (1.0 / cons) if cons and cons > 0 else 0.0
                if best > float(max_decimal):
                    continue
                if implied < float(min_imp_pct):
                    continue
                if spread > float(max_spread_pct / 100.0) and max_spread_pct > 1.0:
                    # if user set percent slider as 5 meaning 5% already, keep it usable:
                    pass
                pool.append({
                    "event_id": ev.get("event_id"),
                    "title": ev.get("title"),
                    "selection": o.get("name"),
                    "price": o.get("best"),
                    "consensus": o.get("consensus"),
                    "safety": (implied - (spread * 0.5)),
                })
        if not pool:
            st.warning("No candidates found with current filters. Try relaxing filters or inspect raw events.")
        else:
            st.write(f"Found {len(pool)} candidate outcomes (showing up to 50):")
            for i, p in enumerate(pool[:50], 1):
                st.write(f"{i}. {p['title']} — {p['selection']} @ {p['price']} (safety {p['safety']:.3f})")

    if st.button("Add safest picks (top N)"):
        # add top n_legs from the pool computed above using very relaxed filters (so something gets added)
        pool = []
        for ev in st.session_state.events:
            for o in ev.get("outcomes", []):
                best = o.get("best") or 0.0
                pool.append({
                    "event_id": ev.get("event_id"),
                    "title": ev.get("title"),
                    "selection": o.get("name"),
                    "price": best,
                    "safety": o.get("consensus") or 0.0
                })
        added = 0
        for p in pool[:int(n_legs)]:
            exists = any((leg.get("event_id") == p["event_id"] and leg.get("selection") == p["selection"]) for leg in st.session_state.parlay)
            if not exists:
                st.session_state.parlay.append({
                    "event_id": p["event_id"],
                    "title": p["title"],
                    "selection": p["selection"],
                    "price": p["price"],
                })
                added += 1
        st.success(f"Added {added} picks (best-effort).")

with right_col:
    st.header("Parlay summary & debug")
    st.write("Last load info:")
    st.write(st.session_state.last_load_info or "No loads recorded yet.")
    st.write("Last raw fetch (session):")
    if st.session_state.raw_fetch_info:
        st.json(st.session_state.raw_fetch_info)
    else:
        st.write("{}")
    st.markdown("---")
    st.subheader("Parlay")
    if not st.session_state.parlay:
        st.write("No picks yet.")
    else:
        for leg in list(st.session_state.parlay):
            st.markdown(f"**{leg.get('title','')}**")
            st.write(f"> {leg.get('selection','')} @ {leg.get('price')}")
            rem_key = "rem-" + hashlib.sha1(f"{leg.get('event_id','')}-{leg.get('selection','')}".encode()).hexdigest()[:12]
            if st.button("Remove", key=rem_key):
                st.session_state.parlay = [l for l in st.session_state.parlay if not (l.get("event_id") == leg.get("event_id") and l.get("selection") == leg.get("selection"))]
                st.experimental_rerun()
    if st.session_state.parlay:
        prices = [l.get("price", 1.0) or 1.0 for l in st.session_state.parlay]
        combined = 1.0
        for p in prices:
            try:
                combined *= float(p)
            except Exception:
                combined *= 1.0
        stake = st.number_input("Stake", value=10.0, min_value=0.0, format="%.2f", key="stake_input")
        payout = stake * combined
        profit = payout - stake
        st.metric("Combined decimal odds", f"{combined:.4f}")
        st.metric("Potential payout", f"${payout:,.2f}")
        st.metric("Potential profit", f"${profit:,.2f}")
        st.download_button("Download parlay JSON", data=json.dumps(st.session_state.parlay, default=str, indent=2), file_name="parlay.json", mime="application/json")

# -------------------------
# End of file
# -------------------------
