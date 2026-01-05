# Parlay Builder — TheOddsAPI (debug + demo fallback)
#
# Paste this entire file into pages/05_Parlay_TheOddsAPI.py (overwrite).
# This version:
# - Robustly handles HTTP responses and rate-limit / auth errors
# - Adds a Self-test and Manual HTTP fetch tools (see Quick debug)
# - Provides a "Use demo data" fallback so you can exercise the UI even
#   when your API key is rate-limited or invalid
# - Keeps safe parsing guards (never mutates non-dict events)
#
# Quick usage:
# 1) Backup current file:
#    cp pages/05_Parlay_TheOddsAPI.py pages/05_Parlay_TheOddsAPI.py.bak
# 2) Overwrite with this file (paste)
# 3) Syntax check:
#    python -m py_compile pages/05_Parlay_TheOddsAPI.py
# 4) Start/reload Streamlit, paste THEODDS_API_KEY into the sidebar or enable "Use demo data"
#
import os
import hashlib
import json
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from statistics import median

import requests
import streamlit as st

st.set_page_config(page_title="Parlay Builder — TheOddsAPI (Debug + Demo)", layout="wide")

# -------------------------
# Demo data (fallback)
# -------------------------
DEMO_RAW_EVENTS = [
    {
        "id": "demo-event-1",
        "title": "Demo Team A vs Demo Team B",
        "teams": ["Demo Team A", "Demo Team B"],
        "commence_time": "2026-01-06T01:00:00Z",
        "bookmakers": [
            {
                "key": "demo_bm_1",
                "title": "DemoBook",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Demo Team A", "price": 1.75},
                            {"name": "Demo Team B", "price": 2.10},
                        ],
                    }
                ],
            },
            {
                "key": "demo_bm_2",
                "title": "OtherBook",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Demo Team A", "price": 1.80},
                            {"name": "Demo Team B", "price": 2.05},
                        ],
                    }
                ],
            },
        ],
    },
    {
        "id": "demo-event-2",
        "title": "Demo Fighter X vs Demo Fighter Y",
        "teams": ["Demo Fighter X", "Demo Fighter Y"],
        "commence_time": "2026-01-07T03:00:00Z",
        "bookmakers": [
            {
                "key": "demo_bm_1",
                "title": "DemoBook",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Demo Fighter X", "price": 1.30},
                            {"name": "Demo Fighter Y", "price": 3.50},
                        ],
                    }
                ],
            }
        ],
    },
]

# -------------------------
# Sidebar / settings
# -------------------------
st.sidebar.header("Settings / TheOddsAPI")
api_key_input = st.sidebar.text_input("TheOddsAPI Key (or set THEODDS_API_KEY env)", type="password")
API_KEY = api_key_input.strip() or os.environ.get("THEODDS_API_KEY", "")

use_demo_data = st.sidebar.checkbox("Use demo data (no API calls)", value=False)

regions = st.sidebar.multiselect("Regions", options=["us", "uk", "eu", "au"], default=["us"])
markets = st.sidebar.multiselect("Markets", options=["h2h", "spreads", "totals"], default=["h2h"])
odds_ttl = st.sidebar.number_input("Odds cache key (change to bust cache)", min_value=1, value=20, step=1)

st.sidebar.markdown("---")
scan_all_sports = st.sidebar.checkbox("Scan all sports for picks (may be slow)", value=False)
filter_bm_before_11_pt = st.sidebar.checkbox("Auto-filter by bookmaker before 11:00 PT", value=False)
st.sidebar.markdown("---")
if not API_KEY and not use_demo_data:
    st.sidebar.warning("No API key set. Use demo data or set THEODDS_API_KEY.")

# -------------------------
# Session defaults
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
if "self_test_result" not in st.session_state:
    st.session_state.self_test_result: Dict[str, Any] = {}

# -------------------------
# Helpers: HTTP + JSON
# -------------------------
def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return {"ok": True, "data": json.loads(text)}
    except Exception as e:
        return {"ok": False, "error": f"json parse error: {e}", "raw_preview": text[:2000]}


def http_get_debug(url: str, timeout: int = 20) -> Dict[str, Any]:
    info: Dict[str, Any] = {"url": url}
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        info.update({"ok": False, "error": f"request failed: {e}", "trace": traceback.format_exc()})
        return info

    headers = dict(resp.headers or {})
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

    # rate headers
    for hk, hv in headers.items():
        kl = hk.lower()
        if any(x in kl for x in ("rate", "limit", "remaining", "retry-after")):
            info.setdefault("rate_headers", {})[hk] = hv
    return info

# -------------------------
# TheOddsAPI wrappers
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
# Parsing helpers (safe)
# -------------------------
def _safe_name(o: Any) -> str:
    return (o.get("name") or o.get("participant") or o.get("label") or "").strip() if isinstance(o, dict) else ""


def _safe_price(o: Any) -> Optional[float]:
    if not isinstance(o, dict):
        return None
    raw = o.get("price") or o.get("decimal") or o.get("odds")
    try:
        return float(raw) if raw is not None else None
    except Exception:
        return None


def extract_events_from_api_json(api_json: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in api_json or []:
        if not isinstance(ev, dict):
            out.append({"event_id": None, "title": repr(ev)[:120], "teams": [], "commence_time": "", "bookmakers": 0, "outcomes": [], "raw_event": ev})
            continue
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("sport_key") or ev.get("title")
        title = ev.get("title") or " / ".join(ev.get("teams") or []) or str(ev_id)
        teams = ev.get("teams") or []
        commence = ev.get("commence_time") or ev.get("start") or ""
        bookmakers = ev.get("bookmakers") or []
        price_map: Dict[str, List[float]] = {}
        for bm in (bookmakers or []):
            if not isinstance(bm, dict):
                continue
            for m in bm.get("markets", []) or []:
                if not isinstance(m, dict):
                    continue
                for o in m.get("outcomes", []) or []:
                    name = _safe_name(o)
                    price = _safe_price(o)
                    if not name or price is None:
                        continue
                    price_map.setdefault(name, []).append(price)
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
            "outcomes": outcomes,
            "raw_event": ev,
        }
        out.append(simplified)
    return out

# -------------------------
# Self-test helper
# -------------------------
def run_self_test(api_key: str, sport_to_test: str, regions_list: List[str], markets_list: List[str]) -> Dict[str, Any]:
    res = {"timestamp": datetime.now(timezone.utc).isoformat(), "api_key_present": bool(api_key)}
    try:
        if not api_key:
            res["error"] = "No API key provided"
            return res
        sports_info = api_sports(api_key)
        res["sports_info"] = {"ok": sports_info.get("ok"), "status": sports_info.get("status_code"), "body_count": sports_info.get("body_count")}
        combos_tried = []
        combos = []
        if regions_list and markets_list:
            combos.append((",".join(regions_list), ",".join(markets_list)))
        combos += [("us", "h2h"), ("us,uk,eu,au", "h2h"), ("us,uk,eu,au", "h2h,spreads,totals")]
        for r, m in combos:
            info = api_events_for_sport(api_key, sport_to_test, r, m)
            entry = {"regions": r, "markets": m, "ok": info.get("ok"), "status": info.get("status_code"), "body_count": info.get("body_count"), "rate_headers": info.get("rate_headers")}
            if info.get("body") and info["body"].get("ok") and isinstance(info["body"].get("data"), list) and info["body"]["data"]:
                sample0 = info["body"]["data"][0]
                entry["first_event_keys"] = list(sample0.keys())[:40] if isinstance(sample0, dict) else ["non-dict event"]
                parse_checks = {"has_teams": bool(sample0.get("teams")) if isinstance(sample0, dict) else False, "bookmakers_len": len(sample0.get("bookmakers") or []) if isinstance(sample0, dict) else 0, "has_outcomes": False}
                if isinstance(sample0, dict):
                    for bm in (sample0.get("bookmakers") or []):
                        for mk in (bm.get("markets") or []):
                            if mk.get("outcomes"):
                                parse_checks["has_outcomes"] = True
                                break
                        if parse_checks["has_outcomes"]:
                            break
                entry["parse_checks_first_event"] = parse_checks
                combos_tried.append(entry)
                res["found_nonempty"] = True
                res["successful_combo"] = {"regions": r, "markets": m}
                break
            combos_tried.append(entry)
        res["combos_tried"] = combos_tried
        return res
    except Exception:
        return {"ok": False, "error": "self-test exception", "trace": traceback.format_exc()}

# -------------------------
# UI: header / debug tools
# -------------------------
st.title("Parlay Builder — TheOddsAPI (Debug + Demo)")
st.write("If no events load: enable 'Use demo data' or run the Self-test to inspect API responses and headers.")

with st.expander("Quick debug & Self-test", expanded=True):
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Test fetch sports"):
            if use_demo_data:
                st.success("Using demo data — sports list simulated.")
                st.session_state.raw_fetch_info = {"demo": True, "sports": ["demo_sport"]}
            elif not API_KEY:
                st.error("No API key provided.")
            else:
                info = api_sports(API_KEY)
                st.session_state.raw_fetch_info = {"sports": info}
                sc = info.get("status_code")
                if sc in (401, 403):
                    rate = info.get("rate_headers") or {}
                    st.error(f"HTTP {sc}: Unauthorized / quota. Rate headers: {rate}")
                elif info.get("ok") and info.get("body", {}).get("ok"):
                    keys = [s.get("key") for s in info["body"]["data"][:12]]
                    st.success(f"Fetched {info.get('body_count') or len(keys)} sports. Sample: {keys}")
                else:
                    st.error("Failed to fetch sports. See 'Last raw fetch' on the right.")
    with col_b:
        if st.button("Show last load info"):
            st.write(st.session_state.last_load_info or "No loads recorded yet.")

    st.markdown("---")
    st.write("Run the self-test (will make a few safe requests). Set sport key to test (default: americanfootball_nfl).")
    test_sport = st.text_input("Sport key for self-test", value="americanfootball_nfl")
    if st.button("Run self-test"):
        if use_demo_data:
            st.info("Demo mode: self-test simulated — demo events available.")
            st.session_state.self_test_result = {"demo": True, "note": "demo data in use"}
        else:
            st.info("Running self-test — please wait...")
            result = run_self_test(API_KEY, test_sport, regions or ["us"], markets or ["h2h"])
            st.session_state.self_test_result = result
            if not result.get("api_key_present"):
                st.error("Self-test: no API key provided.")
            else:
                if result.get("found_nonempty"):
                    sc = result.get("successful_combo") or {}
                    st.success(f"Self-test found events using regions={sc.get('regions')} markets={sc.get('markets')}")
                else:
                    # detect rate-limit or auth issues on tries
                    any_401 = any(c.get("status") in (401, 403) for c in result.get("combos_tried", []))
                    if any_401:
                        st.error("Self-test shows authorization failures (401/403). Check API key or quota.")
                    else:
                        st.warning("Self-test did not find events. See 'Self-test result' on the right for combos tried and rate headers.")

    st.markdown("---")
    st.write("Manual HTTP fetch for a sport (inspects status, headers and sample JSON).")
    manual_sport = st.text_input("Manual sport key", value="americanfootball_nfl", key="manual_sport")
    manual_regions = st.text_input("Manual regions (csv)", value="us", key="manual_regions")
    manual_markets = st.text_input("Manual markets (csv)", value="h2h", key="manual_markets")
    if st.button("Fetch HTTP details for sport"):
        if use_demo_data:
            st.info("Demo mode: showing demo raw event structure.")
            st.session_state.raw_fetch_info = {"demo_fetch": True, "sample": DEMO_RAW_EVENTS[:1]}
            st.json({"sample_top_level_keys": list(DEMO_RAW_EVENTS[0].keys())})
        elif not API_KEY:
            st.error("No API key provided.")
        else:
            debug = api_events_for_sport(API_KEY, manual_sport, manual_regions, manual_markets)
            st.session_state.raw_fetch_info = {"manual_fetch": {"sport": manual_sport, "regions": manual_regions, "markets": manual_markets, "info": debug}}
            sc = debug.get("status_code")
            if sc in (401, 403):
                rate = debug.get("rate_headers") or {}
                st.error(f"HTTP {sc}: Unauthorized / quota. Rate headers: {rate}")
            elif debug.get("ok") and debug.get("body", {}).get("ok") and isinstance(debug["body"]["data"], list):
                st.success(f"HTTP {sc}: fetched {len(debug['body']['data'])} events. Showing first event top-level keys:")
                if debug["body"]["data"]:
                    sample0 = debug["body"]["data"][0]
                    if isinstance(sample0, dict):
                        st.json({k: sample0.get(k) for k in list(sample0.keys())[:16]})
                    else:
                        st.write("First event is not a dict; raw preview:")
                        st.text(repr(sample0)[:1000])
            else:
                st.error(f"HTTP {sc} — could not fetch events. See 'Last raw fetch' on the right for details.")

# -------------------------
# Top controls: sport selector / load
# -------------------------
top_cols = st.columns([3, 2, 1])
with top_cols[0]:
    st.info("Use these controls to load events and then preview/add safe picks.")
with top_cols[1]:
    sport_options: List[str] = []
    sport_titles: Dict[str, str] = {}
    if use_demo_data:
        sport_options = ["demo_sport"]
        sport_titles = {"demo_sport": "Demo Sport"}
    elif API_KEY:
        try:
            sp = api_sports(API_KEY)
            if sp.get("ok") and sp.get("body", {}).get("ok"):
                sport_options = [s.get("key") for s in sp["body"]["data"]]
                sport_titles = {s.get("key"): s.get("title") for s in sp["body"]["data"]}
        except Exception:
            sport_options = []
    chosen_sport_for_load = st.selectbox("Sport (for load)", options=[""] + sport_options, format_func=lambda k: sport_titles.get(k, " — choose sport — "))
with top_cols[2]:
    if st.button("Load odds for sport(s)"):
        try:
            if use_demo_data:
                st.session_state.raw_fetch_info = {"demo_load": True}
                simplified = extract_events_from_api_json(DEMO_RAW_EVENTS)
                st.session_state.events = simplified
                st.session_state.last_load_info = f"Loaded {len(simplified)} demo events at {datetime.now(timezone.utc).isoformat()}"
                st.success(f"Loaded {len(simplified)} demo events.")
            else:
                if not API_KEY:
                    st.error("Set THEODDS_API_KEY first or enable demo data.")
                else:
                    loaded_raw: List[Any] = []
                    if scan_all_sports:
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
                                    if isinstance(ev, dict):
                                        ev_copy = dict(ev)
                                        ev_copy["_sport_key"] = sk
                                        loaded_raw.append(ev_copy)
                                    else:
                                        loaded_raw.append(ev)
                    else:
                        if not chosen_sport_for_load:
                            st.warning("Choose a sport or enable Scan all sports.")
                            loaded_raw = []
                        else:
                            info = api_events_for_sport(API_KEY, chosen_sport_for_load, ",".join(regions or ["us"]), ",".join(markets or ["h2h"]))
                            st.session_state.raw_fetch_info = {"load_call": info}
                            if info.get("ok") and info.get("body", {}).get("ok"):
                                loaded_raw = info["body"]["data"] or []
                            else:
                                loaded_raw = []
                    simplified = extract_events_from_api_json(loaded_raw)
                    st.session_state.events = simplified
                    st.session_state.last_load_info = f"Loaded {len(simplified)} events at {datetime.now(timezone.utc).isoformat()}"
                    st.success(f"Loaded {len(simplified)} events.")
        except Exception as e:
            st.session_state.last_load_info = f"Load failed: {e}"
            st.session_state.raw_fetch_info = {"load_exception": traceback.format_exc()}
            st.error(f"Could not load events: {e}")

# -------------------------
# Main UI: auto-pick, raw events, parlay
# -------------------------
left, right = st.columns([2.5, 1])

with left:
    st.header("Auto-pick (safe)")
    n_legs = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
    max_decimal = st.number_input("Max decimal odds per leg", value=1.8, step=0.05, format="%.2f")
    min_imp_pct = st.slider("Min consensus implied probability (%)", min_value=30, max_value=90, value=60)
    min_imp_frac = min_imp_pct / 100.0
    max_spread_pct = st.slider("Max spread across books (%)", min_value=0, max_value=20, value=5)
    max_spread_frac = max_spread_pct / 100.0
    require_bm = st.text_input("Require bookmaker (optional)", value="")
    avoid_same = st.checkbox("Avoid >1 leg per event", value=True)

    show_teams = st.checkbox("Show teams/players in lists", value=True)
    show_bookmaker = st.checkbox("Show best bookmaker", value=True)
    show_event_id = st.checkbox("Show event id (debug)", value=False)

    st.markdown("---")
    st.subheader("Raw events (first 10)")
    if not st.session_state.events:
        st.info("No events loaded. Use Load odds or Run self-test (or enable demo data).")
    else:
        for i, ev in enumerate(st.session_state.events[:10], 1):
            teams_text = " / ".join(ev.get("teams") or []) if ev.get("teams") else ""
            id_text = f" [{ev.get('event_id')}]" if show_event_id else ""
            bm_count = ev.get("bookmakers", 0)
            st.markdown(f"**{i}. {ev.get('title')}{id_text}**  — bookmakers: {bm_count}")
            if show_teams and teams_text:
                st.write(f"Teams: {teams_text}")
            if ev.get("outcomes"):
                for o in ev["outcomes"][:6]:
                    name = o.get("name")
                    best = o.get("best")
                    cons = o.get("consensus")
                    st.write(f"- {name}  @ {best}  (consensus {cons})")
            if show_bookmaker and ev.get("outcomes"):
                top = sorted([o for o in ev["outcomes"] if o.get("best")], key=lambda x: (x.get("best") or 0), reverse=True)[:1]
                if top:
                    sel = top[0].get("name")
                    found_bm = None
                    for bm in (ev.get("raw_event", {}).get("bookmakers") or []):
                        for m in (bm.get("markets") or []):
                            for o in (m.get("outcomes") or []):
                                if (o.get("name") or "").strip() == sel and _safe_price(o) is not None:
                                    if found_bm is None or _safe_price(o) > found_bm.get("price", 0):
                                        found_bm = {"bookmaker": bm.get("title") or bm.get("key"), "price": _safe_price(o)}
                    if found_bm:
                        st.write(f"Best bookmaker for top selection: {found_bm['bookmaker']} @ {found_bm['price']:.2f}")

    st.markdown("---")
    if st.button("Preview safest picks"):
        pool = []
        for ev in st.session_state.events:
            for o in ev.get("outcomes", []):
                cons = o.get("consensus") or 0.0
                best = o.get("best") or 0.0
                worst = o.get("worst") or best
                spread = 0.0
                try:
                    if cons:
                        spread = abs((worst or best) - best) / cons
                except Exception:
                    spread = 0.0
                implied = (1.0 / cons) if cons and cons > 0 else 0.0
                if best > float(max_decimal):
                    continue
                if implied < float(min_imp_frac):
                    continue
                if spread > float(max_spread_frac):
                    continue
                pool.append({
                    "event_id": ev.get("event_id"),
                    "title": ev.get("title"),
                    "selection": o.get("name"),
                    "price": o.get("best"),
                    "consensus": o.get("consensus"),
                    "safety": (implied - (spread * 0.5)),
                })
        if not pool:
            st.warning("No candidates found. Try relaxing filters or inspect raw events.")
        else:
            st.write(f"Found {len(pool)} candidate outcomes (showing up to 50):")
            for i, p in enumerate(pool[:50], 1):
                st.write(f"{i}. {p['title']} — {p['selection']} @ {p['price']} (safety {p['safety']:.3f})")

    if st.button("Add safest picks (top N)"):
        pool = []
        for ev in st.session_state.events:
            for o in ev.get("outcomes", []):
                best = o.get("best") or 0.0
                pool.append({
                    "event_id": ev.get("event_id"),
                    "title": ev.get("title"),
                    "selection": o.get("name"),
                    "price": best,
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
        st.success(f"Added {added} picks.")

with right:
    st.header("Parlay summary & debug")
    st.write("Last load info:")
    st.write(st.session_state.last_load_info or "No loads recorded yet.")
    st.write("Last raw fetch (session):")
    if st.session_state.raw_fetch_info:
        st.json(st.session_state.raw_fetch_info)
    else:
        st.write("{}")

    st.markdown("---")
    st.subheader("Self-test result (session):")
    if st.session_state.self_test_result:
        st.json(st.session_state.self_test_result)
    else:
        st.write("No self-test run yet.")

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
# End of file
