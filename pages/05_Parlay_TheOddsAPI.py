# Diagnostics-only Parlay — TheOddsAPI (targeted overwrite)
#
# Purpose: replace the page with a focused diagnostics UI that:
# - Runs one-click HTTP checks against TheOddsAPI (sports list + events for a sport)
# - Shows HTTP status, headers (rate-limit info), and a safe JSON preview/body
# - Lets you clear demo/session flags so the main app can run real requests
#
# Paste this entire file into pages/05_Parlay_TheOddsAPI.py (overwrite).
import os
import json
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import requests
import streamlit as st

st.set_page_config(page_title="TheOddsAPI Diagnostics", layout="wide")

# ---- Helpers ----
def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return {"ok": True, "data": json.loads(text)}
    except Exception as e:
        return {"ok": False, "error": f"json parse error: {e}", "raw_preview": text[:2000]}


def http_get_debug(url: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Perform GET and return structured debugging info:
    { ok, status_code, headers, body: {ok,data}|{ok:False,error,raw_preview}, body_count (if list) }
    """
    info: Dict[str, Any] = {"url": url, "ts": datetime.now(timezone.utc).isoformat()}
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

    # pull out relevant rate headers if present
    rate_hdrs = {}
    for k, v in headers.items():
        kl = k.lower()
        if any(x in kl for x in ("rate", "limit", "remaining", "retry-after", "x-requests")):
            rate_hdrs[k] = v
    if rate_hdrs:
        info["rate_headers"] = rate_hdrs

    return info


# ---- TheOddsAPI endpoints ----
def fetch_sports(api_key: str) -> Dict[str, Any]:
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    return http_get_debug(url)


def fetch_events_for_sport(api_key: str, sport_key: str, regions: str = "us", markets: str = "h2h") -> Dict[str, Any]:
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        f"?regions={regions}&markets={markets}&oddsFormat=decimal&apiKey={api_key}"
    )
    return http_get_debug(url)


# ---- UI ----
st.title("TheOddsAPI Diagnostics")
st.markdown(
    "This page is a focused diagnostics tool. It will run simple HTTP checks against TheOddsAPI and show status, headers, and a safe JSON preview. "
    "Use this to confirm your API key, rate limits, and the exact shape of responses before using the main app."
)

# Sidebar: API key + options
st.sidebar.header("Diagnostics settings")
api_key_input = st.sidebar.text_input("TheOddsAPI Key (or set THEODDS_API_KEY env)", type="password")
API_KEY = api_key_input.strip() or os.environ.get("THEODDS_API_KEY", "")
st.sidebar.write("API key present:" , "Yes" if API_KEY else "No")

st.sidebar.markdown("---")
st.sidebar.write("Quick tips:")
st.sidebar.write("- If you see HTTP 401/403: key invalid or quota/permission issue.")
st.sidebar.write("- If you see X-Requests-Remaining: 0 — quota exhausted; wait or use another key.")

# Controls
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("One-click diagnostics")
    if st.button("Fetch sports list (quick)"):
        if not API_KEY:
            st.error("No API key provided. Paste key in sidebar or set THEODDS_API_KEY.")
        else:
            with st.spinner("Fetching sports..."):
                info = fetch_sports(API_KEY)
            st.session_state["diag_last"] = {"type": "sports", "info": info}
            status = info.get("status_code")
            if status in (401, 403):
                st.error(f"HTTP {status} — Unauthorized or quota issue. Rate headers: {info.get('rate_headers')}")
            elif info.get("ok") and info.get("body", {}).get("ok"):
                data = info["body"]["data"]
                st.success(f"Fetched {len(data)} sports. Sample keys: {[s.get('key') for s in data[:12]]}")
                st.json({"sample_sport": data[:6]})
            else:
                st.error("Could not fetch sports. See 'Last raw result' on the right.")

    st.markdown("---")
    st.subheader("Fetch events for a sport")
    sport_key = st.text_input("Sport key (e.g. americanfootball_nfl)", value="americanfootball_nfl")
    regions_csv = st.text_input("Regions (csv)", value="us")
    markets_csv = st.text_input("Markets (csv)", value="h2h")
    if st.button("Fetch events for sport"):
        if not API_KEY:
            st.error("No API key provided. Paste key in sidebar or set THEODDS_API_KEY.")
        else:
            with st.spinner("Fetching events..."):
                info = fetch_events_for_sport(API_KEY, sport_key.strip(), regions_csv.strip(), markets_csv.strip())
            st.session_state["diag_last"] = {"type": "events", "sport": sport_key, "info": info}
            status = info.get("status_code")
            if status in (401, 403):
                st.error(f"HTTP {status} — Unauthorized or quota issue. Rate headers: {info.get('rate_headers')}")
            elif info.get("ok") and info.get("body", {}).get("ok"):
                body = info["body"]["data"]
                # detect likely error-array: elements are dicts with keys like message/error_code
                def looks_like_error_array(arr: List[Any]) -> bool:
                    if not isinstance(arr, list) or not arr:
                        return False
                    sample = arr[:6]
                    err_keys = {"message", "error_code", "details_url", "details"}
                    count = 0
                    for el in sample:
                        if isinstance(el, dict) and set(el.keys()).issubset(err_keys):
                            count += 1
                    return count >= max(1, len(sample) // 2)
                if looks_like_error_array(body):
                    st.error("API returned an array of error-like objects (not events). Sample shown below.")
                    sample_msgs = []
                    for e in body[:5]:
                        if isinstance(e, dict):
                            sample_msgs.append({k: e.get(k) for k in ("message", "error_code", "details_url") if k in e})
                        else:
                            sample_msgs.append(repr(e))
                    st.json({"sample_errors": sample_msgs, "rate_headers": info.get("rate_headers") or {}})
                else:
                    st.success(f"HTTP {status}: fetched {len(body)} events. Showing first item keys and preview:")
                    first = body[0] if body else {}
                    if isinstance(first, dict):
                        st.json({"first_event_keys": list(first.keys())[:40], "first_event_preview": {k: first.get(k) for k in list(first.keys())[:12]}})
                    else:
                        st.text(repr(first)[:1000])
            else:
                st.error(f"HTTP {status} — could not fetch events. See 'Last raw result' on the right.")

with col2:
    st.subheader("Session / quick actions")
    if st.button("Clear demo/session flags"):
        # Clear any keys we used on the main app for demo/test state
        for k in ("events", "raw_fetch_info", "last_load_info", "auto_loaded", "parlay", "self_test_result", "diag_last"):
            if k in st.session_state:
                del st.session_state[k]
        st.success("Cleared demo/session flags. Reload main page to pick up a clean state.")
    st.markdown("Last diagnostics result (session):")
    st.json(st.session_state.get("diag_last") or {"none": True})

st.markdown("---")
st.subheader("How to use this output")
st.markdown(
    "- If you see HTTP 401/403: your API key is invalid or not authorized for this endpoint. Double-check the key, account, and quota.\n"
    "- If you see rate-limit headers (X-Requests-Remaining or similar) with 0: your quota is exhausted. Wait or use a different key.\n"
    "- If the API returns an array of objects whose keys are 'message', 'error_code', etc, that's an error envelope — do not treat it as events.\n"
    "- Paste the full JSON from the green 'Last diagnostics result (session)' block here if you want me to inspect it."
)

st.markdown("---")
st.caption("When diagnostics show valid events, return to the main app page and load events (it should now display real events instead of demo/error arrays).")
