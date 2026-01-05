"""
Parlay Builder — TheOddsAPI prototype

Fixes in this version:
- Ensures every event loaded has sport metadata (_sport_key/_sport_title) so picks always show sport.
- When adding a leg (manual or auto), we now store a 'reason' with the leg so the picked-summary table can display it.
- Manual "Add" button computes a short reason for the selection (consensus vs best) so manual adds look the same as auto-picks.
- Better fallbacks for selection names (participant/label/etc) to avoid empty selection names.
- Minor UI polishing for the picked-summary so sport/title/selection always show and the colored badge maps to sport title.
- Kept previous features (auto-pick by value/safety, DraftKings before 11 PT filter, scan-all-sports).
"""
from typing import Any, Dict, List, Optional
import os
import hashlib
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
st.title("🎯 Parlay Builder — TheOddsAPI prototype")

# ---- Sidebar: API key and options ----
st.sidebar.header("Settings / TheOddsAPI")
api_key_input = st.sidebar.text_input(
    "TheOddsAPI Key (or leave blank to use THEODDS_API_KEY env)", type="password"
)
API_KEY = api_key_input.strip() or os.environ.get("THEODDS_API_KEY", "")

regions = st.sidebar.multiselect("Regions (comma-select)", options=["us", "uk", "eu", "au"], default=["us"])
markets = st.sidebar.multiselect("Markets", options=["h2h", "spreads", "totals"], default=["h2h"])
bookmaker_filter = st.sidebar.text_input("Preferred bookmaker key/title (optional)", value="")
odds_ttl = st.sidebar.number_input("Odds cache key (change to bust cache)", min_value=1, value=20, step=1)
refresh = st.sidebar.button("Reload sports list / clear cache")

# New options
filter_dk_before_11_pt = st.sidebar.checkbox("Auto-filter: DraftKings before 11:00 PT", value=False)
scan_all_sports = st.sidebar.checkbox("Scan all sports for picks (may be slow / rate-limited)", value=False)

if not API_KEY:
    st.sidebar.warning("No API key set. Get one at https://the-odds-api.com/ and put it in THEODDS_API_KEY or paste here.")

# ---- Helpers ----
def parse_iso_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        from dateutil import parser as date_parser  # type: ignore
        dt = date_parser.isoparse(s)
    except Exception:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

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

@st.cache_data()
def fetch_odds_for_sport(api_key: str, sport_key: str, regions_list: List[str], markets_list: List[str], cache_bust: int) -> List[Dict[str, Any]]:
    regions_q = ",".join(regions_list)
    markets_q = ",".join(markets_list)
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions={regions_q}&markets={markets_q}&oddsFormat=decimal&apiKey={api_key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def _safe_name(o: Dict[str, Any]) -> str:
    """Return the best available name for an outcome/participant."""
    return o.get("name") or o.get("participant") or o.get("label") or ""

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

def has_draftkings(bookmakers: Optional[List[Dict[str, Any]]]) -> bool:
    for bm in bookmakers or []:
        key = (bm.get("key") or "").lower()
        title = (bm.get("title") or "").lower()
        if "draftk" in key or "draftk" in title:
            return True
    return False

# --- Value & safety utilities ---
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
    """Compute a short reason string (consensus vs best) for a manual Add action."""
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
        outs = get_consensus_and_best_outcomes(ev)
        for o in outs:
            cons = o["consensus"]
            best = o["best_price"]
            value = score_outcome_value(cons, best)
            implied_consensus = (1.0 / cons) if cons > 0 else None
            favorite_flag = cons < 2.0 if cons else False
            candidates.append({
                "event_id": ev_id,
                "event_title": title,
                "selection": o["name"],
                "price": best,
                "consensus": cons,
                "value": value,
                "implied_consensus": implied_consensus,
                "favorite": favorite_flag,
                "sport_key": ev.get("_sport_key"),
                "sport_title": ev.get("_sport_title"),
                "commence_time": ev.get("commence_time") or ev.get("start"),
            })

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
            cons = m["consensus"]
            implied = m["implied_consensus"] or 0.0
            best = m["best"]
            spread = m["spread_pct"]
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

def kelly_fraction(p: float, decimal_odds: float, f: float = 0.25) -> float:
    b = decimal_odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    numerator = (p * (b + 1) - 1.0)
    denom = b
    if denom == 0:
        return 0.0
    raw = numerator / denom
    return max(0.0, f * raw)

def color_for_prob(p: Optional[float]) -> str:
    if p is None:
        return "#999999"
    if p >= 0.65:
        return "#2ECC71"
    if p >= 0.5:
        return "#F1C40F"
    return "#E67E22"

# ---- UI: sports selector ----
sports: List[Dict[str, Any]] = []
if API_KEY:
    try:
        sports = fetch_sports(API_KEY)
    except Exception as e:
        st.sidebar.error(f"Could not fetch sports list: {e}")

sport_options = {s.get("key"): s.get("title") for s in sports}
selected_sport = st.sidebar.selectbox(
    "Sport (select)",
    options=[""] + list(sport_options.keys()),
    format_func=lambda k: sport_options.get(k, " — choose sport — "),
)

# ---- Load events/odds ----
load_odds_btn = st.sidebar.button("Load odds for sport(s)")
events: List[Dict[str, Any]] = []

if load_odds_btn:
    try:
        if scan_all_sports:
            all_events: List[Dict[str, Any]] = []
            for skey, stitle in sport_options.items():
                try:
                    evs = fetch_odds_for_sport(API_KEY, skey, regions, markets, int(odds_ttl))
                except Exception:
                    continue
                for ev in evs:
                    ev["_sport_key"] = skey
                    ev["_sport_title"] = stitle
                all_events.extend(evs)
            events = all_events
            if not events:
                st.sidebar.info("No events found across sports (API limits or no matches).")
        else:
            if not selected_sport:
                st.sidebar.info("Choose a sport or enable 'Scan all sports'.")
            else:
                raw_events = fetch_odds_for_sport(API_KEY, selected_sport, regions, markets, int(odds_ttl))
                sport_title = sport_options.get(selected_sport, selected_sport)
                for ev in raw_events:
                    ev["_sport_key"] = selected_sport
                    ev["_sport_title"] = sport_title
                events = raw_events
    except Exception as e:
        st.sidebar.error(f"Could not load odds: {e}")
        events = []

# Sample fallback (include sport metadata) if no events
if not events:
    st.info("No live feed loaded — using sample events.")
    events = [
        {
            "id": "ev-1",
            "title": "Ma Long vs Fan Zhendong",
            "commence_time": "2026-01-05T12:00:00Z",
            "teams": ["Ma Long", "Fan Zhendong"],
            "bookmakers": [
                {"key":"draftkings","markets":[{"key":"h2h","outcomes":[{"name":"Ma Long","price":1.45},{"name":"Fan Zhendong","price":2.10}]}]}
            ],
            "_sport_key": "table_tennis",
            "_sport_title": "Table Tennis"
        },
        {
            "id": "ev-2",
            "title": "Player A vs Player B",
            "commence_time": "2026-01-05T13:30:00Z",
            "teams": ["Player A", "Player B"],
            "bookmakers": [
                {"key":"draftkings","markets":[{"key":"h2h","outcomes":[{"name":"Player A","price":1.80},{"name":"Player B","price":2.00}]}]}
            ],
            "_sport_key": "demo_sport",
            "_sport_title": "Demo Sport"
        }
    ]

# ---- Parlay session state ----
if "parlay" not in st.session_state:
    st.session_state.parlay = []

def add_leg(ev_id: str, ev_title: str, sel_name: str, price: float, sport_title: Optional[str] = None, commence_time: Optional[str] = None, reason: Optional[str] = None) -> None:
    # prevent duplicate same-event+selection
    for leg in st.session_state.parlay:
        if leg["event_id"] == ev_id and leg["selection"] == sel_name:
            st.warning("That leg is already in the parlay. Remove it first to add again.")
            return
    st.session_state.parlay.append({
        "event_id": ev_id,
        "sport": sport_title or "",
        "title": ev_title,
        "selection": sel_name,
        "price": price,
        "commence_time": commence_time,
        "reason": reason or ""
    })

def remove_leg(i: int) -> None:
    if 0 <= i < len(st.session_state.parlay):
        st.session_state.parlay.pop(i)

def clear_parlay() -> None:
    st.session_state.parlay = []

# ---- Main layout ----
left, right = st.columns([2, 1])

with left:
    st.header("Available events & outcomes")
    for ev in events:
        ev_id = ev.get("id") or ev.get("event_id") or ev.get("key") or ev.get("title")
        title = ev.get("title") or " vs ".join(ev.get("teams", [])) or str(ev_id)
        sport_title = ev.get("_sport_title") or ev.get("_sport_key") or ""
        commence = ev.get("commence_time") or ev.get("start") or ""
        outs = extract_outcomes(ev, bookmaker_filter=bookmaker_filter)
        if not outs:
            with st.expander(f"{title} (no odds)"):
                st.write("No usable odds/outcomes found for this event.")
            continue

        header = f"{sport_title} — {title}"
        with st.expander(header):
            st.write("Start:", commence)
            for o in outs:
                cols = st.columns([4, 1])
                cols[0].write(f"{o['name']}  —  {o['price']} (market: {o['market']} | bm: {o['bookmaker']})")

                raw_key = f"{ev_id}|{o.get('name','')}|{o.get('market','')}|{o.get('bookmaker','')}"
                add_key = "add-" + hashlib.sha1(raw_key.encode()).hexdigest()[:12]

                if cols[1].button("Add", key=add_key):
                    # compute a reason for manual add so the picked summary has consistent info
                    reason = get_reason_for_selection(ev, o["name"])
                    add_leg(ev_id, title, o["name"], o["price"], sport_title=sport_title, commence_time=commence, reason=reason)

with right:
    st.header("Parlay builder")

    # Picked summary (compact + color)
    st.subheader("Picked summary")
    picked_html = """
    <style>
      .pill {display:inline-block;padding:4px 8px;border-radius:999px;color:#fff;font-size:12px;margin-right:6px;}
      table.picks {width:100%;border-collapse:collapse;}
      table.picks th, table.picks td {padding:6px;border-bottom:1px solid #eee;text-align:left;font-size:13px;}
    </style>
    """
    picked_html += "<table class='picks'><thead><tr><th>Sport</th><th>Event</th><th>Selection</th><th>Start</th><th>Price</th><th>Reason</th></tr></thead><tbody>"
    for leg in st.session_state.parlay:
        prob = None
        try:
            prob = 1.0 / float(leg.get("price", 0))
        except Exception:
            prob = None
        color = color_for_prob(prob)
        sport = leg.get("sport") or ""
        start = leg.get("commence_time") or ""
        picked_html += (
            f"<tr>"
            f"<td><span class='pill' style='background:{color}'>{sport}</span></td>"
            f"<td>{leg.get('title','')}</td>"
            f"<td><strong>{leg.get('selection','')}</strong></td>"
            f"<td>{start}</td>"
            f"<td>{leg.get('price')}</td>"
            f"<td>{leg.get('reason','')}</td>"
            f"</tr>"
        )
    if not st.session_state.parlay:
        picked_html += "<tr><td colspan='6'><em>No picks yet</em></td></tr>"
    picked_html += "</tbody></table>"
    st.markdown(picked_html, unsafe_allow_html=True)

    # controls: auto-pick value & safety
    st.subheader("Auto-pick options")
    auto_n = st.number_input("Number of legs to pick", min_value=1, max_value=10, value=3)
    auto_min_pct = st.slider("Minimum uplift vs consensus (%)", min_value=0, max_value=50, value=2)
    auto_min_value = auto_min_pct / 100.0
    avoid_same_event = st.checkbox("Avoid >1 leg per event", value=True)
    preview_only = st.checkbox("Preview only (don't auto-add)", value=True)

    # Safety pick controls
    st.write("Safety pick filters (conservative):")
    max_decimal_odds = st.number_input("Max decimal odds per leg", value=1.8, step=0.05, format="%.2f")
    min_consensus_prob = st.slider("Minimum consensus implied probability (%)", min_value=30, max_value=90, value=60) / 100.0
    max_spread_pct = st.slider("Max spread across books (%)", min_value=0, max_value=20, value=5) / 100.0
    require_dk = st.checkbox("Require DraftKings for safety picks", value=True)

    if st.button("Auto-pick (value)"):
        filtered_events = events
        if filter_dk_before_11_pt and ZoneInfo is not None:
            tz = ZoneInfo("America/Los_Angeles")
            now = datetime.now(tz)
            cutoff = datetime.combine(now.date(), time(hour=11, minute=0), tzinfo=tz)
            fe: List[Dict[str, Any]] = []
            for ev in events:
                if not has_draftkings(ev.get("bookmakers", [])):
                    continue
                ct = ev.get("commence_time") or ev.get("start")
                dt = parse_iso_datetime(ct)
                if not dt:
                    continue
                if dt.astimezone(tz) < cutoff:
                    fe.append(ev)
            filtered_events = fe

        picks = auto_pick_legs_by_value(filtered_events, n_legs=auto_n, min_value=auto_min_value, avoid_same_event=avoid_same_event)
        if not picks:
            st.warning("No value picks met the criteria.")
        else:
            st.success(f"Found {len(picks)} value pick(s).")
            for p in picks:
                p_sport = p.get("sport_title") or p.get("sport_key") or ""
                p_ct = p.get("commence_time") or ""
                st.markdown(f"**{p_sport}** — {p['event_title']} — **{p['selection']}** @ {p['price']}")
                st.write(p["reason"])
                if not preview_only:
                    add_leg(p['event_id'], p['event_title'], p['selection'], p['price'], sport_title=p_sport, commence_time=p_ct, reason=p.get("reason"))

    if st.button("Auto-pick (safest)"):
        filtered_events = events
        if filter_dk_before_11_pt and ZoneInfo is not None:
            tz = ZoneInfo("America/Los_Angeles")
            now = datetime.now(tz)
            cutoff = datetime.combine(now.date(), time(hour=11, minute=0), tzinfo=tz)
            fe: List[Dict[str, Any]] = []
            for ev in events:
                if not has_draftkings(ev.get("bookmakers", [])):
                    continue
                ct = ev.get("commence_time") or ev.get("start")
                dt = parse_iso_datetime(ct)
                if not dt:
                    continue
                if dt.astimezone(tz) < cutoff:
                    fe.append(ev)
            filtered_events = fe

        picks = auto_pick_safest_legs(
            filtered_events,
            n_legs=auto_n,
            max_decimal_odds=max_decimal_odds,
            min_consensus_prob=min_consensus_prob,
            max_spread_pct=max_spread_pct,
            require_bookmaker="draftkings" if require_dk else None,
            avoid_same_event=avoid_same_event
        )
        if not picks:
            st.warning("No safe picks found with these filters.")
        else:
            st.success(f"Found {len(picks)} safe pick(s).")
            for p in picks:
                st.markdown(f"**{p.get('sport_title') or ''}** — {p['event_title']} — **{p['selection']}** @ {p['price']}")
                st.write(p["reason"])
                if not preview_only:
                    add_leg(p['event_id'], p['event_title'], p['selection'], p['price'], sport_title=p.get("sport_title"), commence_time=p.get("commence_time"), reason=p.get("reason"))

    st.markdown("---")
    # Current parlay UI (detailed)
    st.subheader("Current parlay")
    if st.button("Clear parlay"):
        clear_parlay()
    legs = st.session_state.parlay
    if not legs:
        st.info("No legs added. Use Auto-pick or Add buttons on the left.")
    else:
        for idx, leg in enumerate(legs):
            color = color_for_prob(1.0/leg['price'] if leg.get('price') else None)
            sport_chip = f"<span style='background:{color};color:#fff;padding:4px 8px;border-radius:999px;font-size:12px'>{leg.get('sport','')}</span>"
            start_txt = f" • starts {leg.get('commence_time')}" if leg.get('commence_time') else ""
            st.markdown(f"{sport_chip}  **{leg.get('title','')}**{start_txt}  \n> **{leg.get('selection','')}**  @ {leg.get('price')}", unsafe_allow_html=True)
            rem_raw = f"{idx}|{leg.get('event_id','')}|{leg.get('selection','')}"
            rem_key = "rem-" + hashlib.sha1(rem_raw.encode()).hexdigest()[:12]
            if st.button("Remove", key=rem_key):
                remove_leg(idx)

    st.markdown("---")
    stake = st.number_input("Stake", value=10.0, min_value=0.0, format="%.2f")
    legs = st.session_state.parlay
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

st.caption(
    "Prototype uses TheOddsAPI. Auto-picks are heuristic suggestions (value vs market consensus). "
    "This is a simulation tool only — do not auto-place real bets without user confirmation and proper licensing."
)
