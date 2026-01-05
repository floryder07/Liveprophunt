# Lightweight error / debug helpers used by the app.
# Safe to import (no side-effects). Designed to:
# - format exceptions and tracebacks for UI/logging
# - safely parse JSON from HTTP responses
# - perform a safe HTTP GET that returns structured debug info
# - provide a Streamlit helper to render detailed error info (optional)
#
# Paste this file as `error_handling.py` in your project (overwrite existing).
import json
import traceback
from typing import Any, Dict, Optional

try:
    import requests
except Exception:
    requests = None  # type: ignore


def format_traceback(exc: BaseException) -> str:
    """
    Return a nicely formatted traceback string for the given exception.
    Safe to call from exception handlers.
    """
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(tb_lines)


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse JSON text into a Python object. On failure, return a dict with an
    'error' key and the parse message.
    """
    try:
        return {"ok": True, "data": json.loads(text)}
    except Exception as e:
        return {"ok": False, "error": f"json parse error: {e}", "raw": text[:2000]}


def http_get_debug(url: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Perform a GET request and return structured debug info.
    If `requests` is unavailable, returns an explanatory error.
    This function intentionally does not raise; it always returns a dict.
    """
    if requests is None:
        return {"ok": False, "error": "requests package not available in this environment"}

    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return {"ok": False, "error": f"request failed: {e}"}

    info: Dict[str, Any] = {
        "ok": True,
        "status_code": resp.status_code,
        "headers": {k: v for k, v in resp.headers.items()},
    }

    # Try to parse json body safely (but don't fail while parsing)
    ct = resp.headers.get("Content-Type", "")
    text = ""
    try:
        text = resp.text
    except Exception:
        info["body_read_error"] = "could not read response.text"

    if "application/json" in ct.lower() or (text.strip().startswith("{") or text.strip().startswith("[")):
        parsed = safe_json_loads(text)
        info["body"] = parsed
        # If it's an array, include length for quick inspection
        if parsed.get("ok") and isinstance(parsed.get("data"), list):
            info["body_count"] = len(parsed["data"])
    else:
        info["body_preview"] = text[:2000]

    return info


# Optional: Streamlit UI helper (importing streamlit only when used)
def render_exception_in_streamlit(exc: BaseException, label: Optional[str] = None) -> None:
    """
    Render exception details inside Streamlit. Import Streamlit lazily
    so importing this module doesn't require Streamlit to be installed.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        # If streamlit isn't available, just raise the original exception
        raise exc

    title = f"Error: {label}" if label else "Error"
    st.error(title)
    with st.expander("Show full traceback", expanded=False):
        st.code(format_traceback(exc))


# Simple helper to sanitize values for logging (avoid huge binary blobs)
def summarize_value(v: Any, max_len: int = 400) -> Any:
    """
    Return a compact summary for logging: primitives returned as-is,
    strings truncated, lists/dicts length shown, otherwise the type name.
    """
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, str):
        return v if len(v) <= max_len else v[: max_len // 2] + "..." + v[- max_len // 2 :]
    if isinstance(v, (list, tuple, set)):
        return {"type": type(v).__name__, "len": len(v), "preview": [summarize_value(x, 200) for x in list(v)[:3]]}
    if isinstance(v, dict):
        keys = list(v.keys())[:8]
        return {"type": "dict", "len": len(v), "keys": keys}
    return {"type": type(v).__name__}
