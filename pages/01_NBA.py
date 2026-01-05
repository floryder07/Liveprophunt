import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'app' package can be imported from pages/
repo_root = Path(__file__).resolve().parent.parent  # pages/ -> repo root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import Streamlit and call set_page_config BEFORE any other Streamlit usage/imports
import streamlit as st

# Try to set page config; if Streamlit already set it (multi-page import), ignore the error
try:
    st.set_page_config(page_title="NBA Player Prop Tracker", page_icon="🏀", layout="wide")
except Exception:
    # Streamlit may already have initialized page config during multi-page import.
    # Ignore and continue — this prevents startup crashes.
    pass

# Now safe to import other libraries and local modules that do not do UI at import time
import logging
from typing import Any, Dict, List, Optional
from io import StringIO

import pandas as pd
import requests

from app.utils import parse_players  # ensure app/utils.py does NOT import streamlit at module import
