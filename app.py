import streamlit as st
import requests  # You'll need this to talk to the Sports API

st.title("🔥 Live Player Prop Tracker")

# This part refreshes every 30 seconds automatically
@st.fragment(run_every="30s")
def live_prop_updates():
    # 1. This is where you would call your API (e.g., The Odds API)
    # For now, we simulate the "Live Price" or "Current Stat"
    st.subheader("Current Live Lines")
    
    col1, col2 = st.columns(2)
    col1.metric("LeBron James Points", "24.5", delta="+1.5")
    col2.metric("Kevin Durant Rebounds", "7.5", delta="-0.5")
    
    st.caption("Live updates every 30 seconds...")

live_prop_updates()
