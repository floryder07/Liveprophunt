import streamlit as st
import random
import time

st.title("🎾 Live Tennis Total Points Tracker")

@st.fragment(run_every="15s")
def live_tennis_props():
    # [ ... your full Tennis code here ... ]
    st.subheader("Match Status: Live 🟢")
    pre_match_line = 18.5
    current_total_games = random.randint(9, 19) 
    live_line = 16.5

    st.write(f"Current Total Games: **{current_total_games}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pre-Match O/U Line", f"{pre_match_line}")
    col2.metric("Live O/U Line", f"{live_line}")

    if current_total_games > pre_match_line:
        col3.metric("Pace Indicator", "🔥 OVER PACE", delta_color="normal")
    else:
        col3.metric("Pace Indicator", "⚠️ UNDER PACE", delta_color="inverse")

    st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

live_tennis_props()
