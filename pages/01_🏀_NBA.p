import streamlit as st
import random

st.title("🏀 NBA Live Prop Watch")

@st.fragment(run_every="10s")
def live_tracker():
    st.subheader("Live Performance vs. Line")
    # [ ... your full NBA code here ... ]
    player = "LeBron James"
    pregame_line = 24.5
    current_points = random.randint(10, 28)
    minutes_played = 18
    pace_to_hit = (current_points / minutes_played) * 36

    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-Game Line", f"{pregame_line}")
    c2.metric("Current Points", f"{current_points}")

    if pace_to_hit > pregame_line:
        c3.metric("Projected Finish", f"{pace_to_hit:.1f}", delta="🔥 ON PACE")
    else:
        c3.metric("Projected Finish", f"{pace_to_hit:.1f}", delta="⚠️ BEHIND", delta_color="inverse")

    st.progress(min(current_points / pregame_line, 1.0))

live_tracker()
