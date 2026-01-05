import streamlit as st
import random

st.title("🏀 NBA Live Prop Watch")

# Simulation of Live Data
@st.fragment(run_every="10s")
def live_tracker():
    st.subheader("Live Performance vs. Line")
    
    # Example Player Data
    player = "LeBron James"
    pregame_line = 24.5
    current_points = random.randint(10, 28) # Simulated live stat
    minutes_played = 18
    projected_minutes = 36
    
    # Calculate "Pace"
    pace_to_hit = (current_points / minutes_played) * projected_minutes
    
    # Display the Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-Game Line", f"{pregame_line}")
    c2.metric("Current Points", f"{current_points}")
    
    # Determine if it's a "Good" or "Bad" prop based on pace
    if pace_to_hit > pregame_line:
        c3.metric("Projected Finish", f"{pace_to_hit:.1f}", delta="🔥 ON PACE")
    else:
        c3.metric("Projected Finish", f"{pace_to_hit:.1f}", delta="⚠️ BEHIND", delta_color="inverse")

    # Visual Progress Bar
    progress = min(current_points / pregame_line, 1.0)
    st.write(f"Progress toward Over: {int(progress*100)}%")
    st.progress(progress)

live_tracker()
