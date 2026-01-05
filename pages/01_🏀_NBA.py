import streamlit as st

st.title("🏀 NBA Player Prop Tracker")

# Use columns to show players side-by-side
col1, col2 = st.columns(2)

with col1:
    st.subheader("LeBron James") # The Name stands out here
    st.metric(label="Live Points", value="22", delta="+2.5 vs Line")
    st.caption("Target: 19.5 | Pace: 28.1 🔥")

with col2:
    st.subheader("Kevin Durant")
    st.metric(label="Live Rebounds", value="6", delta="-1.5 vs Line")
    st.caption("Target: 7.5 | Pace: 6.8 ⚠️")
