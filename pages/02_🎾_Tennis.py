import streamlit as st

st.title("🎾 Tennis Match Prop Tracker")

st.header("Match: Djokovic vs. Alcaraz")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Novak Djokovic")
    st.metric("Aces", "12", delta="+3")
    st.write("Live Odds: **-200**")

with c2:
    st.subheader("Carlos Alcaraz")
    st.metric("Aces", "8", delta="-2")
    st.write("Live Odds: **+150**")

st.divider()
st.metric("Total Games (O/U 22.5)", "18", delta="On Pace for Over")
