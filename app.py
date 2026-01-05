import streamlit as st

# This line should ONLY be here, once per application:
st.set_page_config(
    page_title="My Multi-Sport Tracker", 
    page_icon="📊", 
    layout="wide" # You can add the layout here now
)

st.title("Welcome to Your Multi-Sport Prop Tracker!")
st.write("Select a sport from the sidebar to see live props and analytics.")
st.sidebar.success("Choose your sport page above.")
