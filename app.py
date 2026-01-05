import streamlit as st

# 1. Set the page title that appears in the browser tab
st.set_page_config(page_title="Proplivehunter", page_icon="🚀")

# 2. Add a main title and some text
st.title("Liveprophunter!")
st.write("If you can see this, you have successfully fixed the file error.")

# 3. Add an interactive widget (a text input box)
name = st.text_input("Enter your name:")

# 4. Add a button that reacts when clicked
if st.button("Say Hello"):
    if name:
        st.success(f"Hello, {name}! Your Streamlit app is live.")
    else:
        st.warning("Please enter a name first!")

# 5. Add a simple sidebar for extra info
st.sidebar.header("About This Site")
st.sidebar.info("Built with Python and Streamlit in 2026.")
