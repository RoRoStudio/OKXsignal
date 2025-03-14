"""
main.py
📊 Streamlit Dashboard for OKXsignal
"""

import streamlit as st
import sys
import os

# ✅ Ensure the backend folder is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend", "execution"))

from fetch_market import main as fetch_market_data  # ✅ Now, this should work!


# ✅ Configure Streamlit Page
st.set_page_config(page_title="OKXsignal", layout="wide")

st.title("📊 OKXsignal Dashboard")
st.write("Welcome to the trading dashboard!")

# ✅ Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["📈 Market Analysis", "🔍 Portfolio", "⚙️ Settings"])

# ✅ Market Analysis Page
if page == "📈 Market Analysis":
    st.subheader("📊 Market Analysis")
    
    # ✅ Fetch Market Data Button
    if st.button("Fetch Market Data"):
        with st.spinner("Fetching market data... Please wait!"):
            data = fetch_market_data()
            st.success("✅ Market data fetched successfully!")

    # ✅ TODO: Display fetched data here

# ✅ Portfolio Page
elif page == "🔍 Portfolio":
    st.subheader("💰 Portfolio Overview")
    st.write("Your portfolio details will be shown here.")

# ✅ Settings Page
elif page == "⚙️ Settings":
    st.subheader("⚙️ Settings")
    st.write("Configure the app settings here.")
