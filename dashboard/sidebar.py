"""
dashboard/sidebar.py
📌 Sidebar with navigation & buttons
"""

import streamlit as st

def add_sidebar():
    """Adds a sidebar with navigation and actions."""
    st.sidebar.title("📊 OKXsignal Dashboard")
    page = st.sidebar.radio("Navigate", ["Market Analysis", "Portfolio", "Settings"])

    # ✅ Button to fetch market data
    if st.sidebar.button("🔄 Fetch Market Data"):
        st.sidebar.success("Fetching market data... ✅ (Function to be added)")
    
    return page
