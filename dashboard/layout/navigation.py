"""
navigation.py
🧭 Handles sidebar navigation for OKXsignal.
"""

import streamlit as st
from dashboard.utils.session_manager import toggle_theme

def build_sidebar():
    """
    Creates the sidebar for navigation and returns the selected page function.
    """
    with st.sidebar:
        st.title("⚡ OKXsignal")
        st.markdown("---")

        pages = {
            "🏠 Home": "home",
            "📈 Market Analysis": "market_overview",
            "💰 Portfolio": "holdings",
            "⚡ Trade Execution": "trade_execution",
            "⚙️ Settings": "settings",
        }

        selected_page = st.radio("Navigate to:", list(pages.keys()))

        # Theme toggle button
        if st.button("🌗 Toggle Dark Mode"):
            toggle_theme()
            st.rerun()  # Force re-render after toggling theme

        return pages[selected_page]
