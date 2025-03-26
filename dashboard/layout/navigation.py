"""
navigation.py
🧭 Handles sidebar navigation for OKXsignal.
"""

import streamlit as st

def build_sidebar():
    """
    Creates the sidebar for navigation and returns the selected page function.
    """
    with st.sidebar:
        st.title("⚡ OKXsignal")
        st.markdown("---")

        pages = {
            "🏠 Home": "home",
            "📈 Market Analysis": "market_analysis.overview",
            "💰 Portfolio": "portfolio.holdings",
            "⚡ Trade Execution": "trade_execution",
            "⚙️ Settings": "settings.user_prefs",
        }

        selected_page = st.radio("Navigate to:", list(pages.keys()))

        return pages[selected_page]
