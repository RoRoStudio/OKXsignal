"""
session_manager.py
Handles user session state management (theme, selected pair, timeframe, caching).
"""

import streamlit as st

def initialize_session_state():
    """Ensures session variables exist to track user settings."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"  # Default to light mode
    if "selected_pair" not in st.session_state:
        st.session_state["selected_pair"] = "BTC-USDT"  # Default trading pair
    if "timeframe" not in st.session_state:
        st.session_state["timeframe"] = "1H"  # Default timeframe

def toggle_theme():
    """Switch between light and dark themes."""
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

def get_current_theme():
    """Returns the current theme mode (light/dark)."""
    return st.session_state["theme"]

def get_selected_pair():
    """Returns the currently selected trading pair."""
    return st.session_state["selected_pair"]

def get_timeframe():
    """Returns the currently selected timeframe."""
    return st.session_state["timeframe"]

def update_selected_pair(pair: str):
    """Updates the selected trading pair."""
    st.session_state["selected_pair"] = pair

def update_timeframe(timeframe: str):
    """Updates the selected timeframe."""
    st.session_state["timeframe"] = timeframe

# Ensure session state is initialized
initialize_session_state()