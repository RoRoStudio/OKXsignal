"""
filter_form.py
🔍 Provides filtering options (pair, timeframe, date range) for market data.
"""

import streamlit as st
import datetime
from dashboard.utils.data_loader import fetch_trading_pairs

def filter_form():
    """Displays the filtering UI inside a styled container."""
    
    with st.container():
        st.markdown("### 🔍 Filter Data")
        
        # Fetch available trading pairs
        trading_pairs = fetch_trading_pairs()  # Ensure this fetches all 310 pairs!
        
        # Select Trading Pair
        st.selectbox(
            "Trading Pair",
            trading_pairs,
            key="selected_pair"
        )

        # Select Timeframe
        st.radio(
            "Timeframe",
            ["1H", "1D"],
            key="selected_timeframe",
            horizontal=True
        )

        # Select Date Range
        st.date_input(
            "Select Date Range",
            [],
            key="selected_date_range"
        )

# Initialize session state variables if they don't exist
if "selected_pair" not in st.session_state:
    st.session_state["selected_pair"] = "BTC-USDT"
if "selected_timeframe" not in st.session_state:
    st.session_state["selected_timeframe"] = "1H"
if "selected_date_range" not in st.session_state:
    st.session_state["selected_date_range"] = []