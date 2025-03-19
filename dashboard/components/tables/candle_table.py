"""
candle_table.py
📊 Displays fetched market data in a Streamlit table.
"""

import streamlit as st
from dashboard.utils.data_loader import fetch_market_data

def show_candle_table(pair="BTC-USDT", timeframe="1H"):
    """
    Displays a table of candlestick data with indicators.
    """
    st.subheader(f"{pair} - {timeframe} Candles")

    # Fetch Data
    data = fetch_market_data(pair, timeframe, 100)

    if not data:
        st.warning("No data available.")
        return

    # Display Table
    st.dataframe(data, use_container_width=True)

