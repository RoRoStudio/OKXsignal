"""
overview.py
📊 Market Overview Page: Displays the latest 1H & 1D candles with filtering options.
"""

import streamlit as st
import datetime
import pandas as pd
import pytz
from dashboard.components.forms.filter_form import filter_form
from dashboard.utils.data_loader import fetch_market_data, fetch_trading_pairs
from dashboard.components.charts.master_chart import show_master_chart

def show_page():
    st.title("📈 Market Overview")

    # ✅ Step 1: Show Filter Form
    filter_form()

    # ✅ Step 2: Fetch Data (Only when "Apply Filters" is clicked)
    if st.session_state.get("filters_applied", False):
        trading_pairs = fetch_trading_pairs()
        selected_data = fetch_selected_market_data()

        # ✅ Step 3: Show Market Summary (Gainers, Losers, Volume Movers)
        show_market_summary(trading_pairs)

        # ✅ Step 4: Show Selected Pair Data (Instant Update)
        show_filtered_data(selected_data)

        # ✅ Step 5: Show Master Chart (Connected to Filters)
        show_master_chart(selected_data)

        # ✅ Reset applied filter flag to avoid unnecessary refreshes
        st.session_state["filters_applied"] = False
    else:
        st.warning("👆 Adjust filters and click 'Apply Filters' to refresh the table & chart.")

@st.cache_data(ttl=3600)
def fetch_selected_market_data():
    """Fetch data for the currently selected pair & timeframe (cached for 1 hour)."""
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")
    return fetch_market_data(pair=pair, timeframe=timeframe)

def show_market_summary(trading_pairs):
    """Displays key market summary statistics for rapid insight."""
    st.markdown("### 📊 Market Summary")

    summary_data = []
    for pair in trading_pairs[:50]:  # Limit for performance
        market_data = fetch_market_data(pair, "1D", 2)
        if len(market_data) >= 2:
            prev_close, latest_close = market_data[1]["close"], market_data[0]["close"]
            percent_change = ((latest_close - prev_close) / prev_close) * 100
            summary_data.append({
                "pair": pair,
                "close": latest_close,
                "percent_change": percent_change,
                "rsi": market_data[0]["rsi"],
                "volume": market_data[0]["volume"]
            })

    if not summary_data:
        st.warning("No market summary data available.")
        return

    df = pd.DataFrame(summary_data)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🚀 Top 5 Gainers")
        st.dataframe(df.nlargest(5, "percent_change")[["pair", "percent_change"]], use_container_width=True)

    with col2:
        st.markdown("#### 🔻 Top 5 Losers")
        st.dataframe(df.nsmallest(5, "percent_change")[["pair", "percent_change"]], use_container_width=True)

    with col3:
        st.markdown("#### 🔥 Biggest Volume Movers")
        st.dataframe(df.nlargest(5, "volume")[["pair", "volume"]], use_container_width=True)

def show_filtered_data(data):
    """Displays the filtered market data in a table and auto-updates."""
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")
    date_range = st.session_state.get("selected_date_range", [])

    # ✅ Gracefully Handle Partial Date Selection
    start_timestamp, end_timestamp = None, None
    if date_range:
        if isinstance(date_range, list) and len(date_range) > 0:
            if len(date_range) == 1:  # ✅ Only one date selected
                start_timestamp = int(datetime.datetime.combine(date_range[0], datetime.time.min).timestamp() * 1000)
            else:  # ✅ Both start and end dates are selected
                start_timestamp = int(datetime.datetime.combine(date_range[0], datetime.time.min).timestamp() * 1000)
                end_timestamp = int(datetime.datetime.combine(date_range[1], datetime.time.max).timestamp() * 1000)

    # ✅ Apply Date Filtering (Only if Both Dates Exist)
    if start_timestamp and end_timestamp:
        data = [d for d in data if start_timestamp <= d['timestamp_utc'] <= end_timestamp]
    elif start_timestamp:  # ✅ If only start date is selected, show from that date onward
        data = [d for d in data if start_timestamp <= d['timestamp_utc']]
    elif end_timestamp:  # ✅ If only end date is selected, show up to that date
        data = [d for d in data if d['timestamp_utc'] <= end_timestamp]

    # ✅ Convert to DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        df["RSI Alert"] = df["rsi"].apply(lambda x: "🔥 Overbought" if x > 70 else ("💎 Oversold" if x < 30 else ""))
        df["MACD Signal"] = df.apply(lambda row: "📈 Bullish Crossover" if row["macd_line"] > row["macd_signal"] else "📉 Bearish Crossover", axis=1)

    st.dataframe(df, use_container_width=True)