"""
overview.py
📊 Market Overview Page: Displays the latest 1H & 1D candles with filtering options.
"""

import streamlit as st
import datetime
import pandas as pd
import plotly.express as px
import pytz
from dashboard.components.forms.filter_form import filter_form
from dashboard.utils.data_loader import fetch_market_data, fetch_trading_pairs
from dashboard.components.charts.mini_chart import show_mini_chart  # ✅ Separate component

# ✅ Auto-refresh every ~6 minutes past the hour
AUTO_REFRESH_INTERVAL = 360  # 6 minutes in seconds

def show_page():
    st.title("📈 Market Overview")

    # ✅ Step 1: Show Filter Form Instantly
    filter_form()

    # ✅ Step 2: Fetch Data (Cached for Speed)
    trading_pairs = fetch_trading_pairs()  # Fetches once per hour
    selected_data = fetch_selected_market_data()  # Fetches selected pair data

    # ✅ Step 3: Show Market Summary (Gainers, Losers, Volume Movers)
    show_market_summary(trading_pairs)

    # ✅ Step 4: Show Selected Pair Data
    show_filtered_data(selected_data)

    # ✅ Step 5: Show Mini Chart (Now in `mini_chart.py`)
    show_mini_chart()


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

    # 🔥 Top 5 Gainers & Losers
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
    """Displays the filtered market data in a table."""
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")
    date_range = st.session_state.get("selected_date_range", [])

    if date_range:
        start_date, end_date = date_range
        start_timestamp = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp() * 1000)
        end_timestamp = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp() * 1000)
        data = [d for d in data if start_timestamp <= d['timestamp_ms'] <= end_timestamp]

    # 🏷️ Highlight Extreme Indicators (RSI, MACD)
    df = pd.DataFrame(data)
    if not df.empty:
        df["RSI Alert"] = df["rsi"].apply(lambda x: "🔥 Overbought" if x > 70 else ("💎 Oversold" if x < 30 else ""))
        df["MACD Signal"] = df.apply(lambda row: "📈 Bullish Crossover" if row["macd_line"] > row["macd_signal"] else "📉 Bearish Crossover", axis=1)

    st.dataframe(df, use_container_width=True)
