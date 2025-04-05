"""
master_chart.py
📊 The all-in-one Master Chart for OKXsignal, dynamically adjusting zoom & range.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pytz
from dashboard.utils.data_loader import fetch_market_data

# 🎨 Colors
PRIMARY_COLOR = "#03fcb6"  # ✅ Green for bullish candles
BEARISH_COLOR = "#e74c3c"  # 🔻 Red for bearish candles
VOLUME_COLOR = "#3498db"   # 🔵 Blue for volume bars

# ✅ Default indicator settings
DEFAULT_INDICATORS = {
    "Bollinger Bands": True,
    "MACD": True,
    "RSI": True,
    "ATR": True,
}

def show_master_chart(data):
    """Displays the Master Chart with price, volume, and key indicators."""

    # ✅ Fetch updated market data when filters change
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")
    date_range = st.session_state.get("selected_date_range", [])

    # ✅ Fetch new market data based on the selected filters
    data = fetch_market_data(pair=pair, timeframe=timeframe, limit=100)

    if not data:
        st.warning("No data available for this pair and timeframe.")
        return

    # ✅ Convert list of dicts to DataFrame
    df = pd.DataFrame(data)

    # ✅ Convert timestamps to local time
    local_tz = pytz.timezone("Europe/Amsterdam")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(local_tz)

    # ✅ Apply Date Filtering (Handles partial selections)
    if date_range and isinstance(date_range, list):
        if len(date_range) == 1:  # ✅ Only start date selected
            start_timestamp = int(pd.Timestamp(date_range[0]).timestamp() * 1000)
            df = df[df["timestamp_utc"] >= start_timestamp]
        elif len(date_range) == 2:  # ✅ Both start & end date selected
            start_timestamp = int(pd.Timestamp(date_range[0]).timestamp() * 1000)
            end_timestamp = int(pd.Timestamp(date_range[1]).timestamp() * 1000)
            df = df[(df["timestamp_utc"] >= start_timestamp) & (df["timestamp_utc"] <= end_timestamp)]

    # ✅ Auto-Adjust Zoom: Get min & max price for Y-axis
    min_price = df["low"].min() * 0.99
    max_price = df["high"].max() * 1.01

    # ✅ User selection for indicators
    with st.expander("⚙️ Chart Settings", expanded=True):
        st.markdown("**Toggle Indicators**")
        indicators = {key: st.checkbox(key, DEFAULT_INDICATORS[key]) for key in DEFAULT_INDICATORS}

    # ✅ Initialize Plotly figure (Double Height)
    fig = go.Figure()

    # 📈 **Candlestick Chart**
    fig.add_trace(go.Candlestick(
        x=df["timestamp_utc"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing=dict(line=dict(color=PRIMARY_COLOR), fillcolor=PRIMARY_COLOR),
        decreasing=dict(line=dict(color=BEARISH_COLOR), fillcolor=BEARISH_COLOR),
        name="Candles"
    ))

    # 📊 **Volume Bars**
    fig.add_trace(go.Bar(
        x=df["timestamp_utc"],
        y=df["volume"],
        marker_color=VOLUME_COLOR,
        opacity=0.5,
        name="Volume"
    ))

    # 🎚️ **Bollinger Bands**
    if indicators["Bollinger Bands"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["bollinger_upper"], line=dict(color="gray", width=1, dash="dot"), name="Bollinger Upper"))
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["bollinger_middle"], line=dict(color="gray", width=1, dash="dash"), name="Bollinger Middle"))
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["bollinger_lower"], line=dict(color="gray", width=1, dash="dot"), name="Bollinger Lower"))

    # 📉 **MACD**
    if indicators["MACD"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["macd_line"], line=dict(color="blue", width=1), name="MACD Line"))
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["macd_signal"], line=dict(color="orange", width=1, dash="dot"), name="MACD Signal"))

    # 📊 **RSI**
    if indicators["RSI"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["rsi"], line=dict(color="purple", width=1), name="RSI"))

    # 📊 **ATR (Volatility Indicator)**
    if indicators["ATR"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["atr"], line=dict(color="red", width=1), name="ATR (Volatility)"))

    # ✅ Chart Settings (Auto-Adjust Zoom & Secondary Y-Axis for Volume)
    fig.update_layout(
        height=900,
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        yaxis=dict(range=[min_price, max_price]),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1),
        
        # ✅ Volume now has a separate Y-axis
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)
