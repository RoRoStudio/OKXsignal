"""
mini_chart.py
📈 Renders a mini-chart for quick market visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import pytz
from dashboard.utils.data_loader import fetch_market_data

def show_mini_chart():
    """Displays a quick mini-chart for the selected pair & timeframe."""
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")

    st.markdown(f"### 📈 {pair} - {timeframe} Mini Chart")

    data = fetch_market_data(pair=pair, timeframe=timeframe, limit=200)
    df = pd.DataFrame(data)

    if df.empty:
        st.warning("No data available for charting.")
        return

    # ✅ Convert timestamps to local time
    df["timestamp_ms"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(pytz.timezone("Europe/Amsterdam"))

    # ✅ Use primary color for the line
    fig = px.line(df, x="timestamp_ms", y="close", title=f"{pair} - {timeframe} Trend",
                  line_shape="linear", markers=True, color_discrete_sequence=["#03fcb6"])

    st.plotly_chart(fig, use_container_width=True)
