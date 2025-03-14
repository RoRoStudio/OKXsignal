"""
dashboard/charts.py
📈 Shows market analysis charts in Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px

def show_charts():
    """Displays market data as interactive charts."""
    st.title("📈 Market Analysis")

    # ✅ Example Data (Replace with real API data)
    data = pd.DataFrame({
        "Time": pd.date_range(start="2025-03-01", periods=10, freq="D"),
        "BTC-USDT": [50000, 51000, 49500, 50500, 51500, 52000, 51000, 52500, 53000, 54000],
        "ETH-USDT": [3500, 3550, 3400, 3450, 3600, 3700, 3650, 3750, 3800, 3900],
    })

    # ✅ Plot BTC price
    fig = px.line(data, x="Time", y="BTC-USDT", title="Bitcoin (BTC) Price")
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Plot ETH price
    fig = px.line(data, x="Time", y="ETH-USDT", title="Ethereum (ETH) Price")
    st.plotly_chart(fig, use_container_width=True)
