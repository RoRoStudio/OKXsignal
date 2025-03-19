"""
data_loader.py
Efficiently loads candle & indicator data from Supabase, reducing redundant queries.
"""

import pandas as pd
import streamlit as st
from supabase import create_client, Client
import os

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("❌ Supabase credentials missing. Check environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def fetch_market_data(pair: str, timeframe: str = "1H", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch recent market data (candlesticks + indicators) from Supabase.
    
    :param pair: Trading pair, e.g., "BTC-USDT".
    :param timeframe: "1H" or "1D".
    :param limit: Number of most recent rows to fetch.
    :return: Pandas DataFrame containing market data with indicators.
    """
    table_name = f"candles_{timeframe}"

    response = supabase.table(table_name) \
        .select("*") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    return pd.DataFrame(response.data) if response.data else pd.DataFrame()