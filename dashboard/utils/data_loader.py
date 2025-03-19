"""
data_loader.py
🔹 Fetches trading pairs & candle data from Supabase with caching & fixes.
"""

import streamlit as st
from supabase import create_client, Client
import os
import datetime
from dotenv import load_dotenv
from config.config_loader import load_config

# ✅ Load environment variables from credentials.env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "credentials.env"))
load_dotenv(env_path)

# ✅ Load config (ensures SUPABASE_URL comes from config.ini)
config = load_config()

SUPABASE_URL = config.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")  # Loaded from environment

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("❌ Supabase credentials missing. Check config.ini and credentials.env.")

# ✅ Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

@st.cache_data(ttl=3600)
def fetch_trading_pairs():
    """
    Fetches all unique trading pairs from the `distinct_trading_pairs` materialized view.
    Ensures all pairs are retrieved and sorted alphabetically.
    """
    response = supabase.table("distinct_trading_pairs").select("pair").execute()

    if not response.data:
        return ["BTC-USDT"]  # Default fallback if query fails

    pairs = sorted([item["pair"] for item in response.data])
    return pairs


@st.cache_data(ttl=3600)  # Cache market data for 1 hour
def fetch_market_data(pair: str, timeframe: str = "1H", limit: int = 1000):
    """
    Fetches recent market data (candles + indicators) from Supabase.
    
    - ⏳ Caches results for 1 hour (to avoid excessive queries).
    - 📅 Fetches 1H data every hour, 1D data once per day.

    :param pair: Trading pair, e.g., "BTC-USDT".
    :param timeframe: "1H" or "1D".
    :param limit: Number of most recent rows to fetch.
    :return: List of market data rows.
    """
    table_name = f"candles_{timeframe}"

    # Query Supabase
    response = supabase.table(table_name) \
        .select("pair", "timestamp_ms", "open", "high", "low", "close", "volume",
                "rsi", "macd_line", "macd_signal", "macd_hist",
                "bollinger_middle", "bollinger_upper", "bollinger_lower",
                "atr", "stoch_rsi_k", "stoch_rsi_d") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    return response.data if response.data else []

# ✅ Auto-fetch on dashboard start
def auto_fetch_data():
    """Fetches 1H data every hour & 1D data once per day."""
    now = datetime.datetime.utcnow()
    last_fetch_time = st.session_state.get("last_fetch_time", None)

    if not last_fetch_time or (now - last_fetch_time).seconds >= 3600:
        st.session_state["last_fetch_time"] = now
        st.session_state["1H_data"] = fetch_market_data("BTC-USDT", "1H", 1000)
        st.session_state["1D_data"] = fetch_market_data("BTC-USDT", "1D", 365)  # Fetch last 365 days
        print("✅ Market data updated")

# ✅ Auto-run on Streamlit start
auto_fetch_data()
