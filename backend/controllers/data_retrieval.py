"""
data_retrieval.py
Handles fetching price and indicator data from Supabase for the 1H and 1D candles.
"""

import os
import pandas as pd
from supabase import create_client, Client

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("❌ Supabase credentials are missing.")

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
        .select("pair", "timestamp_ms", "open", "high", "low", "close", "volume",
                "rsi", "macd_line", "macd_signal", "macd_hist",
                "bollinger_middle", "bollinger_upper", "bollinger_lower",
                "atr", "stoch_rsi_k", "stoch_rsi_d") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    data = response.data
    return pd.DataFrame(data) if data else pd.DataFrame()

if __name__ == "__main__":
    df = fetch_market_data("BTC-USDT", "1H", 5)
    print(df)
