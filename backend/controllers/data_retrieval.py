"""
data_retrieval.py
Handles supabase table retrieval for the 1H and 1D candles.

Requires:
  pip install supabase-py

Assumes you have the following environment variables or direct config:
    SUPABASE_URL
    SUPABASE_ANON_KEY
"""

import os
import pandas as pd
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "<your-supabase-url>")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "<your-anon-key>")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("Supabase credentials missing. Set SUPABASE_URL and SUPABASE_ANON_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def get_candles_1H(pair: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch candle data from the 'candles_1H' table for a given pair.
    :param pair: e.g. "BTC-USDT"
    :param limit: how many recent rows to fetch
    :return: pandas DataFrame with columns:
        [pair, timestamp_ms, open, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote]
    """
    response = supabase.table("candles_1H") \
        .select("*") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    data = response.data  # list of dict
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values("timestamp_ms", ascending=True).reset_index(drop=True)
    return df

def get_candles_1D(pair: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch candle data from the 'candles_1D' table for a given pair.
    :param pair: e.g. "BTC-USDT"
    :param limit: how many recent rows to fetch
    :return: pandas DataFrame with columns:
        [pair, timestamp_ms, open, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote]
    """
    response = supabase.table("candles_1D") \
        .select("*") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    data = response.data
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values("timestamp_ms", ascending=True).reset_index(drop=True)
    return df


def get_recent_candles(pair: str, timeframe: str = "1H", limit: int = 1000) -> pd.DataFrame:
    """
    Simple wrapper to fetch either 1H or 1D from a single function.
    """
    if timeframe == "1H":
        return get_candles_1H(pair, limit)
    elif timeframe == "1D":
        return get_candles_1D(pair, limit)
    else:
        raise ValueError("Unsupported timeframe. Use '1H' or '1D'.")

# Optional: You can add more specialized queries for date ranges, e.g. timestamp_ms >= ...
# For example:

def get_candles_by_range(pair: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetch candles within a specific timestamp range (inclusive).
    :param start_ts: earliest timestamp in ms
    :param end_ts: latest timestamp in ms
    """
    table_name = "candles_1H" if timeframe == "1H" else "candles_1D"
    response = supabase.table(table_name) \
        .select("*") \
        .eq("pair", pair) \
        .gte("timestamp_ms", start_ts) \
        .lte("timestamp_ms", end_ts) \
        .order("timestamp_ms", desc=False).execute()

    data = response.data
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Simple test
    df_1h = get_candles_1H("BTC-USDT", limit=5)
    print("=== Last 5 of 1H ===")
    print(df_1h)

    df_1d = get_candles_1D("BTC-USDT", limit=5)
    print("=== Last 5 of 1D ===")
    print(df_1d)
