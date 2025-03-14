"""
fetch_market.py
Fetches historical market data (candlesticks) for all USDT spot pairs on OKX.
Designed to be triggered by Streamlit.
"""

import asyncio
import time
import random
import pandas as pd
import streamlit as st
from tqdm.asyncio import tqdm  # Async progress bar
from okx_api.rest_client import OKXRestClient

CANDLE_LIMIT = 100  # OKX API max per request
TOTAL_CANDLES = 1000  # Number of daily candles we want
TIMEFRAME = "1D"  # 1-day timeframe
MAX_CONCURRENT_REQUESTS = 20  # ✅ Rate-limiting

async def fetch_usdt_pairs():
    """Fetches active USDT spot trading pairs from OKX."""
    client = OKXRestClient()
    response = client.get_instruments(instType="SPOT")

    raw_pairs = [x["instId"] for x in response["data"] if x["instId"].endswith("-USDT") and x["state"] == "live"]

    valid_pairs = []
    batch_size = 10
    for i in range(0, len(raw_pairs), batch_size):
        batch = raw_pairs[i : i + batch_size]

        for pair in batch:
            test_response = client.get_ticker(pair)
            if "code" in test_response and test_response["code"] == "51001":
                st.warning(f"🛑 Skipping non-existent pair: {pair}")
            else:
                valid_pairs.append(pair)

        time.sleep(1)  # ✅ Add delay to avoid rate-limiting

    st.success(f"✅ Found {len(valid_pairs)} valid USDT trading pairs.")
    return valid_pairs

async def fetch_candles(pair, semaphore, progress_bar):
    """Fetches historical candlestick data for a trading pair."""
    client = OKXRestClient()
    all_data = []

    async with semaphore:
        while len(all_data) < TOTAL_CANDLES:
            response = client.get_candlesticks(instId=pair, bar=TIMEFRAME, limit=CANDLE_LIMIT)

            if "code" in response and response["code"] == "50011":
                st.warning(f"⚠️ Rate limited. Retrying {pair}...")
                await asyncio.sleep(random.uniform(3, 5))  
                continue  

            if "data" not in response or len(response["data"]) == 0:
                return pair, pd.DataFrame()

            all_data.extend(response["data"])
            await asyncio.sleep(random.uniform(0.2, 0.5))  # ✅ Avoid hammering API

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")

    progress_bar.update(1)
    return pair, df[:TOTAL_CANDLES]

async def fetch_all_pairs():
    """Fetches market data for all USDT trading pairs."""
    pairs = await fetch_usdt_pairs()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    progress_bar = st.progress(0)
    total_pairs = len(pairs)

    async def limited_fetch(pair):
        return await fetch_candles(pair, semaphore, progress_bar)

    tasks = [limited_fetch(pair) for pair in pairs]
    results = await asyncio.gather(*tasks)

    progress_bar.empty()
    return {pair: df for pair, df in results}

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(fetch_all_pairs())

