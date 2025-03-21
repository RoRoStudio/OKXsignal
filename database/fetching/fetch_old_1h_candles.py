"""
fetch_old_1h_candles.py
Finds and fetches older 1-hour candles from OKX API and stores them in PostgreSQL.
"""

import requests
import time
from database.db import fetch_data, execute_batch
from config.config_loader import load_config

# ✅ Load configuration settings
config = load_config()

# ✅ OKX API Endpoints
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit Settings (20 requests per 2 seconds)
HISTORY_CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2


def fetch_active_pairs():
    """Fetch active USDT trading pairs from OKX API."""
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()

    if "data" in data:
        return [
            inst["instId"]
            for inst in data["data"]
            if inst["quoteCcy"] == "USDT" and inst["state"] == "live"
        ]
    return []


def fetch_oldest_timestamp(pair):
    """Fetch the oldest available timestamp for a given pair in PostgreSQL."""
    query = "SELECT MIN(timestamp_ms) FROM public.candles_1h WHERE pair = %s;"
    result = fetch_data(query, (pair,))
    return result[0]["min"] if result and result[0]["min"] else None


def fetch_candles(pair, after_timestamp_ms):
    """Fetch older 1H candles from OKX using `after` for pagination."""
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
        "after": str(int(after_timestamp_ms))
    }

    print(f"🔍 Fetching {pair} older candles from {after_timestamp_ms}...")

    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)

    try:
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []


def insert_candles(pair, candles):
    """Insert older fetched 1H candles into PostgreSQL."""
    query = """
    INSERT INTO public.candles_1h 
    (pair, timestamp_ms, open_1h, high_1h, low_1h, close_1h, volume_1h, quote_volume_1h, taker_buy_base_1h)
    VALUES %s
    ON CONFLICT (pair, timestamp_ms) DO NOTHING;
    """

    rows = [
        (
            pair,
            int(c[0]),  # timestamp_ms
            float(c[1]),  # open
            float(c[2]),  # high
            float(c[3]),  # low
            float(c[4]),  # close
            float(c[5]),  # volume
            float(c[6]),  # quote volume
            float(c[7]),  # taker buy base volume
        )
        for c in candles
    ]

    if rows:
        execute_batch(query, rows)  # ✅ Efficient batch insert
        return len(rows)
    return 0


def enforce_rate_limit(request_count, start_time):
    """Ensure API rate limits are respected."""
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def main():
    """Find and fetch older 1H candles using OKX API."""
    print("🚀 Script started: Fetching older 1H candles...")

    pairs = fetch_active_pairs()
    if not pairs:
        print("✅ No active pairs found. Exiting.")
        return

    print(f"🚀 Fetching historical 1H candles for {len(pairs)} pairs...")

    total_fixed = 0
    failed_pairs = []
    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        try:
            oldest_timestamp_ms = fetch_oldest_timestamp(pair)
            if oldest_timestamp_ms is None:
                print(f"⚠️ No timestamp found for {pair}, skipping...")
                continue

            print(f"⏳ {pair} → Fetching candles older than {oldest_timestamp_ms}")

            # ✅ Start from the oldest available timestamp and move forward
            while True:
                candles = fetch_candles(pair, after_timestamp_ms=oldest_timestamp_ms)

                if not candles:
                    print(f"⏳ No more older candles found for {pair}, stopping.")
                    break

                inserted = insert_candles(pair, candles)
                total_fixed += inserted
                oldest_timestamp_ms = int(candles[-1][0])  # ✅ Move forward in time

                print(f"📌 {pair} → Inserted {inserted} older candles.")

                request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_HISTORY_CANDLES_URL], start_time)

            # ✅ Log progress every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Fixed: {total_fixed}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Fixed={total_fixed}, Failed={len(failed_pairs)}")


if __name__ == "__main__":
    main()
