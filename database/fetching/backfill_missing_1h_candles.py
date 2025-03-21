"""
backfill_missing_1h_candles.py
Finds missing 1-hour candles and backfills them using OKX API, storing data in PostgreSQL.
"""

import requests
import time
from database.db import fetch_data, execute_batch
from config.config_loader import load_config

# ✅ Load configuration settings
config = load_config()

# ✅ OKX API Endpoint
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit Settings (20 requests per 2 seconds)
CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2


def fetch_missing_pairs():
    """Fetch pairs that exist in 1D but are missing from 1H using PostgreSQL."""
    query = """
    SELECT DISTINCT pair FROM public.candles_1h 
    WHERE pair NOT IN (SELECT DISTINCT pair FROM public.candles_1h);
    """
    return [row["pair"] for row in fetch_data(query)]


def fetch_latest_timestamp(pair):
    """Fetch the most recent timestamp stored for a given pair in PostgreSQL."""
    query = "SELECT MAX(timestamp_ms) FROM public.candles_1h WHERE pair = %s;"
    result = fetch_data(query, (pair,))
    return result[0]["max"] if result and result[0]["max"] else None


def fetch_candles(pair, after_timestamp=None):
    """Fetch 1H historical candles from OKX using `before` for pagination."""
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
    }
    if after_timestamp:
        params["before"] = str(after_timestamp)  # ✅ Use `before` for correct pagination

    print(f"📡 Sending request to OKX: {params}")  # Debugging output

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON for {pair}: {e}")
        return None


def insert_candles(pair, candles):
    """Insert fetched candles into PostgreSQL."""
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
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def main():
    """Find missing 1H candles and backfill them using OKX API."""
    print("🚀 Script started: Backfilling missing 1H candles...")

    missing_pairs = fetch_missing_pairs()
    if not missing_pairs:
        print("✅ No missing pairs found. Exiting.")
        return

    print(f"🚀 Backfilling {len(missing_pairs)} missing 1H candles...")

    total_fixed = 0
    failed_pairs = []
    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(missing_pairs, start=1):
        try:
            print(f"🔍 Fetching {pair} missing candles...")

            # ✅ Start from the latest known candle in PostgreSQL
            latest_timestamp = fetch_latest_timestamp(pair)

            # ✅ If no data, start from latest OKX candle
            if latest_timestamp is None:
                first_candle = fetch_candles(pair, after_timestamp=None)  # Get latest candle
                if not first_candle:
                    print(f"⚠️ {pair}: No candles found in OKX.")
                    continue
                latest_timestamp = int(first_candle[0][0])

            # ✅ Backfill candles using `before=<timestamp>`
            while True:
                candles = fetch_candles(pair, after_timestamp=latest_timestamp)

                if not candles:
                    print(f"⏳ No more missing candles found for {pair}, stopping.")
                    break

                inserted = insert_candles(pair, candles)
                total_fixed += inserted

                # ✅ Use the earliest candle timestamp instead of the latest
                latest_timestamp = int(candles[-1][0])  # ✅ Use oldest timestamp in batch for proper backfilling

                print(f"📌 {pair} → Inserted {inserted} missing candles. Now fetching before {latest_timestamp}...")

                request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_CANDLES_URL], start_time)

            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(missing_pairs)} | Fixed: {total_fixed}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(missing_pairs)}, Fixed={total_fixed}, Failed={len(failed_pairs)}")


if __name__ == "__main__":
    main()
