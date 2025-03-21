"""
fetch_new_1h_candles.py
Fetches new 1-hour candles from the OKX API and inserts them into PostgreSQL.
"""

import requests
import time
from config.config_loader import load_config
from database.processing.compute_indicators import process_new_candles

# ✅ Load configuration settings
config = load_config()

# ✅ OKX API Endpoint
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"

# ✅ Rate Limit Settings (40 requests per 2 seconds)
CANDLES_RATE_LIMIT = 40
BATCH_INTERVAL = 2


def fetch_active_pairs():
    """Fetch active USDT trading pairs from OKX API."""
    OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
    
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    
    if "data" in data:
        return [
            inst["instId"]
            for inst in data["data"]
            if inst["quoteCcy"] == "USDT" and inst["state"] == "live"
        ]
    return []


def fetch_latest_timestamp(pair):
    """Fetch the latest timestamp for a given pair from PostgreSQL."""
    query = "SELECT MAX(timestamp_ms) FROM public.candles_1h WHERE pair = %s;"
    result = fetch_data(query, (pair,))
    return result[0]["max"] if result and result[0]["max"] else None


def fetch_candles(pair, after_timestamp_ms=None):
    """Fetch new 1-hour candles from OKX using `before` for pagination."""
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,  # Fetch max 100 candles per request
    }
    if after_timestamp_ms:
        params["before"] = str(after_timestamp_ms)

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []


def insert_candles(pair, candles):
    """Insert new 1H candles into PostgreSQL, ensuring data matches schema."""
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
        execute_batch(query, rows)  # ✅ Fix: Use batch insert function
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
    """Main function to fetch and insert new 1-hour candles for all active pairs."""
    print("🚀 Fetching new 1H candles from OKX...")

    pairs = fetch_active_pairs()
    if not pairs:
        print("⚠️ No active pairs found. Exiting.")
        return

    total_inserted = 0
    failed_pairs = []

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        try:
            latest_timestamp_ms = fetch_latest_timestamp(pair)

            candles = fetch_candles(pair, after_timestamp_ms=latest_timestamp_ms)
            inserted = insert_candles(pair, candles)

            total_inserted += inserted

            request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(
                request_count[OKX_CANDLES_URL], start_time
            )

            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Inserted: {total_inserted}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Inserted={total_inserted}, Failed={len(failed_pairs)}")


if __name__ == "__main__":
    main()

print("🚀 Triggering indicator computation after fetching new candles...")
process_new_candles()