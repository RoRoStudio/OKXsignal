"""
fetch_old_1h_candles.py
Finds and fetches older 1-hour candles from OKX API and stores them in PostgreSQL.
"""

import requests
import time
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

# ✅ Load configuration settings
config = load_config()

# ✅ OKX API Endpoints
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit Settings
HISTORY_CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

def fetch_active_pairs():
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
    query = "SELECT MIN(timestamp_utc) FROM public.candles_1h WHERE pair = %s;"
    result = fetch_data(query, (pair,))
    return result[0]["min"] if result and result[0]["min"] else None

def fetch_candles(pair, after_timestamp_utc):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
        "after": str(int(after_timestamp_utc.timestamp() * 1000))
    }

    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []

def insert_candles(pair, candles):
    query = """
    INSERT INTO public.candles_1h 
    (pair, timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h, quote_volume_1h, taker_buy_base_1h)
    VALUES %s
    ON CONFLICT (pair, timestamp_utc) DO NOTHING;
    """

    rows = []
    for c in candles:
        try:
            utc_ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc) - timedelta(hours=8)  # HK → UTC
            row = (
                pair,
                utc_ts,
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
                float(c[6]),
                float(c[7])
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Malformed candle for {pair}: {e} | Raw: {c}")

    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} historical candles for {pair} | {rows[0][1]} → {rows[-1][1]}")
            return rows[-1][1]  # Return latest timestamp in batch
        except Exception as e:
            print(f"❌ Insert failed for {pair}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    return None

def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

def main():
    print("🚀 Fetching older 1H candles from OKX...")

    pairs = fetch_active_pairs()
    print(f"✅ {len(pairs)} active USDT spot pairs found")

    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        print(f"\n🔁 Processing {pair}")
        after = fetch_oldest_timestamp(pair)
        if after is None:
            print(f"⚠️ No local data for {pair}, skipping.")
            continue

        total = 0

        while True:
            candles = fetch_candles(pair, after_timestamp_utc=after)
            if not candles:
                print(f"⛔ No more older candles for {pair}")
                break

            after = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
            inserted_timestamp = insert_candles(pair, candles)
            if not inserted_timestamp:
                break

            total += len(candles)

            request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(
                request_count[OKX_HISTORY_CANDLES_URL], start_time
            )

        print(f"📦 Finished {pair}: Inserted {total} older candles")

if __name__ == "__main__":
    main()
