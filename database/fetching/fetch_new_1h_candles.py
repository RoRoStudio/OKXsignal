"""
fetch_new_1h_candles.py
Fetches all historical 1-hour candles from OKX API and inserts them into PostgreSQL.
"""
import logging
import os
import requests
import time
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()

# Runtime logging
start_time = datetime.now()
os.makedirs("logs", exist_ok=True)  # Create logs folder if missing
runtime_log_path = os.path.join("logs", "runtime_fetch.log")

def log_duration():
    duration = datetime.now() - start_time
    with open(runtime_log_path, "a") as f:
        f.write(f"[{datetime.now()}] fetch_new_1h_candles.py completed in {duration.total_seconds():.2f} seconds\n")
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
CANDLES_RATE_LIMIT = 40
BATCH_INTERVAL = 2

def fetch_active_pairs():
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

def fetch_candles(pair, before_timestamp_utc=None):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100
    }
    if before_timestamp_utc:
        params["after"] = str(int(before_timestamp_utc.timestamp() * 1000))

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error fetching {pair}: {e}")
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
            # OKX timestamps are in ms, HK-time-aligned → subtract 8h to convert to UTC
            utc_ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc) - timedelta(hours=8)
            row = (
                pair,
                utc_ts,
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
                float(c[6]),
                float(c[7]),
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Malformed row for {pair}: {e} | Raw: {c}")

    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} candles for {pair} | {rows[-1][1]} → {rows[0][1]}")
            return rows[-1][1]  # Return earliest timestamp in this batch
        except Exception as e:
            print(f"❌ Insert failed for {pair}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    return None

def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            print(f"⏳ Sleeping for {BATCH_INTERVAL - elapsed:.2f}s (rate limit)")
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

def main():
    print("🚀 Backfilling full 1H candle history from OKX...")

    pairs = fetch_active_pairs()
    print(f"✅ {len(pairs)} active USDT spot pairs found")

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        print(f"\n🔁 Processing {pair}")
        before = None
        total_rows = 0
        earliest = None

        while True:
            candles = fetch_candles(pair, before)
            if not candles:
                print(f"⛔ No more candles for {pair}. Total inserted: {total_rows}")
                break

            before = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
            inserted_timestamp = insert_candles(pair, candles)
            if not inserted_timestamp:
                break

            earliest = inserted_timestamp
            total_rows += len(candles)

            request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(
                request_count[OKX_CANDLES_URL], start_time
            )

        print(f"📦 Finished {pair}: {total_rows} candles (oldest = {earliest})")

if __name__ == "__main__":
    main()
    log_duration()