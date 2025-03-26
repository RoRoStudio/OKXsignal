"""
backfill_missing_1h_candles.py
Finds and fills missing 1-hour candles in PostgreSQL using OKX API.
"""

import requests
import time
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2


def fetch_all_pairs():
    query = "SELECT DISTINCT pair FROM public.candles_1h;"
    return [row["pair"] for row in fetch_data(query)]


def fetch_timestamps(pair):
    query = """
    SELECT timestamp_utc FROM public.candles_1h 
    WHERE pair = %s ORDER BY timestamp_utc ASC;
    """
    return [row["timestamp_utc"] for row in fetch_data(query, (pair,))]


def find_gaps(timestamps):
    gaps = []
    expected_delta = timedelta(hours=1)
    for i in range(1, len(timestamps)):
        current = timestamps[i]
        prev = timestamps[i - 1]
        delta = current - prev
        if delta > expected_delta:
            missing_start = prev + expected_delta
            while missing_start < current:
                gaps.append(missing_start)
                missing_start += expected_delta
    return gaps


def fetch_candles(pair, after_utc):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
        "after": str(int(after_utc.timestamp() * 1000))
    }

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error fetching {pair} after {after_utc}: {e}")
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
            print(f"⚠️ Malformed candle for {pair}: {e} | Raw: {c}")

    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} gap candles for {pair} | {rows[0][1]} → {rows[-1][1]}")
            return len(rows)
        except Exception as e:
            print(f"❌ Insert failed for {pair}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    return 0


def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def main():
    print("🚀 Scanning for gaps in 1H candle history...")

    pairs = fetch_all_pairs()
    print(f"✅ Found {len(pairs)} pairs with existing data")

    total_inserted = 0
    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        try:
            print(f"\n🔁 Checking {pair}")
            timestamps = fetch_timestamps(pair)
            if len(timestamps) < 2:
                print(f"⚠️ Not enough data to find gaps for {pair}")
                continue

            gaps = find_gaps(timestamps)
            print(f"🧩 Found {len(gaps)} missing 1H timestamps for {pair}")

            for gap_start in gaps:
                candles = fetch_candles(pair, gap_start)
                inserted = insert_candles(pair, candles)
                total_inserted += inserted

                request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(
                    request_count[OKX_CANDLES_URL], start_time
                )

            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Total inserted: {total_inserted}")

        except Exception as e:
            print(f"❌ Failed to process {pair}: {e}")

    print(f"\n✅ Backfill complete: Inserted {total_inserted} missing candles across {len(pairs)} pairs")


if __name__ == "__main__":
    main()
