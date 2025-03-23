"""
fetch_new_1h_candles.py
Fetches new 1-hour candles from OKX API and inserts only unseen rows into PostgreSQL.
"""

import os
import requests
import time
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()

OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
CANDLES_RATE_LIMIT = 40
BATCH_INTERVAL = 2

def get_known_timestamps(pair):
    query = "SELECT timestamp_utc FROM candles_1h WHERE pair = %s;"
    return set(row["timestamp_utc"] for row in fetch_data(query, (pair,)))

def fetch_active_pairs():
    response = requests.get("https://www.okx.com/api/v5/public/instruments?instType=SPOT")
    data = response.json()
    return [
        inst["instId"]
        for inst in data.get("data", [])
        if inst["quoteCcy"] == "USDT" and inst["state"] == "live"
    ]

def fetch_candles(pair, direction, ref_ts=None):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100
    }
    if direction == "before":
        print(f"📤 Fetching latest candles using BEFORE for {pair}")
    elif direction == "after" and ref_ts:
        params["after"] = str(int(ref_ts.timestamp() * 1000))
        print(f"📤 Fetching older candles using AFTER={ref_ts} for {pair}")
    else:
        raise ValueError("Invalid fetch direction")

    response = requests.get(OKX_CANDLES_URL, params=params)
    response.raise_for_status()
    return response.json().get("data", [])

def insert_candles(pair, candles, known_ts):
    query = """
    INSERT INTO public.candles_1h
    (pair, timestamp_utc, open_1h, high_1h, low_1h, close_1h,
     volume_1h, quote_volume_1h, taker_buy_base_1h)
    VALUES %s
    ON CONFLICT (pair, timestamp_utc) DO NOTHING;
    """
    rows = []
    for c in candles:
        try:
            utc_ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc) - timedelta(hours=8)
            if utc_ts in known_ts:
                continue  # skip already-known
            row = (
                pair, utc_ts, float(c[1]), float(c[2]), float(c[3]), float(c[4]),
                float(c[5]), float(c[6]), float(c[7])
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Skipping malformed row: {e} | Raw: {c}")

    if not rows:
        return None, 0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        execute_values(cursor, query, rows)
        conn.commit()
        print(f"✅ Inserted {len(rows)} new candles for {pair} | {rows[-1][1]} → {rows[0][1]}")
        return rows[-1][1], len(rows)
    except Exception as e:
        print(f"❌ Insert failed for {pair}: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    return None, 0

def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            print(f"⏳ Sleeping {BATCH_INTERVAL - elapsed:.2f}s to honor rate limit")
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

def main():
    print("🚀 Fetching latest 1H candles from OKX...\n")
    pairs = fetch_active_pairs()
    print(f"✅ {len(pairs)} pairs found\n")

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for pair in pairs:
        print(f"\n🔁 Processing {pair}")
        known_ts = get_known_timestamps(pair)

        # Initial call → latest candles (newest to oldest)
        candles = fetch_candles(pair, direction="before")
        if not candles:
            print(f"⛔ No candles returned for {pair}")
            continue

        after_ts = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
        inserted_ts, inserted = insert_candles(pair, candles, known_ts)

        total_inserted = inserted
        if inserted == 0:
            print(f"🛑 No new data for {pair}. Skipping pagination.")
            continue

        # Paginate backward using AFTER
        while True:
            candles = fetch_candles(pair, direction="after", ref_ts=after_ts)
            if not candles:
                break

            after_ts = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
            inserted_ts, inserted = insert_candles(pair, candles, known_ts)
            total_inserted += inserted

            if inserted == 0:
                print(f"🛑 Reached known data for {pair}")
                break

            request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(
                request_count[OKX_CANDLES_URL], start_time
            )

        print(f"📦 Finished {pair}: {total_inserted} candles inserted")

if __name__ == "__main__":
    main()
