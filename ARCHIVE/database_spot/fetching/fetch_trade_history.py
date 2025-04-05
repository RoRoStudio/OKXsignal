# GET /api/v5/market/history-trades

#Purpose: Returns individual trade-level data (price, size, direction) for last 3 months
#Backfill range: ❗Only last 3 months
#Pagination: via tradeId (type=1) or ts (type=2)

#Use:

#Train the slippage model

#Label high-slippage conditions per candle (e.g. price impact)

#Detect microstructure patterns pre-trade

"""
fetch_trade_history.py
Fetches recent trade-level data (last 3 months) from OKX API for all USDT spot pairs.
Inserts new records into `slippage_training_data`, avoiding duplicates by tradeId.
"""

import requests
import time
from datetime import datetime
from config.config_loader import load_config
from database_spot.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()
CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2
HISTORY_TRADES_URL = "https://www.okx.com/api/v5/market/history-trades"
INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"


def fetch_active_usdt_pairs():
    response = requests.get(INSTRUMENTS_URL)
    data = response.json()
    if "data" not in data:
        return []
    return [
        inst["instId"]
        for inst in data["data"]
        if inst["quoteCcy"] == "USDT" and inst["state"] == "live"
    ]


def fetch_existing_trade_ids(pair):
    query = "SELECT trade_id FROM slippage_training_data WHERE pair = %s;"
    result = fetch_data(query, (pair,))
    return {row["trade_id"] for row in result}


def fetch_trades(pair, before_trade_id=None):
    params = {
        "instId": pair,
        "limit": 100,
    }
    if before_trade_id:
        params["after"] = before_trade_id

    response = requests.get(HISTORY_TRADES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error fetching trades for {pair}: {e}")
        return []


def insert_trades(pair, trades, existing_ids):
    query = """
    INSERT INTO raw_trades
    (pair, trade_id, price, quantity, side, timestamp_utc)
    VALUES %s
    ON CONFLICT DO NOTHING;
    """
    rows = []
    for t in trades:
        try:
            if t["tradeId"] in existing_ids:
                continue
            row = (
                pair,
                t["tradeId"],
                float(t["px"]),
                float(t["sz"]),
                t["side"],
                datetime.utcfromtimestamp(int(t["ts"]) / 1000),
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Skipped malformed trade: {t} | {e}")

    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} trades for {pair}")
        except Exception as e:
            print(f"❌ Insert failed for {pair}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()


def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def main():
    print("🚀 Fetching recent trade-level data for slippage training...")
    pairs = fetch_active_usdt_pairs()
    print(f"✅ {len(pairs)} USDT pairs found.")

    request_count = {HISTORY_TRADES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        print(f"\n🔁 {index}/{len(pairs)} | Fetching trades for {pair}")
        existing_ids = fetch_existing_trade_ids(pair)
        print(f"🧠 Existing trade IDs: {len(existing_ids)}")

        before = None
        total_inserted = 0

        while True:
            trades = fetch_trades(pair, before_trade_id=before)

            if not trades:
                break

            insert_trades(pair, trades, existing_ids)
            total_inserted += len(trades)

            before = trades[-1]["tradeId"]
            request_count[HISTORY_TRADES_URL], start_time = enforce_rate_limit(
                request_count[HISTORY_TRADES_URL], start_time
            )

            if total_inserted >= 1000:  # Limit daily run
                print(f"🛑 Reached daily fetch limit for {pair}")
                break

    print("✅ Done fetching slippage trades!")


if __name__ == "__main__":
    main()
