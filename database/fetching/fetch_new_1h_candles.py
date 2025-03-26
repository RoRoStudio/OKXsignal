import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()

OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
CANDLES_RATE_LIMIT = 40
BATCH_INTERVAL = 2

# Configure logging
logging.basicConfig(level=config['LOG_LEVEL'].upper(), format='%(asctime)s - %(levelname)s - %(message)s')

async def get_known_timestamps(pair):
    query = "SELECT timestamp_utc FROM candles_1h WHERE pair = %s;"
    return {row["timestamp_utc"] for row in await fetch_data(query, (pair,))}

async def fetch_active_pairs():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://www.okx.com/api/v5/public/instruments?instType=SPOT") as response:
            data = await response.json()
            return [
                inst["instId"]
                for inst in data.get("data", [])
                if inst["quoteCcy"] == "USDT" and inst["state"] == "live"
            ]

async def fetch_candles(session, pair, direction, ref_ts=None):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100
    }
    if direction == "before":
        logging.info(f"Fetching latest candles using BEFORE for {pair}")
    elif direction == "after" and ref_ts:
        params["after"] = str(int(ref_ts.timestamp() * 1000))
        logging.info(f"Fetching older candles using AFTER={ref_ts} for {pair}")
    else:
        raise ValueError("Invalid fetch direction")

    async with session.get(OKX_CANDLES_URL, params=params) as response:
        response.raise_for_status()
        return await response.json().get("data", [])

async def insert_candles(pair, candles, known_ts):
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
            logging.warning(f"Skipping malformed row: {e} | Raw: {c}")

    if not rows:
        return None, 0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        execute_values(cursor, query, rows)
        conn.commit()
        logging.info(f"Inserted {len(rows)} new candles for {pair} | {rows[-1][1]} → {rows[0][1]}")
        return rows[-1][1], len(rows)
    except Exception as e:
        logging.error(f"Insert failed for {pair}: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    return None, 0

async def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            logging.info(f"Sleeping {BATCH_INTERVAL - elapsed:.2f}s to honor rate limit")
            await asyncio.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

async def process_pair(pair, session, request_count, start_time):
    logging.info(f"Processing {pair}")
    known_ts = await get_known_timestamps(pair)

    # Initial call → latest candles (newest to oldest)
    candles = await fetch_candles(session, pair, direction="before")
    if not candles:
        logging.warning(f"No candles returned for {pair}")
        return

    after_ts = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
    inserted_ts, inserted = await insert_candles(pair, candles, known_ts)

    total_inserted = inserted
    if inserted == 0:
        logging.info(f"No new data for {pair}. Skipping pagination.")
        return

    # Paginate backward using AFTER
    while True:
        candles = await fetch_candles(session, pair, direction="after", ref_ts=after_ts)
        if not candles:
            break

        after_ts = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
        inserted_ts, inserted = await insert_candles(pair, candles, known_ts)
        total_inserted += inserted

        if inserted == 0:
            logging.info(f"Reached known data for {pair}")
            break

        request_count[OKX_CANDLES_URL], start_time = await enforce_rate_limit(
            request_count[OKX_CANDLES_URL], start_time
        )

    logging.info(f"Finished {pair}: {total_inserted} candles inserted")

async def main():
    logging.info("Fetching latest 1H candles from OKX...")
    pairs = await fetch_active_pairs()
    logging.info(f"{len(pairs)} pairs found")

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_pair(pair, session, request_count, start_time)
            for pair in pairs
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
