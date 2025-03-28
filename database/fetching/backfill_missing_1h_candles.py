import asyncio
import aiohttp
import time
from datetime import datetime, timedelta, timezone
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
RATE_LIMIT = 20
INTERVAL = 2
MAX_CONCURRENT_PAIRS = 5
semaphore = None

def fetch_all_pairs():
    query = "SELECT DISTINCT pair FROM public.candles_1h;"
    return [row["pair"] for row in fetch_data(query)]

def fetch_existing_timestamps(pair):
    query = "SELECT timestamp_utc FROM public.candles_1h WHERE pair = %s;"
    return {
        row["timestamp_utc"].replace(minute=0, second=0, microsecond=0)
        for row in fetch_data(query, (pair,))
    }

def find_missing_timestamps(timestamps):
    if len(timestamps) < 2:
        return []

    timestamps = sorted(timestamps)
    missing = []
    expected_delta = timedelta(hours=1)

    for i in range(1, len(timestamps)):
        current = timestamps[i]
        prev = timestamps[i - 1]
        while prev + expected_delta < current:
            prev += expected_delta
            missing.append(prev)
    return missing

async def fetch_candles(session, pair, after_utc):
    async with semaphore:
        params = {
            "instId": pair,
            "bar": "1H",
            "limit": 100,
            "after": str(int(after_utc.timestamp() * 1000))
        }
        try:
            async with session.get(OKX_CANDLES_URL, params=params) as response:
                if response.status == 429:
                    print(f"\u26a0\ufe0f Rate limited for {pair}, sleeping {INTERVAL}s...")
                    await asyncio.sleep(INTERVAL)
                    return await fetch_candles(session, pair, after_utc)
                response.raise_for_status()
                data = await response.json()
                return data.get("data", [])
        except Exception as e:
            print(f"❌ Error fetching {pair} after {after_utc}: {e}")
            return []

def insert_candles(pair, candles, existing_ts):
    query = """
    INSERT INTO public.candles_1h
    (pair, timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h, quote_volume_1h)
    VALUES %s
    ON CONFLICT (pair, timestamp_utc) DO NOTHING;
    """
    rows = []
    for c in candles:
        try:
            from_zone = timezone.utc
            to_zone = timezone(timedelta(hours=1))
            utc_ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=from_zone).astimezone(to_zone)


            print(f"📦 Fetched candle UTC {utc_ts} for {pair}")
            if utc_ts in existing_ts:
                print(f"⏩ Skipped duplicate candle {utc_ts} for {pair}")
                continue
            else:
                print(f"⬆️ New candle {utc_ts} for {pair}, will insert")

            row = (
                pair, utc_ts, float(c[1]), float(c[2]), float(c[3]), float(c[4]),
                float(c[5]), float(c[6])
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠\ufe0f Malformed candle for {pair}: {e} | Raw: {c}")

    if not rows:
        return 0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        execute_values(cursor, query, rows)
        conn.commit()
        print(f"✅ Inserted {len(rows)} candles for {pair} | {rows[0][1]} → {rows[-1][1]}")
        return len(rows)
    except Exception as e:
        print(f"❌ Insert failed for {pair}: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    return 0

async def process_pair(pair, session):
    try:
        print(f"\n🔁 Checking {pair}")
        existing_ts = fetch_existing_timestamps(pair)
        now = datetime.now(timezone.utc)

        missing_ts = find_missing_timestamps(existing_ts)
        missing_ts = [ts for ts in missing_ts if ts < now]

        if not missing_ts:
            print(f"✔️ No gaps for {pair}")
            return 0

        print(f"🧩 Found {len(missing_ts)} missing 1H timestamps for {pair}")
        inserted_total = 0

        while True:
            earliest_gap = min(find_missing_timestamps(existing_ts), default=None)
            if not earliest_gap or earliest_gap >= now:
                break

            print(f"⏳ Fetching from earliest gap: {earliest_gap}")
            candles = await fetch_candles(session, pair, earliest_gap - timedelta(seconds=1))

            if not candles:
                print(f"⚠️ No candles returned for {pair} after {earliest_gap}")
                break

            inserted = insert_candles(pair, candles, existing_ts)
            if inserted > 0:
                new_ts = {datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc) for c in candles}
                existing_ts.update(new_ts)
                inserted_total += inserted
            else:
                print(f"⚠️ No new inserts from candles fetched after {earliest_gap}")

            await asyncio.sleep(INTERVAL / RATE_LIMIT)

        if inserted_total == 0:
            print(f"✔️ No candles inserted for {pair}")
        else:
            print(f"✅ Total inserted for {pair}: {inserted_total}")

        return inserted_total

    except Exception as e:
        print(f"❌ Failed to process {pair}: {e}")
        return 0


async def main():
    global semaphore
    semaphore = asyncio.Semaphore(RATE_LIMIT)

    print("🚀 Scanning for gaps in 1H candle history...")
    pairs = fetch_all_pairs()
    print(f"📈 Found {len(pairs)} pairs with existing data")

    total_inserted = 0
    queue = asyncio.Queue()
    for p in pairs:
        await queue.put(p)

    async def worker():
        nonlocal total_inserted
        async with aiohttp.ClientSession() as session:
            while not queue.empty():
                pair = await queue.get()
                inserted = await process_pair(pair, session)
                total_inserted += inserted
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENT_PAIRS)]
    await queue.join()

    for w in workers:
        w.cancel()

    print(f"\n✅ Backfill complete: Inserted {total_inserted} missing candles")

if __name__ == "__main__":
    asyncio.run(main())
