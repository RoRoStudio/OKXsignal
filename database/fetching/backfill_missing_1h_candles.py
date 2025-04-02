import asyncio
import aiohttp
import time
import logging
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("backfill")

config = load_config()
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
RATE_LIMIT = 15
INTERVAL = 2.5
MAX_CONCURRENT_PAIRS = 2
BATCH_SIZE = 20  # Process missing candles in batches

semaphore = None

def convert_hk_timestamp_to_utc(unix_timestamp_ms):
    """
    Properly converts an OKX Hong Kong timestamp (Unix milliseconds) to UTC datetime
    """
    hk_timezone = timezone(timedelta(hours=8))
    hk_ts = datetime.fromtimestamp(int(unix_timestamp_ms) / 1000, tz=hk_timezone)
    return hk_ts.astimezone(timezone.utc)

def fetch_all_pairs():
    """Fetch all pairs from the database that have at least some candle data"""
    query = "SELECT DISTINCT pair FROM public.candles_1h;"
    return [row["pair"] for row in fetch_data(query)]

def find_all_gaps(pair):
    """
    Find ALL gaps in hourly data, not just exact 2-hour gaps.
    This will detect any missing hours, regardless of gap size.
    """
    # First, just get all timestamps for this pair, ordered
    query = """
    SELECT timestamp_utc
    FROM public.candles_1h
    WHERE pair = %s
    ORDER BY timestamp_utc;
    """
    
    results = fetch_data(query, (pair,))
    timestamps = [row["timestamp_utc"] for row in results]
    
    if len(timestamps) < 2:
        logger.warning(f"Not enough data for {pair} to find gaps")
        return []
    
    # Find all gaps (greater than 1 hour) between consecutive timestamps
    missing_hours = []
    
    for i in range(1, len(timestamps)):
        current = timestamps[i]
        previous = timestamps[i-1]
        gap_seconds = (current - previous).total_seconds()
        
        # If gap is more than 1 hour (3600 seconds), there are missing hours
        if gap_seconds > 3600:
            # Calculate how many hours are missing
            hours_missing = int(gap_seconds / 3600) - 1
            
            if hours_missing > 0:
                # Generate all missing hours in this gap
                for h in range(1, hours_missing + 1):
                    missing_hour = previous + timedelta(hours=h)
                    missing_hours.append(missing_hour)
                
                logger.info(f"Found gap of {hours_missing + 1} hours between {previous} and {current}")
    
    # Add logging for all identified missing hours
    if missing_hours:
        logger.info(f"Found {len(missing_hours)} total missing hours for {pair}")
        logger.info(f"First few missing hours:")
        for i in range(min(10, len(missing_hours))):
            logger.info(f"  - {missing_hours[i]}")
    else:
        logger.info(f"No missing hours found for {pair}")
    
    return missing_hours

def log_candle_details(candle):
    """Log details of a candle for debugging"""
    if not candle:
        return "None"
    
    # Convert timestamp to readable format
    ts_ms = int(candle[0])
    utc_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    cet_time = utc_time.astimezone(timezone(timedelta(hours=1)))  # CET is UTC+1
    
    details = f"Timestamp: {candle[0]} (UTC: {utc_time}, CET: {cet_time}), "
    details += f"OHLC: {candle[1]}/{candle[2]}/{candle[3]}/{candle[4]}, "
    details += f"Volume: {candle[5]}"
    
    return details

async def fetch_candles_direct(session, pair, timestamp, fetch_one=False):
    """
    Fetch candles directly using millisecond timestamps
    
    The OKX API has counterintuitive behavior:
    - after=X returns candles BEFORE X
    - before=Y returns candles AFTER Y
    
    OKX API uses Hong Kong time (UTC+8), while our database uses CET (UTC+1)
    We need to adjust for this timezone difference
    """
    async with semaphore:
        # Ensure timestamp has timezone info
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Convert timestamp to UTC
        timestamp_utc = timestamp.astimezone(timezone.utc)
        
        # Convert to milliseconds for OKX API 
        # We need to add one hour to get the next candle as the 'after' parameter
        target_ts_ms = int(timestamp_utc.timestamp() * 1000)
        after_ms = str(target_ts_ms + 3600000)  # Add one hour
        
        params = {
            "instId": pair,
            "bar": "1H",
            "limit": 1 if fetch_one else 100,  # Fetch just one candle if needed
            "after": after_ms
        }
        
        logger.debug(f"Fetching candles with params: {params}")
        
        try:
            async with session.get(OKX_CANDLES_URL, params=params) as response:
                if response.status == 429:
                    logger.warning(f"⚠️ Rate limited for {pair}, sleeping {INTERVAL}s...")
                    await asyncio.sleep(INTERVAL)
                    return await fetch_candles_direct(session, pair, timestamp, fetch_one)
                
                response.raise_for_status()
                data = await response.json()
                
                if data.get("code") != "0":
                    logger.error(f"❌ API Error for {pair}: {data.get('msg', 'Unknown error')}")
                    return []
                
                candles = data.get("data", [])
                
                if candles:
                    logger.debug(f"Retrieved {len(candles)} candles")
                    # Log first and last candle
                    logger.debug(f"First candle: {log_candle_details(candles[0])}")
                    if len(candles) > 1:
                        logger.debug(f"Last candle: {log_candle_details(candles[-1])}")
                    
                    # Filter to find the exact candle we want
                    if fetch_one:
                        for candle in candles:
                            # Convert timestamp to datetime in UTC
                            candle_ts_ms = int(candle[0])
                            candle_ts_utc = datetime.fromtimestamp(candle_ts_ms / 1000, tz=timezone.utc)
                            
                            # Compare the absolute timestamp values, not the components
                            # First, ensure both timestamps have timezone info
                            if timestamp.tzinfo is None:
                                timestamp_with_tz = timestamp.replace(tzinfo=timezone.utc)
                            else:
                                timestamp_with_tz = timestamp
                            
                            # Convert timestamp to UTC for comparison
                            timestamp_utc = timestamp_with_tz.astimezone(timezone.utc)
                            
                            # Find the difference in seconds
                            diff_seconds = abs((candle_ts_utc - timestamp_utc).total_seconds())
                            
                            # Allow a small tolerance (e.g., a few seconds) for timestamp comparison
                            # If the timestamps are within 5 minutes of each other, consider it a match
                            if diff_seconds < 300:  # 5 minutes in seconds
                                logger.debug(f"Found candle match: {log_candle_details(candle)}")
                                logger.debug(f"Candle time (UTC): {candle_ts_utc}, Target time (adj. to UTC): {timestamp_utc}")
                                return [candle]
                        
                        logger.debug(f"No exact match found for {timestamp} (UTC: {timestamp.astimezone(timezone.utc)})")
                        return []
                
                else:
                    logger.warning(f"No candles returned for query with after={after_ms}")
                
                return candles
        
        except Exception as e:
            logger.error(f"❌ Error fetching candles: {e}")
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
            # Use the helper function for consistent conversion
            utc_ts = convert_hk_timestamp_to_utc(c[0])
            
            # Check if we already have this timestamp
            if utc_ts in existing_ts:
                print(f"⏩ Skipped duplicate candle {utc_ts} for {pair}")
                continue
            
            # This is a new candle
            print(f"⬆️ New candle {utc_ts} for {pair}, will insert")
            
            # Parse candle data
            # OKX returns [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            # We need the first 7 fields
            if len(c) < 7:
                print(f"⚠️ Malformed candle for {pair}: insufficient data | Raw: {c}")
                continue
                
            row = (
                pair, utc_ts, float(c[1]), float(c[2]), float(c[3]), float(c[4]),
                float(c[5]), float(c[6])
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Malformed candle for {pair}: {e} | Raw: {c}")

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

async def process_missing_hour(pair, missing_hour, session):
    """Process a single missing hour"""
    try:
        # Fetch the specific candle for this hour
        candles = await fetch_candles_direct(session, pair, missing_hour, fetch_one=True)
        
        if not candles:
            logger.warning(f"⚠️ Could not find candle for {pair} at {missing_hour}")
            return 0
        
        # Insert the candles (should be just one)
        inserted = insert_candles(pair, candles)
        
        # Rate limiting
        await asyncio.sleep(INTERVAL / RATE_LIMIT)
        
        return inserted
    
    except Exception as e:
        logger.error(f"❌ Error processing {pair} at {missing_hour}: {e}")
        return 0

async def process_pair(pair, session):
    """Process a single pair to find and fill gaps"""
    try:
        logger.info(f"🔁 Processing {pair}")
        
        # Find missing hours using the direct gap detection method
        missing_hours = find_all_gaps(pair)
        
        if not missing_hours:
            logger.info(f"✓ No missing hours found for {pair}")
            return 0
        
        # Group missing hours by date for better logging
        date_groups = {}
        for hour in missing_hours:
            day = hour.strftime("%Y-%m-%d")
            if day not in date_groups:
                date_groups[day] = []
            date_groups[day].append(hour)
        
        logger.info(f"Missing hours grouped into {len(date_groups)} days")
        for day, hours in date_groups.items():
            logger.info(f"Day {day}: {len(hours)} missing hours")
        
        # Process missing hours in batches
        total_inserted = 0
        logger.info(f"Attempting to fill {len(missing_hours)} gaps for {pair}")
        
        # Process in batches to avoid too many concurrent tasks
        for i in range(0, len(missing_hours), BATCH_SIZE):
            batch = missing_hours[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(missing_hours) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            # Process each missing hour in the batch
            batch_tasks = [process_missing_hour(pair, hour, session) for hour in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Update the total inserted count
            batch_inserted = sum(batch_results)
            total_inserted += batch_inserted
            
            logger.info(f"Batch complete: {batch_inserted} candles inserted")
            
            # Brief pause between batches
            await asyncio.sleep(1)
        
        logger.info(f"✅ Total inserted for {pair}: {total_inserted}")
        return total_inserted
        
    except Exception as e:
        logger.error(f"❌ Error processing {pair}: {e}", exc_info=True)
        return 0

async def main():
    """Main function to process all pairs"""
    global semaphore
    semaphore = asyncio.Semaphore(RATE_LIMIT)
    
    logger.info("🚀 Scanning for gaps in hourly candle history...")
    pairs = fetch_all_pairs()
    logger.info(f"📈 Found {len(pairs)} pairs with existing data")
    
    # Process command-line arguments
    filter_pairs = None
    
    for arg in sys.argv[1:]:
        if arg == '--debug':
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        elif not arg.startswith('--'):
            # Assume it's a list of pairs
            filter_pairs = arg.split(',')
    
    if filter_pairs:
        pairs = [p for p in pairs if p in filter_pairs]
        logger.info(f"Filtered to {len(pairs)} pairs: {pairs}")
    
    total_inserted = 0
    queue = asyncio.Queue()
    for p in pairs:
        await queue.put(p)
    
    async def worker():
        nonlocal total_inserted
        async with aiohttp.ClientSession() as session:
            while not queue.empty():
                pair = await queue.get()
                try:
                    inserted = await process_pair(pair, session)
                    total_inserted += inserted
                except Exception as e:
                    logger.error(f"Unhandled error processing {pair}: {e}", exc_info=True)
                finally:
                    queue.task_done()
    
    workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENT_PAIRS)]
    
    try:
        await queue.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Cancelling workers...")
        for w in workers:
            w.cancel()
    
    for w in workers:
        w.cancel()
    
    logger.info(f"\n✅ Backfill complete: {total_inserted} candles inserted")
    
    if total_inserted == 0:
        logger.info("\nNOTE: No candles were inserted. Possible reasons:")
        logger.info("1. No gaps were found in the data")
        logger.info("2. All identified gaps already had candles inserted in a previous run")
        logger.info("3. There were issues connecting to the OKX API")

if __name__ == "__main__":
    # Fix possible import issue with psycopg2.pool
    try:
        import psycopg2.pool
    except AttributeError:
        logger.warning("psycopg2.pool not available. Connection pooling will be disabled.")
    
    asyncio.run(main())