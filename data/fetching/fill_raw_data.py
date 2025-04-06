import requests
import time
import logging
from datetime import datetime, timedelta, timezone
from psycopg2.extras import execute_values
from data.database import get_connection
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('okx_data_fetcher')

# --- Constants
PERPETUAL_PAIRS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "ADA-USDT-SWAP", "DOGE-USDT-SWAP", 
    "XRP-USDT-SWAP", "LTC-USDT-SWAP", "SOL-USDT-SWAP", "DOT-USDT-SWAP", "BNB-USDT-SWAP"
]

def get_all_usdt_swap_pairs():
    """Fetch all SWAP pairs with USDT as quote currency"""
    try:
        logger.info("Fetching all available USDT SWAP pairs...")
        response = requests.get(f"{BASE_URL}/api/v5/public/instruments", params={"instType": "SWAP"}, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        
        # Filter for pairs with USDT as the quote currency
        usdt_pairs = []
        for instrument in data:
            inst_id = instrument.get("instId", "")
            if inst_id.endswith("-USDT-SWAP"):
                usdt_pairs.append(inst_id)
        
        logger.info(f"Found {len(usdt_pairs)} USDT SWAP pairs")
        return usdt_pairs
    except Exception as e:
        logger.error(f"Error fetching USDT SWAP pairs: {e}")
        # Fall back to predefined pairs if we can't fetch all of them
        logger.warning(f"Falling back to predefined list of {len(PERPETUAL_PAIRS)} pairs")
        return PERPETUAL_PAIRS

DAYS_BACK = 180
BASE_URL = "https://www.okx.com"

# --- Rate limit state
last_request_time = {}
request_counts = {}

# Calculate target start date
UTC_NOW = datetime.now(timezone.utc)
TARGET_START_DATE = UTC_NOW - timedelta(days=DAYS_BACK)

def wait_if_needed(key, rate_limit):
    """Implement rate limiting for API requests"""
    rate, interval = rate_limit
    now = time.time()
    if key not in request_counts:
        request_counts[key] = 0
        last_request_time[key] = now
        
    request_counts[key] += 1
    if request_counts[key] >= rate:
        elapsed = now - last_request_time[key]
        if elapsed < interval:
            time.sleep(interval - elapsed)
        request_counts[key] = 0
        last_request_time[key] = time.time()

def parse_timestamp(ts_ms):
    """Convert millisecond timestamp to UTC datetime object"""
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)

def format_date(dt):
    """Format a datetime to a readable date string"""
    if isinstance(dt, (int, float)):
        dt = parse_timestamp(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def insert_data(table, rows, columns):
    """Insert data into database with conflict handling"""
    if not rows:
        return 0
        
    query = f"""
    INSERT INTO {table} ({', '.join(columns)}) 
    VALUES %s 
    ON CONFLICT DO NOTHING;
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    try:
        execute_values(cursor, query, rows)
        conn.commit()
        return len(rows)
    except Exception as e:
        logger.error(f"Failed to insert into {table}: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        conn.close()

def get_earliest_data_timestamp(table, symbol, date_column=None):
    """Get the earliest timestamp for a symbol in a table"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # If date_column is not specified, use default column name based on table
    if date_column is None:
        if table == "funding_rates_raw":
            date_column = "funding_time"
        else:
            date_column = "timestamp"
    
    try:
        query = f"""
        SELECT MIN({date_column})
        FROM {table} 
        WHERE symbol = %s
        """
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()[0]
        
        return result
    except Exception as e:
        logger.error(f"Error checking earliest data for {symbol} in {table}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def api_request_with_retry(endpoint, params, rate_limit_key, rate_limit):
    """Make API request with retry logic"""
    max_retries = 5
    
    for retry in range(max_retries):
        try:
            wait_if_needed(rate_limit_key, rate_limit)
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if retry < max_retries - 1:
                sleep_time = 2 ** retry
                logger.warning(f"API request failed (attempt {retry+1}/{max_retries}): {e}")
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
    
    raise Exception("Failed to make API request")

def get_candle_data_for_timeframe(table_name, pair, start_date, end_date):
    """Get candle data for a specific timeframe from the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = f"""
        SELECT * FROM {table_name}
        WHERE symbol = %s 
          AND timestamp >= %s 
          AND timestamp <= %s
        ORDER BY timestamp
        """
        cursor.execute(query, (pair, start_date, end_date))
        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return result
    except Exception as e:
        logger.error(f"Error fetching {table_name} data: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def fetch_complete_candles_for_pair(pair, table_name, endpoint, params_func, row_parser, columns, rate_limit):
    """Fetch complete candle data for one pair with proper tracking"""
    
    # For index price candles, we need to check existing data with the index symbol
    check_symbol = pair
    if table_name == "index_price_candles_raw":
        check_symbol = pair.replace("-SWAP", "")
        logger.info(f"Using index symbol {check_symbol} for database check")
    
    # First check what's already in the database
    earliest_timestamp = get_earliest_data_timestamp(table_name, check_symbol)
    
    if earliest_timestamp and earliest_timestamp <= TARGET_START_DATE:
        logger.info(f"✓ {check_symbol} already has complete {table_name} data starting from {format_date(earliest_timestamp)}")
        return True
    
    # If we have some data but not enough, adjust our target
    actual_target_date = TARGET_START_DATE
    if earliest_timestamp:
        logger.info(f"Partial data exists for {check_symbol} starting from {format_date(earliest_timestamp)}")
        logger.info(f"Will fetch {table_name} from {format_date(TARGET_START_DATE)} to {format_date(earliest_timestamp)}")
    else:
        logger.info(f"No existing {table_name} data for {check_symbol}")
        logger.info(f"Will fetch complete history from {format_date(TARGET_START_DATE)} to {format_date(UTC_NOW)}")
    
    # Need to track if we've reached our target date
    reached_target_date = False
    total_records = 0
    
    # For progress tracking
    estimated_records = DAYS_BACK * 24 * 4  # 15-minute intervals for 180 days
    
    # Create a progress bar
    with tqdm(total=estimated_records, desc=f"{check_symbol} {table_name}", unit="candles") as pbar:
        # Start from current time and go backwards
        after = None
        before = None
        
        while not reached_target_date:
            try:
                params = params_func(pair, after, before)
                data = api_request_with_retry(endpoint, params, table_name, rate_limit).get("data", [])
                
                if not data:
                    logger.warning(f"No more data available for {check_symbol} - API returned empty response")
                    break
                
                # Process and insert the data
                rows = []
                for item in data:
                    row = row_parser(item, pair)
                    if row:
                        rows.append(row)
                
                # Insert the rows
                inserted = insert_data(table_name, rows, columns)
                total_records += inserted
                pbar.update(len(data))
                
                # Get the timestamp of the oldest record in this batch
                if isinstance(data[-1], list):  # Candle data format
                    oldest_ts = int(data[-1][0])
                    oldest_dt = parse_timestamp(oldest_ts)
                else:  # Funding rate and other data format
                    oldest_ts = int(data[-1].get("ts") or data[-1].get("fundingTime"))
                    oldest_dt = parse_timestamp(oldest_ts)
                
                # Update progress description to show the date range
                pbar.set_description(f"{check_symbol} {table_name} ({format_date(oldest_dt)})")
                
                # Check if we've reached our target date
                if oldest_dt <= TARGET_START_DATE:
                    logger.info(f"✓ Reached target date: {format_date(oldest_dt)}")
                    reached_target_date = True
                    break
                
                # If we're using 'after', we're paginating backward
                after = oldest_ts - 1
                
            except Exception as e:
                logger.error(f"Error fetching {check_symbol} {table_name}: {e}")
                time.sleep(5)  # Pause on error
    
    logger.info(f"Completed {check_symbol} {table_name}: Inserted {total_records} records")
    return True

def process_data_type(data_type_info, main_progress=None, pairs=None):
    """Process one data type for all pairs"""
    table_name, endpoint, params_func, row_parser, columns, rate_limit = data_type_info
    
    # Use provided pairs or default to PERPETUAL_PAIRS
    pairs_to_process = pairs if pairs is not None else PERPETUAL_PAIRS
    
    logger.info(f"\n{'=' * 30}")
    logger.info(f"Processing {table_name} for {len(pairs_to_process)} pairs")
    logger.info(f"{'=' * 30}")
    
    for idx, pair in enumerate(pairs_to_process):
        logger.info(f"[{idx+1}/{len(pairs_to_process)}] Processing {pair}")
        
        fetch_complete_candles_for_pair(
            pair=pair,
            table_name=table_name,
            endpoint=endpoint,
            params_func=params_func,
            row_parser=row_parser,
            columns=columns,
            rate_limit=rate_limit
        )
        
        # Update main progress bar if provided
        if main_progress:
            increment = 100 / (len(pairs_to_process) * len(DATA_TYPES))
            main_progress.update(increment)

def fetch_premium_aligned_to_candles():
    """Fetch premium data and align it with 15m candles"""
    for pair_idx, pair in enumerate(PERPETUAL_PAIRS):
        logger.info(f"[{pair_idx+1}/{len(PERPETUAL_PAIRS)}] Processing premium data for {pair}")
        
        # Check if we already have complete premium data
        earliest_timestamp = get_earliest_data_timestamp("premium_history_raw", pair)
        
        if earliest_timestamp and earliest_timestamp <= TARGET_START_DATE:
            logger.info(f"✓ {pair} already has complete premium data starting from {format_date(earliest_timestamp)}")
            continue
        
        # Get all 15m candle timestamps for this pair
        candle_data = get_candle_data_for_timeframe(
            "candles_15m_raw", 
            pair, 
            TARGET_START_DATE, 
            UTC_NOW
        )
        
        if not candle_data:
            logger.warning(f"No 15m candles found for {pair}. Skipping premium alignment.")
            continue
        
        logger.info(f"Found {len(candle_data)} candle timestamps to align premium data with")
        
        # Extract all timestamps
        candle_timestamps = [row['timestamp'] for row in candle_data]
        
        # Fetch all premium data for the date range
        all_premium_data = []
        
        # Create a progress bar for premium fetching
        with tqdm(
            desc=f"Fetching premium data for {pair}",
            total=DAYS_BACK  # Approximate - just for display
        ) as premium_pbar:
            
            # Start from latest time and go backwards
            after = None
            
            # Track progress for display
            last_reported_date = UTC_NOW
            days_fetched = 0
            
            while True:
                try:
                    params = {"instId": pair, "limit": 100}
                    if after:
                        params["after"] = after
                    
                    response = api_request_with_retry(
                        f"{BASE_URL}/api/v5/public/premium-history",
                        params,
                        "premium_history_raw",
                        (10, 1)
                    )
                    
                    data = response.get("data", [])
                    
                    if not data:
                        logger.info(f"No more premium data available for {pair}")
                        break
                    
                    # Add to our collection
                    for item in data:
                        ts = int(item["ts"])
                        dt = parse_timestamp(ts)
                        all_premium_data.append({
                            'timestamp': dt,
                            'premium': float(item["premium"])
                        })
                    
                    # Get the oldest timestamp from this batch
                    oldest_ts = int(data[-1]["ts"])
                    oldest_dt = parse_timestamp(oldest_ts)
                    
                    # Update progress based on days fetched
                    if (last_reported_date - oldest_dt).days > days_fetched:
                        days_fetched = (last_reported_date - oldest_dt).days
                        premium_pbar.update(days_fetched - premium_pbar.n)
                        last_reported_date = oldest_dt
                    
                    # If we've reached our target date, we're done
                    if oldest_dt <= TARGET_START_DATE:
                        premium_pbar.update(DAYS_BACK - premium_pbar.n)  # Complete the progress bar
                        logger.info(f"✓ Reached target date for premium data: {format_date(oldest_dt)}")
                        break
                    
                    # Move backward for next batch
                    after = oldest_ts - 1
                    
                except Exception as e:
                    logger.error(f"Error fetching premium data: {e}")
                    time.sleep(5)
                    continue
        
        if not all_premium_data:
            logger.warning(f"No premium data fetched for {pair}")
            continue
            
        logger.info(f"Fetched {len(all_premium_data)} premium data points")
        
        # Sort premium data by timestamp
        all_premium_data.sort(key=lambda x: x['timestamp'])
        
        # Convert to pandas DataFrame for more efficient alignment
        premium_df = pd.DataFrame(all_premium_data)
        
        # Align premium data with candle timestamps
        aligned_data = []
        
        with tqdm(
            total=len(candle_timestamps),
            desc=f"Aligning premium data for {pair}"
        ) as align_pbar:
            
            if not premium_df.empty:
                for candle_ts in candle_timestamps:
                    # Find the premium data point closest to this candle timestamp
                    # but not exceeding it
                    mask = premium_df['timestamp'] <= candle_ts
                    if mask.any():
                        closest_idx = premium_df.loc[mask, 'timestamp'].idxmax()
                        premium_value = premium_df.loc[closest_idx, 'premium']
                        
                        aligned_data.append((
                            candle_ts, pair, premium_value
                        ))
                    
                    align_pbar.update(1)
        
        # Insert the aligned data
        if aligned_data:
            inserted = insert_data("premium_history_raw", aligned_data, [
                "timestamp", "symbol", "premium"
            ])
            logger.info(f"✓ Inserted {inserted} aligned premium records for {pair}")
        else:
            logger.warning(f"No premium data could be aligned for {pair}")

# Define data types to process
DATA_TYPES = [
    # Table name, endpoint, params_func, row_parser, columns, rate_limit
    (
        "candles_15m_raw",
        f"{BASE_URL}/api/v5/market/history-candles",
        lambda pair, after, before: {
            "instId": pair,
            "bar": "15m",
            "after": after,
            "before": before,
            "limit": 100
        },
        lambda d, pair: (
            parse_timestamp(d[0]), pair,
            float(d[1]), float(d[2]), float(d[3]), float(d[4]),
            float(d[5]), float(d[6]), float(d[7]), d[8] == "1"
        ) if len(d) >= 9 else None,
        ["timestamp", "symbol", "open", "high", "low", "close", 
         "volume", "volume_ccy", "volume_quote", "confirmed"],
        (20, 2)
    ),
    (
        "funding_rates_raw",
        f"{BASE_URL}/api/v5/public/funding-rate-history",
        lambda pair, after, before: {
            "instId": pair,
            "after": after,
            "before": before,
            "limit": 100
        },
        lambda d, pair: (
            parse_timestamp(d["fundingTime"]), pair,
            float(d["fundingRate"]), float(d.get("realizedRate") or 0.0),
            d.get("formulaType"), d.get("method")
        ),
        ["funding_time", "symbol", "funding_rate", "realized_rate", 
         "formula_type", "method"],
        (5, 1)
    ),
    (
        "mark_price_candles_raw",
        f"{BASE_URL}/api/v5/market/history-mark-price-candles",  # Using the historical endpoint
        lambda pair, after, before: {
            "instId": pair,
            "bar": "15m",
            "after": after,
            "before": before,
            "limit": 100
        },
        lambda d, pair: (
            parse_timestamp(d[0]), pair,
            float(d[1]), float(d[2]), float(d[3]), float(d[4]), d[5] == "1"
        ) if len(d) >= 6 else None,
        ["timestamp", "symbol", "open", "high", "low", "close", "confirmed"],
        (10, 2)  # Updated rate limit according to docs
    ),
    (
        "index_price_candles_raw",
        f"{BASE_URL}/api/v5/market/history-index-candles",  # Using the historical endpoint
        lambda pair, after, before: {
            "instId": pair.replace("-SWAP", ""),  # Convert to index ID format
            "bar": "15m",
            "after": after,
            "before": before,
            "limit": 100
        },
        lambda d, pair: (
            parse_timestamp(d[0]), pair.replace("-SWAP", ""),  # Store the index symbol, not the SWAP symbol
            float(d[1]), float(d[2]), float(d[3]), float(d[4]), d[5] == "1"
        ) if len(d) >= 6 else None,
        ["timestamp", "symbol", "open", "high", "low", "close", "confirmed"],
        (10, 2)  # Updated rate limit according to docs
    )
]

def main():
    """Main function to fetch all data types for all pairs"""
    start_time = time.time()
    
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting data fetch for {len(PERPETUAL_PAIRS)} pairs, {DAYS_BACK} days of history")
    logger.info(f"Target start date: {format_date(TARGET_START_DATE)}")
    logger.info(f"{'=' * 60}")
    
    # Process each data type for the 9 main pairs
    for data_type_info in DATA_TYPES:
        process_data_type(data_type_info)
    
    # Handle premium data as a special case
    logger.info("\n" + "=" * 30)
    logger.info("Processing premium data with 15m alignment")
    logger.info("=" * 30)
    
    fetch_premium_aligned_to_candles()
    
    total_duration = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"All data fetching completed in {total_duration/60:.1f} minutes!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()