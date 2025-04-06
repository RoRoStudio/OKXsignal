import requests
import time
import logging
from datetime import datetime, timedelta, timezone
from psycopg2.extras import execute_values
from data.database import get_connection
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('funding_rates_fetcher')

# --- Constants
BASE_URL = "https://www.okx.com"
DAYS_BACK = 180

# Calculate target start date
UTC_NOW = datetime.now(timezone.utc)
TARGET_START_DATE = UTC_NOW - timedelta(days=DAYS_BACK)

# --- Rate limit state
last_request_time = {}
request_counts = {}

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
        return []

def insert_data(rows, columns):
    """Insert data into funding_rates_raw table with conflict handling"""
    if not rows:
        return 0
        
    query = f"""
    INSERT INTO funding_rates_raw ({', '.join(columns)}) 
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
        logger.error(f"Failed to insert into funding_rates_raw: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        conn.close()

def get_earliest_data_timestamp(symbol):
    """Get the earliest funding_time for a symbol in the funding_rates_raw table"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
        SELECT MIN(funding_time)
        FROM funding_rates_raw 
        WHERE symbol = %s
        """
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()[0]
        
        return result
    except Exception as e:
        logger.error(f"Error checking earliest data for {symbol}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def api_request_with_retry(params, rate_limit=(5, 1)):
    """Make API request with retry logic"""
    max_retries = 5
    endpoint = f"{BASE_URL}/api/v5/public/funding-rate-history"
    
    for retry in range(max_retries):
        try:
            wait_if_needed("funding_rates_raw", rate_limit)
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

def fetch_funding_rates_for_pair(pair):
    """Fetch complete funding rate history for a single pair"""
    
    # First check what's already in the database
    earliest_timestamp = get_earliest_data_timestamp(pair)
    
    if earliest_timestamp and earliest_timestamp <= TARGET_START_DATE:
        logger.info(f"✓ {pair} already has complete funding rate data starting from {format_date(earliest_timestamp)}")
        return True
    
    # If we have some data but not enough, adjust our target
    actual_target_date = TARGET_START_DATE
    if earliest_timestamp:
        logger.info(f"Partial data exists for {pair} starting from {format_date(earliest_timestamp)}")
        logger.info(f"Will fetch from {format_date(TARGET_START_DATE)} to {format_date(earliest_timestamp)}")
    else:
        logger.info(f"No existing funding rate data for {pair}")
        logger.info(f"Will fetch complete history from {format_date(TARGET_START_DATE)} to {format_date(UTC_NOW)}")
    
    # Need to track if we've reached our target date
    reached_target_date = False
    total_records = 0
    
    # For progress tracking - funding rates are every 8 hours, so approximately 3 per day
    estimated_records = DAYS_BACK * 3
    
    # Create a progress bar
    with tqdm(total=estimated_records, desc=f"{pair}", unit="rates") as pbar:
        # Start from current time and go backwards
        after = None
        
        while not reached_target_date:
            try:
                params = {"instId": pair, "limit": 100}
                if after:
                    params["after"] = after
                
                data = api_request_with_retry(params).get("data", [])
                
                if not data:
                    logger.warning(f"No more data available for {pair} - API returned empty response")
                    break
                
                # Process and insert the data
                rows = []
                for item in data:
                    try:
                        funding_time = parse_timestamp(item["fundingTime"])
                        funding_rate = float(item["fundingRate"])
                        realized_rate = float(item.get("realizedRate") or 0.0)
                        formula_type = item.get("formulaType")
                        method = item.get("method")
                        
                        rows.append((
                            funding_time, pair, funding_rate, realized_rate, formula_type, method
                        ))
                    except Exception as e:
                        logger.error(f"Error processing funding rate record: {e}")
                        continue
                
                # Insert the rows
                columns = ["funding_time", "symbol", "funding_rate", "realized_rate", "formula_type", "method"]
                inserted = insert_data(rows, columns)
                total_records += inserted
                pbar.update(len(data))
                
                # Get the timestamp of the oldest record in this batch
                oldest_ts = int(data[-1]["fundingTime"])
                oldest_dt = parse_timestamp(oldest_ts)
                
                # Update progress description to show the date range
                pbar.set_description(f"{pair} ({format_date(oldest_dt)})")
                
                # Check if we've reached our target date
                if oldest_dt <= TARGET_START_DATE:
                    logger.info(f"✓ Reached target date: {format_date(oldest_dt)}")
                    reached_target_date = True
                    break
                
                # Move backward for next batch
                after = oldest_ts - 1
                
            except Exception as e:
                logger.error(f"Error fetching {pair} funding rates: {e}")
                time.sleep(5)  # Pause on error
    
    logger.info(f"Completed {pair}: Inserted {total_records} funding rate records")
    return True

def main():
    """Main function to fetch funding rates for all USDT SWAP pairs"""
    start_time = time.time()
    
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting funding rate data fetch for all USDT SWAP pairs")
    logger.info(f"Target start date: {format_date(TARGET_START_DATE)}")
    logger.info(f"{'=' * 60}")
    
    # Get all available USDT SWAP pairs
    all_pairs = get_all_usdt_swap_pairs()
    
    if not all_pairs:
        logger.error("Failed to fetch any USDT SWAP pairs. Exiting.")
        return
    
    # Process each pair
    for idx, pair in enumerate(all_pairs):
        logger.info(f"[{idx+1}/{len(all_pairs)}] Processing {pair}")
        fetch_funding_rates_for_pair(pair)
    
    total_duration = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"All funding rate data fetching completed in {total_duration/60:.1f} minutes!")
    logger.info(f"Total pairs processed: {len(all_pairs)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()