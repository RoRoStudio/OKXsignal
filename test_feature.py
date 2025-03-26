#!/usr/bin/env python3
"""
Test script for computing features on a single pair
- Simpler version of compute_features.py for testing
- Focuses on just one pair without threading
"""

import os
import sys
import logging
import argparse
import time
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Setup logging
def setup_logging(log_dir="logs", log_level="INFO"):
    """Set up application logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = os.path.join(log_dir, f"feature_test_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(levelname)s] %(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("feature_test")

def get_db_connection():
    """Get a database connection using environment variables"""
    # Load environment variables
    load_dotenv(os.path.join(root_dir, "config", "credentials.env"))
    
    # Try to load from config.ini first
    try:
        from config.config_loader import load_config
        config = load_config()
        db_host = config.get('DB_HOST', os.getenv('DB_HOST', 'localhost'))
        db_port = config.get('DB_PORT', os.getenv('DB_PORT', '5432'))
        db_name = config.get('DB_NAME', os.getenv('DB_NAME', 'okxsignal'))
    except:
        # Fall back to environment variables
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'okxsignal')
    
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')
    
    # Log connection details
    logging.info(f"Connecting to database {db_name} on {db_host}:{db_port} as {db_user}")
    
    try:
        # Create connection
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        raise

def verify_database():
    """Verify that the database connection works and has the expected tables"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check database version
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        logging.info(f"Connected to PostgreSQL: {version}")
        
        # Check for candles_1h table
        cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'candles_1h'
        )
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logging.error("Table candles_1h does not exist!")
            return False
            
        # Check for data in candles_1h
        cursor.execute("SELECT COUNT(*) FROM candles_1h")
        row_count = cursor.fetchone()[0]
        logging.info(f"Found {row_count} rows in candles_1h table")
        
        # Get available pairs
        cursor.execute("SELECT DISTINCT pair FROM candles_1h")
        pairs = [row[0] for row in cursor.fetchall()]
        logging.info(f"Found {len(pairs)} distinct pairs in candles_1h table")
        
        if len(pairs) == 0:
            logging.error("No pairs found in candles_1h!")
            return False
            
        cursor.close()
        return True
    except Exception as e:
        logging.error(f"Database verification failed: {e}")
        return False
    finally:
        if conn is not None:
            conn.close()

def process_single_pair(pair, window=100):
    """Process a single pair without threading"""
    logging.info(f"Processing pair {pair} with window {window}")
    
    try:
        # Import required modules
        from features.config import ConfigManager
        from features.optimized.feature_processor import OptimizedFeatureProcessor
        from features.db_operations import (
            fetch_data_numpy, 
            batch_update_features,
            get_database_columns
        )
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Create a database connection
        conn = get_db_connection()
        
        # Create feature processor
        feature_processor = OptimizedFeatureProcessor(
            use_numba=True,
            use_gpu=False
        )
        
        # Get database columns
        db_columns = get_database_columns(conn, 'candles_1h')
        logging.info(f"Found {len(db_columns)} columns in candles_1h table")
        
        # Fetch data
        start_time = time.time()
        price_data = fetch_data_numpy(conn, pair, window + 50)
        logging.info(f"Fetched {len(price_data['closes'])} rows in {time.time() - start_time:.3f}s")
        
        if not price_data:
            logging.error(f"No data found for {pair}")
            return False
            
        # Determine enabled feature groups
        enabled_features = {
            'price_action', 'momentum', 'volatility', 'volume', 
            'statistical', 'pattern', 'time', 'labels'
        }
        
        # Process features
        start_time = time.time()
        feature_results = feature_processor.process_features(price_data, enabled_features)
        compute_time = time.time() - start_time
        logging.info(f"Computed features in {compute_time:.3f}s")
        
        # Take only the newest rows for updating
        if len(price_data['closes']) > window:
            start_idx = len(price_data['closes']) - window
            for key in feature_results:
                feature_results[key] = feature_results[key][start_idx:]
                
            update_timestamps = price_data['raw_timestamps'][start_idx:]
        else:
            update_timestamps = price_data['raw_timestamps']
            
        # Define columns to update
        reserved_columns = {'id', 'pair', 'timestamp_utc', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 
                          'volume_1h', 'quote_volume_1h', 'taker_buy_base_1h'}
        
        # Filter out columns that don't exist in the database
        columns_for_update = [
            col for col in feature_results.keys() 
            if col in db_columns and col not in reserved_columns
        ]
        
        logging.info(f"Found {len(columns_for_update)} columns to update")
        
        # Check if we want to actually update the database
        if '--dry-run' in sys.argv:
            logging.info("Dry run - not updating database")
            return True
            
        # Update database
        start_time = time.time()
        updated_rows = batch_update_features(
            conn, pair, update_timestamps, feature_results, columns_for_update
        )
        update_time = time.time() - start_time
        
        logging.info(f"Updated {updated_rows} rows in {update_time:.3f}s")
        
        # Close connection
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"Error processing pair {pair}: {e}", exc_info=True)
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test feature computation on a single pair')
    parser.add_argument('--pair', type=str, default='BTC-USDT', help='Pair to process')
    parser.add_argument('--window', type=int, default=100, help='Window size')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    parser.add_argument('--dry-run', action='store_true', help='Do not update database')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    logger.info(f"Starting feature test with pair {args.pair} and window {args.window}")
    
    # Verify database
    if not verify_database():
        logger.error("Database verification failed. Exiting.")
        sys.exit(1)
        
    # Process the pair
    if process_single_pair(args.pair, args.window):
        logger.info("Feature computation test completed successfully")
    else:
        logger.error("Feature computation test failed")
        sys.exit(1)
    
if __name__ == '__main__':
    main()