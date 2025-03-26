#!/usr/bin/env python3
"""
Comprehensive Database Connection and Data Retrieval Test Script
"""

import os
import sys
import logging
from pathlib import Path
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('database_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
def load_env_vars():
    """Load environment variables from credentials file"""
    credentials_path = project_root / 'config' / 'credentials.env'
    config_path = project_root / 'config' / 'config.ini'
    
    logger.info(f"Looking for credentials at: {credentials_path}")
    logger.info(f"Looking for config at: {config_path}")
    
    if credentials_path.exists():
        load_dotenv(dotenv_path=credentials_path)
        logger.info("Credentials loaded successfully")
    else:
        logger.error(f"Credentials file not found at {credentials_path}")
        return None
    
    return credentials_path

# Database connection test
def test_database_connection():
    """Test database connection with detailed logging"""
    try:
        # Read database configuration
        import configparser
        config = configparser.ConfigParser()
        config.read(project_root / 'config' / 'config.ini')
        
        # Get connection parameters
        db_params = {
            'host': config['DATABASE']['DB_HOST'],
            'port': config['DATABASE']['DB_PORT'],
            'dbname': config['DATABASE']['DB_NAME'],
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        # Log connection details (mask password)
        masked_params = db_params.copy()
        masked_params['password'] = '****'
        logger.info(f"Connection Parameters: {masked_params}")
        
        # Attempt connection
        with psycopg2.connect(**db_params) as conn:
            logger.info("Database connection successful!")
            
            # Create cursor and run test query
            with conn.cursor() as cur:
                # Get server version
                cur.execute("SELECT version();")
                version = cur.fetchone()
                logger.info(f"PostgreSQL Server Version: {version[0]}")
                
                # Test data retrieval
                cur.execute("SELECT COUNT(*) FROM candles_1h WHERE pair = 'BTC-USDT';")
                count = cur.fetchone()[0]
                logger.info(f"Total records for BTC-USDT: {count}")
                
                # Additional diagnostic query
                cur.execute("""
                    SELECT 
                        MIN(timestamp_utc) as earliest_timestamp, 
                        MAX(timestamp_utc) as latest_timestamp,
                        COUNT(DISTINCT timestamp_utc) as unique_timestamps
                    FROM candles_1h 
                    WHERE pair = 'BTC-USDT';
                """)
                diag_info = cur.fetchone()
                logger.info(f"BTC-USDT Timestamp Diagnostics:")
                logger.info(f"  Earliest Timestamp: {diag_info[0]}")
                logger.info(f"  Latest Timestamp:   {diag_info[1]}")
                logger.info(f"  Unique Timestamps: {diag_info[2]}")
                
    except psycopg2.Error as e:
        logger.error(f"Database Connection Error: {e}")
        logger.error(f"Error Code: {e.pgcode}")
        logger.error(f"Error Details: {e.pgerror}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False
    
    return True

# Pandas-based data retrieval test
def test_pandas_retrieval():
    """Test data retrieval using pandas"""
    try:
        import sqlalchemy
        from sqlalchemy import create_engine
        
        # Create SQLAlchemy engine
        db_params = {
            'host': '192.168.144.1',  # Explicitly use the Windows host IP
            'port': '5432',
            'dbname': 'okxsignal',
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        connection_string = (
            f"postgresql://{db_params['user']}:{db_params['password']}@"
            f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
        engine = create_engine(connection_string)
        
        # Test query with pandas
        query = "SELECT * FROM candles_1h WHERE pair = 'BTC-USDT' LIMIT 5;"
        df = pd.read_sql(query, engine)
        
        logger.info("Pandas DataFrame Retrieval Test:")
        logger.info(f"Records retrieved: {len(df)}")
        logger.info("DataFrame Columns:")
        logger.info(df.columns.tolist())
        logger.info("\nFirst few rows:")
        logger.info(df.head().to_string())
        
    except Exception as e:
        logger.error(f"Pandas Retrieval Error: {e}", exc_info=True)
        return False
    
    return True

def main():
    """Main test function"""
    logger.info("Starting Comprehensive Database Test")
    
    # Load environment variables
    load_env_vars()
    
    # Run tests
    db_connection_result = test_database_connection()
    pandas_retrieval_result = test_pandas_retrieval()
    
    # Final summary
    logger.info("\n--- TEST SUMMARY ---")
    logger.info(f"Database Connection: {'✓ PASSED' if db_connection_result else '✗ FAILED'}")
    logger.info(f"Pandas Data Retrieval: {'✓ PASSED' if pandas_retrieval_result else '✗ FAILED'}")

if __name__ == '__main__':
    main()