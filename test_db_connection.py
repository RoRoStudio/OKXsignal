#!/usr/bin/env python3
"""
Test script for database connections and connection pooling
- Tests whether the connection pool is working correctly
- Can be used to diagnose issues with compute_features.py
"""

import os
import sys
import logging
import argparse
import time
import traceback
import threading
import psycopg2
import psycopg2.pool
from threading import Thread, local as thread_local_class
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

# Thread-local storage for connections
thread_local = thread_local_class()

# Global connection pool
connection_pool = None

# Setup logging
def setup_logging(log_dir="logs", log_level="INFO"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_connection_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(levelname)s] %(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("test_connection")

def initialize_connection_pool(db_host, db_port, db_name, db_user, db_password, 
                              min_conn=3, max_conn=10):
    """Initialize the connection pool"""
    global connection_pool
    
    try:
        # Create a ThreadedConnectionPool
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            application_name='connection_test',
            client_encoding='UTF8',
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
        
        # Test the connection pool
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        connection_pool.putconn(conn)
        
        logging.info(f"Connection pool initialized with {min_conn}-{max_conn} connections")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize connection pool: {e}")
        logging.error(traceback.format_exc())
        return False

def get_thread_connection():
    """Get a connection for the current thread from the pool"""
    global connection_pool, thread_local
    
    if connection_pool is None:
        raise RuntimeError("Connection pool not initialized")
    
    if not hasattr(thread_local, 'connection') or thread_local.connection is None:
        try:
            thread_local.connection = connection_pool.getconn()
            logging.debug(f"Thread {threading.current_thread().name} got a new connection")
        except Exception as e:
            logging.error(f"Failed to get connection from pool: {e}")
            raise
    
    return thread_local.connection

def release_thread_connection():
    """Release the connection for the current thread back to the pool"""
    global connection_pool, thread_local
    
    if connection_pool is not None and hasattr(thread_local, 'connection') and thread_local.connection is not None:
        try:
            connection_pool.putconn(thread_local.connection)
            thread_local.connection = None
            logging.debug(f"Thread {threading.current_thread().name} released its connection")
        except Exception as e:
            logging.error(f"Failed to release connection to pool: {e}")

def test_connection_in_thread(thread_id, sleep_time=2):
    """Test function to run in a thread"""
    try:
        logging.info(f"Thread {thread_id} starting")
        
        # Get a connection
        conn = get_thread_connection()
        
        # Use the connection
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candles_1h")
        result = cursor.fetchone()
        cursor.close()
        
        logging.info(f"Thread {thread_id} found {result[0]} rows in candles_1h")
        
        # Sleep to simulate work
        time.sleep(sleep_time)
        
        # Use the connection again
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pair FROM candles_1h LIMIT 5")
        pairs = cursor.fetchall()
        cursor.close()
        
        logging.info(f"Thread {thread_id} found pairs: {[p[0] for p in pairs]}")
        
        return True
    except Exception as e:
        logging.error(f"Error in thread {thread_id}: {e}")
        logging.error(traceback.format_exc())
        return False
    finally:
        # Release the connection
        release_thread_connection()
        logging.info(f"Thread {thread_id} finished")

def main():
    parser = argparse.ArgumentParser(description='Test database connections')
    parser.add_argument('--log-level', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    parser.add_argument('--log-dir', type=str, default='logs', 
                      help='Directory for log files')
    parser.add_argument('--threads', type=int, default=5, 
                      help='Number of threads to test with')
    parser.add_argument('--min-conn', type=int, default=3, 
                      help='Minimum connections in pool')
    parser.add_argument('--max-conn', type=int, default=10, 
                      help='Maximum connections in pool')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.log_level)
    logger.info(f"Starting connection pool test with {args.threads} threads")
    
    # Load environment variables from credentials file
    load_dotenv(os.path.join(os.path.dirname(__file__), "config", "credentials.env"))
    
    # Get database credentials from environment
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'okxsignal')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')
    
    logger.info(f"Connecting to database {db_name} on {db_host}:{db_port}")
    
    # Initialize the connection pool
    if not initialize_connection_pool(
        db_host, db_port, db_name, db_user, db_password,
        min_conn=args.min_conn, max_conn=args.max_conn
    ):
        logger.critical("Failed to initialize database connection pool")
        sys.exit(1)
    
    # Test with multiple threads
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(test_connection_in_thread, i): i
            for i in range(args.threads)
        }
        
        for future in as_completed(futures):
            thread_id = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"Thread {thread_id} raised exception: {e}")
    
    logger.info(f"Test completed: {success_count}/{args.threads} threads succeeded")
    
    # Close the connection pool
    global connection_pool
    if connection_pool is not None:
        connection_pool.closeall()
        logger.info("Connection pool closed")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        print(traceback.format_exc())
        sys.exit(1)