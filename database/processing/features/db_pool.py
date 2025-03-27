# database/processing/features/db_pool.py
#!/usr/bin/env python3
"""
Database connection pool utilities
"""

import logging
import threading
import time
import psycopg2
from psycopg2 import pool
import psycopg2.extras

# Thread-local storage for connections
THREAD_LOCAL = threading.local()

# Global connection pool
CONNECTION_POOL = None
MAX_RETRIES = 5
RETRY_BACKOFF = 1.5  # Seconds

def initialize_pool(config_manager, min_connections=3, max_connections=None):
    """
    Initialize the database connection pool
    
    Args:
        config_manager: Configuration manager with DB settings
        min_connections: Minimum number of connections to keep in the pool
        max_connections: Maximum number of connections allowed (default: CPU count * 4)
        
    Returns:
        Connection pool object
    """
    global CONNECTION_POOL
    
    # Only initialize once
    if CONNECTION_POOL is not None:
        return CONNECTION_POOL
    
    # Get database parameters from config
    db_params = config_manager.get_db_params()
    
    # Add extra parameters for performance
    db_params.update({
        'application_name': 'feature_compute',
        'client_encoding': 'UTF8',
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5
    })
    
    # Automatically determine max connections if not specified
    if max_connections is None:
        import os
        import psutil
        
        # Get available CPU cores and memory
        cpu_count = os.cpu_count() or 8
        
        # Start with CPU count * 1.5 (plus overhead) for worker threads
        suggested_max = int(cpu_count * 1.5) + 5
        
        # Check if DB has connection limits
        try:
            # Create a temporary connection to check DB settings
            temp_conn = psycopg2.connect(**db_params)
            cursor = temp_conn.cursor()
            
            # Get PostgreSQL max_connections setting
            cursor.execute("SHOW max_connections")
            pg_max_conn = int(cursor.fetchone()[0])
            
            # We should use at most 50% of available connections
            db_max = max(3, int(pg_max_conn * 0.5))
            
            # Use the smaller of the two limits
            max_connections = min(suggested_max, db_max)
            
            cursor.close()
            temp_conn.close()
        except Exception as e:
            logging.warning(f"Could not determine optimal connection pool size: {e}")
            # Use a conservative default
            max_connections = suggested_max
    
    # Ensure min_connections is at least 3
    min_connections = max(3, min_connections)
    
    # Ensure max_connections is at least min_connections + 1
    max_connections = max(min_connections + 1, max_connections)
    
    # Create the connection pool
    try:
        CONNECTION_POOL = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            **db_params
        )
        
        logging.info(f"Database connection pool initialized with {min_connections}-{max_connections} connections")
        return CONNECTION_POOL
    except Exception as e:
        logging.error(f"Failed to initialize connection pool: {e}")
        raise

def get_connection():
    """
    Get a connection from the pool with retry logic
    
    Returns:
        Database connection
    """
    if CONNECTION_POOL is None:
        raise ValueError("Connection pool not initialized")
    
    retry_count = 0
    retry_delay = RETRY_BACKOFF
    
    while retry_count < MAX_RETRIES:
        try:
            conn = CONNECTION_POOL.getconn()
            return conn
        except psycopg2.pool.PoolError as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logging.error(f"Error getting connection from pool: {e}")
                raise
            
            # Log and retry with exponential backoff
            logging.warning(f"Connection pool busy, retrying in {retry_delay:.2f}s ({retry_count}/{MAX_RETRIES})")
            time.sleep(retry_delay)
            retry_delay *= RETRY_BACKOFF
    
    raise RuntimeError("Failed to get database connection after retries")

def return_connection(conn):
    """
    Return a connection to the pool
    
    Args:
        conn: Connection to return
    """
    if conn and CONNECTION_POOL:
        try:
            # Make sure we're not in a transaction
            if conn.status != psycopg2.extensions.STATUS_READY:
                conn.rollback()
                
            CONNECTION_POOL.putconn(conn)
        except Exception as e:
            logging.warning(f"Error returning connection to pool: {e}")
            try:
                conn.close()
            except:
                pass

def close_all_connections():
    """Close all connections in the pool"""
    global CONNECTION_POOL
    
    if CONNECTION_POOL:
        CONNECTION_POOL.closeall()
        CONNECTION_POOL = None
        logging.info("All connections in the pool have been closed")

# Thread-local connection management
def get_thread_connection():
    """
    Get or create a connection for the current thread
    
    Returns:
        Database connection
    """
    if not hasattr(THREAD_LOCAL, 'connection'):
        THREAD_LOCAL.connection = get_connection()
    
    return THREAD_LOCAL.connection

def close_thread_connection():
    """Close the connection for the current thread"""
    if hasattr(THREAD_LOCAL, 'connection'):
        return_connection(THREAD_LOCAL.connection)
        del THREAD_LOCAL.connection

def get_db_connection():
    """
    Get a regular database connection from the pool
    
    Returns:
        Database connection
    """
    return get_connection()

def get_connection():
    """
    Get a connection from the pool with retry logic
    
    Returns:
        Database connection
    """
    if CONNECTION_POOL is None:
        raise ValueError("Connection pool not initialized")
    
    retry_count = 0
    retry_delay = RETRY_BACKOFF
    
    while retry_count < MAX_RETRIES:
        try:
            conn = CONNECTION_POOL.getconn()
            return conn
        except psycopg2.pool.PoolError as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logging.error(f"Error getting connection from pool: {e}")
                raise
            
            # Log and retry with exponential backoff
            logging.warning(f"Connection pool busy, retrying in {retry_delay:.2f}s ({retry_count}/{MAX_RETRIES})")
            time.sleep(retry_delay)
            retry_delay *= RETRY_BACKOFF
    
    raise RuntimeError("Failed to get database connection after retries")
