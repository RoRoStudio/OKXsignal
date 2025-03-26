#!/usr/bin/env python3
"""
Database connection pool management
- Provides thread-safe connection pooling
- Avoids circular imports by isolating connection management
"""

import logging
import threading
import psycopg2
import time
from psycopg2 import pool
from contextlib import contextmanager

# Thread-local storage
thread_local = threading.local()

# Global connection pool
connection_pool = None
max_pool_connections = 0

def initialize_connection_pool(db_params, min_connections=2, max_connections=10):
    """
    Initialize the global connection pool
    
    Args:
        db_params: Dictionary of database connection parameters
        min_connections: Minimum number of connections in the pool
        max_connections: Maximum number of connections in the pool
    
    Returns:
        Initialized connection pool
    """
    global connection_pool, max_pool_connections
    
    # Add performance parameters
    db_params.update({
        'application_name': 'feature_compute',
        'client_encoding': 'UTF8',
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5
    })
    
    # Create connection pool
    connection_pool = pool.ThreadedConnectionPool(
        minconn=min_connections,
        maxconn=max_connections,
        **db_params
    )
    
    max_pool_connections = max_connections
    
    logging.info(f"Initialized database connection pool with min={min_connections}, max={max_connections} connections")
    return connection_pool

@contextmanager
def get_db_connection():
    """
    Context manager for getting a connection from the pool
    and ensuring it's returned properly
    
    Usage:
        with get_db_connection() as conn:
            # Use conn here
    
    Returns:
        Database connection from the pool
    """
    conn = None
    try:
        if connection_pool is None:
            raise ValueError("Connection pool not initialized. Call initialize_connection_pool first.")
            
        # Get connection from pool
        conn = connection_pool.getconn()
        
        # Set autocommit false for better control
        conn.autocommit = False
        
        # Yield connection to caller
        yield conn
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        raise
    finally:
        # Always return connection to pool
        if conn is not None:
            try:
                # Always rollback any uncommitted transaction before returning
                if not conn.closed:
                    conn.rollback()
                    
                connection_pool.putconn(conn)
            except Exception as e:
                logging.error(f"Error returning connection to pool: {e}")

def monitor_pool_usage(interval=30):
    """Start a background thread to monitor pool usage"""
    if connection_pool is None:
        logging.warning("Cannot monitor connection pool - not initialized")
        return
    
    def _monitor():
        while True:
            status = get_pool_status()
            if status['used_connections'] > status['max_connections'] * 0.8:
                logging.warning(f"Connection pool nearing capacity: {status}")
            time.sleep(interval)
    
    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    logging.info(f"Started connection pool monitoring thread (interval: {interval}s)")

def get_connection_with_timeout(timeout=5.0):
    """Get a connection with timeout to prevent deadlocks"""
    if connection_pool is None:
        raise ValueError("Connection pool not initialized")
    
    start_time = time.time()
    last_exc = None
    
    while time.time() - start_time < timeout:
        try:
            conn = connection_pool.getconn()
            return conn
        except psycopg2.pool.PoolError as e:
            last_exc = e
            # Log and sleep before retrying
            logging.warning(f"Pool exhausted, waiting for connection (used: {len(connection_pool._used)})")
            time.sleep(0.5)
    
    # If we get here, we timed out
    status = get_pool_status()
    logging.error(f"Timed out waiting for connection. Pool status: {status}")
    raise last_exc if last_exc else psycopg2.pool.PoolError("Timed out waiting for connection")

# Then update get_thread_connection to use the timeout version:

def get_thread_connection():
    """
    Get a thread-local database connection
    
    Returns:
        Connection from the pool, stored in thread-local storage
    """
    # Check if thread already has a connection
    if not hasattr(thread_local, 'connection'):
        try:
            # If not, get a new connection from the pool with timeout
            thread_local.connection = get_connection_with_timeout(timeout=10.0)
            thread_local.connection.autocommit = False
        except Exception as e:
            logging.error(f"Error getting connection from pool: {e}")
            raise
    
    return thread_local.connection

def release_thread_connection():
    """
    Release the thread-local connection back to the pool
    """
    if hasattr(thread_local, 'connection'):
        try:
            conn = thread_local.connection
            
            # Always rollback any uncommitted transaction
            if conn and not conn.closed:
                conn.rollback()
                
            # Return connection to pool
            if connection_pool is not None and conn is not None:
                connection_pool.putconn(conn)
                
            # Remove connection from thread-local storage
            del thread_local.connection
        except Exception as e:
            logging.error(f"Error releasing thread connection: {e}")

def get_pool_status():
    """Get current connection pool status"""
    if connection_pool is None:
        return "Not initialized"
        
    used = connection_pool._used
    pool = connection_pool._pool
    
    return {
        "used_connections": len(used) if used else 0,
        "available_connections": len(pool) if pool else 0,
        "max_connections": max_pool_connections
    }

