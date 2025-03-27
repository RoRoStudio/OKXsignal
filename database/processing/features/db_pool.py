#!/usr/bin/env python3
"""
Database connection pool manager
- Provides thread-safe connection management
- Handles connection pooling and timeouts
"""

import logging
import threading
import time
import psycopg2
import psycopg2.pool
from contextlib import contextmanager

# Thread-local storage for connections
thread_local = threading.local()
connection_pool = None
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

def initialize_pool(db_params, min_connections=2, max_connections=None):
    """
    Initialize the connection pool
    
    Args:
        db_params: Database connection parameters
        min_connections: Minimum number of connections
        max_connections: Maximum number of connections
    """
    global connection_pool
    
    # Default max connections to CPU count * 2.5, capped at 50
    if max_connections is None:
        import os
        cpu_count = os.cpu_count() or 4
        max_connections = min(50, int(cpu_count * 2.5))
    
    if connection_pool is not None:
        logging.warning("Connection pool already initialized, closing existing pool")
        try:
            connection_pool.closeall()
        except:
            pass
    
    try:
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            **db_params
        )
        
        logging.info(f"Initialized database connection pool with min={min_connections}, max={max_connections} connections")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize connection pool: {e}")
        return False

def get_connection_with_timeout(timeout=10.0):
    """
    Get a connection from the pool with timeout
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        Database connection
    """
    if connection_pool is None:
        raise ValueError("Connection pool not initialized")
    
    start_time = time.time()
    last_exc = None
    
    while time.time() - start_time < timeout:
        try:
            conn = connection_pool.getconn()
            # Test connection is valid
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return conn
        except psycopg2.pool.PoolError as e:
            # Pool exhausted, wait and retry
            last_exc = e
            time.sleep(0.1)
        except psycopg2.Error as e:
            # Connection invalid, close and retry
            connection_pool.putconn(conn, close=True)
            last_exc = e
            time.sleep(0.1)
    
    # Log pool status
    used = connection_pool._used
    pool_size = connection_pool._pool
    logging.error(f"Timed out waiting for connection. Pool status: {{'used_connections': {len(used)}, 'available_connections': {len(pool_size)}, 'max_connections': {connection_pool.maxconn}}}")
    
    raise last_exc if last_exc else psycopg2.pool.PoolError("Timed out waiting for connection")

def get_thread_connection():
    """
    Get a connection for the current thread
    
    Returns:
        Database connection
    """
    global thread_local
    
    # Create thread local storage if it doesn't exist
    if not hasattr(thread_local, 'connection'):
        thread_local.connection = get_connection_with_timeout(timeout=10.0)
    
    # Test connection is valid
    try:
        with thread_local.connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    except Exception:
        # Connection is not valid, get a new one
        try:
            connection_pool.putconn(thread_local.connection, close=True)
        except Exception:
            pass
        thread_local.connection = get_connection_with_timeout(timeout=10.0)
    
    return thread_local.connection

def release_thread_connection():
    """Release the thread's connection back to the pool"""
    global thread_local
    
    if hasattr(thread_local, 'connection'):
        try:
            connection_pool.putconn(thread_local.connection)
        except Exception as e:
            logging.warning(f"Error returning connection to pool: {e}")
        finally:
            delattr(thread_local, 'connection')

@contextmanager
def get_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = get_thread_connection()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            release_thread_connection()

def close_pool():
    """Close all connections in the pool"""
    global connection_pool
    
    if connection_pool is not None:
        connection_pool.closeall()
        connection_pool = None