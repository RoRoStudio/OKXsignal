#!/usr/bin/env python3
"""
Database connection pool management
- Provides efficient connection pooling for PostgreSQL
- Optimizes database connections for high-throughput operations
"""

import logging
import threading
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor

# Global connection pool
CONNECTION_POOL = None

# Thread-local storage for thread-specific connections
THREAD_LOCAL = threading.local()

def initialize_connection_pool(db_params, min_connections=5, max_connections=20):
    """
    Initialize a PostgreSQL connection pool
    
    Args:
        db_params: Dictionary with database connection parameters
        min_connections: Minimum number of connections to keep in the pool
        max_connections: Maximum number of connections allowed in the pool
        
    Returns:
        Connection pool object
    """
    global CONNECTION_POOL
    
    try:
        if CONNECTION_POOL is None:
            # Create threaded connection pool
            CONNECTION_POOL = pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                **db_params
            )
            
            logging.info(f"Database connection pool initialized with {min_connections}-{max_connections} connections")
        return CONNECTION_POOL
    except Exception as e:
        logging.error(f"Error initializing connection pool: {e}")
        raise

def get_db_connection():
    """
    Get a database connection that can be used with 'with' statement
    
    Returns:
        Database connection with context manager support
    """
    class ConnectionContextManager:
        def __enter__(self):
            self.conn = get_connection()
            return self.conn
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            release_connection(self.conn)
            self.conn = None
            
    return ConnectionContextManager()

def get_connection():
    """
    Get a connection from the pool
    
    Returns:
        Database connection
    """
    global CONNECTION_POOL
    
    if CONNECTION_POOL is None:
        raise ValueError("Connection pool not initialized. Call initialize_connection_pool first.")
        
    try:
        conn = CONNECTION_POOL.getconn()
        logging.debug("Retrieved connection from pool")
        return conn
    except Exception as e:
        logging.error(f"Error getting connection from pool: {e}")
        raise

def release_connection(conn):
    """
    Return a connection to the pool
    
    Args:
        conn: Database connection to return
    """
    global CONNECTION_POOL
    
    if CONNECTION_POOL is None:
        logging.warning("Connection pool not initialized, cannot release connection")
        return
        
    try:
        CONNECTION_POOL.putconn(conn)
        logging.debug("Connection returned to pool")
    except Exception as e:
        logging.error(f"Error returning connection to pool: {e}")

def close_all_connections():
    """Close all connections in the pool"""
    global CONNECTION_POOL
    
    if CONNECTION_POOL is None:
        return
        
    try:
        CONNECTION_POOL.closeall()
        logging.info("All connections in the pool have been closed")
    except Exception as e:
        logging.error(f"Error closing connection pool: {e}")

def get_optimized_connection_from_pool():
    """
    Get an optimized connection from the pool with performance settings
    
    Returns:
        Optimized database connection
    """
    conn = get_connection()
    
    try:
        # Set performance-related parameters
        cursor = conn.cursor()
        
        # Increase work memory for this connection
        cursor.execute("SET work_mem = '64MB'")
        
        # Set appropriate isolation level for read-heavy operations
        cursor.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        
        # Disable synchronous commit for better performance when safe to do so
        # cursor.execute("SET synchronous_commit = off")  # Uncomment if needed
        
        cursor.close()
        
        logging.debug("Connection optimized for performance")
        return conn
    except Exception as e:
        release_connection(conn)
        logging.error(f"Error optimizing connection: {e}")
        raise

def execute_query(query, params=None, fetch_all=True, cursor_factory=None):
    """
    Execute a query using a connection from the pool
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
        fetch_all: Whether to fetch all results (True) or just one (False)
        cursor_factory: Custom cursor factory (optional)
        
    Returns:
        Query results
    """
    conn = get_connection()
    
    try:
        if cursor_factory:
            cursor = conn.cursor(cursor_factory=cursor_factory)
        else:
            cursor = conn.cursor()
            
        cursor.execute(query, params or ())
        
        if fetch_all:
            result = cursor.fetchall()
        else:
            result = cursor.fetchone()
            
        cursor.close()
        return result
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        raise
    finally:
        release_connection(conn)

def execute_query_with_dict_result(query, params=None, fetch_all=True):
    """
    Execute a query and return results as dictionaries
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
        fetch_all: Whether to fetch all results (True) or just one (False)
        
    Returns:
        Query results as dictionaries
    """
    return execute_query(query, params, fetch_all, cursor_factory=DictCursor)


# Thread-specific connection management
def get_thread_connection():
    """
    Get a connection that is specific to the current thread
    
    Returns:
        Database connection for this thread
    """
    global THREAD_LOCAL
    
    # Check if this thread already has a connection
    if not hasattr(THREAD_LOCAL, "connection") or THREAD_LOCAL.connection is None:
        # Get a new connection from the pool
        THREAD_LOCAL.connection = get_connection()
        logging.debug(f"Created new thread connection for thread {threading.get_ident()}")
    
    return THREAD_LOCAL.connection

def release_thread_connection():
    """
    Release the thread-specific connection back to the pool
    """
    global THREAD_LOCAL
    
    if hasattr(THREAD_LOCAL, "connection") and THREAD_LOCAL.connection is not None:
        release_connection(THREAD_LOCAL.connection)
        THREAD_LOCAL.connection = None
        logging.debug(f"Released thread connection for thread {threading.get_ident()}")