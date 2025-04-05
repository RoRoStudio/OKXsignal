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

# Connection cache for better reuse
CONNECTION_CACHE = {}
CACHE_LOCK = threading.Lock()

def initialize_pool(config_or_params, min_connections=3, max_connections=None):
    """
    Initialize the database connection pool
    
    Args:
        config_or_params: Either a ConfigManager object or a dictionary with DB parameters
        min_connections: Minimum number of connections to keep in the pool
        max_connections: Maximum number of connections allowed (default: CPU count * 4)
        
    Returns:
        Connection pool object
    """
    global CONNECTION_POOL
    
    # Only initialize once
    if CONNECTION_POOL is not None:
        return CONNECTION_POOL
        
    # Check if we received a config manager or direct params
    if hasattr(config_or_params, 'get_db_params'):
        # It's a config manager
        db_params = config_or_params.get_db_params()
    else:
        # It's already a dictionary with parameters
        db_params = config_or_params
    
    # Add extra parameters for performance
    db_params.update({
        'application_name': 'feature_compute',
        'client_encoding': 'UTF8',
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5,
        'options': "-c statement_timeout=300000 -c timezone=UTC"  # 5 minute timeout
    })
    
    # Automatically determine max connections if not specified
    if max_connections is None:
        import os
        import psutil
        
        # Get available CPU cores and memory
        cpu_count = os.cpu_count() or 8
        
        # Use CPU count * 4 for worker threads plus overhead
        suggested_max = cpu_count * 4 + 10
        
        # Check if DB has connection limits
        try:
            # Create a temporary connection to check DB settings
            temp_conn = psycopg2.connect(**db_params)
            cursor = temp_conn.cursor()
            
            # Get PostgreSQL max_connections setting
            cursor.execute("SHOW max_connections")
            pg_max_conn = int(cursor.fetchone()[0])
            
            # We should use at most 75% of available connections
            db_max = max(5, int(pg_max_conn * 0.75))
            
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
        
        # Pre-warm the connection pool
        connections = []
        for _ in range(min_connections):
            try:
                conn = CONNECTION_POOL.getconn()
                connections.append(conn)
                
                # Optimize database parameters and setup schema improvements
                cursor = conn.cursor()
                
                # Optimize work memory for better query performance
                cursor.execute("SET work_mem = '128MB'")
                
                # Create/update indices for better query performance if missing
                cursor.execute("SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'candles_1h_pair_timestamp_utc_idx'")
                if cursor.fetchone()[0] == 0:
                    logging.info("Creating optimized indices for faster queries")
                    # Primary lookup index
                    cursor.execute("CREATE INDEX IF NOT EXISTS candles_1h_pair_timestamp_utc_idx ON candles_1h(pair, timestamp_utc)")
                    # Index for time range queries
                    cursor.execute("CREATE INDEX IF NOT EXISTS candles_1h_timestamp_utc_idx ON candles_1h(timestamp_utc)")
                    # Supporting index for cross-pair features
                    cursor.execute("CREATE INDEX IF NOT EXISTS candles_1h_timestamp_utc_pair_idx ON candles_1h(timestamp_utc, pair)")
                    
                    # Update table statistics
                    cursor.execute("ANALYZE candles_1h")
                    
                    # Optimize table settings
                    cursor.execute("""
                        ALTER TABLE candles_1h SET (
                            autovacuum_vacuum_scale_factor = 0.05,
                            autovacuum_analyze_scale_factor = 0.02,
                            fillfactor = 90
                        )
                    """)
                    conn.commit()
                
                cursor.close()
            except Exception as e:
                logging.warning(f"Error during pool warmup: {e}")
        
        # Return connections to the pool
        for conn in connections:
            CONNECTION_POOL.putconn(conn)
            
        logging.info(f"Connection pool pre-warmed with {len(connections)} connections")
        
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
    
    # Use connection cache when possible
    thread_id = threading.current_thread().ident
    if thread_id in CONNECTION_CACHE:
        with CACHE_LOCK:
            cached_conn = CONNECTION_CACHE.get(thread_id)
            if cached_conn is not None:
                try:
                    # Check if connection is still valid
                    cursor = cached_conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return cached_conn
                except Exception:
                    # Connection is no longer valid, remove from cache
                    del CONNECTION_CACHE[thread_id]
    
    # Normal pool connection acquisition with retries
    retry_count = 0
    retry_delay = RETRY_BACKOFF
    
    while retry_count < MAX_RETRIES:
        try:
            conn = CONNECTION_POOL.getconn(key=thread_id)
            
            # Add to cache
            with CACHE_LOCK:
                CONNECTION_CACHE[thread_id] = conn
                
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
            # Get the thread ID that owns this connection
            thread_id = None
            with CACHE_LOCK:
                for tid, cached_conn in CONNECTION_CACHE.items():
                    if conn is cached_conn:
                        thread_id = tid
                        break
            
            # Make sure we're not in a transaction
            if conn.status != psycopg2.extensions.STATUS_READY:
                conn.rollback()
            
            # Return connection to pool with the correct key
            CONNECTION_POOL.putconn(conn, key=thread_id)
        except Exception as e:
            logging.warning(f"Error returning connection to pool: {e}")
            try:
                conn.close()
            except:
                pass
            
            # Remove from cache if present
            with CACHE_LOCK:
                for tid, cached_conn in list(CONNECTION_CACHE.items()):
                    if conn is cached_conn:
                        del CONNECTION_CACHE[tid]
                        break

def close_all_connections():
    """Close all connections in the pool"""
    global CONNECTION_POOL
    
    # Clear the connection cache
    with CACHE_LOCK:
        CONNECTION_CACHE.clear()
    
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
    if not hasattr(THREAD_LOCAL, 'connection') or THREAD_LOCAL.connection is None:
        try:
            # Check if existing connection is valid before returning
            if hasattr(THREAD_LOCAL, 'connection'):
                try:
                    # Test if connection is still valid
                    cursor = THREAD_LOCAL.connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return THREAD_LOCAL.connection
                except Exception:
                    # Connection is broken, get a new one
                    logging.debug("Thread connection invalid, getting new connection")
                    try:
                        return_connection(THREAD_LOCAL.connection)
                    except Exception:
                        pass
            
            # Get a new connection
            THREAD_LOCAL.connection = get_connection()
        except Exception as e:
            logging.error(f"Error getting thread connection: {e}")
            # Force a new connection on next attempt
            THREAD_LOCAL.connection = None
            raise
    
    return THREAD_LOCAL.connection

def close_thread_connection():
    """Close the connection for the current thread"""
    if hasattr(THREAD_LOCAL, 'connection') and THREAD_LOCAL.connection is not None:
        try:
            return_connection(THREAD_LOCAL.connection)
        except Exception as e:
            logging.warning(f"Error returning thread connection to pool: {e}")
            try:
                THREAD_LOCAL.connection.close()
            except Exception:
                pass
        finally:
            THREAD_LOCAL.connection = None

def close_thread_connection():
    """Close the connection for the current thread"""
    if hasattr(THREAD_LOCAL, 'connection'):
        return_connection(THREAD_LOCAL.connection)
        del THREAD_LOCAL.connection

def get_db_connection():
    """
    Get a regular database connection from the pool
    
    Returns:
        Database connection wrapped in a context manager
    """
    # Use a context manager for automatic connection return
    class ConnectionContextManager:
        def __enter__(self):
            self.conn = get_connection()
            return self.conn
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            return_connection(self.conn)
            
    return ConnectionContextManager()

def get_optimized_connection_from_pool():
    """Get an optimized connection from the pool for high-performance operations"""
    conn = get_connection()
    
    try:
        # Set optimize parameters for this specific connection
        cursor = conn.cursor()
        
        # Set work_mem higher for complex operations
        cursor.execute("SET work_mem = '128MB'")
        
        # Set maintenance_work_mem higher for bulk operations
        cursor.execute("SET maintenance_work_mem = '256MB'")
        
        # Increase temp_buffers for better temp table handling
        cursor.execute("SET temp_buffers = '64MB'")
        
        # Commit the settings
        conn.commit()
        cursor.close()
    except Exception as e:
        logging.warning(f"Could not optimize connection parameters: {e}")
    
    return conn