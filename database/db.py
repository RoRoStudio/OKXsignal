import psycopg2
import os
import time
import logging
from dotenv import load_dotenv
from config.config_loader import load_config

logger = logging.getLogger("compute")

# Load environment variables
credentials_path = os.path.join(os.path.dirname(__file__), "..", "config", "credentials.env")
if os.path.exists(credentials_path):
    load_dotenv(credentials_path)
    logger.info(f"Loaded credentials from {credentials_path}")
else:
    logger.warning(f"Credentials file not found: {credentials_path}")

# Try to load configuration settings
try:
    config = load_config()
    DB_HOST = config.get("DB_HOST", os.getenv("DB_HOST", "localhost"))
    DB_PORT = config.get("DB_PORT", os.getenv("DB_PORT", "5432"))
    DB_NAME = config.get("DB_NAME", os.getenv("DB_NAME", "okxsignal"))
except Exception as e:
    logger.warning(f"Error loading config: {e}. Using environment variables or defaults.")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "okxsignal")

# Always get these from environment
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Global connection pool flag and object
POOL_AVAILABLE = False
connection_pool = None

# Check if connection pooling is available
try:
    if hasattr(psycopg2, 'pool'):
        POOL_AVAILABLE = True
        logger.info("Connection pooling is available")
    else:
        logger.warning("psycopg2.pool not available. Install psycopg2-binary for connection pooling support.")
except Exception as e:
    logger.warning(f"Error checking for connection pooling: {e}")

def initialize_connection_pool(min_conn=3, max_conn=10):
    """Initialize a connection pool for better performance if available"""
    global connection_pool, POOL_AVAILABLE
    
    if not POOL_AVAILABLE:
        logger.warning("Connection pooling not available. Skipping pool initialization.")
        return False
        
    if connection_pool is not None:
        logger.warning("Connection pool already initialized")
        return True
        
    try:
        logger.info(f"Initializing connection pool with {min_conn}-{max_conn} connections")
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            application_name='okxsignal',
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
        
        logger.info(f"Connection pool initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        connection_pool = None
        POOL_AVAILABLE = False  # Disable pool on error
        return False

def get_connection():
    """Establishes and returns a PostgreSQL database connection."""
    global connection_pool, POOL_AVAILABLE
    
    # Try to use connection pool if available
    if POOL_AVAILABLE and connection_pool is not None:
        try:
            conn = connection_pool.getconn()
            logger.debug("Got connection from pool")
            return conn
        except Exception as e:
            logger.warning(f"Error getting connection from pool: {e}. Falling back to direct connection.")
            POOL_AVAILABLE = False  # Disable pool on error
    
    # Fall back to direct connection
    logger.debug("Using direct connection")
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def return_connection(conn):
    """Return a connection to the pool if it came from the pool"""
    global connection_pool, POOL_AVAILABLE
    
    if POOL_AVAILABLE and connection_pool is not None:
        try:
            connection_pool.putconn(conn)
            logger.debug("Returned connection to pool")
            return True
        except Exception as e:
            logger.warning(f"Error returning connection to pool: {e}")
            return False
    
    # If not using pool, close the connection
    conn.close()
    return False

def close_all_connections():
    """Close all connections in the pool"""
    global connection_pool, POOL_AVAILABLE
    
    if POOL_AVAILABLE and connection_pool is not None:
        try:
            connection_pool.closeall()
            logger.info("All connections in the pool closed")
            return True
        except Exception as e:
            logger.error(f"Error closing all connections: {e}")
            return False
    
    return False

def fetch_data(query, params=None):
    """Fetches data from PostgreSQL and returns results as a list of dictionaries."""
    conn = None
    cursor = None
    results = []
    try:
        conn = get_connection()
        
        # Import extras module here to handle case where it might not exist
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except AttributeError:
            # Fall back to regular cursor if extras not available
            cursor = conn.cursor()
            logger.warning("psycopg2.extras not available, using regular cursor")
            
        cursor.execute(query, params or ())
        results = cursor.fetchall()
    except Exception as e:
        logger.error(f"Database fetch error: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            if not return_connection(conn):
                conn.close()
    return results

def execute_query(query, params=None):
    """Executes a query (INSERT, UPDATE, DELETE) on PostgreSQL."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        conn.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        if conn is not None:
            conn.rollback()
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            if not return_connection(conn):
                conn.close()

def execute_copy_update(temp_table_name, column_names, values, update_query):
    """
    Performs a high-speed COPY into a temp table and executes an UPDATE ... FROM ... join.
    """
    import io
    
    conn = None
    cursor = None
    
    try:
        start_all = time.time()
        
        conn = get_connection()
        cursor = conn.cursor()

        logger.info(f"Creating temporary table: {temp_table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
        create_stmt = f"""
        CREATE TEMP TABLE {temp_table_name} (
            {', '.join(f"{col} double precision" if col != 'id' else 'id bigint' for col in column_names)}
        ) ON COMMIT DROP;
        """
        cursor.execute(create_stmt)

        logger.info("Starting COPY INTO temp table...")
        output = io.StringIO()
        for row in values:
            output.write("\t".join("" if v is None else str(v) for v in row) + "\n")
        output.seek(0)

        copy_start = time.time()
        cursor.copy_from(output, temp_table_name, sep="\t", null="")
        logger.info(f"COPY completed in {time.time() - copy_start:.2f}s")

        logger.info("Running UPDATE FROM temp table...")
        update_start = time.time()
        cursor.execute(update_query.format(temp_table=temp_table_name))
        logger.info(f"UPDATE completed in {time.time() - update_start:.2f}s")

        conn.commit()
        logger.info(f"Total COPY + UPDATE time: {time.time() - start_all:.2f}s")
        
        return True
    except Exception as e:
        logger.error(f"Error during COPY+UPDATE: {e}")
        if conn is not None:
            conn.rollback()
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            if not return_connection(conn):
                conn.close()

# Try to initialize connection pool at module load time, but continue if it fails
try:
    initialize_connection_pool()
except Exception as e:
    logger.warning(f"Could not initialize connection pool at startup: {e}")