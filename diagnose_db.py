#!/usr/bin/env python3
"""
Diagnostic script for database connection issues
- Tests database connectivity
- Checks for active connections
- Verifies table existence and permissions
"""

import os
import sys
import time
import logging
import psycopg2
import psycopg2.extras
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Setup logging
def setup_logging(log_dir="logs", log_level="INFO"):
    """Set up logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = os.path.join(log_dir, f"db_diagnostic_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(levelname)s] %(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("db_diagnostic")

def check_db_connection(host, port, dbname, user, password):
    """Test basic database connectivity"""
    logger.info(f"Testing connection to {dbname} on {host}:{port}")
    
    try:
        # Test connection
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=10
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        logger.info(f"Connected successfully. PostgreSQL version: {version}")
        
        # Check for active connections
        cursor.execute("""
        SELECT count(*) 
        FROM pg_stat_activity 
        WHERE datname = %s
        """, (dbname,))
        active_connections = cursor.fetchone()[0]
        logger.info(f"Active connections to {dbname}: {active_connections}")
        
        # Check if this user is superuser
        cursor.execute("SELECT usesuper FROM pg_user WHERE usename = %s", (user,))
        is_superuser = cursor.fetchone()[0] if cursor.rowcount > 0 else False
        logger.info(f"User {user} is superuser: {is_superuser}")
        
        # Check table existence and permissions
        check_table_permissions(conn, user)
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False

def check_table_permissions(conn, user):
    """Check table existence and permissions"""
    cursor = conn.cursor()
    
    # Check for candles_1h table
    try:
        cursor.execute("""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_name = 'candles_1h'
        )
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            logger.info("Table candles_1h exists")
            
            # Check for data in the table
            cursor.execute("SELECT COUNT(*) FROM candles_1h")
            count = cursor.fetchone()[0]
            logger.info(f"candles_1h contains {count} rows")
            
            # Check for pairs in the table
            cursor.execute("SELECT COUNT(DISTINCT pair) FROM candles_1h")
            pair_count = cursor.fetchone()[0]
            logger.info(f"candles_1h contains {pair_count} distinct pairs")
            
            # Check table permissions
            cursor.execute("""
            SELECT grantee, privilege_type
            FROM information_schema.table_privileges
            WHERE table_name = 'candles_1h'
            ORDER BY grantee, privilege_type
            """)
            
            permissions = cursor.fetchall()
            logger.info("Permissions on candles_1h:")
            for grantee, privilege in permissions:
                logger.info(f"  {grantee}: {privilege}")
        else:
            logger.warning("Table candles_1h does not exist!")
    except Exception as e:
        logger.error(f"Error checking table permissions: {e}")
    
    # Check for user's permission on various PostgreSQL objects
    try:
        cursor.execute("""
        SELECT table_name, privilege_type
        FROM information_schema.table_privileges
        WHERE grantee = %s
        GROUP BY table_name, privilege_type
        ORDER BY table_name, privilege_type
        """, (user,))
        
        user_permissions = cursor.fetchall()
        logger.info(f"User {user} has the following permissions:")
        for table, privilege in user_permissions:
            logger.info(f"  {table}: {privilege}")
    except Exception as e:
        logger.error(f"Error checking user permissions: {e}")
    
    cursor.close()

def test_batch_operations(conn):
    """Test batch database operations"""
    cursor = conn.cursor()
    
    try:
        # Create a test table
        cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS test_batch_ops (
            id SERIAL PRIMARY KEY,
            test_name VARCHAR(100),
            test_value DOUBLE PRECISION
        )
        """)
        
        # Test batch insert
        logger.info("Testing batch operations...")
        start_time = time.time()
        
        test_data = [
            ("test1", 1.1),
            ("test2", 2.2),
            ("test3", 3.3),
            ("test4", 4.4),
            ("test5", 5.5)
        ]
        
        psycopg2.extras.execute_batch(
            cursor,
            "INSERT INTO test_batch_ops (test_name, test_value) VALUES (%s, %s)",
            test_data
        )
        
        # Test batch select
        cursor.execute("SELECT * FROM test_batch_ops")
        results = cursor.fetchall()
        
        logger.info(f"Batch operations completed in {time.time() - start_time:.4f} seconds")
        logger.info(f"Inserted and retrieved {len(results)} rows")
        
        # Test temporary table handling
        cursor.execute("""
        CREATE TEMPORARY TABLE test_temp (
            id SERIAL,
            value DOUBLE PRECISION
        ) ON COMMIT DROP
        """)
        
        for i in range(10):
            cursor.execute("INSERT INTO test_temp (value) VALUES (%s)", (i * 1.5,))
            
        cursor.execute("SELECT COUNT(*) FROM test_temp")
        temp_count = cursor.fetchone()[0]
        logger.info(f"Temporary table test successful: {temp_count} rows inserted")
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Batch operations test failed: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()

def test_connection_pool(host, port, dbname, user, password, pool_size=5):
    """Test connection pooling"""
    logger.info(f"Testing connection pool with {pool_size} connections")
    
    try:
        # Create a connection pool
        pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=pool_size,
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        
        # Get multiple connections
        connections = []
        for i in range(pool_size):
            try:
                conn = pool.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                connections.append(conn)
                logger.info(f"Pool connection {i+1} successful")
            except Exception as e:
                logger.error(f"Failed to get connection {i+1} from pool: {e}")
        
        # Return connections to the pool
        for i, conn in enumerate(connections):
            pool.putconn(conn)
            logger.info(f"Connection {i+1} returned to pool")
        
        # Close the pool
        pool.closeall()
        logger.info("Connection pool test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Connection pool test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Diagnose database connection issues')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for log files')
    parser.add_argument('--env-file', type=str,
                       help='Path to .env file with database credentials')
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_dir, args.log_level)
    logger.info("Starting database diagnostic")
    
    # Load environment variables
    if args.env_file and os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        logger.info(f"Loaded environment from {args.env_file}")
    else:
        default_env_path = os.path.join(os.path.dirname(__file__), "config", "credentials.env")
        if os.path.exists(default_env_path):
            load_dotenv(default_env_path)
            logger.info(f"Loaded environment from {default_env_path}")
        else:
            logger.warning("No environment file found. Using environment variables if set.")
    
    # Get database connection parameters
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'okxsignal')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    
    # Validate required parameters
    if not db_user or not db_password:
        logger.error("DB_USER and DB_PASSWORD environment variables must be set")
        sys.exit(1)
    
    logger.info(f"Testing connection to: {db_host}:{db_port}/{db_name}")
    
    # Test basic connection
    if not check_db_connection(db_host, db_port, db_name, db_user, db_password):
        logger.error("Basic connection test failed. Exiting.")
        sys.exit(1)
    
    # Test connection pool
    if not test_connection_pool(db_host, db_port, db_name, db_user, db_password):
        logger.error("Connection pool test failed")
    
    # Test batch operations
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        if not test_batch_operations(conn):
            logger.error("Batch operations test failed")
        conn.close()
    except Exception as e:
        logger.error(f"Failed to connect for batch operations test: {e}")
    
    logger.info("Database diagnostic completed")

if __name__ == "__main__":
    main()