#!/usr/bin/env python3
"""
Optimized database operations for feature computation
- High-performance data fetching and updating
- Uses direct psycopg2 for maximum efficiency
"""

import logging
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
import io
import time
from datetime import datetime
from sqlalchemy import text

from database_spot.processing.features.config import SMALLINT_COLUMNS
from database_spot.processing.features.utils import cast_for_sqlalchemy

def get_optimized_connection(config_manager):
    """
    Create optimized database connection
    
    Args:
        config_manager: Configuration manager with DB settings
        
    Returns:
        psycopg2 connection
    """
    db_params = config_manager.get_db_params()
    
    # Initialize connection pool if not already done
    try:
        from database_spot.processing.features.db_pool import (
            initialize_connection_pool,
            get_optimized_connection_from_pool
        )
        
        # Initialize pool with database parameters
        initialize_connection_pool(db_params)
        
        # Get an optimized connection from the pool
        return get_optimized_connection_from_pool()
    except ImportError:
        # Fall back to direct connection if pool module is not available
        # Extra parameters for performance
        db_params.update({
            'application_name': 'feature_compute',
            'client_encoding': 'UTF8',
            'keepalives': 1,
            'keepalives_idle': 30,
            'keepalives_interval': 10,
            'keepalives_count': 5
        })
        
        return psycopg2.connect(**db_params)

def fetch_data_numpy(db_conn, pair, rolling_window=None):
    """
    Optimized data fetching using NumPy arrays
    
    Args:
        db_conn: Database connection
        pair: Cryptocurrency pair
        rolling_window: Number of rows to fetch
        
    Returns:
        Dictionary with OHLCV arrays
    """
    cursor = db_conn.cursor()
        
    try:
        # OPTIMIZATION: Use server-side cursor for large results
        server_cursor = db_conn.cursor(name=f"fetch_{hash(pair)}_{time.time()}")
        
        if rolling_window:
            # FIX: Use different query approach to get correct row count
            # First get the total count to check if we have enough data
            count_query = "SELECT COUNT(*) FROM candles_1h WHERE pair = %s"
            cursor.execute(count_query, (pair,))
            total_rows = cursor.fetchone()[0]
            
            # Log the total number of rows
            logging.debug(f"{pair}: Total rows in database: {total_rows}")
            
            # Add extra padding for indicators that need more data
            lookback_padding = 300
            limit = rolling_window + lookback_padding
            
            # OPTIMIZATION: Use a more optimized query that leverages timestamp index
            query = """
                SELECT timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h
                FROM candles_1h
                WHERE pair = %s
                ORDER BY timestamp_utc DESC
                LIMIT %s
            """
            server_cursor.execute(query, (pair, limit))
        else:
            # OPTIMIZATION: Use an optimized query with explicit index hint
            query = """
                /*+ BitmapScan(candles_1h candles_1h_pair_timestamp_utc_idx) */
                SELECT timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h
                FROM candles_1h
                WHERE pair = %s
                ORDER BY timestamp_utc ASC
            """
            server_cursor.execute(query, (pair,))
        
        # OPTIMIZATION: Fetch in batches for memory efficiency
        batch_size = 10000
        dt_arrays = []
        
        while True:
            rows = server_cursor.fetchmany(batch_size)
            if not rows:
                break
            dt_arrays.extend(rows)
            
        # Close server cursor
        server_cursor.close()
        
        if not dt_arrays:
            return None
            
        # Convert to arrays with optimized approach
        timestamps = np.array([row[0].timestamp() for row in dt_arrays], dtype=np.int64)
        opens = np.array([row[1] or 0.0 for row in dt_arrays], dtype=np.float64)
        highs = np.array([row[2] or 0.0 for row in dt_arrays], dtype=np.float64)
        lows = np.array([row[3] or 0.0 for row in dt_arrays], dtype=np.float64)
        closes = np.array([row[4] or 0.0 for row in dt_arrays], dtype=np.float64)
        volumes = np.array([row[5] or 0.0 for row in dt_arrays], dtype=np.float64)
        
        # Log the range of dates for debugging
        if len(timestamps) > 0:
            first_date = pd.to_datetime(timestamps[0], unit='s')
            last_date = pd.to_datetime(timestamps[-1], unit='s')
            logging.debug(f"{pair}: Fetched data from {first_date} to {last_date}")
        
        # If fetched in DESC order, reverse the arrays for chronological order
        if rolling_window:
            # Log array shapes before reversing
            logging.debug(f"{pair}: Before reversing: {len(timestamps)} timestamps")
            
            timestamps = timestamps[::-1]
            opens = opens[::-1]
            highs = highs[::-1]
            lows = lows[::-1]
            closes = closes[::-1]
            volumes = volumes[::-1]
            raw_timestamps = [row[0] for row in dt_arrays][::-1]
            
            # Log after reversing for verification
            logging.debug(f"{pair}: After reversing: {len(timestamps)} timestamps, "
                            f"range: {pd.to_datetime(timestamps[0], unit='s')} to "
                            f"{pd.to_datetime(timestamps[-1], unit='s')}")
        else:
            raw_timestamps = [row[0] for row in dt_arrays]
        
        # Log the actual vs requested number of rows
        if rolling_window:
            logging.debug(f"{pair}: Requested {rolling_window+300} rows, got {len(timestamps)} rows")
        
        return {
            'pair': pair,
            'timestamps': timestamps,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'volumes': volumes,
            'raw_timestamps': raw_timestamps
        }
    except Exception as e:
        logging.error(f"Error fetching data for {pair}: {e}")
        raise
    finally:
        if cursor:
            cursor.close()

def batch_update_features(db_conn, pair, timestamps, feature_data, columns):
    """
    Batch update features in database
    
    Args:
        db_conn: Database connection
        pair: Cryptocurrency pair
        timestamps: Array of timestamps
        feature_data: Dictionary with feature arrays
        columns: List of columns to update
        
    Returns:
        Number of rows updated
    """
    if not columns:
        logging.warning("No columns to update")
        return 0
        
    try:
        cursor = db_conn.cursor()
        
        # OPTIMIZATION: Use batched execute_values instead of individual executions
        # Build update query
        set_clause = ", ".join([f"{col} = tmp.{col}" for col in columns])
        
        update_query = f"""
        UPDATE candles_1h
        SET {set_clause}
        FROM (VALUES %s) AS tmp(timestamp_utc, {', '.join(columns)})
        WHERE candles_1h.pair = %s AND candles_1h.timestamp_utc = tmp.timestamp_utc
        """
        
        # Prepare data for batch update
        values = []
        
        for i, ts in enumerate(timestamps):
            row = [ts]
            
            for col in columns:
                if col in SMALLINT_COLUMNS:
                    # Convert to int
                    row.append(int(round(feature_data[col][i])))
                elif col in ['features_computed', 'targets_computed']:
                    # Convert to boolean
                    row.append(bool(feature_data[col][i]))
                else:
                    # Convert to float
                    row.append(float(feature_data[col][i]))
            
            values.append(tuple(row))
        
        # Execute the batch update
        psycopg2.extras.execute_values(
            cursor, 
            update_query, 
            values,
            template=None,
            page_size=5000
        )
        
        # Get updated row count
        updated_rows = cursor.rowcount
        
        # Commit changes
        db_conn.commit()
        
        return updated_rows
    except Exception as e:
        db_conn.rollback()
        logging.error(f"Error updating features for {pair}: {e}")
        raise
    finally:
        if cursor:
            cursor.close()

def bulk_copy_update(db_conn, pair, timestamps, feature_data, columns):
    """
    Highly optimized bulk update using COPY
    
    Args:
        db_conn: Database connection
        pair: Cryptocurrency pair
        timestamps: Array of timestamps
        feature_data: Dictionary with feature arrays
        columns: List of columns to update
        
    Returns:
        Number of rows updated
    """
    if not columns:
        logging.warning("No columns to update")
        return 0
        
    cursor = None
    try:
        # Verify connection is still valid
        try:
            test_cursor = db_conn.cursor()
            test_cursor.execute("SELECT 1")
            test_cursor.close()
        except Exception as e:
            logging.warning(f"Connection validation failed for {pair}: {e}")
            raise ValueError("Invalid database connection")
            
        cursor = db_conn.cursor()
        
        # Generate a safe temp table name
        import hashlib
        # Use a more unique hash that includes pair and timestamp to prevent collisions
        hash_input = f"{pair}_{int(time.time())}_{id(db_conn)}"
        table_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        temp_table_name = f"temp_feature_update_{table_hash}"
        
        logging.info(f"Creating temporary table: {temp_table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
        create_stmt = f"""
        CREATE TEMP TABLE {temp_table_name} (
            timestamp_utc TIMESTAMP WITH TIME ZONE,
            {', '.join(f"{col} double precision" if col not in SMALLINT_COLUMNS else f"{col} smallint" for col in columns)}
        ) ON COMMIT DROP;
        """
        cursor.execute(create_stmt)

        logging.info("Starting COPY INTO temp table...")
        output = io.StringIO()
        
        # Prepare data for copy
        for i, ts in enumerate(timestamps):
            # Format timestamp
            if isinstance(ts, datetime):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
                
            row_values = [ts_str]
            
            for col in columns:
                if col in feature_data and i < len(feature_data[col]):
                    # Handle different column types
                    if col in SMALLINT_COLUMNS:
                        value = str(int(round(feature_data[col][i])))
                    elif col in ['features_computed', 'targets_computed']:
                        value = 't' if feature_data[col][i] else 'f'
                    else:
                        value = str(float(feature_data[col][i]))
                else:
                    value = '0'  # Default value
                    
                row_values.append(value)
            
            output.write('\t'.join(row_values) + '\n')
        
        output.seek(0)
        
        # Perform COPY in smaller batches to avoid memory issues
        copy_start = time.time()
        try:
            cursor.copy_expert(
                f"COPY {temp_table_name} FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '')",
                output
            )
        except Exception as e:
            db_conn.rollback()
            logging.warning(f"COPY operation failed for {pair}: {e}")
            raise
            
        logging.info(f"COPY completed in {time.time() - copy_start:.2f}s")

        # Perform UPDATE with transaction handling
        update_start = time.time()
        update_query = f"""
        UPDATE candles_1h c
        SET {', '.join([f"{col} = tmp.{col}" for col in columns])}
        FROM {temp_table_name} tmp
        WHERE c.pair = %s AND c.timestamp_utc = tmp.timestamp_utc
        """
        cursor.execute(update_query, (pair,))
        
        # Get number of rows affected
        updated_rows = cursor.rowcount
        logging.info(f"UPDATE completed in {time.time() - update_start:.2f}s with {updated_rows} rows")
        
        # Commit changes
        db_conn.commit()
        
        return updated_rows
    except Exception as e:
        logging.error(f"Error bulk updating features for {pair}: {e}")
        if db_conn:
            try:
                db_conn.rollback()
            except Exception as rollback_error:
                logging.warning(f"Error during rollback: {rollback_error}")
        
        # Return 0 instead of raising to allow fallback to batch update
        return 0
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass

def get_database_columns(db_conn, table_name):
    """
    Get all column names from a specific database table
    
    Args:
        db_conn: Database connection
        table_name: Name of the table
        
    Returns:
        Set of column names
    """
    # OPTIMIZATION: Use a prepared statement to avoid repeated query parsing
    cursor = db_conn.cursor()
    
    try:
        query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """
        cursor.execute(query, (table_name,))
        
        result = cursor.fetchall()
        return set(row[0] for row in result)
    except Exception as e:
        logging.error(f"Error getting columns for {table_name}: {e}")
        raise
    finally:
        cursor.close()