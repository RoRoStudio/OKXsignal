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
from datetime import datetime
from sqlalchemy import text
from io import StringIO

from database.processing.features.config import SMALLINT_COLUMNS
from database.processing.features.utils import cast_for_sqlalchemy

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
        from database.processing.features.db_pool import (
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
        if rolling_window:
            # Add extra padding for indicators that need more data
            lookback_padding = 300
            limit = rolling_window + lookback_padding
            
            query = """
                SELECT timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h
                FROM candles_1h
                WHERE pair = %s
                ORDER BY timestamp_utc DESC
                LIMIT %s
            """
            cursor.execute(query, (pair, limit))
        else:
            query = """
                SELECT timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h
                FROM candles_1h
                WHERE pair = %s
                ORDER BY timestamp_utc ASC
            """
            cursor.execute(query, (pair,))
        
        # Fetch all rows at once
        rows = cursor.fetchall()
        
        if not rows:
            return None
            
        # Get raw arrays
        dt_arrays = np.array(rows, dtype=object)
        
        # Convert to appropriate types
        timestamps = np.array([row[0].timestamp() for row in rows], dtype=np.int64)
        opens = np.array([row[1] for row in rows], dtype=np.float64)
        highs = np.array([row[2] for row in rows], dtype=np.float64)
        lows = np.array([row[3] for row in rows], dtype=np.float64)
        closes = np.array([row[4] for row in rows], dtype=np.float64)
        volumes = np.array([row[5] for row in rows], dtype=np.float64)
        
        # If fetched in DESC order, reverse the arrays
        if rolling_window:
            timestamps = timestamps[::-1]
            opens = opens[::-1]
            highs = highs[::-1]
            lows = lows[::-1]
            closes = closes[::-1]
            volumes = volumes[::-1]
        
        return {
            'pair': pair,
            'timestamps': timestamps,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'volumes': volumes,
            'raw_timestamps': [row[0] for row in rows][::-1] if rolling_window else [row[0] for row in rows]
        }
    except Exception as e:
        logging.error(f"Error fetching data for {pair}: {e}")
        raise
    finally:
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
        
        # Build update query - use named parameters instead of %s
        set_clause = ", ".join([f"{col} = %(data_{col})s" for col in columns])
        
        update_query = f"""
        UPDATE candles_1h
        SET {set_clause}
        WHERE pair = %(pair)s AND timestamp_utc = %(timestamp)s
        """
        
        # Execute updates one by one
        updated_rows = 0
        
        for i, ts in enumerate(timestamps):
            params = {'pair': pair, 'timestamp': ts}
            
            # Add data parameters
            for col in columns:
                if col in SMALLINT_COLUMNS:
                    # Convert to int
                    params[f'data_{col}'] = int(round(feature_data[col][i]))
                elif col in ['features_computed', 'targets_computed']:
                    # Convert to boolean
                    params[f'data_{col}'] = bool(feature_data[col][i])
                else:
                    # Convert to float
                    params[f'data_{col}'] = float(feature_data[col][i])
            
            # Execute the query
            cursor.execute(update_query, params)
            updated_rows += cursor.rowcount
        
        # Commit changes
        db_conn.commit()
        
        return updated_rows
    except Exception as e:
        db_conn.rollback()
        logging.error(f"Error updating features for {pair}: {e}")
        raise
    finally:
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
        
    try:
        cursor = db_conn.cursor()
        
        # Create a temporary table with the same structure
        # Use absolute value of hash to avoid negative numbers in table name
        temp_table_name = f"temp_feature_update_{abs(hash(pair))}_{int(datetime.now().timestamp())}"
        
        # Determine column types
        column_defs = []
        for col in columns:
            if col in SMALLINT_COLUMNS:
                col_type = "SMALLINT"
            elif col in ['features_computed', 'targets_computed']:
                col_type = "BOOLEAN"
            else:
                col_type = "DOUBLE PRECISION"
            column_defs.append(f"{col} {col_type}")
        
        # Create temp table
        create_temp_table_query = f"""
        CREATE TEMP TABLE {temp_table_name} (
            timestamp_utc TIMESTAMP WITH TIME ZONE,
            {', '.join(column_defs)}
        ) ON COMMIT DROP
        """
        cursor.execute(create_temp_table_query)
        
        # Use COPY for bulk insertion
        copy_file = StringIO()
        
        for i, ts in enumerate(timestamps):
            # Format timestamp
            if isinstance(ts, datetime):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
                
            row_values = [ts_str]
            
            for col in columns:
                if col in feature_data:
                    if col in SMALLINT_COLUMNS:
                        # Convert to int
                        value = str(int(round(feature_data[col][i])))
                    elif col in ['features_computed', 'targets_computed']:
                        # Convert to boolean
                        value = 't' if feature_data[col][i] else 'f'
                    else:
                        # Convert to float
                        value = str(float(feature_data[col][i]))
                else:
                    value = '0'  # Default value
                
                row_values.append(value)
            
            copy_file.write('\t'.join(row_values) + '\n')
        
        copy_file.seek(0)
        
        # Execute COPY
        cursor.copy_expert(
            f"COPY {temp_table_name} FROM STDIN WITH NULL ''",
            copy_file
        )
        
        # Update from temp table
        update_query = f"""
        UPDATE candles_1h
        SET {', '.join([f"{col} = tmp.{col}" for col in columns])}
        FROM {temp_table_name} tmp
        WHERE candles_1h.pair = %s 
          AND candles_1h.timestamp_utc = tmp.timestamp_utc
        """
        cursor.execute(update_query, (pair,))
        
        # Get number of rows affected
        updated_rows = cursor.rowcount
        
        # Commit changes
        db_conn.commit()
        
        return updated_rows
    except Exception as e:
        db_conn.rollback()
        logging.error(f"Error bulk updating features for {pair}: {e}")
        raise
    finally:
        cursor.close()

def get_database_columns(db_conn, table_name):
    """
    Get all column names from a specific database table
    
    Args:
        db_conn: Database connection
        table_name: Name of the table
        
    Returns:
        Set of column names
    """
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