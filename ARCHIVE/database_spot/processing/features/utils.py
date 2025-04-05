#!/usr/bin/env python3
"""
Utility functions for feature computation
"""

import os
import gc
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd

# ---------------------------
# Database Utilities
# ---------------------------
def get_database_columns(db_engine, table_name):
    """Get all column names from a specific database table"""
    with db_engine.connect() as conn:
        query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        """
        result = pd.read_sql(query, conn)
        return set(result['column_name'].tolist())

# Define smallint columns for proper type conversion
SMALLINT_COLUMNS = {
    'performance_rank_btc_1h', 'performance_rank_eth_1h',
    'volatility_rank_1h', 'volume_rank_1h',
    'hour_of_day', 'day_of_week', 'month_of_year',
    'was_profitable_12h', 'is_weekend', 'asian_session',
    'european_session', 'american_session',
    'pattern_doji', 'pattern_engulfing', 'pattern_hammer',
    'pattern_morning_star', 'profit_target_1pct', 'profit_target_2pct'
}

# ---------------------------
# Data Type Conversion
# ---------------------------
def cast_for_sqlalchemy(col_name, val, smallint_columns=None):
    """
    Convert values to appropriate Python types for SQLAlchemy
    
    Args:
        col_name: Name of the column
        val: Value to convert
        smallint_columns: Set of column names that should be smallint
        
    Returns:
        Converted value
    """
    smallint_columns = smallint_columns or SMALLINT_COLUMNS
    
    # Handle null values first
    if val is None or pd.isna(val):
        return None

    # Special handling for boolean columns
    if col_name in ['features_computed', 'targets_computed']:
        if isinstance(val, (int, float, bool, np.bool_)):
            return bool(val)  # Ensure proper boolean type
        else:
            return True if str(val).lower() in ('true', 't', 'yes', 'y', '1') else False

    # Convert numpy types
    if isinstance(val, (np.integer)):
        val = int(val)
    elif isinstance(val, (np.floating)):
        val = float(val)
    elif isinstance(val, (np.datetime64, pd.Timestamp)):
        val = pd.to_datetime(val).to_pydatetime()
    elif isinstance(val, (np.bool_)):
        val = bool(val)

    # Special handling for smallint columns
    if col_name in smallint_columns and val is not None:
        try:
            val = int(round(float(val))) if val is not None else None
        except (ValueError, TypeError):
            logging.warning(f"Failed to convert {col_name} value {val} to int")
            val = None

    # Final type check
    if not isinstance(val, (int, float, bool, datetime, str, type(None))):
        logging.warning(f"Unhandled type {type(val)} for column {col_name}, converting to string")
        val = str(val)

    return val

# ---------------------------
# Memory Management
# ---------------------------
def clean_memory():
    """Force garbage collection to free memory"""
    gc.collect()

# ---------------------------
# DataFrame Operations
# ---------------------------
def safe_indicator_assign(df, column_name, indicator_result):
    """
    Safely assign an indicator result to a DataFrame column, handling index misalignment.
    
    Args:
        df: DataFrame to assign to
        column_name: Name of the column to create/update
        indicator_result: Result from a pandas_ta or TA-Lib calculation
    
    Returns:
        DataFrame with the indicator assigned
    """
    # Initialize with zeros
    df[column_name] = 0.0
    
    try:
        # If it's already a DataFrame column or Series
        if isinstance(indicator_result, (pd.Series, pd.DataFrame)):
            # Use DataFrame column if multi-column result
            if isinstance(indicator_result, pd.DataFrame) and indicator_result.shape[1] > 1:
                indicator_result = indicator_result.iloc[:, 0]
            
            # Align and assign
            aligned_result = indicator_result.reindex(df.index, fill_value=0)
            df[column_name] = aligned_result
        
        # Handle numpy array results
        elif isinstance(indicator_result, np.ndarray):
            if len(indicator_result) == len(df):
                df[column_name] = indicator_result
            else:
                # If lengths don't match, attempt to assign where possible
                length = min(len(df), len(indicator_result))
                df.iloc[:length, df.columns.get_loc(column_name)] = indicator_result[:length]
        
        # Handle dict-like results from pandas_ta
        elif hasattr(indicator_result, 'keys') and len(indicator_result) > 0:
            # Get the first key if the result is a dict-like object with multiple columns
            first_key = next(iter(indicator_result))
            result_series = indicator_result[first_key]
            
            # Align and assign
            aligned_result = result_series.reindex(df.index, fill_value=0)
            df[column_name] = aligned_result
    
    except Exception as e:
        logging.warning(f"Error assigning indicator {column_name}: {e}")
    
    # Ensure any NaN values are filled with zeros
    df[column_name] = df[column_name].fillna(0)
    return df

def validate_future_features(results, config=None):
    """
    Validate that future-dependent features are properly zeroed out
    for recent candles where complete future data isn't available.
    
    Args:
        results: Dictionary with computed features
        config: Configuration parameters
        
    Returns:
        Dictionary with validated features
    """
    from database_spot.processing.features.config import LABEL_PARAMS
    
    # Get parameters
    max_return_window = LABEL_PARAMS['max_return_window']
    max_drawdown_window = LABEL_PARAMS['max_drawdown_window']
    
    # Validate future returns
    for horizon_name, shift in LABEL_PARAMS['horizons'].items():
        key = f'future_return_{horizon_name}_pct'
        if key in results:
            results[key][-shift:] = 0
    
    # Validate max return/drawdown
    if 'future_max_return_24h_pct' in results:
        results['future_max_return_24h_pct'][-max_return_window:] = 0
        
    if 'future_max_drawdown_12h_pct' in results:
        results['future_max_drawdown_12h_pct'][-max_drawdown_window:] = 0
    
    # Validate dependent features
    if 'was_profitable_12h' in results:
        results['was_profitable_12h'][-LABEL_PARAMS['horizons']['12h']:] = 0
        
    if 'future_risk_adj_return_12h' in results:
        results['future_risk_adj_return_12h'][-LABEL_PARAMS['horizons']['12h']:] = 0
        
    if 'profit_target_1pct' in results:
        results['profit_target_1pct'][-max_return_window:] = 0
        
    if 'profit_target_2pct' in results:
        results['profit_target_2pct'][-max_return_window:] = 0
    
    return results

def check_gpu_available():
    """Check if CuPy is available and GPU can be used"""
    try:
        import cupy
        # Basic test to check if GPU is available
        x = cupy.array([1, 2, 3])
        y = x * 2
        cupy.cuda.Stream.null.synchronize()
        return True
    except Exception:
        return False