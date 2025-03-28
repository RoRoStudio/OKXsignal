#!/usr/bin/env python3
"""
Data Completeness Validator
- Checks for missing values (NaN or None)
- Detects gaps in the timestamp series (should be exactly 1-hour apart)
- Verifies that no future-return label is computed if future data is unavailable
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from database.validation.validation_utils import main_validator
import logging

def validate_completeness(df, pair):
    """
    Validate data completeness for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {'missing_values': {'count': 0}, 'timestamp_gaps': {'count': 0}, 'future_return_issues': {'count': 0}}
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Calculate total number of elements
    total_elements = df.shape[0] * df.shape[1]
    
    # Check for missing values (NaN or None)
    # SKIP certain columns that are known to sometimes be legitimately NULL/NaN
    # These could be feature columns that aren't computed for every pair
    columns_to_skip = [
        'performance_rank_btc_1h', 'performance_rank_eth_1h',  # These can be NULL for BTC-USDT or ETH-USDT
        'btc_corr_24h',  # Can be NULL for BTC-USDT or during start of data collection
        'prev_volume_rank'  # Can be NULL for first candle
    ]
    
    required_columns = [
        'timestamp_utc', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h'
    ]
    
    # Check primary/required columns first
    for column in required_columns:
        if column not in df.columns:
            issue_summary['missing_values']['count'] += df.shape[0]  # Entire column missing
            issues.append({
                'issue_type': 'missing_column',
                'column': column,
                'details': f"Required column {column} is missing"
            })
            continue
            
        null_count = df[column].isnull().sum()
        
        if null_count > 0:
            issue_summary['missing_values']['count'] += null_count
            
            # Record first few missing values for reporting
            null_rows = df[df[column].isnull()]
            for idx, row in null_rows.head(5).iterrows():
                issues.append({
                    'issue_type': 'missing_value',
                    'column': column,
                    'timestamp': row['timestamp_utc'] if 'timestamp_utc' in row and not pd.isna(row['timestamp_utc']) else None,
                    'details': f"Missing value in {column} (required column)"
                })
    
    # Then check other columns, skipping those in the skip list
    other_columns = [col for col in df.columns if col not in required_columns and col not in columns_to_skip]
    
    for column in other_columns:
        null_count = df[column].isnull().sum()
        
        if null_count > 0:
            issue_summary['missing_values']['count'] += null_count
            
            # Record first few missing values for reporting
            null_rows = df[df[column].isnull()]
            for idx, row in null_rows.head(5).iterrows():
                issues.append({
                    'issue_type': 'missing_value',
                    'column': column,
                    'timestamp': row['timestamp_utc'] if 'timestamp_utc' in row and not pd.isna(row['timestamp_utc']) else None,
                    'details': f"Missing value in {column}"
                })
    
    # Detect gaps in timestamp series (should be exactly 1 hour apart)
    if 'timestamp_utc' in df.columns and len(df) > 1:
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp_utc')
        
        # Calculate time differences
        df['time_diff'] = df['timestamp_utc'].diff()
        
        # Find gaps (expecting 1 hour intervals with 10% tolerance)
        expected_interval = timedelta(hours=1)
        tolerance = expected_interval * 0.1  # 10% tolerance
        
        gaps = df[df['time_diff'] > expected_interval + tolerance]
        
        if not gaps.empty:
            issue_summary['timestamp_gaps']['count'] = len(gaps)
            
            # Record gaps for reporting
            for idx, row in gaps.head(10).iterrows():
                issues.append({
                    'issue_type': 'timestamp_gap',
                    'timestamp': row['timestamp_utc'],
                    'previous_timestamp': row['timestamp_utc'] - row['time_diff'],
                    'gap_hours': row['time_diff'].total_seconds() / 3600,
                    'details': f"Gap of {row['time_diff'].total_seconds() / 3600:.2f} hours"
                })
    
    # Verify that no future-return label is computed if future data is unavailable
    future_return_columns = [col for col in df.columns if col.startswith('future_return_') and col.endswith('_pct')]
    
    for col in future_return_columns:
        # Extract horizon from column name (e.g., '1h', '4h', '12h')
        horizon_parts = col.split('_')
        if len(horizon_parts) >= 3:
            horizon = horizon_parts[2]
            
            # Determine the number of hours based on the horizon
            hours_ahead = 0
            if horizon == '1h':
                hours_ahead = 1
            elif horizon == '4h':
                hours_ahead = 4
            elif horizon == '12h':
                hours_ahead = 12
            elif horizon == '1d':
                hours_ahead = 24
            elif horizon == '3d':
                hours_ahead = 72
            elif horizon == '1w':
                hours_ahead = 168
            elif horizon == '2w':
                hours_ahead = 336
            
            # Check that recent candles have zero future return (as data isn't available yet)
            if hours_ahead > 0:
                # Find the most recent candle with data
                max_timestamp = df['timestamp_utc'].max()
                cutoff_timestamp = max_timestamp - timedelta(hours=hours_ahead)
                
                # Check if any candles beyond the cutoff have non-zero future returns
                recent_candles = df[df['timestamp_utc'] > cutoff_timestamp]
                # Check for values that are non-zero and not NaN (can be NULL/NaN for recent candles)
                incorrect_future_returns = recent_candles[(recent_candles[col] != 0) & (~recent_candles[col].isnull())]
                
                if not incorrect_future_returns.empty:
                    issue_summary['future_return_issues']['count'] += len(incorrect_future_returns)
                    
                    # Record first few issues for reporting
                    for idx, row in incorrect_future_returns.head(5).iterrows():
                        issues.append({
                            'issue_type': 'future_return_issue',
                            'column': col,
                            'timestamp': row['timestamp_utc'],
                            'value': row[col],
                            'details': f"Non-zero future return for recent candle in {col}"
                        })
    
    # Calculate total issues
    total_issues = sum(category['count'] for category in issue_summary.values())
    
    # Only report overall missing values if they exceed a reasonable threshold (e.g., 5%)
    missing_pct = (issue_summary['missing_values']['count'] / total_elements) * 100 if total_elements > 0 else 0
    
    if missing_pct < 5 and issue_summary['missing_values']['count'] > 0:
        logging.info(f"{pair}: {missing_pct:.2f}% missing values - below threshold, not reporting as issue")
        # Don't count low percentage of missing values as issues
        total_issues -= issue_summary['missing_values']['count']
        issue_summary['missing_values']['count'] = 0
        # Filter out missing value issues
        issues = [issue for issue in issues if issue['issue_type'] != 'missing_value']
    
    return {
        'pair': pair,
        'status': 'completed',
        'issues_count': total_issues,
        'issue_summary': issue_summary,
        'issues': issues,
        'total_elements': total_elements,
        'missing_pct': missing_pct
    }

if __name__ == "__main__":
    main_validator(validate_completeness, "Data Completeness Validator", 
                  "Validates data completeness, checks for missing values, and detects timestamp gaps")