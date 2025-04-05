#!/usr/bin/env python3
"""
Labels Validator
- Recalculates: future_return_*, future_max_return_24h_pct, future_max_drawdown_12h_pct, was_profitable_12h
- Ensures no future-looking leakage (e.g., using future values for current indicators)
"""

import pandas as pd
import numpy as np
from database_spot.validation.validation_utils import main_validator

def calculate_future_return(df, column, shift):
    """Calculate future return for a specific horizon"""
    close = df['close_1h'].values
    future_return = np.zeros(len(close))
    
    # For points that have future data available
    for i in range(len(close) - shift):
        if close[i] > 0:  # Avoid division by zero
            future_return[i] = (close[i + shift] - close[i]) / close[i]
    
    return future_return

def calculate_max_future_return(df, window=25):
    """Calculate maximum future return in a window"""
    close = df['close_1h'].values
    high = df['high_1h'].values
    max_future_return = np.zeros(len(close))
    
    for i in range(len(close) - 1):
        # Limit to available future points
        end_idx = min(i + window, len(high))
        
        if i + 1 < end_idx:  # At least one future point is needed
            max_high = np.max(high[i+1:end_idx])
            
            if close[i] > 0:  # Avoid division by zero
                max_future_return[i] = (max_high - close[i]) / close[i]
    
    return max_future_return

def calculate_max_future_drawdown(df, window=13):
    """Calculate maximum future drawdown in a window"""
    close = df['close_1h'].values
    low = df['low_1h'].values
    max_future_drawdown = np.zeros(len(close))
    
    for i in range(len(close) - 1):
        # Limit to available future points
        end_idx = min(i + window, len(low))
        
        if i + 1 < end_idx:  # At least one future point is needed
            min_low = np.min(low[i+1:end_idx])
            
            if close[i] > 0:  # Avoid division by zero
                max_future_drawdown[i] = (min_low - close[i]) / close[i]
    
    return max_future_drawdown

def validate_labels(df, pair):
    """
    Validate label/target features for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {}
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Check if required OHLC columns exist
    required_columns = ['close_1h', 'high_1h', 'low_1h']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return {
            'pair': pair,
            'status': 'missing_base_columns',
            'issues_count': len(missing_columns),
            'missing_columns': missing_columns
        }
    
    # Define all possible label columns
    label_columns = [
        'future_return_1h_pct', 'future_return_4h_pct', 'future_return_12h_pct',
        'future_return_1d_pct', 'future_return_3d_pct', 'future_return_1w_pct',
        'future_return_2w_pct', 'future_max_return_24h_pct', 'future_max_drawdown_12h_pct',
        'was_profitable_12h'
    ]
    
    # Check which label columns are present
    present_labels = [col for col in label_columns if col in df.columns]
    
    if not present_labels:
        return {
            'pair': pair,
            'status': 'no_label_columns',
            'issues_count': 0
        }
    
    # Initialize issue counters for each label
    for label in present_labels:
        issue_summary[f'{label}_issues'] = {'count': 0}
    
    # Add future leakage counter
    issue_summary['future_leakage'] = {'count': 0}
    
    # Threshold for considering values as different (allowing for minor floating-point differences)
    threshold = 1e-4
    
    # Validate future returns for different horizons
    horizon_map = {
        'future_return_1h_pct': 1,
        'future_return_4h_pct': 4,
        'future_return_12h_pct': 12,
        'future_return_1d_pct': 24,
        'future_return_3d_pct': 72,
        'future_return_1w_pct': 168,
        'future_return_2w_pct': 336
    }
    
    for label, shift in [(l, horizon_map[l]) for l in present_labels if l in horizon_map]:
        # Calculate expected future return
        expected_return = calculate_future_return(df, label, shift)
        
        # Convert to Series for easier comparison
        expected_return_series = pd.Series(expected_return, index=df.index)
        
        # Calculate absolute differences
        return_diff = np.abs(df[label] - expected_return_series)
        return_issues = df[return_diff > threshold]
        
        # Count issues
        issue_count = len(return_issues)
        if issue_count > 0:
            issue_summary[f'{label}_issues']['count'] = issue_count
            
            # Record first few issues for reporting
            for idx, row in return_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'future_return_issue',
                    'label': label,
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_return_series.loc[idx]),
                    'actual': float(row[label]),
                    'diff': float(return_diff.loc[idx]),
                    'details': f"{label} calculation discrepancy"
                })
        
        # Check for future leakage: non-zero values near the end of the dataframe
        # where future data would not be available
        recent_cutoff = len(df) - shift
        if recent_cutoff > 0:
            recent_data = df.iloc[recent_cutoff:]
            # Set a small epsilon to handle floating point precision issues
            epsilon = 1e-10
            non_zero_recent = recent_data[abs(recent_data[label]) > epsilon]
            
            if not non_zero_recent.empty:
                issue_summary['future_leakage']['count'] += len(non_zero_recent)
                
                # Record first few issues for reporting
                for idx, row in non_zero_recent.head(5).iterrows():
                    issues.append({
                        'issue_type': 'future_leakage',
                        'label': label,
                        'timestamp': row['timestamp_utc'],
                        'value': float(row[label]),
                        'details': f"Future leakage detected: {label} has non-zero value without enough future data"
                    })
            
    # Validate future_max_return_24h_pct
    if 'future_max_return_24h_pct' in present_labels:
        # Calculate expected maximum future return
        expected_max_return = calculate_max_future_return(df, window=25)  # 24h = 24 candles in 1h timeframe + 1 for safety
        
        # Convert to Series for easier comparison
        expected_max_return_series = pd.Series(expected_max_return, index=df.index)
        
        # Calculate absolute differences
        max_return_diff = np.abs(df['future_max_return_24h_pct'] - expected_max_return_series)
        max_return_issues = df[max_return_diff > threshold]
        
        issue_count = len(max_return_issues)
        if issue_count > 0:
            issue_summary['future_max_return_24h_pct_issues']['count'] = issue_count
            
            # Record first few issues for reporting
            for idx, row in max_return_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'max_return_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_max_return_series.loc[idx]),
                    'actual': float(row['future_max_return_24h_pct']),
                    'diff': float(max_return_diff.loc[idx]),
                    'details': f"future_max_return_24h_pct calculation discrepancy"
                })
        
        # Check for future leakage
        recent_cutoff = len(df) - 24  # 24h
        if recent_cutoff > 0:
            recent_data = df.iloc[recent_cutoff:]
            non_zero_recent = recent_data[(recent_data['future_max_return_24h_pct'] != 0) & 
                                         (~recent_data['future_max_return_24h_pct'].isna())]
            
            if not non_zero_recent.empty:
                issue_summary['future_leakage']['count'] += len(non_zero_recent)
                
                # Record first few issues for reporting
                for idx, row in non_zero_recent.head(5).iterrows():
                    issues.append({
                        'issue_type': 'future_leakage',
                        'label': 'future_max_return_24h_pct',
                        'timestamp': row['timestamp_utc'],
                        'value': float(row['future_max_return_24h_pct']),
                        'details': f"Future leakage detected: future_max_return_24h_pct has non-zero value without enough future data"
                    })
    
    # Validate future_max_drawdown_12h_pct
    if 'future_max_drawdown_12h_pct' in present_labels:
        # Calculate expected maximum future drawdown
        expected_max_drawdown = calculate_max_future_drawdown(df, window=13)  # 12h = 12 candles in 1h timeframe + 1 for safety
        
        # Convert to Series for easier comparison
        expected_max_drawdown_series = pd.Series(expected_max_drawdown, index=df.index)
        
        # Calculate absolute differences
        max_drawdown_diff = np.abs(df['future_max_drawdown_12h_pct'] - expected_max_drawdown_series)
        max_drawdown_issues = df[max_drawdown_diff > threshold]
        
        issue_count = len(max_drawdown_issues)
        if issue_count > 0:
            issue_summary['future_max_drawdown_12h_pct_issues']['count'] = issue_count
            
            # Record first few issues for reporting
            for idx, row in max_drawdown_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'max_drawdown_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_max_drawdown_series.loc[idx]),
                    'actual': float(row['future_max_drawdown_12h_pct']),
                    'diff': float(max_drawdown_diff.loc[idx]),
                    'details': f"future_max_drawdown_12h_pct calculation discrepancy"
                })
        
        # Check for future leakage
        recent_cutoff = len(df) - 12  # 12h
        if recent_cutoff > 0:
            recent_data = df.iloc[recent_cutoff:]
            non_zero_recent = recent_data[(recent_data['future_max_drawdown_12h_pct'] != 0) &
                                         (~recent_data['future_max_drawdown_12h_pct'].isna())]
            
            if not non_zero_recent.empty:
                issue_summary['future_leakage']['count'] += len(non_zero_recent)
                
                # Record first few issues for reporting
                for idx, row in non_zero_recent.head(5).iterrows():
                    issues.append({
                        'issue_type': 'future_leakage',
                        'label': 'future_max_drawdown_12h_pct',
                        'timestamp': row['timestamp_utc'],
                        'value': float(row['future_max_drawdown_12h_pct']),
                        'details': f"Future leakage detected: future_max_drawdown_12h_pct has non-zero value without enough future data"
                    })
    
    # Validate was_profitable_12h
    if 'was_profitable_12h' in present_labels and 'future_return_12h_pct' in present_labels:
        # Expected value: 1 if future_return_12h_pct > 0, else 0
        expected_profitable = (df['future_return_12h_pct'] > 0).astype(int)
        
        # Find mismatches
        profitable_issues = df[df['was_profitable_12h'] != expected_profitable]
        
        issue_count = len(profitable_issues)
        if issue_count > 0:
            issue_summary['was_profitable_12h_issues']['count'] = issue_count
            
            # Record first few issues for reporting
            for idx, row in profitable_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'was_profitable_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': int(expected_profitable.loc[idx]),
                    'actual': int(row['was_profitable_12h']),
                    'future_return': float(row['future_return_12h_pct']),
                    'details': f"was_profitable_12h inconsistent with future_return_12h_pct"
                })
    
    # Calculate total issues
    total_issues = sum(category['count'] for category in issue_summary.values())
    
    # Calculate issue percentage based on number of candles
    issue_percentage = (total_issues / len(df)) * 100 if len(df) > 0 else 0
    
    return {
        'pair': pair,
        'status': 'completed',
        'issues_count': total_issues,
        'candles_count': len(df),
        'issue_percentage': issue_percentage,
        'issue_summary': issue_summary,
        'issues': issues
    }

if __name__ == "__main__":
    main_validator(validate_labels, "Labels Validator", 
                  "Validates future return and target labels by recalculating them from price data")