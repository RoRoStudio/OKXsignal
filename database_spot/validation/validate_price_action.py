#!/usr/bin/env python3
"""
Price Action Validator
- Validates: candle_body_size = |close - open|
- Validates: upper_shadow, lower_shadow, relative_close_position
- Confirms expected range bounds (e.g., 0-1 for ratios)
"""

import pandas as pd
import numpy as np
from database_spot.validation.validation_utils import main_validator

def validate_price_action(df, pair):
    """
    Validate price action features for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {
        'body_size_issues': {'count': 0},
        'upper_shadow_issues': {'count': 0},
        'lower_shadow_issues': {'count': 0},
        'rel_position_issues': {'count': 0},
        'range_violation': {'count': 0}
    }
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Check if required columns exist
    required_columns = ['open_1h', 'high_1h', 'low_1h', 'close_1h', 
                        'candle_body_size', 'upper_shadow', 'lower_shadow', 
                        'relative_close_position']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return {
            'pair': pair,
            'status': 'missing_columns',
            'issues_count': len(missing_columns),
            'missing_columns': missing_columns
        }
    
    # Calculate expected values independently
    expected_body_size = np.abs(df['close_1h'] - df['open_1h'])
    expected_upper_shadow = df.apply(
        lambda row: row['high_1h'] - max(row['open_1h'], row['close_1h']), 
        axis=1
    )
    expected_lower_shadow = df.apply(
        lambda row: min(row['open_1h'], row['close_1h']) - row['low_1h'], 
        axis=1
    )
    
    # Calculate expected relative close position
    # Should be (close - low) / (high - low) if high != low, else 0.5
    expected_rel_position = df.apply(
        lambda row: (row['close_1h'] - row['low_1h']) / (row['high_1h'] - row['low_1h']) 
                    if (row['high_1h'] - row['low_1h']) > 0 else 0.5,
        axis=1
    )
    
    # Validate candle_body_size
    # Allow for small floating-point differences (1e-8)
    body_size_diff = np.abs(df['candle_body_size'] - expected_body_size)
    body_size_issues = df[body_size_diff > 1e-8]
    
    if not body_size_issues.empty:
        issue_summary['body_size_issues']['count'] = len(body_size_issues)
        
        # Record first few issues for reporting
        for idx, row in body_size_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'body_size_issue',
                'timestamp': row['timestamp_utc'],
                'expected': expected_body_size.loc[idx],
                'actual': row['candle_body_size'],
                'diff': body_size_diff.loc[idx],
                'details': f"Incorrect candle body size"
            })
    
    # Validate upper_shadow
    upper_shadow_diff = np.abs(df['upper_shadow'] - expected_upper_shadow)
    upper_shadow_issues = df[upper_shadow_diff > 1e-8]
    
    if not upper_shadow_issues.empty:
        issue_summary['upper_shadow_issues']['count'] = len(upper_shadow_issues)
        
        # Record first few issues for reporting
        for idx, row in upper_shadow_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'upper_shadow_issue',
                'timestamp': row['timestamp_utc'],
                'expected': expected_upper_shadow.loc[idx],
                'actual': row['upper_shadow'],
                'diff': upper_shadow_diff.loc[idx],
                'details': f"Incorrect upper shadow"
            })
    
    # Validate lower_shadow
    lower_shadow_diff = np.abs(df['lower_shadow'] - expected_lower_shadow)
    lower_shadow_issues = df[lower_shadow_diff > 1e-8]
    
    if not lower_shadow_issues.empty:
        issue_summary['lower_shadow_issues']['count'] = len(lower_shadow_issues)
        
        # Record first few issues for reporting
        for idx, row in lower_shadow_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'lower_shadow_issue',
                'timestamp': row['timestamp_utc'],
                'expected': expected_lower_shadow.loc[idx],
                'actual': row['lower_shadow'],
                'diff': lower_shadow_diff.loc[idx],
                'details': f"Incorrect lower shadow"
            })
    
    # Validate relative_close_position
    rel_pos_diff = np.abs(df['relative_close_position'] - expected_rel_position)
    rel_pos_issues = df[rel_pos_diff > 1e-8]
    
    if not rel_pos_issues.empty:
        issue_summary['rel_position_issues']['count'] = len(rel_pos_issues)
        
        # Record first few issues for reporting
        for idx, row in rel_pos_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'rel_position_issue',
                'timestamp': row['timestamp_utc'],
                'expected': expected_rel_position.loc[idx],
                'actual': row['relative_close_position'],
                'diff': rel_pos_diff.loc[idx],
                'details': f"Incorrect relative close position"
            })
    
    # Confirm expected range bounds (e.g., 0-1 for relative_close_position)
    rel_pos_range_issues = df[(df['relative_close_position'] < 0) | (df['relative_close_position'] > 1)]
    
    if not rel_pos_range_issues.empty:
        issue_summary['range_violation']['count'] += len(rel_pos_range_issues)
        
        # Record first few issues for reporting
        for idx, row in rel_pos_range_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'range_violation',
                'column': 'relative_close_position',
                'timestamp': row['timestamp_utc'],
                'value': row['relative_close_position'],
                'details': f"relative_close_position outside [0,1] range: {row['relative_close_position']}"
            })
    
    # Check for negative shadows (should not happen if OHLC data is consistent)
    negative_upper_shadow = df[df['upper_shadow'] < 0]
    negative_lower_shadow = df[df['lower_shadow'] < 0]
    
    if not negative_upper_shadow.empty:
        issue_summary['range_violation']['count'] += len(negative_upper_shadow)
        
        # Record first few issues for reporting
        for idx, row in negative_upper_shadow.head(5).iterrows():
            issues.append({
                'issue_type': 'range_violation',
                'column': 'upper_shadow',
                'timestamp': row['timestamp_utc'],
                'value': row['upper_shadow'],
                'details': f"Negative upper shadow: {row['upper_shadow']}"
            })
    
    if not negative_lower_shadow.empty:
        issue_summary['range_violation']['count'] += len(negative_lower_shadow)
        
        # Record first few issues for reporting
        for idx, row in negative_lower_shadow.head(5).iterrows():
            issues.append({
                'issue_type': 'range_violation',
                'column': 'lower_shadow',
                'timestamp': row['timestamp_utc'],
                'value': row['lower_shadow'],
                'details': f"Negative lower shadow: {row['lower_shadow']}"
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
    main_validator(validate_price_action, "Price Action Validator", 
                  "Validates price action features: candle body size, shadows, and relative positioning")