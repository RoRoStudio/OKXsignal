#!/usr/bin/env python3
"""
Raw OHLCV Data Validator
- Ensures that open_1h <= high_1h, low_1h <= close_1h, etc.
- Ensures all base volume values are non-negative
- Checks taker_buy_base_1h <= volume_1h
"""

import pandas as pd
import numpy as np
from database.validation.validation_utils import main_validator

def validate_raw_ohlcv(df, pair):
    """
    Validate raw OHLCV data for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {
        'high_low_issues': {'count': 0},
        'open_high_issues': {'count': 0},
        'low_close_issues': {'count': 0},
        'negative_volume': {'count': 0},
        'taker_buy_issues': {'count': 0}
    }
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Check for OHLC price relationship issues
    
    # 1. high_1h should be >= low_1h
    high_low_issues = df[df['high_1h'] < df['low_1h']]
    if not high_low_issues.empty:
        issue_summary['high_low_issues']['count'] = len(high_low_issues)
        
        # Record first few issues for reporting
        for idx, row in high_low_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'high_low_issue',
                'timestamp': row['timestamp_utc'],
                'high': row['high_1h'],
                'low': row['low_1h'],
                'details': f"High ({row['high_1h']}) < Low ({row['low_1h']})"
            })
    
    # 2. high_1h should be >= open_1h
    open_high_issues = df[df['high_1h'] < df['open_1h']]
    if not open_high_issues.empty:
        issue_summary['open_high_issues']['count'] = len(open_high_issues)
        
        # Record first few issues for reporting
        for idx, row in open_high_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'open_high_issue',
                'timestamp': row['timestamp_utc'],
                'open': row['open_1h'],
                'high': row['high_1h'],
                'details': f"High ({row['high_1h']}) < Open ({row['open_1h']})"
            })
    
    # 3. low_1h should be <= close_1h
    low_close_issues = df[df['low_1h'] > df['close_1h']]
    if not low_close_issues.empty:
        issue_summary['low_close_issues']['count'] = len(low_close_issues)
        
        # Record first few issues for reporting
        for idx, row in low_close_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'low_close_issue',
                'timestamp': row['timestamp_utc'],
                'low': row['low_1h'],
                'close': row['close_1h'],
                'details': f"Low ({row['low_1h']}) > Close ({row['close_1h']})"
            })
    
    # 4. high_1h should be >= close_1h (this check is redundant if above checks pass)
    high_close_issues = df[df['high_1h'] < df['close_1h']]
    if not high_close_issues.empty:
        for idx, row in high_close_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'high_close_issue',
                'timestamp': row['timestamp_utc'],
                'high': row['high_1h'],
                'close': row['close_1h'],
                'details': f"High ({row['high_1h']}) < Close ({row['close_1h']})"
            })
    
    # 5. low_1h should be <= open_1h (this check is redundant if above checks pass)
    low_open_issues = df[df['low_1h'] > df['open_1h']]
    if not low_open_issues.empty:
        for idx, row in low_open_issues.head(5).iterrows():
            issues.append({
                'issue_type': 'low_open_issue',
                'timestamp': row['timestamp_utc'],
                'low': row['low_1h'],
                'open': row['open_1h'],
                'details': f"Low ({row['low_1h']}) > Open ({row['open_1h']})"
            })
    
    # Check for negative volume in BASE VOLUME ONLY
    # Exclude derivative metrics that can legitimately be negative
    exclude_terms = ['change_pct', 'oscillator', 'zone', 'trend', 'slope', '_price_', 'correlation']
    base_volume_columns = ['volume_1h', 'quote_volume_1h', 'taker_buy_base_1h']
    
    for col in base_volume_columns:
        if col in df.columns:
            negative_volume = df[df[col] < 0]
            
            if not negative_volume.empty:
                issue_summary['negative_volume']['count'] += len(negative_volume)
                
                # Record first few issues for reporting
                for idx, row in negative_volume.head(5).iterrows():
                    issues.append({
                        'issue_type': 'negative_volume',
                        'column': col,
                        'timestamp': row['timestamp_utc'],
                        'value': row[col],
                        'details': f"Negative volume in {col}: {row[col]}"
                    })
    
    # Check taker_buy_base_1h <= volume_1h if both columns exist
    if 'taker_buy_base_1h' in df.columns and 'volume_1h' in df.columns:
        taker_buy_issues = df[df['taker_buy_base_1h'] > df['volume_1h']]
        
        if not taker_buy_issues.empty:
            issue_summary['taker_buy_issues']['count'] = len(taker_buy_issues)
            
            # Record first few issues for reporting
            for idx, row in taker_buy_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'taker_buy_issue',
                    'timestamp': row['timestamp_utc'],
                    'taker_buy': row['taker_buy_base_1h'],
                    'volume': row['volume_1h'],
                    'details': f"Taker buy ({row['taker_buy_base_1h']}) > Volume ({row['volume_1h']})"
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
    main_validator(validate_raw_ohlcv, "Raw OHLCV Data Validator", 
                  "Validates raw OHLCV data, ensuring price and volume relationships")