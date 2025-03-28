#!/usr/bin/env python3
"""
Cross-Pair Features Validator
- Confirms: volume_rank, volatility_rank, performance_rank_btc/eth, btc_corr_24h
- Ensures BTC correlation ≈ 1 for BTC-USDT
- Checks percentile rank range (0-100)
"""

import pandas as pd
import numpy as np
from database.validation.validation_utils import main_validator

def validate_cross_pair_features(df, pair):
    """
    Validate cross-pair features for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {
        'rank_range_issues': {'count': 0},
        'btc_correlation_issues': {'count': 0}
    }
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # List of cross-pair features to validate
    cross_pair_features = [
        'volume_rank_1h', 'volatility_rank_1h', 
        'performance_rank_btc_1h', 'performance_rank_eth_1h',
        'btc_corr_24h'
    ]
    
    # Check which features are present in the dataframe
    present_features = [col for col in cross_pair_features if col in df.columns]
    
    if not present_features:
        return {
            'pair': pair,
            'status': 'no_cross_pair_features',
            'issues_count': 0
        }
    
    # Check rank features are within valid range (0-100)
    rank_features = [
        'volume_rank_1h', 'volatility_rank_1h', 
        'performance_rank_btc_1h', 'performance_rank_eth_1h'
    ]
    
    for feature in [f for f in rank_features if f in present_features]:
        invalid_ranks = df[(df[feature] < 0) | (df[feature] > 100)]
        
        if not invalid_ranks.empty:
            issue_summary['rank_range_issues']['count'] += len(invalid_ranks)
            
            # Record first few issues for reporting
            for idx, row in invalid_ranks.head(5).iterrows():
                issues.append({
                    'issue_type': 'rank_range_issue',
                    'feature': feature,
                    'timestamp': row['timestamp_utc'],
                    'value': float(row[feature]),
                    'details': f"{feature} out of valid range (0-100)"
                })
    
    # Check BTC correlation for BTC-USDT
    if 'btc_corr_24h' in present_features and pair == 'BTC-USDT':
        # BTC's correlation with itself should be approximately 1
        # Allow for small numerical precision issues
        invalid_corr = df[~((df['btc_corr_24h'] > 0.95) | (df['btc_corr_24h'] == 0))]
        
        if not invalid_corr.empty:
            issue_summary['btc_correlation_issues']['count'] += len(invalid_corr)
            
            # Record first few issues for reporting
            for idx, row in invalid_corr.head(5).iterrows():
                issues.append({
                    'issue_type': 'btc_correlation_issue',
                    'timestamp': row['timestamp_utc'],
                    'value': float(row['btc_corr_24h']),
                    'details': f"BTC-USDT correlation with itself should be ~1, got {row['btc_corr_24h']}"
                })
    
    # Check correlation range for any pair (-1 to 1)
    if 'btc_corr_24h' in present_features:
        invalid_corr_range = df[(df['btc_corr_24h'] < -1) | (df['btc_corr_24h'] > 1)]
        
        if not invalid_corr_range.empty:
            issue_summary['btc_correlation_issues']['count'] += len(invalid_corr_range)
            
            # Record first few issues for reporting
            for idx, row in invalid_corr_range.head(5).iterrows():
                issues.append({
                    'issue_type': 'correlation_range_issue',
                    'timestamp': row['timestamp_utc'],
                    'value': float(row['btc_corr_24h']),
                    'details': f"BTC correlation out of valid range (-1 to 1)"
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
    main_validator(validate_cross_pair_features, "Cross-Pair Features Validator", 
                  "Validates cross-pair features like volume rank, volatility rank, and BTC correlation")