#!/usr/bin/env python3
"""
Data Sanity Validator
- Global check for any absurd values: rsi > 100, volume < 0, etc.
- Detects floating-point overflows, division by zero
- Allows legitimate negative values for oscillator metrics
"""

import pandas as pd
import numpy as np
import math
from database_spot.validation.validation_utils import main_validator

def validate_data_sanity(df, pair):
    """
    Validate data sanity for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {
        'range_violations': {'count': 0},
        'infinity_issues': {'count': 0},
        'nan_issues': {'count': 0},
        'absurd_values': {'count': 0}
    }
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Define valid ranges and constraints for different feature types
    feature_constraints = {
        # RSI and similar oscillators (0-100)
        'oscillators': {
            'min': 0, 
            'max': 100,
            'columns': ['rsi_1h', 'rsi_4h', 'rsi_1d', 'stoch_k_14', 'stoch_d_14',
                       'money_flow_index_1h']
        },
        # Correlation coefficients (-1 to 1)
        'correlations': {
            'min': -1,
            'max': 1,
            'columns': ['btc_corr_24h', 'autocorr_1']
        },
        # Percentile ranks (0-100)
        'ranks': {
            'min': 0,
            'max': 100,
            'columns': ['volume_rank_1h', 'volatility_rank_1h', 
                       'performance_rank_btc_1h', 'performance_rank_eth_1h']
        },
        # Binary indicators (0 or 1)
        'binary': {
            'allowed_values': [0, 1],
            'columns': ['pattern_doji', 'pattern_engulfing', 'pattern_hammer', 'pattern_morning_star',
                       'was_profitable_12h', 'is_weekend', 'asian_session', 'european_session', 
                       'american_session', 'profit_target_1pct', 'profit_target_2pct',
                       'volume_price_confirmation']
        },
        # Time features
        'time': {
            'hour_of_day': {'min': 0, 'max': 23},
            'day_of_week': {'min': 0, 'max': 6},
            'month_of_year': {'min': 1, 'max': 12}
        },
        # Volume and price must be non-negative
        'non_negative': {
            'min': 0,
            'columns': ['volume_1h', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 
                      'atr_1h', 'bollinger_width_1h', 'donchian_channel_width_1h',
                      'body_size', 'upper_shadow', 'lower_shadow']
        }
    }
    
    # Define metrics that can legitimately be negative 
    # These should be excluded from range checks
    legitimetely_negative_metrics = [
        # Price derivatives
        'log_return', 'gap_open', 'price_velocity', 'price_acceleration',
        # Volume change metrics
        'volume_change_pct_1h', 'volume_change_pct_4h', 'volume_change_pct_1d',
        # Oscillator metrics
        'volume_zone_oscillator', 'volume_price_trend', 'volume_oscillator',
        'macd_slope_1h', 'macd_hist_slope_1h', 'rsi_slope_1h',
        'williams_r_14', 'cci_14', 'roc_10', 'tsi', 'awesome_oscillator', 'ppo',
        'chaikin_money_flow', 'klinger_oscillator',
        # Statistical metrics
        'skewness_20', 'kurtosis_20', 'z_score_20', 'hurst_exponent',
        # Return metrics
        'future_return_1h_pct', 'future_return_4h_pct', 'future_return_12h_pct',
        'future_return_1d_pct', 'future_return_3d_pct', 'future_return_1w_pct',
        'future_return_2w_pct', 'future_max_drawdown_12h_pct', 'future_risk_adj_return_12h'
    ]
    
    # Columns to ignore for absurd value checks
    # These are metrics where large values are normal or expected
    exclude_from_absurd_checks = [
        'quote_volume_1h', 'obv_1h', 'future_risk_adj_return_12h',
        'awesome_oscillator', 'volume_price_trend'
    ]
    
    # Check for NaN values in any column
    nan_counts = df.isna().sum()
    columns_with_nan = nan_counts[nan_counts > 0]
    
    # Exclude performance rank columns which can legitimately be NaN
    # for pairs like BTC-USDT or ETH-USDT
    excluded_nan_columns = ['performance_rank_btc_1h', 'performance_rank_eth_1h', 
                          'btc_corr_24h', 'prev_volume_rank']
    
    filtered_nan_columns = {col: count for col, count in columns_with_nan.items() 
                           if col not in excluded_nan_columns}
    
    if filtered_nan_columns:
        issue_summary['nan_issues']['count'] = sum(filtered_nan_columns.values())
        
        # Record NaN issues by column
        for col, count in filtered_nan_columns.items():
            issues.append({
                'issue_type': 'nan_issue',
                'column': col,
                'count': int(count),
                'details': f"Found {count} NaN values in {col}"
            })
    
    # Check for infinity values
    inf_mask = df.replace([np.inf, -np.inf], np.nan).isna() & ~df.isna()
    inf_counts = inf_mask.sum()
    columns_with_inf = inf_counts[inf_counts > 0]
    
    if not columns_with_inf.empty:
        issue_summary['infinity_issues']['count'] = columns_with_inf.sum()
        
        # Record infinity issues by column
        for col, count in columns_with_inf.items():
            issues.append({
                'issue_type': 'infinity_issue',
                'column': col,
                'count': int(count),
                'details': f"Found {count} infinity values in {col}"
            })
    
    # Check each constraint type
    # Oscillators (0-100)
    for constraint_type, constraint_info in feature_constraints.items():
        if constraint_type == 'oscillators' or constraint_type == 'correlations' or constraint_type == 'ranks' or constraint_type == 'non_negative':
            min_val = constraint_info['min']
            max_val = constraint_info.get('max', float('inf'))
            
            for col in [c for c in constraint_info['columns'] if c in df.columns]:
                # Skip if this column can legitimately be negative
                if col in legitimetely_negative_metrics:
                    continue
                    
                # Find values outside valid range
                invalid_rows = df[(df[col] < min_val) | (df[col] > max_val)]
                
                if not invalid_rows.empty:
                    issue_summary['range_violations']['count'] += len(invalid_rows)
                    
                    # Record first few issues for reporting
                    for idx, row in invalid_rows.head(5).iterrows():
                        issues.append({
                            'issue_type': 'range_violation',
                            'constraint_type': constraint_type,
                            'column': col,
                            'timestamp': row['timestamp_utc'],
                            'value': float(row[col]),
                            'valid_range': f"[{min_val}, {max_val}]",
                            'details': f"{col} outside valid range [{min_val}, {max_val}]"
                        })
        
        # Binary indicators (0 or 1)
        elif constraint_type == 'binary':
            allowed_values = constraint_info['allowed_values']
            
            for col in [c for c in constraint_info['columns'] if c in df.columns]:
                # Find values that are not in allowed values
                invalid_rows = df[~df[col].isin(allowed_values)]
                
                if not invalid_rows.empty:
                    issue_summary['range_violations']['count'] += len(invalid_rows)
                    
                    # Record first few issues for reporting
                    for idx, row in invalid_rows.head(5).iterrows():
                        issues.append({
                            'issue_type': 'range_violation',
                            'constraint_type': constraint_type,
                            'column': col,
                            'timestamp': row['timestamp_utc'],
                            'value': float(row[col]),
                            'allowed_values': str(allowed_values),
                            'details': f"{col} not in allowed values {allowed_values}"
                        })
        
        # Time features (specific ranges)
        elif constraint_type == 'time':
            for col, range_info in constraint_info.items():
                if col in df.columns:
                    min_val = range_info['min']
                    max_val = range_info['max']
                    
                    # Find values outside valid range
                    invalid_rows = df[(df[col] < min_val) | (df[col] > max_val)]
                    
                    if not invalid_rows.empty:
                        issue_summary['range_violations']['count'] += len(invalid_rows)
                        
                        # Record first few issues for reporting
                        for idx, row in invalid_rows.head(5).iterrows():
                            issues.append({
                                'issue_type': 'range_violation',
                                'constraint_type': 'time',
                                'column': col,
                                'timestamp': row['timestamp_utc'],
                                'value': int(row[col]),
                                'valid_range': f"[{min_val}, {max_val}]",
                                'details': f"{col} outside valid range [{min_val}, {max_val}]"
                            })
    
    # Check for absurdly large values in numerical columns
    # But exclude certain metrics where large values are normal
    for col in df.select_dtypes(include=[np.number]).columns:
        # Skip non-numeric or specifically excluded columns
        if col in ['timestamp_utc', 'id'] or col in exclude_from_absurd_checks:
            continue
            
        # Determine a reasonable threshold based on column statistics
        # Default absurd threshold is 1 million, but adjust based on column type
        if col == 'volume_1h':
            absurd_threshold = df[col].mean() * 100  # 100x the mean volume
        elif col in ['open_1h', 'high_1h', 'low_1h', 'close_1h']:
            absurd_threshold = df[col].mean() * 10  # 10x the mean price
        elif col.startswith('future_return') or col.endswith('_pct'):
            absurd_threshold = 2.0  # 200% change is very large
        elif 'rank' in col:
            absurd_threshold = 100  # Ranks shouldn't exceed 100
        else:
            # General threshold
            absurd_threshold = 1e6
            
        # Check for values that exceed the threshold
        absurd_rows = df[df[col] > absurd_threshold]
        
        if not absurd_rows.empty:
            issue_summary['absurd_values']['count'] += len(absurd_rows)
            
            # Record first few issues for reporting
            for idx, row in absurd_rows.head(5).iterrows():
                issues.append({
                    'issue_type': 'absurd_value',
                    'column': col,
                    'timestamp': row['timestamp_utc'],
                    'value': float(row[col]),
                    'threshold': absurd_threshold,
                    'details': f"Absurdly large value in {col}: {row[col]} > {absurd_threshold}"
                })
    
    # Calculate total issues
    total_issues = sum(category['count'] for category in issue_summary.values())
    
    # Calculate issue percentage based on number of cells
    issue_percentage = (total_issues / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
    
    return {
        'pair': pair,
        'status': 'completed',
        'issues_count': total_issues,
        'candles_count': len(df),
        'cells_count': len(df) * len(df.columns),
        'issue_percentage': issue_percentage,
        'issue_summary': issue_summary,
        'issues': issues
    }

if __name__ == "__main__":
    main_validator(validate_data_sanity, "Data Sanity Validator", 
                  "Validates general data sanity, checking for absurd values, infinities, and NaNs")