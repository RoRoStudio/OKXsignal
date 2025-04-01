#!/usr/bin/env python3
"""
Pattern Recognition Validator
- Ensures patterns are properly identified
- Validates pattern detection logic
"""

import pandas as pd
import numpy as np
from database.validation.validation_utils import main_validator

def validate_pattern_recognition(df, pair):
    """
    Validate pattern recognition features for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {
        'pattern_logic': {'count': 0}
    }
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Identify pattern columns
    pattern_columns = [col for col in df.columns if col.startswith('pattern_')]
    
    if not pattern_columns:
        return {
            'pair': pair,
            'status': 'no_pattern_columns',
            'issues_count': 0
        }
    
    # Check if required OHLC columns exist for pattern revalidation
    required_columns = ['open_1h', 'high_1h', 'low_1h', 'close_1h']
    missing_columns = [col for col in required_columns if col not in df.columns]
    has_ohlc = len(missing_columns) == 0
    
    # Revalidate pattern logic if OHLC data is available
    if has_ohlc:
        # Revalidate Doji pattern
        if 'pattern_doji' in pattern_columns:
            # Doji: open and close are almost equal
            body_size = np.abs(df['close_1h'] - df['open_1h'])
            total_size = df['high_1h'] - df['low_1h']
            
            # Avoid division by zero
            doji_ratio = body_size / total_size.replace(0, np.inf)
            
            # Default threshold (can be adjusted)
            doji_threshold = 0.1
            
            expected_doji = (doji_ratio < doji_threshold).astype(int)
            
            # Find mismatches
            doji_issues = df[df['pattern_doji'] != expected_doji]
            
            if not doji_issues.empty:
                issue_summary['pattern_logic']['count'] += len(doji_issues)
                
                # Record first few issues for reporting
                for idx, row in doji_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': 'pattern_logic',
                        'pattern': 'doji',
                        'timestamp': row['timestamp_utc'],
                        'expected': int(expected_doji.loc[idx]),
                        'actual': int(row['pattern_doji']),
                        'ratio': float(doji_ratio.loc[idx]),
                        'details': f"Doji pattern logic mismatch: ratio={doji_ratio.loc[idx]:.4f}, threshold={doji_threshold}"
                    })
        
        # Revalidate Engulfing pattern
        if 'pattern_engulfing' in pattern_columns:
            # Engulfing: current candle's body completely engulfs previous candle's body
            prev_body_low = np.minimum(df['open_1h'].shift(1), df['close_1h'].shift(1))
            prev_body_high = np.maximum(df['open_1h'].shift(1), df['close_1h'].shift(1))
            curr_body_low = np.minimum(df['open_1h'], df['close_1h'])
            curr_body_high = np.maximum(df['open_1h'], df['close_1h'])
            
            # Bullish or bearish engulfing
            bullish_engulfing = (df['close_1h'] > df['open_1h']) & (curr_body_low < prev_body_low) & (curr_body_high > prev_body_high)
            bearish_engulfing = (df['close_1h'] < df['open_1h']) & (curr_body_low < prev_body_low) & (curr_body_high > prev_body_high)
            expected_engulfing = (bullish_engulfing | bearish_engulfing).astype(int)
            
            # Find mismatches
            engulfing_issues = df[df['pattern_engulfing'] != expected_engulfing]
            
            if not engulfing_issues.empty:
                issue_summary['pattern_logic']['count'] += len(engulfing_issues)
                
                # Record first few issues for reporting
                for idx, row in engulfing_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': 'pattern_logic',
                        'pattern': 'engulfing',
                        'timestamp': row['timestamp_utc'],
                        'expected': int(expected_engulfing.loc[idx]),
                        'actual': int(row['pattern_engulfing']),
                        'details': f"Engulfing pattern logic mismatch"
                    })
        
        # Revalidate Hammer pattern
        if 'pattern_hammer' in pattern_columns:
            # Hammer: small body at the top, long lower shadow
            upper_shadow = df['high_1h'] - np.maximum(df['open_1h'], df['close_1h'])
            lower_shadow = np.minimum(df['open_1h'], df['close_1h']) - df['low_1h']
            
            # Safe body size for division
            body_size = np.maximum(np.abs(df['close_1h'] - df['open_1h']), 1e-8)
            total_size = np.maximum(df['high_1h'] - df['low_1h'], 1e-8)
            
            # Default thresholds
            body_threshold = 0.3
            shadow_threshold = 2.0
            
            expected_hammer = (
                (body_size / total_size < body_threshold) & 
                (lower_shadow / body_size > shadow_threshold) & 
                (upper_shadow < 0.1 * total_size)
            ).astype(int)
            
            # Find mismatches
            hammer_issues = df[df['pattern_hammer'] != expected_hammer]
            
            if not hammer_issues.empty:
                issue_summary['pattern_logic']['count'] += len(hammer_issues)
                
                # Record first few issues for reporting
                for idx, row in hammer_issues.head(5).iterrows():
                    body_ratio = body_size.loc[idx] / total_size.loc[idx]
                    shadow_ratio = lower_shadow.loc[idx] / body_size.loc[idx] if body_size.loc[idx] > 0 else float('inf')
                    
                    issues.append({
                        'issue_type': 'pattern_logic',
                        'pattern': 'hammer',
                        'timestamp': row['timestamp_utc'],
                        'expected': int(expected_hammer.loc[idx]),
                        'actual': int(row['pattern_hammer']),
                        'body_ratio': float(body_ratio),
                        'shadow_ratio': float(shadow_ratio),
                        'details': f"Hammer pattern logic mismatch: body_ratio={body_ratio:.4f}, shadow_ratio={shadow_ratio:.4f}"
                    })
        
        # Revalidate Morning Star pattern
        if 'pattern_morning_star' in pattern_columns:
            # Day 1: Bearish candle with large body
            bearish_day1 = (df['close_1h'].shift(2) < df['open_1h'].shift(2)) & (np.abs(df['close_1h'].shift(2) - df['open_1h'].shift(2)) > 0.5 * (df['high_1h'].shift(2) - df['low_1h'].shift(2)))
            
            # Day 2: Small body with gap down
            day2_body = np.abs(df['close_1h'].shift(1) - df['open_1h'].shift(1))
            day1_body = np.abs(df['close_1h'].shift(2) - df['open_1h'].shift(2))
            small_body_day2 = day2_body < (0.3 * day1_body.replace(0, np.inf))
            gap_down = df['high_1h'].shift(1) < df['close_1h'].shift(2)
            
            # Day 3: Bullish candle that closes above the midpoint of Day 1
            bullish_day3 = df['close_1h'] > df['open_1h']
            midpoint_day1 = (df['open_1h'].shift(2) + df['close_1h'].shift(2)) / 2
            closes_above = df['close_1h'] > midpoint_day1
            
            # Combined pattern
            expected_morning_star = (bearish_day1 & small_body_day2 & gap_down & bullish_day3 & closes_above).astype(int)
            
            # Find mismatches
            morning_star_issues = df[df['pattern_morning_star'] != expected_morning_star]
            
            if not morning_star_issues.empty:
                issue_summary['pattern_logic']['count'] += len(morning_star_issues)
                
                # Record first few issues for reporting
                for idx, row in morning_star_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': 'pattern_logic',
                        'pattern': 'morning_star',
                        'timestamp': row['timestamp_utc'],
                        'expected': int(expected_morning_star.loc[idx]),
                        'actual': int(row['pattern_morning_star']),
                        'details': f"Morning Star pattern logic mismatch"
                    })
    
    # Calculate total issues
    total_issues = sum(category['count'] for category in issue_summary.values())
    
    return {
        'pair': pair,
        'status': 'completed',
        'issues_count': total_issues,
        'issue_summary': issue_summary,
        'issues': issues
    }

if __name__ == "__main__":
    main_validator(validate_pattern_recognition, "Pattern Recognition Validator", 
                  "Validates pattern recognition flags and logic")