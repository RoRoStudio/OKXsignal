#!/usr/bin/env python3
"""
Volatility Indicators Validator
- Validates: atr_*, bollinger_*, true_range, historical_vol_30, normalized_atr_14, keltner_channel_width, etc.
- Checks for expected range consistency (e.g., volatility not zero during market movement)
"""

import pandas as pd
import numpy as np
from database.validation.validation_utils import main_validator

def calculate_true_range(high, low, close):
    """Calculate True Range independently"""
    tr = pd.Series(index=high.index)
    
    # First row has no previous close, so use high-low
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    
    # For subsequent rows, calculate true range
    for i in range(1, len(high)):
        tr1 = high.iloc[i] - low.iloc[i]
        tr2 = abs(high.iloc[i] - close.iloc[i-1])
        tr3 = abs(low.iloc[i] - close.iloc[i-1])
        tr.iloc[i] = max(tr1, tr2, tr3)
    
    return tr

def calculate_atr(high, low, close, length=14):
    """Calculate ATR independently"""
    # Calculate True Range first
    tr_series = calculate_true_range(high, low, close)
    
    # Initialize ATR with explicit dtype to avoid warning
    atr = pd.Series(np.nan, index=tr_series.index, dtype=float)
    
    # First ATR is simple average of first 'length' TRs
    if len(tr_series) >= length:
        atr.iloc[length-1] = tr_series.iloc[:length].mean()
        
        # Use Wilder's smoothing for subsequent values
        for i in range(length, len(tr_series)):
            atr.iloc[i] = (atr.iloc[i-1] * (length-1) + tr_series.iloc[i]) / length
    
    # Return a dictionary to match expected structure in validate_volatility()
    return {
        'tr': tr_series,
        'atr': atr.fillna(0)  # Fill NaN values with 0
    }

def calculate_bollinger_bands(close, length=20, num_std=2):
    """Calculate Bollinger Bands independently"""
    # Use SMA for middle band
    sma = close.rolling(window=length).mean()
    
    # Calculate standard deviation
    std_dev = close.rolling(window=length).std()
    
    # Calculate bands
    upper_band = sma + (std_dev * num_std)
    lower_band = sma - (std_dev * num_std)
    
    # Calculate bandwidth (using proper normalization)
    bandwidth = (upper_band - lower_band) / sma.replace(0, np.nan)
    
    # Calculate %B with proper handling of division by zero
    percent_b = (close - lower_band) / (upper_band - lower_band).replace(0, np.nan)
    
    return {
        'middle_band': sma.fillna(close),
        'upper_band': upper_band.fillna(close),
        'lower_band': lower_band.fillna(close),
        'width': (upper_band - lower_band).fillna(0),
        'bandwidth': bandwidth.fillna(0),
        'percent_b': percent_b.fillna(0.5)  # Default to 0.5 for NaN
    }

def calculate_historical_volatility(close, length=30):
    """Calculate Historical Volatility independently"""
    # Calculate daily returns
    returns = close.pct_change().fillna(0)
    
    # Calculate standard deviation of returns
    vol = returns.rolling(window=length).std()
    
    # Annualize (multiply by sqrt(365*24) for hourly)
    annual_factor = np.sqrt(365 * 24)  # For hourly data
    hist_vol = vol * annual_factor
    
    return hist_vol.fillna(0)

def calculate_keltner_channels(high, low, close, ema_length=20, atr_length=14, atr_multiplier=2):
    """Calculate Keltner Channels independently"""
    # Calculate EMA of closing prices
    ema = close.ewm(span=ema_length, adjust=False).mean()
    
    # Calculate ATR
    atr_series = calculate_atr(high, low, close, atr_length)
    
    # Calculate upper and lower bands
    upper_band = ema + (atr_series * atr_multiplier)
    lower_band = ema - (atr_series * atr_multiplier)
    
    # Calculate width
    width = upper_band - lower_band
    
    return {
        'middle_line': ema,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'width': width
    }

def calculate_donchian_channels(high, low, length=20):
    """Calculate Donchian Channels independently"""
    upper_band = high.rolling(window=length).max()
    lower_band = low.rolling(window=length).min()
    middle_band = (upper_band + lower_band) / 2
    
    return {
        'upper_band': upper_band.fillna(high),
        'lower_band': lower_band.fillna(low),
        'middle_band': middle_band.fillna((high + low) / 2),
        'width': (upper_band - lower_band).fillna(0)
    }

def calculate_z_score(close, window=20):
    """Calculate z-score independently"""
    # Calculate moving average
    ma = close.rolling(window=window).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=window).std()
    
    # Calculate z-score
    z_score = (close - ma) / std.replace(0, np.nan)
    
    return z_score.fillna(0)

def validate_volatility(df, pair):
    """
    Validate volatility indicators for a cryptocurrency pair
    
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
    
    # Check if required base columns exist
    required_base_columns = ['open_1h', 'high_1h', 'low_1h', 'close_1h']
    missing_base_columns = [col for col in required_base_columns if col not in df.columns]
    
    if missing_base_columns:
        return {
            'pair': pair,
            'status': 'missing_base_columns',
            'issues_count': len(missing_base_columns),
            'missing_columns': missing_base_columns
        }
    
    # Higher threshold to allow for implementation differences
    threshold = 0.05
    
    # Validate ATR and True Range
    if 'atr_1h' in df.columns or 'true_range' in df.columns:
        # Calculate true range
        expected_tr = calculate_true_range(df['high_1h'], df['low_1h'], df['close_1h'])
        
        # Validate True Range
        if 'true_range' in df.columns:
            # Calculate absolute differences
            tr_diff = np.abs(df['true_range'] - expected_tr)
            relative_diff = tr_diff / expected_tr.replace(0, 1).abs()
            tr_issues = df[relative_diff > threshold]
            
            issue_count = len(tr_issues)
            if issue_count > 0:
                issue_summary['tr_issues'] = {'count': issue_count}
                
                # Record first few issues for reporting
                for idx, row in tr_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': 'true_range_issue',
                        'timestamp': row['timestamp_utc'],
                        'expected': float(expected_tr.loc[idx]),
                        'actual': float(row['true_range']),
                        'diff': float(tr_diff.loc[idx]),
                        'details': f"True Range calculation discrepancy"
                    })
        
        # Validate ATR
        if 'atr_1h' in df.columns:
            expected_atr = calculate_atr(df['high_1h'], df['low_1h'], df['close_1h'])
            
            # Calculate absolute differences with relative threshold
            atr_diff = np.abs(df['atr_1h'] - expected_atr)
            relative_diff = atr_diff / expected_atr.replace(0, 1).abs()
            atr_issues = df[relative_diff > threshold]
            
            issue_count = len(atr_issues)
            if issue_count > 0:
                issue_summary['atr_issues'] = {'count': issue_count}
                
                # Record first few issues for reporting
                for idx, row in atr_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': 'atr_issue',
                        'timestamp': row['timestamp_utc'],
                        'expected': float(expected_atr.loc[idx]),
                        'actual': float(row['atr_1h']),
                        'diff': float(atr_diff.loc[idx]),
                        'details': f"ATR calculation discrepancy"
                    })
    
    # Validate Normalized ATR
    if 'normalized_atr_14' in df.columns and 'atr_1h' in df.columns:
        # Calculate expected normalized ATR (ATR / Close)
        expected_norm_atr = df['atr_1h'] / df['close_1h'].replace(0, np.inf)
        expected_norm_atr = expected_norm_atr.replace([np.inf, -np.inf], 0)
        
        # Calculate absolute differences
        norm_atr_diff = np.abs(df['normalized_atr_14'] - expected_norm_atr)
        relative_diff = norm_atr_diff / expected_norm_atr.replace(0, 1).abs()
        norm_atr_issues = df[relative_diff > threshold]
        
        issue_count = len(norm_atr_issues)
        if issue_count > 0:
            issue_summary['norm_atr_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in norm_atr_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'norm_atr_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_norm_atr.loc[idx]),
                    'actual': float(row['normalized_atr_14']),
                    'diff': float(norm_atr_diff.loc[idx]),
                    'details': f"Normalized ATR calculation discrepancy"
                })
    
    # Validate Bollinger Band Width
    if 'bollinger_width_1h' in df.columns:
        # Calculate expected Bollinger Bands
        bb_results = calculate_bollinger_bands(df['close_1h'])
        expected_bb_width = bb_results['width']
        
        # Calculate relative differences
        bb_width_diff = np.abs(df['bollinger_width_1h'] - expected_bb_width)
        relative_diff = bb_width_diff / expected_bb_width.replace(0, 1).abs()
        bb_width_issues = df[relative_diff > threshold]
        
        issue_count = len(bb_width_issues)
        if issue_count > 0:
            issue_summary['bb_width_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in bb_width_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'bb_width_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_bb_width.loc[idx]),
                    'actual': float(row['bollinger_width_1h']),
                    'diff': float(bb_width_diff.loc[idx]),
                    'details': f"Bollinger Band Width calculation discrepancy"
                })
    
    # Validate Bollinger Percent B
    if 'bollinger_percent_b' in df.columns:
        expected_percent_b = bb_results['percent_b']
        
        # Calculate absolute differences
        percent_b_diff = np.abs(df['bollinger_percent_b'] - expected_percent_b)
        percent_b_issues = df[percent_b_diff > threshold]
        
        issue_count = len(percent_b_issues)
        if issue_count > 0:
            issue_summary['percent_b_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in percent_b_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'percent_b_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_percent_b.loc[idx]),
                    'actual': float(row['bollinger_percent_b']),
                    'diff': float(percent_b_diff.loc[idx]),
                    'details': f"Bollinger %B calculation discrepancy"
                })
    
    # Validate Donchian Channel Width
    if 'donchian_channel_width_1h' in df.columns:
        # Calculate expected Donchian Channels
        dc_results = calculate_donchian_channels(df['high_1h'], df['low_1h'])
        expected_dc_width = dc_results['width']
        
        # Calculate relative differences
        dc_width_diff = np.abs(df['donchian_channel_width_1h'] - expected_dc_width)
        relative_diff = dc_width_diff / expected_dc_width.replace(0, 1).abs()
        dc_width_issues = df[relative_diff > threshold]
        
        issue_count = len(dc_width_issues)
        if issue_count > 0:
            issue_summary['dc_width_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in dc_width_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'dc_width_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_dc_width.loc[idx]),
                    'actual': float(row['donchian_channel_width_1h']),
                    'diff': float(dc_width_diff.loc[idx]),
                    'details': f"Donchian Channel Width calculation discrepancy"
                })
    
    # Validate Keltner Channel Width
    if 'keltner_channel_width' in df.columns:
        # Calculate expected Keltner Channels using high/low/close
        kc_results = calculate_keltner_channels(df['high_1h'], df['low_1h'], df['close_1h'])
        expected_kc_width = kc_results['width']
        
        # Calculate relative differences
        kc_width_diff = np.abs(df['keltner_channel_width'] - expected_kc_width)
        relative_diff = kc_width_diff / expected_kc_width.replace(0, 1).abs()
        kc_width_issues = df[relative_diff > threshold]
        
        issue_count = len(kc_width_issues)
        if issue_count > 0:
            issue_summary['kc_width_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in kc_width_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'kc_width_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_kc_width.loc[idx]),
                    'actual': float(row['keltner_channel_width']),
                    'diff': float(kc_width_diff.loc[idx]),
                    'details': f"Keltner Channel Width calculation discrepancy"
                })
    
    # Validate Historical Volatility
    if 'historical_vol_30' in df.columns:
        # Calculate expected Historical Volatility
        expected_hist_vol = calculate_historical_volatility(df['close_1h'])
        
        # Calculate relative differences
        hist_vol_diff = np.abs(df['historical_vol_30'] - expected_hist_vol)
        relative_diff = hist_vol_diff / expected_hist_vol.replace(0, 0.001).abs()
        hist_vol_issues = df[relative_diff > threshold]
        
        issue_count = len(hist_vol_issues)
        if issue_count > 0:
            issue_summary['hist_vol_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in hist_vol_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'hist_vol_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_hist_vol.loc[idx]),
                    'actual': float(row['historical_vol_30']),
                    'diff': float(hist_vol_diff.loc[idx]),
                    'details': f"Historical Volatility calculation discrepancy"
                })
    
    # Validate Z-Score
    if 'z_score_20' in df.columns:
        # Calculate expected Z-Score
        expected_z_score = calculate_z_score(df['close_1h'])
        
        # Calculate absolute differences (use higher threshold for z-score)
        z_score_diff = np.abs(df['z_score_20'] - expected_z_score)
        z_score_issues = df[z_score_diff > threshold * 5]  # Higher threshold for z-score
        
        issue_count = len(z_score_issues)
        if issue_count > 0:
            issue_summary['z_score_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in z_score_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'z_score_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_z_score.loc[idx]),
                    'actual': float(row['z_score_20']),
                    'diff': float(z_score_diff.loc[idx]),
                    'details': f"Z-Score calculation discrepancy"
                })
    
    # Check for consistency issues (volatility not zero during market movement)
    
    # 1. Check if ATR is zero when there is price movement
    if 'atr_1h' in df.columns:
        price_movement = df['high_1h'] - df['low_1h']
        consistency_issues = df[(df['atr_1h'] == 0) & (price_movement > price_movement.mean() * 0.5)]
        
        issue_count = len(consistency_issues)
        if issue_count > 0:
            issue_summary['atr_consistency_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in consistency_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'atr_consistency_issue',
                    'timestamp': row['timestamp_utc'],
                    'high': float(row['high_1h']),
                    'low': float(row['low_1h']),
                    'atr': float(row['atr_1h']),
                    'details': f"ATR is zero despite significant price movement"
                })
    
    # 2. Check if Bollinger Width is zero when there is price volatility
    if 'bollinger_width_1h' in df.columns:
        # Calculate rolling standard deviation of close prices
        rolling_std = df['close_1h'].rolling(window=20).std()
        
        consistency_issues = df[(df['bollinger_width_1h'] == 0) & (rolling_std > rolling_std.mean() * 0.5)]
        
        issue_count = len(consistency_issues)
        if issue_count > 0:
            issue_summary['bb_consistency_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in consistency_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'bb_consistency_issue',
                    'timestamp': row['timestamp_utc'],
                    'bollinger_width': float(row['bollinger_width_1h']),
                    'rolling_std': float(rolling_std.loc[idx]),
                    'details': f"Bollinger Width is zero despite price volatility"
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
    main_validator(validate_volatility, "Volatility Indicators Validator", 
                 "Validates volatility technical indicators by recomputing and comparing them to stored values")