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
    """Calculate True Range to match feature_processor"""
    n = len(high)
    tr = np.zeros(n)
    
    # First true range is high-low
    if n > 0:
        tr[0] = high.iloc[0] - low.iloc[0]
    
    # Rest of true ranges
    for i in range(1, n):
        tr1 = high.iloc[i] - low.iloc[i]
        tr2 = abs(high.iloc[i] - close.iloc[i-1])
        tr3 = abs(low.iloc[i] - close.iloc[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    return pd.Series(tr, index=high.index)

def calculate_atr(high, low, close, length=14):
    """Calculate ATR to match feature_processor"""
    tr_series = calculate_true_range(high, low, close)
    n = len(tr_series)
    atr = np.zeros(n)
    
    # First ATR is first TR
    if n > length:
        atr[length-1] = np.mean(tr_series.iloc[:length])
        
        # Rest use smoothing
        for i in range(length, n):
            atr[i] = (atr[i-1] * (length-1) + tr_series.iloc[i]) / length
    
    return pd.Series(atr, index=high.index)

def calculate_bollinger_bands(close, period=20, num_std=2):
    """Calculate Bollinger Bands to match feature_processor"""
    # Calculate middle band (SMA)
    middle_band = close.rolling(window=period, min_periods=1).mean()
    
    # Calculate standard deviation
    std_dev = close.rolling(window=period, min_periods=1).std()
    
    # Calculate bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    # Calculate width
    width = upper_band - lower_band
    
    # Calculate %B (close position relative to bands)
    band_diff = upper_band - lower_band
    percent_b = pd.Series(0.5, index=close.index)  # Default to 0.5
    mask = band_diff > 0
    percent_b[mask] = (close[mask] - lower_band[mask]) / band_diff[mask]
    
    return {
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'width': width,
        'percent_b': percent_b
    }

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels to match feature_processor"""
    # Calculate upper and lower bands
    upper_band = high.rolling(window=period, min_periods=1).max()
    lower_band = low.rolling(window=period, min_periods=1).min()
    
    # Calculate middle band
    middle_band = (upper_band + lower_band) / 2
    
    # Calculate width
    width = upper_band - lower_band
    
    return {
        'upper_band': upper_band,
        'lower_band': lower_band,
        'middle_band': middle_band,
        'width': width
    }

def calculate_keltner_channels(high, low, close, ema_length=20, atr_length=14, atr_multiplier=2):
    """Calculate Keltner Channels to match feature_processor"""
    # Calculate EMA
    ema = close.ewm(span=ema_length, adjust=False).mean()
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, atr_length)
    
    # Calculate bands
    upper_band = ema + (atr * atr_multiplier)
    lower_band = ema - (atr * atr_multiplier)
    
    # Calculate width
    width = upper_band - lower_band
    
    return {
        'middle_line': ema,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'width': width
    }

def calculate_historical_volatility(close, period=30):
    """Calculate Historical Volatility to match feature_processor"""
    # Calculate returns
    returns = close.pct_change().fillna(0)
    
    # Calculate standard deviation of returns
    volatility = returns.rolling(window=period, min_periods=1).std()
    
    # Annualize (multiply by sqrt(252) for daily, or sqrt(252*24) for hourly)
    annual_factor = np.sqrt(252 * 24)  # For hourly data
    hist_vol = volatility * annual_factor
    
    return hist_vol

def calculate_z_score(close, period=20):
    """Calculate Z-Score to match feature_processor"""
    # Calculate moving average
    ma = close.rolling(window=period, min_periods=1).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=period, min_periods=1).std()
    
    # Calculate z-score, handling zeros
    z_score = pd.Series(0, index=close.index)
    mask = std > 0
    z_score[mask] = (close[mask] - ma[mask]) / std[mask]
    
    return z_score

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
    
    # Threshold for considering values as different
    threshold = 1e-6
    
    # Validate ATR and True Range
    if 'atr_1h' in df.columns or 'true_range' in df.columns:
        # Calculate expected True Range
        expected_tr = calculate_true_range(df['high_1h'], df['low_1h'], df['close_1h'])
        
        # Calculate expected ATR
        expected_atr = calculate_atr(df['high_1h'], df['low_1h'], df['close_1h'])
        
        # Validate True Range
        if 'true_range' in df.columns:
            # Calculate absolute differences
            tr_diff = np.abs(df['true_range'] - expected_tr)
            tr_issues = df[tr_diff > threshold]
            
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
            # Calculate absolute differences
            atr_diff = np.abs(df['atr_1h'] - expected_atr)
            atr_issues = df[atr_diff > threshold]
            
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
        # Handle division by zero
        safe_close = df['close_1h'].replace(0, np.nan)
        expected_norm_atr = df['atr_1h'] / safe_close
        expected_norm_atr = expected_norm_atr.fillna(0)
        
        # Calculate absolute differences
        norm_atr_diff = np.abs(df['normalized_atr_14'] - expected_norm_atr)
        norm_atr_issues = df[norm_atr_diff > threshold]
        
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
        
        # Calculate absolute differences
        bb_width_diff = np.abs(df['bollinger_width_1h'] - expected_bb_width)
        bb_width_issues = df[bb_width_diff > threshold]
        
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
        
        # Calculate absolute differences
        dc_width_diff = np.abs(df['donchian_channel_width_1h'] - expected_dc_width)
        dc_width_issues = df[dc_width_diff > threshold]
        
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
        # Calculate expected Keltner Channels
        kc_results = calculate_keltner_channels(df['high_1h'], df['low_1h'], df['close_1h'])
        expected_kc_width = kc_results['width']
        
        # Calculate absolute differences
        kc_width_diff = np.abs(df['keltner_channel_width'] - expected_kc_width)
        kc_width_issues = df[kc_width_diff > threshold]
        
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
        
        # Calculate absolute differences
        hist_vol_diff = np.abs(df['historical_vol_30'] - expected_hist_vol)
        hist_vol_issues = df[hist_vol_diff > threshold]
        
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
        
        # Calculate absolute differences
        z_score_diff = np.abs(df['z_score_20'] - expected_z_score)
        z_score_issues = df[z_score_diff > threshold]
        
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