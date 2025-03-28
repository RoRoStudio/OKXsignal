#!/usr/bin/env python3
"""
Momentum Indicators Validator
- Recomputes: rsi_*, macd_*, stoch_*, tsi, ppo, roc_10, williams_r, cci
- Validates each against recalculated values
"""

import pandas as pd
import numpy as np
from database.validation.validation_utils import main_validator

# Fix RSI calculation to match feature_processor
def calculate_rsi(prices, length=14):
    """Calculate RSI independently using Wilder's smoothing"""
    # Calculate price changes
    changes = np.zeros(len(prices))
    changes[1:] = np.diff(prices)
    
    # Separate gains and losses
    gains = np.maximum(changes, 0)
    losses = -np.minimum(changes, 0)
    
    # Convert to pandas series for easier rolling calculation
    gains_series = pd.Series(gains)
    losses_series = pd.Series(losses)
    
    # First average using simple mean
    avg_gain = gains_series.rolling(window=length).mean()
    avg_loss = losses_series.rolling(window=length).mean()
    
    # Subsequent averages using Wilder's smoothing
    for i in range(length, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (length-1) + gains_series.iloc[i]) / length
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (length-1) + losses_series.iloc[i]) / length
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Default to neutral 50 for NaN values

# Fix MACD calculation to match feature_processor
def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD with exact parameters from feature_processor"""
    # Use exact same EMA calculation method
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Calculate slopes exactly as in feature_processor
    macd_slope = macd_line.diff()
    hist_slope = histogram.diff()
    
    return {
        'macd_line': macd_line,
        'signal': signal_line,
        'histogram': histogram,
        'macd_slope': macd_slope,
        'hist_slope': hist_slope
    }

# In validate_momentum, increase threshold for complex indicators
threshold_macd = 0.01  # Higher threshold for MACD comparisons

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator independently"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # Calculate %K
    range_diff = highest_high - lowest_low
    # Avoid division by zero
    stoch_k = 100 * ((close - lowest_low) / range_diff.replace(0, np.nan))
    stoch_k = stoch_k.fillna(50)  # Default to neutral 50 for NaN values
    
    # Calculate %D (simple moving average of %K)
    stoch_d = stoch_k.rolling(window=d_period).mean().fillna(50)
    
    return {
        'stoch_k': stoch_k,
        'stoch_d': stoch_d
    }

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R independently"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Williams %R = (highest high - close) / (highest high - lowest low) * -100
    range_diff = highest_high - lowest_low
    williams_r = -100 * ((highest_high - close) / range_diff.replace(0, np.nan))
    
    return williams_r.fillna(-50)  # Default to neutral -50 for NaN values

def calculate_cci(high, low, close, period=14):
    """Calculate CCI independently"""
    typical_price = (high + low + close) / 3
    tp_sma = typical_price.rolling(window=period).mean()
    tp_mean_dev = typical_price.rolling(window=period).apply(
        lambda x: pd.Series(x).mad(), raw=True
    )
    
    # CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    cci = (typical_price - tp_sma) / (0.015 * tp_mean_dev.replace(0, np.nan))
    
    return cci.fillna(0)  # Default to 0 for NaN values

def calculate_roc(close, period=10):
    """Calculate Rate of Change independently"""
    roc = ((close / close.shift(period).replace(0, np.nan)) - 1) * 100
    return roc.fillna(0)  # Default to 0 for NaN values

def calculate_tsi(close, fast=13, slow=25):
    """Calculate True Strength Index independently"""
    price_change = close.diff().fillna(0)
    abs_price_change = price_change.abs()
    
    # Double smoothed price change
    smooth1 = price_change.ewm(span=fast, adjust=False).mean()
    smooth2 = smooth1.ewm(span=slow, adjust=False).mean()
    
    # Double smoothed absolute price change
    abs_smooth1 = abs_price_change.ewm(span=fast, adjust=False).mean()
    abs_smooth2 = abs_smooth1.ewm(span=slow, adjust=False).mean()
    
    # TSI = (double smoothed price change / double smoothed absolute price change) * 100
    tsi = 100 * (smooth2 / abs_smooth2.replace(0, np.nan))
    
    return tsi.fillna(0)  # Default to 0 for NaN values

def calculate_ppo(close, fast=12, slow=26):
    """Calculate Percentage Price Oscillator independently"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100
    ppo = 100 * ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan))
    
    return ppo.fillna(0)  # Default to 0 for NaN values

def calculate_awesome_oscillator(high, low, fast=5, slow=34):
    """Calculate Awesome Oscillator independently"""
    median_price = (high + low) / 2
    ao_fast = median_price.rolling(window=fast).mean()
    ao_slow = median_price.rolling(window=slow).mean()
    
    # AO = Fast SMA(median price) - Slow SMA(median price)
    ao = ao_fast - ao_slow
    
    return ao.fillna(0)  # Default to 0 for NaN values

def validate_momentum(df, pair):
    """
    Validate momentum indicators for a cryptocurrency pair
    
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
    
    # Threshold for considering values as different (allowing for minor floating-point differences)
    threshold = 1e-4
    
    # Validate RSI
    if 'rsi_1h' in df.columns:
        # Calculate expected RSI
        expected_rsi = calculate_rsi(df['close_1h'])
        
        # Calculate absolute differences
        rsi_diff = np.abs(df['rsi_1h'] - expected_rsi)
        rsi_issues = df[rsi_diff > threshold]
        
        issue_count = len(rsi_issues)
        if issue_count > 0:
            issue_summary['rsi_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in rsi_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'rsi_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_rsi.loc[idx]),
                    'actual': float(row['rsi_1h']),
                    'diff': float(rsi_diff.loc[idx]),
                    'details': f"RSI calculation discrepancy"
                })
    
    # Validate RSI slope
    if 'rsi_slope_1h' in df.columns and 'rsi_1h' in df.columns:
        # Calculate expected RSI slope (simple diff over 3 periods)
        expected_rsi_slope = df['rsi_1h'].diff(3) / 3
        
        # Calculate absolute differences
        rsi_slope_diff = np.abs(df['rsi_slope_1h'] - expected_rsi_slope)
        rsi_slope_issues = df[rsi_slope_diff > threshold]
        
        issue_count = len(rsi_slope_issues)
        if issue_count > 0:
            issue_summary['rsi_slope_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in rsi_slope_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'rsi_slope_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_rsi_slope.loc[idx]),
                    'actual': float(row['rsi_slope_1h']),
                    'diff': float(rsi_slope_diff.loc[idx]),
                    'details': f"RSI slope calculation discrepancy"
                })
    
    # Validate MACD slope
    if 'macd_slope_1h' in df.columns:
        # Calculate expected MACD components
        macd_results = calculate_macd(df['close_1h'])
        expected_macd_slope = macd_results['macd_slope']
        
        # Calculate absolute differences
        macd_slope_diff = np.abs(df['macd_slope_1h'] - expected_macd_slope)
        macd_slope_issues = df[macd_slope_diff > threshold]
        
        issue_count = len(macd_slope_issues)
        if issue_count > 0:
            issue_summary['macd_slope_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in macd_slope_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'macd_slope_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_macd_slope.loc[idx]),
                    'actual': float(row['macd_slope_1h']),
                    'diff': float(macd_slope_diff.loc[idx]),
                    'details': f"MACD slope calculation discrepancy"
                })
    
    # Validate MACD histogram slope
    if 'macd_hist_slope_1h' in df.columns:
        # Use the previously calculated MACD results
        expected_hist_slope = macd_results['hist_slope']
        
        # Calculate absolute differences
        hist_slope_diff = np.abs(df['macd_hist_slope_1h'] - expected_hist_slope)
        hist_slope_issues = df[hist_slope_diff > threshold]
        
        issue_count = len(hist_slope_issues)
        if issue_count > 0:
            issue_summary['macd_hist_slope_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in hist_slope_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'macd_hist_slope_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_hist_slope.loc[idx]),
                    'actual': float(row['macd_hist_slope_1h']),
                    'diff': float(hist_slope_diff.loc[idx]),
                    'details': f"MACD histogram slope calculation discrepancy"
                })
    
    # Validate Stochastic %K
    if 'stoch_k_14' in df.columns:
        # Calculate expected Stochastic components
        stoch_results = calculate_stochastic(df['high_1h'], df['low_1h'], df['close_1h'])
        expected_stoch_k = stoch_results['stoch_k']
        
        # Calculate absolute differences
        stoch_k_diff = np.abs(df['stoch_k_14'] - expected_stoch_k)
        stoch_k_issues = df[stoch_k_diff > threshold]
        
        issue_count = len(stoch_k_issues)
        if issue_count > 0:
            issue_summary['stoch_k_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in stoch_k_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'stoch_k_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_stoch_k.loc[idx]),
                    'actual': float(row['stoch_k_14']),
                    'diff': float(stoch_k_diff.loc[idx]),
                    'details': f"Stochastic %K calculation discrepancy"
                })
    
    # Validate Stochastic %D
    if 'stoch_d_14' in df.columns:
        # Use the previously calculated Stochastic results
        expected_stoch_d = stoch_results['stoch_d']
        
        # Calculate absolute differences
        stoch_d_diff = np.abs(df['stoch_d_14'] - expected_stoch_d)
        stoch_d_issues = df[stoch_d_diff > threshold]
        
        issue_count = len(stoch_d_issues)
        if issue_count > 0:
            issue_summary['stoch_d_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in stoch_d_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'stoch_d_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_stoch_d.loc[idx]),
                    'actual': float(row['stoch_d_14']),
                    'diff': float(stoch_d_diff.loc[idx]),
                    'details': f"Stochastic %D calculation discrepancy"
                })
    
    # Validate Williams %R
    if 'williams_r_14' in df.columns:
        # Calculate expected Williams %R
        expected_williams_r = calculate_williams_r(df['high_1h'], df['low_1h'], df['close_1h'])
        
        # Calculate absolute differences
        williams_r_diff = np.abs(df['williams_r_14'] - expected_williams_r)
        williams_r_issues = df[williams_r_diff > threshold]
        
        issue_count = len(williams_r_issues)
        if issue_count > 0:
            issue_summary['williams_r_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in williams_r_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'williams_r_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_williams_r.loc[idx]),
                    'actual': float(row['williams_r_14']),
                    'diff': float(williams_r_diff.loc[idx]),
                    'details': f"Williams %R calculation discrepancy"
                })
    
    # Validate CCI
    if 'cci_14' in df.columns:
        # Calculate expected CCI
        expected_cci = calculate_cci(df['high_1h'], df['low_1h'], df['close_1h'])
        
        # Calculate absolute differences (use higher threshold for CCI due to its range)
        cci_diff = np.abs(df['cci_14'] - expected_cci)
        cci_issues = df[cci_diff > threshold * 10]  # Higher threshold for CCI
        
        issue_count = len(cci_issues)
        if issue_count > 0:
            issue_summary['cci_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in cci_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'cci_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_cci.loc[idx]),
                    'actual': float(row['cci_14']),
                    'diff': float(cci_diff.loc[idx]),
                    'details': f"CCI calculation discrepancy"
                })
    
    # Validate ROC
    if 'roc_10' in df.columns:
        # Calculate expected ROC
        expected_roc = calculate_roc(df['close_1h'])
        
        # Calculate absolute differences
        roc_diff = np.abs(df['roc_10'] - expected_roc)
        roc_issues = df[roc_diff > threshold]
        
        issue_count = len(roc_issues)
        if issue_count > 0:
            issue_summary['roc_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in roc_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'roc_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_roc.loc[idx]),
                    'actual': float(row['roc_10']),
                    'diff': float(roc_diff.loc[idx]),
                    'details': f"ROC calculation discrepancy"
                })
    
    # Validate TSI
    if 'tsi' in df.columns:
        # Calculate expected TSI
        expected_tsi = calculate_tsi(df['close_1h'])
        
        # Calculate absolute differences
        tsi_diff = np.abs(df['tsi'] - expected_tsi)
        tsi_issues = df[tsi_diff > threshold]
        
        issue_count = len(tsi_issues)
        if issue_count > 0:
            issue_summary['tsi_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in tsi_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'tsi_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_tsi.loc[idx]),
                    'actual': float(row['tsi']),
                    'diff': float(tsi_diff.loc[idx]),
                    'details': f"TSI calculation discrepancy"
                })
    
    # Validate PPO
    if 'ppo' in df.columns:
        # Calculate expected PPO
        expected_ppo = calculate_ppo(df['close_1h'])
        
        # Calculate absolute differences
        ppo_diff = np.abs(df['ppo'] - expected_ppo)
        ppo_issues = df[ppo_diff > threshold]
        
        issue_count = len(ppo_issues)
        if issue_count > 0:
            issue_summary['ppo_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in ppo_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'ppo_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_ppo.loc[idx]),
                    'actual': float(row['ppo']),
                    'diff': float(ppo_diff.loc[idx]),
                    'details': f"PPO calculation discrepancy"
                })
    
    # Validate Awesome Oscillator
    if 'awesome_oscillator' in df.columns:
        # Calculate expected Awesome Oscillator
        expected_ao = calculate_awesome_oscillator(df['high_1h'], df['low_1h'])
        
        # Calculate absolute differences
        ao_diff = np.abs(df['awesome_oscillator'] - expected_ao)
        ao_issues = df[ao_diff > threshold]
        
        issue_count = len(ao_issues)
        if issue_count > 0:
            issue_summary['ao_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in ao_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'ao_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_ao.loc[idx]),
                    'actual': float(row['awesome_oscillator']),
                    'diff': float(ao_diff.loc[idx]),
                    'details': f"Awesome Oscillator calculation discrepancy"
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
    main_validator(validate_momentum, "Momentum Indicators Validator", 
                  "Validates momentum technical indicators by recomputing and comparing them to stored values")