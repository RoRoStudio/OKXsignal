#!/usr/bin/env python3
"""
Momentum Indicators Validator
- Recomputes: rsi_*, macd_*, stoch_*, tsi, ppo, roc_10, williams_r, cci
- Validates each against recalculated values
"""

import pandas as pd
import numpy as np
from database.validation.validation_utils import main_validator

def calculate_rsi(prices, length=14):
    """Calculate RSI to match exactly with Numba implementation"""
    n = len(prices)
    rsi = np.zeros(n)
    
    # Default value for beginning periods
    rsi[:length] = 50.0
    
    if n <= length:
        return pd.Series(rsi, index=prices.index)
    
    # Calculate price changes
    changes = np.zeros(n)
    changes[1:] = np.diff(prices)
    
    # Separate gains and losses
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(1, n):
        if changes[i] > 0:
            gains[i] = changes[i]
        else:
            losses[i] = -changes[i]
    
    # Initial averages
    avg_gain = np.sum(gains[1:length+1]) / length
    avg_loss = np.sum(losses[1:length+1]) / length
    
    # First RSI calculation
    if avg_loss == 0:
        rsi[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[length] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate remaining RSI values with Wilder's smoothing
    for i in range(length+1, n):
        avg_gain = ((avg_gain * (length-1)) + gains[i]) / length
        avg_loss = ((avg_loss * (length-1)) + losses[i]) / length
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return pd.Series(rsi, index=prices.index)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD to match feature_processor implementation"""
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

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator to match feature_processor"""
    n = len(close)
    stoch_k = np.zeros(n)
    stoch_d = np.zeros(n)
    
    # Calculate %K (fast stochastic)
    for i in range(k_period - 1, n):
        lowest_low = np.min(low.iloc[i-k_period+1:i+1])
        highest_high = np.max(high.iloc[i-k_period+1:i+1])
        
        # Avoid division by zero
        range_diff = highest_high - lowest_low
        if range_diff > 0:
            stoch_k[i] = 100 * ((close.iloc[i] - lowest_low) / range_diff)
        else:
            stoch_k[i] = 50  # Default when range is zero
    
    # Calculate %D (moving average of %K)
    for i in range(d_period - 1, n):
        stoch_d[i] = np.mean(stoch_k[i-d_period+1:i+1])
    
    # Convert to Series for compatibility with validation code
    return {
        'stoch_k': pd.Series(stoch_k, index=close.index).fillna(50),
        'stoch_d': pd.Series(stoch_d, index=close.index).fillna(50)
    }

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R to match feature_processor"""
    n = len(close)
    will_r = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high = np.max(high.iloc[i-period+1:i+1])
        lowest_low = np.min(low.iloc[i-period+1:i+1])
        
        # Avoid division by zero
        range_diff = highest_high - lowest_low
        if range_diff > 0:
            will_r[i] = -100 * ((highest_high - close.iloc[i]) / range_diff)
        else:
            will_r[i] = -50  # Default when range is zero
    
    return pd.Series(will_r, index=close.index).fillna(-50)

def calculate_cci(high, low, close, period=14):
    """Calculate CCI to match feature_processor"""
    n = len(close)
    cci = np.zeros(n)
    
    for i in range(period - 1, n):
        # Calculate typical price for the window
        tp = (high.iloc[i-period+1:i+1] + low.iloc[i-period+1:i+1] + close.iloc[i-period+1:i+1]) / 3
        tp_mean = np.mean(tp)
        
        # Calculate mean deviation
        mad = np.mean(np.abs(tp - tp_mean))
        
        # Current typical price
        current_tp = (high.iloc[i] + low.iloc[i] + close.iloc[i]) / 3
        
        # Calculate CCI with proper handling of division by zero
        if mad > 0:
            cci[i] = (current_tp - tp_mean) / (0.015 * mad)
            
    return pd.Series(cci, index=close.index).fillna(0)

def calculate_roc(close, period=10):
    """Calculate Rate of Change to match feature_processor"""
    n = len(close)
    roc = np.zeros(n)
    
    for i in range(period, n):
        if close.iloc[i-period] > 0:  # Avoid division by zero
            roc[i] = ((close.iloc[i] / close.iloc[i-period]) - 1) * 100
    
    return pd.Series(roc, index=close.index).fillna(0)

def calculate_tsi(close, fast=13, slow=25):
    """Calculate True Strength Index to match feature_processor"""
    # Calculate price changes
    price_change = close.diff().fillna(0)
    abs_price_change = price_change.abs()
    
    # Double smoothed price change
    smooth1 = price_change.ewm(span=fast, adjust=False).mean()
    smooth2 = smooth1.ewm(span=slow, adjust=False).mean()
    
    # Double smoothed absolute price change
    abs_smooth1 = abs_price_change.ewm(span=fast, adjust=False).mean()
    abs_smooth2 = abs_smooth1.ewm(span=slow, adjust=False).mean()
    
    # Calculate TSI, avoiding division by zero
    tsi = 100 * (smooth2 / abs_smooth2.replace(0, np.nan))
    
    return tsi.fillna(0)

def calculate_ppo(close, fast=12, slow=26):
    """Calculate Percentage Price Oscillator to match feature_processor"""
    # Calculate EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # Calculate PPO, avoiding division by zero
    ppo = 100 * ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan))
    
    return ppo.fillna(0)

def calculate_awesome_oscillator(high, low, fast=5, slow=34):
    """Calculate Awesome Oscillator to match feature_processor"""
    # Calculate median price
    median_price = (high + low) / 2
    
    # Calculate simple moving averages
    fast_ma = median_price.rolling(window=fast, min_periods=1).mean()
    slow_ma = median_price.rolling(window=slow, min_periods=1).mean()
    
    # Calculate Awesome Oscillator
    ao = fast_ma - slow_ma
    
    return ao.fillna(0)

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
    
    # Adjusted threshold to allow for small floating-point differences
    threshold = 1e-6
    
    # Validate RSI
    if 'rsi_1h' in df.columns:
        # Calculate expected RSI
        expected_rsi = calculate_rsi(df['close_1h'])
        
        # Handle NaN values for proper comparison
        expected_rsi = expected_rsi.fillna(50.0)
        rsi_actual = df['rsi_1h'].fillna(50.0)
        
        # Calculate absolute differences
        rsi_diff = np.abs(rsi_actual - expected_rsi)
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
                    'expected': float(expected_rsi_slope.loc[idx]) if not pd.isna(expected_rsi_slope.loc[idx]) else None,
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
                    'expected': float(expected_macd_slope.loc[idx]) if not pd.isna(expected_macd_slope.loc[idx]) else None,
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
                    'expected': float(expected_hist_slope.loc[idx]) if not pd.isna(expected_hist_slope.loc[idx]) else None,
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
                    'expected': float(expected_stoch_k.loc[idx]) if not pd.isna(expected_stoch_k.loc[idx]) else None,
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
                    'expected': float(expected_stoch_d.loc[idx]) if not pd.isna(expected_stoch_d.loc[idx]) else None,
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
                    'expected': float(expected_williams_r.loc[idx]) if not pd.isna(expected_williams_r.loc[idx]) else None,
                    'actual': float(row['williams_r_14']),
                    'diff': float(williams_r_diff.loc[idx]),
                    'details': f"Williams %R calculation discrepancy"
                })
    
    # Validate CCI
    if 'cci_14' in df.columns:
        # Calculate expected CCI
        expected_cci = calculate_cci(df['high_1h'], df['low_1h'], df['close_1h'])
        
        # Calculate absolute differences
        cci_diff = np.abs(df['cci_14'] - expected_cci)
        cci_issues = df[cci_diff > threshold]
        
        issue_count = len(cci_issues)
        if issue_count > 0:
            issue_summary['cci_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in cci_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'cci_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_cci.loc[idx]) if not pd.isna(expected_cci.loc[idx]) else None,
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
                    'expected': float(expected_roc.loc[idx]) if not pd.isna(expected_roc.loc[idx]) else None,
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
                    'expected': float(expected_tsi.loc[idx]) if not pd.isna(expected_tsi.loc[idx]) else None,
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
                    'expected': float(expected_ppo.loc[idx]) if not pd.isna(expected_ppo.loc[idx]) else None,
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
                    'expected': float(expected_ao.loc[idx]) if not pd.isna(expected_ao.loc[idx]) else None,
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