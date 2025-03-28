#!/usr/bin/env python3
"""
Statistical Features Validator
- Recalculates: std_dev_returns_20, skewness_20, kurtosis_20, hurst_exponent, shannon_entropy, autocorr_1
- Verifies all are within mathematically valid bounds
"""

import pandas as pd
import numpy as np
from scipy import stats
import math
from database.validation.validation_utils import main_validator

def calculate_std_dev_returns(returns, window=20):
    """Calculate standard deviation of returns independently"""
    return returns.rolling(window=window).std().fillna(0)

def calculate_skewness(returns, window=20):
    """Calculate skewness independently"""
    return returns.rolling(window=window).apply(
        lambda x: stats.skew(x) if len(x) > 3 else 0,
        raw=True
    ).fillna(0)

def calculate_kurtosis(returns, window=20):
    """Calculate kurtosis independently"""
    return returns.rolling(window=window).apply(
        lambda x: stats.kurtosis(x) if len(x) > 4 else 0,
        raw=True
    ).fillna(0)

def calculate_autocorrelation(returns, window=20, lag=1):
    """Calculate autocorrelation independently"""
    return returns.rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0,
        raw=False
    ).fillna(0)

def calculate_hurst_exponent(prices, window=100, max_lag=20):
    """
    Calculate Hurst exponent independently
    
    The Hurst exponent measures the long-term memory of a time series.
    H < 0.5 indicates mean-reverting series
    H = 0.5 indicates random walk
    H > 0.5 indicates trending series
    """
    hurst = np.zeros(len(prices))
    
    if len(prices) < window:
        return pd.Series(hurst, index=prices.index)
    
    # Process each window of data
    for i in range(window, len(prices)):
        # Get window
        ts = prices.iloc[i-window:i].values
        
        # Calculate log returns
        lags = range(2, max_lag + 1)
        tau = []; lagvec = []
        
        # Step through different lags
        for lag in lags:
            # Compute price difference for the lag
            pp = np.subtract(ts[lag:], ts[:-lag])
            
            # Write the different lags into a vector
            lagvec.append(lag)
            
            # Calculate the variance of the difference
            tau.append(np.sqrt(np.std(pp)))
        
        # Calculate Hurst exponent using linear regression in log-log space
        if len(tau) > 0 and len(lagvec) > 0:
            # Check for valid tau
            if np.all(np.array(tau) > 0):
                # Convert to log space
                m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
                # The Hurst exponent is the slope
                hurst[i] = m[0]
            else:
                hurst[i] = 0.5  # Default value for invalid cases
        else:
            hurst[i] = 0.5  # Default value for invalid cases
    
    return pd.Series(hurst, index=prices.index)

def calculate_shannon_entropy(returns, window=20, bins=10):
    """
    Calculate Shannon entropy independently
    
    Shannon entropy measures the unpredictability of the returns distribution
    Higher entropy = more random/unpredictable
    Lower entropy = more predictable/structured
    """
    entropy = np.zeros(len(returns))
    
    if len(returns) < window:
        return pd.Series(entropy, index=returns.index)
    
    # Process each window of data
    for i in range(window, len(returns)):
        # Get window of returns
        window_data = returns.iloc[i-window:i]
        
        # Create histogram of returns
        hist, _ = np.histogram(window_data, bins=bins)
        
        # Convert to probability
        prob = hist / np.sum(hist)
        
        # Remove zeros (log(0) is undefined)
        prob = prob[prob > 0]
        
        # Calculate entropy
        if len(prob) > 0:
            entropy[i] = -np.sum(prob * np.log(prob))
        else:
            entropy[i] = 0
    
    return pd.Series(entropy, index=returns.index)

def validate_statistical(df, pair):
    """
    Validate statistical features for a cryptocurrency pair
    
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
    required_base_columns = ['close_1h', 'log_return']
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
    
    # Validate Standard Deviation of Returns
    if 'std_dev_returns_20' in df.columns:
        # Calculate expected std_dev
        expected_std_dev = calculate_std_dev_returns(df['log_return'])
        
        # Calculate absolute differences
        std_dev_diff = np.abs(df['std_dev_returns_20'] - expected_std_dev)
        std_dev_issues = df[std_dev_diff > threshold]
        
        issue_count = len(std_dev_issues)
        if issue_count > 0:
            issue_summary['std_dev_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in std_dev_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'std_dev_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_std_dev.loc[idx]) if not pd.isna(expected_std_dev.loc[idx]) else None,
                    'actual': float(row['std_dev_returns_20']),
                    'diff': float(std_dev_diff.loc[idx]),
                    'details': f"Standard Deviation of Returns calculation discrepancy"
                })
    
    # Validate Skewness
    if 'skewness_20' in df.columns:
        # Calculate expected skewness
        expected_skewness = calculate_skewness(df['log_return'])
        
        # Calculate absolute differences (larger threshold for skewness)
        skewness_diff = np.abs(df['skewness_20'] - expected_skewness)
        skewness_issues = df[skewness_diff > threshold * 10]  # Higher threshold for skewness
        
        issue_count = len(skewness_issues)
        if issue_count > 0:
            issue_summary['skewness_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in skewness_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'skewness_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_skewness.loc[idx]) if not pd.isna(expected_skewness.loc[idx]) else None,
                    'actual': float(row['skewness_20']),
                    'diff': float(skewness_diff.loc[idx]),
                    'details': f"Skewness calculation discrepancy"
                })
    
    # Validate Kurtosis
    if 'kurtosis_20' in df.columns:
        # Calculate expected kurtosis
        expected_kurtosis = calculate_kurtosis(df['log_return'])
        
        # Calculate absolute differences (larger threshold for kurtosis)
        kurtosis_diff = np.abs(df['kurtosis_20'] - expected_kurtosis)
        kurtosis_issues = df[kurtosis_diff > threshold * 10]  # Higher threshold for kurtosis
        
        issue_count = len(kurtosis_issues)
        if issue_count > 0:
            issue_summary['kurtosis_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in kurtosis_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'kurtosis_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_kurtosis.loc[idx]) if not pd.isna(expected_kurtosis.loc[idx]) else None,
                    'actual': float(row['kurtosis_20']),
                    'diff': float(kurtosis_diff.loc[idx]),
                    'details': f"Kurtosis calculation discrepancy"
                })
    
    # Validate Autocorrelation
    if 'autocorr_1' in df.columns:
        # Calculate expected autocorrelation
        expected_autocorr = calculate_autocorrelation(df['log_return'])
        
        # Calculate absolute differences
        autocorr_diff = np.abs(df['autocorr_1'] - expected_autocorr)
        autocorr_issues = df[autocorr_diff > threshold]
        
        issue_count = len(autocorr_issues)
        if issue_count > 0:
            issue_summary['autocorr_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in autocorr_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'autocorr_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_autocorr.loc[idx]) if not pd.isna(expected_autocorr.loc[idx]) else None,
                    'actual': float(row['autocorr_1']),
                    'diff': float(autocorr_diff.loc[idx]),
                    'details': f"Autocorrelation calculation discrepancy"
                })
    
    # Validate Hurst Exponent
    if 'hurst_exponent' in df.columns and len(df) >= 100:
        # Calculate expected Hurst exponent
        expected_hurst = calculate_hurst_exponent(df['close_1h'])
        
        # Calculate absolute differences
        hurst_diff = np.abs(df['hurst_exponent'] - expected_hurst)
        hurst_issues = df[hurst_diff > threshold * 10]  # Higher threshold for Hurst
        
        issue_count = len(hurst_issues)
        if issue_count > 0:
            issue_summary['hurst_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in hurst_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'hurst_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_hurst.loc[idx]) if not pd.isna(expected_hurst.loc[idx]) else None,
                    'actual': float(row['hurst_exponent']),
                    'diff': float(hurst_diff.loc[idx]),
                    'details': f"Hurst Exponent calculation discrepancy"
                })
    
    # Validate Shannon Entropy
    if 'shannon_entropy' in df.columns:
        # Calculate expected Shannon entropy
        expected_entropy = calculate_shannon_entropy(df['log_return'])
        
        # Calculate absolute differences
        entropy_diff = np.abs(df['shannon_entropy'] - expected_entropy)
        entropy_issues = df[entropy_diff > threshold]
        
        issue_count = len(entropy_issues)
        if issue_count > 0:
            issue_summary['entropy_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in entropy_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'entropy_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_entropy.loc[idx]) if not pd.isna(expected_entropy.loc[idx]) else None,
                    'actual': float(row['shannon_entropy']),
                    'diff': float(entropy_diff.loc[idx]),
                    'details': f"Shannon Entropy calculation discrepancy"
                })
    
    # Check for range consistency
    
    # 1. Hurst Exponent should be between 0 and 1
    if 'hurst_exponent' in df.columns:
        invalid_hurst = df[(df['hurst_exponent'] < 0) | (df['hurst_exponent'] > 1)]
        
        if not invalid_hurst.empty:
            if 'range_issues' not in issue_summary:
                issue_summary['range_issues'] = {'count': 0}
            
            issue_summary['range_issues']['count'] += len(invalid_hurst)
            
            # Record first few issues for reporting
            for idx, row in invalid_hurst.head(5).iterrows():
                issues.append({
                    'issue_type': 'range_violation',
                    'column': 'hurst_exponent',
                    'timestamp': row['timestamp_utc'],
                    'value': float(row['hurst_exponent']),
                    'details': f"Hurst Exponent outside valid range [0,1]: {row['hurst_exponent']}"
                })
    
    # 2. Shannon Entropy should be non-negative
    if 'shannon_entropy' in df.columns:
        invalid_entropy = df[df['shannon_entropy'] < 0]
        
        if not invalid_entropy.empty:
            if 'range_issues' not in issue_summary:
                issue_summary['range_issues'] = {'count': 0}
            
            issue_summary['range_issues']['count'] += len(invalid_entropy)
            
            # Record first few issues for reporting
            for idx, row in invalid_entropy.head(5).iterrows():
                issues.append({
                    'issue_type': 'range_violation',
                    'column': 'shannon_entropy',
                    'timestamp': row['timestamp_utc'],
                    'value': float(row['shannon_entropy']),
                    'details': f"Shannon Entropy is negative: {row['shannon_entropy']}"
                })
    
    # 3. Autocorrelation should be between -1 and 1
    if 'autocorr_1' in df.columns:
        invalid_autocorr = df[(df['autocorr_1'] < -1) | (df['autocorr_1'] > 1)]
        
        if not invalid_autocorr.empty:
            if 'range_issues' not in issue_summary:
                issue_summary['range_issues'] = {'count': 0}
            
            issue_summary['range_issues']['count'] += len(invalid_autocorr)
            
            # Record first few issues for reporting
            for idx, row in invalid_autocorr.head(5).iterrows():
                issues.append({
                    'issue_type': 'range_violation',
                    'column': 'autocorr_1',
                    'timestamp': row['timestamp_utc'],
                    'value': float(row['autocorr_1']),
                    'details': f"Autocorrelation outside valid range [-1,1]: {row['autocorr_1']}"
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
    main_validator(validate_statistical, "Statistical Features Validator", 
                  "Validates statistical features by recomputing and comparing them to stored values")