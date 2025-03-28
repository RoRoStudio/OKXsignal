#!/usr/bin/env python3
"""
Data Distributions Validator
- Summary stats, histograms, and sanity bounds for each feature group
- Helps spot outliers visually/statistically
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from database.validation.validation_utils import main_validator

def compute_distribution_statistics(df, column):
    """Compute distribution statistics for a column"""
    data = df[column].dropna()
    
    if len(data) == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'min': None,
            'max': None,
            'std': None,
            'skew': None,
            'kurtosis': None,
            'q1': None,
            'q3': None,
            'iqr': None,
            'outliers_count': 0,
            'zeros_count': 0,
            'unique_count': 0
        }
    
    # Calculate basic statistics
    stats_dict = {
        'count': len(data),
        'mean': float(data.mean()),
        'median': float(data.median()),
        'min': float(data.min()),
        'max': float(data.max()),
        'std': float(data.std()),
        'skew': float(data.skew()),
        'kurtosis': float(data.kurtosis()),
        'q1': float(data.quantile(0.25)),
        'q3': float(data.quantile(0.75)),
        'zeros_count': int((data == 0).sum()),
        'unique_count': int(data.nunique())
    }
    
    # Calculate IQR and outliers
    stats_dict['iqr'] = stats_dict['q3'] - stats_dict['q1']
    
    # Count outliers (using 1.5 * IQR rule)
    lower_bound = stats_dict['q1'] - 1.5 * stats_dict['iqr']
    upper_bound = stats_dict['q3'] + 1.5 * stats_dict['iqr']
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    stats_dict['outliers_count'] = len(outliers)
    
    return stats_dict

def create_histogram(df, column, output_dir, pair):
    """Create a histogram for a column"""
    plt.figure(figsize=(10, 6))
    
    # Drop NaN values
    data = df[column].dropna()
    
    if len(data) == 0:
        plt.close()
        return None
    
    # Create histogram
    plt.hist(data, bins=50, alpha=0.7, color='steelblue')
    
    # Add titles and labels
    plt.title(f"{pair} - {column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    # Add key statistics as text
    stats_text = f"Mean: {data.mean():.4f}\n"
    stats_text += f"Median: {data.median():.4f}\n"
    stats_text += f"Std Dev: {data.std():.4f}\n"
    stats_text += f"Min: {data.min():.4f}\n"
    stats_text += f"Max: {data.max():.4f}\n"
    stats_text += f"Skew: {data.skew():.4f}\n"
    stats_text += f"Outliers: {((data < data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))) | (data > data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)))).sum()}"
    
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="steelblue", alpha=0.8),
                 verticalalignment='top')
    
    # Save the histogram
    os.makedirs(f"{output_dir}/histograms/{pair}", exist_ok=True)
    filename = f"{output_dir}/histograms/{pair}/{column.replace('/', '_')}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def detect_outliers(df, column):
    """Detect outliers in a column using the IQR method"""
    data = df[column].dropna()
    
    if len(data) == 0:
        return pd.DataFrame()
    
    # Calculate IQR
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    # Define bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Find outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers

def validate_data_distributions(df, pair, output_dir="reports"):
    """
    Validate data distributions for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        output_dir: Directory to save reports and visualizations
        
    Returns:
        Dictionary with validation results
    """
    # Create output directory for histograms
    os.makedirs(f"{output_dir}/histograms", exist_ok=True)
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Group columns by feature type
    feature_groups = {
        'price': ['open_1h', 'high_1h', 'low_1h', 'close_1h'],
        'volume': ['volume_1h', 'taker_buy_base_1h'],
        'price_action': ['body_size', 'upper_shadow', 'lower_shadow', 'relative_close_position', 
                       'log_return', 'gap_open', 'price_velocity', 'price_acceleration'],
        'momentum': ['rsi_1h', 'rsi_slope_1h', 'macd_slope_1h', 'macd_hist_slope_1h', 
                   'stoch_k_14', 'stoch_d_14', 'williams_r_14', 'cci_14', 'roc_10', 
                   'tsi', 'awesome_oscillator', 'ppo'],
        'volatility': ['atr_1h', 'true_range', 'normalized_atr_14', 'bollinger_width_1h',
                     'bollinger_percent_b', 'donchian_channel_width_1h', 'keltner_channel_width',
                     'historical_vol_30', 'chaikin_volatility', 'z_score_20'],
        'volume_indicators': ['money_flow_index_1h', 'obv_1h', 'obv_slope_1h', 'volume_change_pct_1h',
                            'vwma_20', 'chaikin_money_flow', 'klinger_oscillator',
                            'volume_oscillator', 'volume_price_trend', 'volume_zone_oscillator'],
        'statistical': ['std_dev_returns_20', 'skewness_20', 'kurtosis_20', 'hurst_exponent',
                      'shannon_entropy', 'autocorr_1'],
        'patterns': ['pattern_doji', 'pattern_engulfing', 'pattern_hammer', 'pattern_morning_star'],
        'time': ['hour_of_day', 'day_of_week', 'month_of_year', 'is_weekend', 
               'asian_session', 'european_session', 'american_session'],
        'cross_pair': ['volume_rank_1h', 'volatility_rank_1h', 'performance_rank_btc_1h',
                     'performance_rank_eth_1h', 'btc_corr_24h', 'prev_volume_rank'],
        'labels': ['future_return_1h_pct', 'future_return_4h_pct', 'future_return_12h_pct',
                 'future_return_1d_pct', 'future_max_return_24h_pct', 'future_max_drawdown_12h_pct',
                 'was_profitable_12h', 'future_risk_adj_return_12h', 'profit_target_1pct', 'profit_target_2pct']
    }
    
    # Compute distribution statistics for each column
    distribution_stats = {}
    significant_outliers = {}
    histograms = {}
    
    # Process each feature group
    for group, columns in feature_groups.items():
        group_stats = {}
        group_outliers = {}
        group_histograms = {}
        
        # Process each column in the group
        for col in columns:
            if col in df.columns:
                # Compute statistics
                col_stats = compute_distribution_statistics(df, col)
                group_stats[col] = col_stats
                
                # Create histogram
                if col_stats['count'] > 0:
                    histogram_path = create_histogram(df, col, output_dir, pair)
                    group_histograms[col] = histogram_path
                
                # Check for significant outliers
                if col_stats['outliers_count'] > 0:
                    outliers_pct = col_stats['outliers_count'] / col_stats['count'] * 100
                    
                    # If more than 1% of values are outliers, consider it significant
                    if outliers_pct > 1:
                        outlier_rows = detect_outliers(df, col)
                        group_outliers[col] = {
                            'count': col_stats['outliers_count'],
                            'percentage': outliers_pct,
                            'min_value': float(outlier_rows[col].min()),
                            'max_value': float(outlier_rows[col].max()),
                            'first_timestamp': outlier_rows['timestamp_utc'].min().isoformat() if not outlier_rows.empty else None,
                            'last_timestamp': outlier_rows['timestamp_utc'].max().isoformat() if not outlier_rows.empty else None
                        }
        
        # Store results for this group
        if group_stats:
            distribution_stats[group] = group_stats
        if group_outliers:
            significant_outliers[group] = group_outliers
        if group_histograms:
            histograms[group] = group_histograms
    
    # Calculate total outliers
    total_outliers = sum(
        sum(o['count'] for o in group.values())
        for group in significant_outliers.values()
    )
    
    # Calculate feature coverage
    all_expected_features = [col for group in feature_groups.values() for col in group]
    present_features = [col for col in all_expected_features if col in df.columns]
    feature_coverage = len(present_features) / len(all_expected_features) * 100
    
    # Calculate total zeros percentage by feature group
    zeros_by_group = {}
    for group, columns in feature_groups.items():
        group_columns = [col for col in columns if col in df.columns]
        if group_columns:
            total_cells = sum(distribution_stats.get(group, {}).get(col, {}).get('count', 0) for col in group_columns)
            total_zeros = sum(distribution_stats.get(group, {}).get(col, {}).get('zeros_count', 0) for col in group_columns)
            zeros_by_group[group] = {
                'total_cells': total_cells,
                'zeros_count': total_zeros,
                'zeros_pct': (total_zeros / total_cells * 100) if total_cells > 0 else 0
            }
    
    # Identify potential issues
    issues = []
    
    # Check for high percentage of zeros in feature groups
    for group, zeros_info in zeros_by_group.items():
        if zeros_info['zeros_pct'] > 40:  # Arbitrary threshold, adjust as needed
            issues.append({
                'issue_type': 'high_zeros_percentage',
                'feature_group': group,
                'zeros_count': zeros_info['zeros_count'],
                'total_cells': zeros_info['total_cells'],
                'percentage': zeros_info['zeros_pct'],
                'details': f"High percentage of zeros ({zeros_info['zeros_pct']:.2f}%) in {group} feature group"
            })
    
    # Check for extreme skewness in distributions
    for group, group_stats in distribution_stats.items():
        for col, stats in group_stats.items():
            if stats['skew'] is not None:
                if abs(stats['skew']) > 10:  # Arbitrary threshold for extreme skewness
                    issues.append({
                        'issue_type': 'extreme_skewness',
                        'feature_group': group,
                        'feature': col,
                        'skewness': stats['skew'],
                        'details': f"Extreme skewness ({stats['skew']:.2f}) in {col} distribution"
                    })
    
    # Add significant outliers to issues
    for group, group_outliers in significant_outliers.items():
        for col, outlier_info in group_outliers.items():
            if outlier_info['percentage'] > 5:  # Threshold for concerning outlier percentage
                issues.append({
                    'issue_type': 'high_outliers_percentage',
                    'feature_group': group,
                    'feature': col,
                    'outliers_count': outlier_info['count'],
                    'percentage': outlier_info['percentage'],
                    'min_value': outlier_info['min_value'],
                    'max_value': outlier_info['max_value'],
                    'details': f"High percentage of outliers ({outlier_info['percentage']:.2f}%) in {col}"
                })
    
    # Create summary
    issue_summary = {
        'feature_coverage': feature_coverage,
        'total_outliers': total_outliers,
        'outliers_by_group': {group: sum(o['count'] for o in g.values()) for group, g in significant_outliers.items()},
        'zeros_by_group': zeros_by_group
    }
    
    return {
        'pair': pair,
        'status': 'completed',
        'issues_count': len(issues),
        'candles_count': len(df),
        'issue_summary': issue_summary,
        'distribution_stats': distribution_stats,
        'significant_outliers': significant_outliers,
        'histograms': histograms,
        'issues': issues
    }

if __name__ == "__main__":
    main_validator(validate_data_distributions, "Data Distributions Validator", 
                  "Validates distribution characteristics and identifies statistical outliers")