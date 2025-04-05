#!/usr/bin/env python3
"""
Targets and Risk Validator
- Validates: profit_target_1pct, profit_target_2pct, future_risk_adj_return_12h
- Ensures 1 = target hit, 0 = not hit, -1 = N/A
"""

import pandas as pd
import numpy as np
from database_spot.validation.validation_utils import main_validator

def validate_targets_and_risk(df, pair):
    """
    Validate target and risk features for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {
        'profit_target_issues': {'count': 0},
        'risk_adj_return_issues': {'count': 0}
    }
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Check for required columns
    required_columns = ['future_max_return_24h_pct', 'future_return_12h_pct', 'future_max_drawdown_12h_pct']
    target_columns = ['profit_target_1pct', 'profit_target_2pct', 'future_risk_adj_return_12h']
    
    # Check which columns are present
    missing_req_columns = [col for col in required_columns if col not in df.columns]
    present_target_columns = [col for col in target_columns if col in df.columns]
    
    # Skip if no target columns are present
    if not present_target_columns:
        return {
            'pair': pair,
            'status': 'no_target_columns',
            'issues_count': 0
        }
    
    # Validate profit targets
    profit_targets = {
        'profit_target_1pct': 0.01,
        'profit_target_2pct': 0.02
    }
    
    for target_col, threshold in profit_targets.items():
        if target_col in present_target_columns:
            if 'future_max_return_24h_pct' not in df.columns:
                # We need future_max_return to validate profit targets
                continue
            
            # Expected target hit: 1 if future_max_return_24h_pct > threshold, else 0
            expected_target = (df['future_max_return_24h_pct'] > threshold).astype(int)
            
            # Find mismatches
            target_issues = df[df[target_col] != expected_target]
            
            issue_count = len(target_issues)
            if issue_count > 0:
                issue_summary['profit_target_issues']['count'] += issue_count
                
                # Record first few issues for reporting
                for idx, row in target_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': 'profit_target_issue',
                        'target': target_col,
                        'timestamp': row['timestamp_utc'],
                        'expected': int(expected_target.loc[idx]),
                        'actual': int(row[target_col]),
                        'max_return': float(row['future_max_return_24h_pct']),
                        'threshold': threshold,
                        'details': f"{target_col} inconsistent with future_max_return_24h_pct"
                    })
            
            # Validate target values: should be only 0 or 1 (not -1 or other values)
            invalid_values = df[(df[target_col] != 0) & (df[target_col] != 1)]
            
            if not invalid_values.empty:
                issue_summary['profit_target_issues']['count'] += len(invalid_values)
                
                # Record first few issues for reporting
                for idx, row in invalid_values.head(5).iterrows():
                    issues.append({
                        'issue_type': 'invalid_target_value',
                        'target': target_col,
                        'timestamp': row['timestamp_utc'],
                        'value': int(row[target_col]),
                        'details': f"{target_col} should be 0 or 1, found {row[target_col]}"
                    })
    
    # Validate future_risk_adj_return_12h
    if 'future_risk_adj_return_12h' in present_target_columns:
        if 'future_return_12h_pct' in df.columns and 'future_max_drawdown_12h_pct' in df.columns:
            # Expected risk-adjusted return: future_return_12h_pct / abs(future_max_drawdown_12h_pct)
            # Handle division by zero by replacing zero drawdown with a small value
            safe_drawdown = np.abs(df['future_max_drawdown_12h_pct'].replace(0, 1e-8))
            expected_risk_adj = df['future_return_12h_pct'] / safe_drawdown
            
            # Remove inf and NaN values
            expected_risk_adj = expected_risk_adj.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Calculate absolute differences
            risk_adj_diff = np.abs(df['future_risk_adj_return_12h'] - expected_risk_adj)
            risk_adj_issues = df[risk_adj_diff > 1e-4]  # Using a slightly higher threshold for division operations
            
            issue_count = len(risk_adj_issues)
            if issue_count > 0:
                issue_summary['risk_adj_return_issues']['count'] = issue_count
                
                # Record first few issues for reporting
                for idx, row in risk_adj_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': 'risk_adj_return_issue',
                        'timestamp': row['timestamp_utc'],
                        'expected': float(expected_risk_adj.loc[idx]),
                        'actual': float(row['future_risk_adj_return_12h']),
                        'return': float(row['future_return_12h_pct']),
                        'drawdown': float(row['future_max_drawdown_12h_pct']),
                        'diff': float(risk_adj_diff.loc[idx]),
                        'details': f"future_risk_adj_return_12h calculation discrepancy"
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
    main_validator(validate_targets_and_risk, "Targets and Risk Validator", 
                  "Validates profit targets and risk-adjusted return calculations")