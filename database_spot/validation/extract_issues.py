#!/usr/bin/env python3
"""
Extract and analyze specific validation issues for focused debugging
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

# Import validation utilities
sys.path.append('/root/OKXsignal')
from database_spot.validation.validation_utils import load_config, get_db_connection, fetch_data

def extract_issues(report_path, validator_name, max_examples=5, issue_type=None):
    """
    Extract specific issue examples from a validation report
    
    Args:
        report_path: Path to the JSON report file
        validator_name: Name of the validator to extract issues from
        max_examples: Maximum number of examples to extract
        issue_type: Optional filter for specific issue type
        
    Returns:
        List of issue examples
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error loading report: {e}")
        return []
    
    issues = []
    
    # Extract issues from the report
    for pair, pair_results in report.get('results', {}).items():
        if validator_name in pair_results and 'issues' in pair_results[validator_name]:
            for issue in pair_results[validator_name]['issues']:
                # Filter by issue type if specified
                if issue_type and issue.get('issue_type') != issue_type:
                    continue
                    
                # Add pair information to the issue
                issue['pair'] = pair
                issues.append(issue)
                
                # Break if we have enough examples
                if len(issues) >= max_examples:
                    break
    
    return issues

def fetch_issue_data(conn, pair, timestamp, window=5):
    """
    Fetch data around the timestamp of an issue
    
    Args:
        conn: Database connection
        pair: Cryptocurrency pair
        timestamp: Timestamp of the issue
        window: Number of hours before and after to include
        
    Returns:
        DataFrame with data around the issue
    """
    try:
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        query = """
        SELECT * FROM candles_1h
        WHERE pair = %s AND timestamp_utc BETWEEN %s - INTERVAL %s HOUR AND %s + INTERVAL %s HOUR
        ORDER BY timestamp_utc
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (pair, timestamp, window, timestamp, window))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Format timestamp column
        if 'timestamp_utc' in df.columns:
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        
        return df
    
    except Exception as e:
        print(f"{Fore.RED}Error fetching data: {e}")
        return pd.DataFrame()

def get_computation_function(validator_name, issue_type):
    """Get the computation function based on validator name and issue type"""
    # Import validation functions based on validator name
    if validator_name == "Volatility":
        from database_spot.validation.validate_volatility import calculate_atr, calculate_true_range, calculate_bollinger_bands
        
        if 'atr' in issue_type.lower():
            return calculate_atr
        elif 'true_range' in issue_type.lower():
            return calculate_true_range
        elif 'bollinger' in issue_type.lower():
            return calculate_bollinger_bands
            
    elif validator_name == "Momentum":
        from database_spot.validation.validate_momentum import calculate_rsi, calculate_macd, calculate_stochastic
        
        if 'rsi' in issue_type.lower():
            return calculate_rsi
        elif 'macd' in issue_type.lower():
            return calculate_macd
        elif 'stoch' in issue_type.lower():
            return calculate_stochastic
            
    elif validator_name == "Volume Indicators":
        from database_spot.validation.validate_volume_indicators import calculate_obv, calculate_money_flow_index
        
        if 'obv' in issue_type.lower():
            return calculate_obv
        elif 'mfi' in issue_type.lower() or 'money_flow' in issue_type.lower():
            return calculate_money_flow_index
    
    return None

def analyze_issue(issue, df):
    """
    Analyze a specific issue and show relevant data
    
    Args:
        issue: Issue dictionary
        df: DataFrame with data around the issue
        
    Returns:
        Analysis results
    """
    try:
        # Get issue details
        issue_type = issue.get('issue_type', '')
        timestamp = issue.get('timestamp')
        expected = issue.get('expected')
        actual = issue.get('actual')
        details = issue.get('details', '')
        
        # Find the row with the issue
        issue_row = None
        if timestamp and 'timestamp_utc' in df.columns:
            issue_row = df[df['timestamp_utc'] == timestamp]
        
        # Print issue details
        print(f"\n{Fore.CYAN}Issue Analysis{Style.RESET_ALL}")
        print(f"Type: {issue_type}")
        print(f"Details: {details}")
        
        if expected is not None and actual is not None:
            diff = abs(float(expected) - float(actual))
            print(f"Expected: {expected}")
            print(f"Actual: {actual}")
            print(f"Difference: {diff}")
        
        # Show the data row with the issue
        if issue_row is not None and not issue_row.empty:
            print(f"\n{Fore.CYAN}Issue Row Data:{Style.RESET_ALL}")
            
            # Format for better readability
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            # Filter columns based on issue type for focused viewing
            if 'atr' in issue_type.lower() or 'true_range' in issue_type.lower():
                cols = ['timestamp_utc', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 'atr_1h', 'true_range']
            elif 'rsi' in issue_type.lower():
                cols = ['timestamp_utc', 'close_1h', 'rsi_1h', 'rsi_slope_1h']
            elif 'macd' in issue_type.lower():
                cols = ['timestamp_utc', 'close_1h', 'macd_slope_1h', 'macd_hist_slope_1h']
            elif 'volume' in issue_type.lower():
                cols = ['timestamp_utc', 'close_1h', 'volume_1h', 'taker_buy_base_1h']
            else:
                cols = df.columns
            
            # Select only columns that exist in the dataframe
            display_cols = [col for col in cols if col in df.columns]
            
            print(issue_row[display_cols].to_string(index=False))
        
        # Attempt to recalculate the expected value
        compute_fn = get_computation_function(issue.get('validator', ''), issue_type)
        
        if compute_fn and issue_row is not None and not issue_row.empty:
            print(f"\n{Fore.CYAN}Computation Analysis:{Style.RESET_ALL}")
            print(f"Using function: {compute_fn.__name__}")
            
            # Extract required columns for computation based on issue type
            if 'atr' in issue_type.lower() or 'true_range' in issue_type.lower():
                if all(col in df.columns for col in ['high_1h', 'low_1h', 'close_1h']):
                    result = compute_fn(df['high_1h'], df['low_1h'], df['close_1h'])
                    print(f"Recalculated result: {result}")
            
            # Add more computation cases as needed
        
        return {
            'issue_type': issue_type,
            'expected': expected,
            'actual': actual,
            'difference': abs(float(expected) - float(actual)) if expected is not None and actual is not None else None
        }
    
    except Exception as e:
        print(f"{Fore.RED}Error analyzing issue: {e}")
        return {}

def suggest_fixes(validator_name, issue_type):
    """
    Suggest possible fixes based on validator and issue type
    
    Args:
        validator_name: Name of the validator
        issue_type: Type of issue
        
    Returns:
        Suggestions for fixing the issue
    """
    suggestions = []
    
    # Volatility issues
    if validator_name == "Volatility":
        if 'atr' in issue_type.lower():
            suggestions = [
                "Check the ATR calculation method in feature_processor.py",
                "Ensure Wilder's smoothing is applied correctly",
                "Verify the handling of NaN values and initial ATR calculation"
            ]
        elif 'bollinger' in issue_type.lower():
            suggestions = [
                "Verify the standard deviation calculation",
                "Check the moving average method (SMA vs EMA)",
                "Ensure the multiplication factor is correct (usually 2)"
            ]
    
    # Momentum issues
    elif validator_name == "Momentum":
        if 'rsi' in issue_type.lower():
            suggestions = [
                "Check the RSI smoothing method",
                "Verify the handling of first RSI value",
                "Ensure proper averaging of gains and losses"
            ]
        elif 'macd' in issue_type.lower():
            suggestions = [
                "Verify the EMA periods (12, 26, 9)",
                "Check the calculation of the signal line",
                "Ensure proper calculation of histogram and slopes"
            ]
    
    # Volume issues
    elif validator_name == "Volume Indicators":
        if 'obv' in issue_type.lower():
            suggestions = [
                "Check the OBV cumulative calculation",
                "Verify the price comparison logic",
                "Ensure proper handling of flat prices"
            ]
    
    # Raw OHLCV issues
    elif validator_name == "Raw OHLCV":
        suggestions = [
            "Check for data consistency (high >= low, high >= open, high >= close, etc.)",
            "Verify volume values are non-negative",
            "Ensure taker buy values <= volume"
        ]
    
    # Default suggestions
    if not suggestions:
        suggestions = [
            f"Review the calculation method in feature_processor.py for {issue_type}",
            "Compare with the validation function in the validator",
            "Check for differences in handling edge cases or NaN values"
        ]
    
    return suggestions

def main():
    parser = argparse.ArgumentParser(description="Extract and analyze validation issues for debugging")
    parser.add_argument('report', type=str, help='Path to validation report JSON file')
    parser.add_argument('validator', type=str, help='Name of the validator to extract issues from')
    parser.add_argument('--issue-type', type=str, help='Filter by specific issue type')
    parser.add_argument('--max-examples', type=int, default=5, help='Maximum number of examples to extract')
    parser.add_argument('--window', type=int, default=5, help='Hours before/after issue to include in data')
    args = parser.parse_args()
    
    # Extract issues
    issues = extract_issues(args.report, args.validator, args.max_examples, args.issue_type)
    
    if not issues:
        print(f"{Fore.YELLOW}No issues found for {args.validator}")
        if args.issue_type:
            print(f"Try without the --issue-type filter or check the issue type spelling.")
        return
    
    # Show issue summary
    print(f"{Fore.GREEN}Found {len(issues)} issues for {args.validator}")
    
    # Connect to database
    config_manager = load_config()
    conn = get_db_connection(config_manager)
    
    try:
        # Analyze each issue
        for i, issue in enumerate(issues):
            print(f"\n{Fore.CYAN}Issue {i+1}/{len(issues)}{Style.RESET_ALL}")
            print(f"Pair: {issue['pair']}")
            
            # Add validator name to issue for computation
            issue['validator'] = args.validator
            
            # Fetch data around the issue
            if 'timestamp' in issue:
                issue_data = fetch_issue_data(conn, issue['pair'], issue['timestamp'], args.window)
                
                if not issue_data.empty:
                    # Analyze the issue
                    analyze_issue(issue, issue_data)
                else:
                    print(f"{Fore.YELLOW}No data found around issue timestamp")
            else:
                print(f"{Fore.YELLOW}No timestamp in issue data")
        
        # Suggest fixes
        if issues and 'issue_type' in issues[0]:
            print(f"\n{Fore.GREEN}Suggested Fixes:{Style.RESET_ALL}")
            for suggestion in suggest_fixes(args.validator, issues[0]['issue_type']):
                print(f"- {suggestion}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()