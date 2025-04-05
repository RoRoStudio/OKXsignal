#!/usr/bin/env python3
"""
Shared utilities for all data validation scripts
"""

import os
import logging
import argparse
import sys
import time
import json
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from tabulate import tabulate
from colorama import init, Fore, Back, Style
from tqdm import tqdm

# Initialize colorama
init(autoreset=True)

# ---------------------------
# Configuration 
# ---------------------------
DEFAULT_CONFIG_PATH = os.path.expanduser("~/OKXsignal/config/config.ini")
DEFAULT_CREDENTIALS_PATH = os.path.expanduser("~/OKXsignal/config/credentials.env")

# Load configuration
def load_config(config_path=None, credentials_path=None):
    """Load configuration from config.ini and credentials.env"""
    sys.path.append('/root/OKXsignal')
    
    try:
        from database_spot.processing.features.config import ConfigManager
        
        config_manager = ConfigManager(
            config_path=config_path or DEFAULT_CONFIG_PATH,
            credentials_path=credentials_path or DEFAULT_CREDENTIALS_PATH
        )
        
        return config_manager
    except ImportError:
        print(f"Error: Unable to import ConfigManager. Make sure OKXsignal is correctly installed.")
        sys.exit(1)

# ---------------------------
# Database Connection
# ---------------------------
def get_db_connection(config_manager):
    """Get a database connection using parameters from config"""
    db_params = config_manager.get_db_params()
    
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

# ---------------------------
# Data Fetching
# ---------------------------
def fetch_all_pairs(conn):
    """Fetch all distinct pairs from the database"""
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT DISTINCT pair FROM candles_1h ORDER BY pair")
        pairs = [row[0] for row in cursor.fetchall()]
        return pairs
    except Exception as e:
        print(f"Error fetching pairs: {e}")
        return []
    finally:
        cursor.close()

def fetch_data(conn, pair, limit=None):
    """
    Fetch data for a specific pair with optional limit
    
    Args:
        conn: Database connection
        pair: Cryptocurrency pair symbol
        limit: Optional limit for most recent records
        
    Returns:
        Pandas DataFrame with data
    """
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    try:
        if limit:
            query = """
            SELECT * FROM candles_1h
            WHERE pair = %s
            ORDER BY timestamp_utc DESC
            LIMIT %s
            """
            cursor.execute(query, (pair, limit))
        else:
            query = """
            SELECT * FROM candles_1h
            WHERE pair = %s
            ORDER BY timestamp_utc DESC
            """
            cursor.execute(query, (pair,))
        
        rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        # Convert to DataFrame
        # Use column names from cursor description rather than relying on DictCursor
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert timestamp to datetime if the column exists
        if 'timestamp_utc' in df.columns:
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)  # Add utc=True here
        else:
            # If timestamp_utc doesn't exist, log a warning and return empty DataFrame
            logging.warning(f"timestamp_utc column not found for {pair}")
            return pd.DataFrame()
        
        # Sort by timestamp (ascending)
        df = df.sort_values('timestamp_utc').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"Error fetching data for {pair}: {e}")
        return pd.DataFrame()
    
    finally:
        cursor.close()

# ---------------------------
# Progress Display
# ---------------------------
class ValidationProgress:
    """Class to handle progress display during validation"""
    
    def __init__(self, total_pairs, validator_name):
        """Initialize progress display"""
        self.total_pairs = total_pairs
        self.validator_name = validator_name
        self.pbar = tqdm(total=total_pairs, desc=f"Running {validator_name}")
        self.results = {}
        self.start_time = time.time()
    
    def update(self, pair, result):
        """Update progress and store result"""
        self.results[pair] = result
        self.pbar.update(1)
        
        # Show brief status in description
        if result.get('status') == 'error':
            self.pbar.set_description(f"{self.validator_name}: {pair} - Error")
        elif 'issues_count' in result and result['issues_count'] > 0:
            self.pbar.set_description(f"{self.validator_name}: {pair} - {result['issues_count']} issues")
        else:
            self.pbar.set_description(f"{self.validator_name}: {pair} - OK")
    
    def finish(self):
        """Finish progress tracking"""
        self.pbar.close()
        return {
            'results': self.results,
            'duration_seconds': time.time() - self.start_time,
            'total_pairs': self.total_pairs,
            'validator_name': self.validator_name,
            'timestamp': datetime.now().isoformat()
        }

# ---------------------------
# Reporting Functions
# ---------------------------
def generate_report(validation_results, validator_name, output_dir="reports"):
    """
    Generate validation report
    
    Args:
        validation_results: Dictionary with validation results
        validator_name: Name of the validator
        output_dir: Directory to save reports
        
    Returns:
        Path to the report files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{validator_name.lower().replace(' ', '_')}_{timestamp}"
    json_path = os.path.join(output_dir, f"{report_name}.json")
    txt_path = os.path.join(output_dir, f"{report_name}.txt")
    
    # Calculate summary statistics
    total_issues = 0
    pairs_with_issues = 0
    error_count = 0
    
    for pair, result in validation_results['results'].items():
        if result.get('status') == 'error':
            error_count += 1
        elif 'issues_count' in result:
            if result['issues_count'] > 0:
                pairs_with_issues += 1
                total_issues += result['issues_count']
    
    # Add summary to results
    validation_results['summary'] = {
        'total_pairs': validation_results['total_pairs'],
        'pairs_with_issues': pairs_with_issues,
        'total_issues': total_issues,
        'error_count': error_count,
        'duration_seconds': validation_results['duration_seconds']
    }
    
    # Save JSON report
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    # Generate text report - ADD encoding='utf-8' here
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"OKXsignal Validation Report: {validator_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {validation_results['duration_seconds']:.2f} seconds\n\n")
        
        f.write("Summary\n")
        f.write("-------\n")
        f.write(f"Total pairs analyzed: {validation_results['total_pairs']}\n")
        f.write(f"Pairs with issues: {pairs_with_issues}\n")
        f.write(f"Total issues found: {total_issues}\n")
        f.write(f"Pairs with errors: {error_count}\n\n")
        
        # Display overall status
        if total_issues == 0 and error_count == 0:
            f.write("VALIDATION RESULT: PASSED ✓\n\n")
        elif pairs_with_issues < validation_results['total_pairs'] * 0.05 and total_issues < 100:
            f.write("VALIDATION RESULT: MINOR ISSUES ⚠\n\n")
        else:
            f.write("VALIDATION RESULT: SIGNIFICANT ISSUES ✗\n\n")
        
        # List pairs with issues
        if pairs_with_issues > 0:
            f.write("Pairs with Issues\n")
            f.write("----------------\n")
            
            # Sort pairs by issue count
            issue_pairs = []
            for pair, result in validation_results['results'].items():
                if result.get('status') != 'error' and result.get('issues_count', 0) > 0:
                    issue_pairs.append((pair, result.get('issues_count', 0)))
            
            # Sort by issue count (descending)
            issue_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # List pairs with highest issues first
            for pair, count in issue_pairs[:20]:  # Show top 20 pairs
                f.write(f"{pair}: {count} issues\n")
                
                # Add brief description of issues if available
                if 'issue_summary' in validation_results['results'][pair]:
                    for category, details in validation_results['results'][pair]['issue_summary'].items():
                        f.write(f"  - {category}: {details['count']} issues\n")
            
            if len(issue_pairs) > 20:
                f.write(f"... and {len(issue_pairs) - 20} more pairs with issues\n")
            
            f.write("\n")
        
        # List pairs with errors
        if error_count > 0:
            f.write("Pairs with Errors\n")
            f.write("----------------\n")
            
            for pair, result in validation_results['results'].items():
                if result.get('status') == 'error':
                    f.write(f"{pair}: {result.get('error_message', 'Unknown error')}\n")
            
            f.write("\n")
        
        # Show detailed analysis if available
        if 'analysis' in validation_results:
            f.write("Detailed Analysis\n")
            f.write("-----------------\n")
            
            for category, details in validation_results['analysis'].items():
                f.write(f"\n{category}:\n")
                
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, dict):
                            f.write(f"  {key}:\n")
                            for subkey, subval in value.items():
                                f.write(f"    {subkey}: {subval}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {details}\n")
    
    print(f"\nReport saved to {txt_path} and {json_path}")
    return txt_path, json_path

# ---------------------------
# Result Formatting
# ---------------------------
def format_issue_table(issues, max_rows=10):
    """Format a table of issues for console display"""
    if not issues:
        return "No issues found."
    
    # Prepare table headers and rows
    headers = list(issues[0].keys())
    rows = [list(issue.values()) for issue in issues[:max_rows]]
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    
    if len(issues) > max_rows:
        table += f"\n... and {len(issues) - max_rows} more issues"
    
    return table

def print_validation_summary(results):
    """Print a summary of validation results to console"""
    total_pairs = len(results)
    pairs_with_issues = sum(1 for r in results.values() if r.get('issues_count', 0) > 0)
    pairs_with_errors = sum(1 for r in results.values() if r.get('status') == 'error')
    total_issues = sum(r.get('issues_count', 0) for r in results.values() if r.get('status') != 'error')
    
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs analyzed: {total_pairs}")
    print(f"Pairs with issues:    {pairs_with_issues} ({(pairs_with_issues/total_pairs)*100:.1f}%)")
    print(f"Total issues found:   {total_issues}")
    print(f"Pairs with errors:    {pairs_with_errors}")
    
    # Overall status with color
    if total_issues == 0 and pairs_with_errors == 0:
        print(f"\n{Fore.GREEN}VALIDATION RESULT: PASSED ✓{Style.RESET_ALL}")
    elif pairs_with_issues < total_pairs * 0.05 and total_issues < 100:
        print(f"\n{Fore.YELLOW}VALIDATION RESULT: MINOR ISSUES ⚠{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}VALIDATION RESULT: SIGNIFICANT ISSUES ✗{Style.RESET_ALL}")
    
    return {
        'total_pairs': total_pairs,
        'pairs_with_issues': pairs_with_issues,
        'total_issues': total_issues,
        'pairs_with_errors': pairs_with_errors
    }

# ---------------------------
# Common Argument Parsing
# ---------------------------
def parse_common_args(description):
    """Parse common command line arguments for all validators"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--rolling-window', type=int, default=450,
                       help='Number of recent candles to validate')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                       help='Path to config.ini file')
    parser.add_argument('--credentials', type=str, default=DEFAULT_CREDENTIALS_PATH,
                       help='Path to credentials.env file')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Directory for validation reports')
    parser.add_argument('--pairs', type=str,
                       help='Comma-separated list of pairs to validate (default: all)')
    
    return parser.parse_args()

# ---------------------------
# Common Functions for Validators
# ---------------------------
def validate_all_pairs(validator_function, validator_name, args):
    """
    Generic function to run validation across all pairs
    
    Args:
        validator_function: Function that validates a single pair
        validator_name: Name of the validator for reporting
        args: Command-line arguments
        
    Returns:
        Validation results dictionary
    """
    # Load configuration
    config_manager = load_config(args.config, args.credentials)
    
    # Get database connection
    conn = get_db_connection(config_manager)
    
    try:
        # Get pairs to validate
        if args.pairs:
            pairs = [p.strip() for p in args.pairs.split(',')]
            print(f"Validating {len(pairs)} specified pairs")
        else:
            pairs = fetch_all_pairs(conn)
            print(f"Found {len(pairs)} pairs to validate")
        
        if not pairs:
            print("No pairs found. Exiting.")
            return None
        
        # Initialize progress tracking
        progress = ValidationProgress(len(pairs), validator_name)
        
        # Validate each pair
        for pair in pairs:
            df = fetch_data(conn, pair, args.rolling_window)
            
            if df.empty:
                result = {
                    'pair': pair,
                    'status': 'no_data',
                    'issues_count': 0
                }
            else:
                try:
                    result = validator_function(df, pair)
                except Exception as e:
                    result = {
                        'pair': pair,
                        'status': 'error',
                        'error_message': str(e)
                    }
            
            progress.update(pair, result)
        
        # Finish progress tracking
        validation_results = progress.finish()
        
        # Generate report
        generate_report(validation_results, validator_name, args.output_dir)
        
        # Print summary
        results_summary = print_validation_summary(validation_results['results'])
        validation_results['summary'] = results_summary
        
        return validation_results
        
    finally:
        conn.close()

# Main function for individual validators
def main_validator(validator_function, validator_name, description):
    """
    Main function template for validators
    
    Args:
        validator_function: Function that validates a single pair
        validator_name: Name of the validator for reporting
        description: Description for the argument parser
    """
    args = parse_common_args(description)
    
    print(f"\n{Fore.CYAN}Running {validator_name}...{Style.RESET_ALL}")
    validate_all_pairs(validator_function, validator_name, args)