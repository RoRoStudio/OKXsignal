#!/usr/bin/env python3
"""
Run all validation scripts sequentially and generate a comprehensive report
"""

import os
import sys
import argparse
import json
import time
import datetime
from tabulate import tabulate
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# Import validators
sys.path.append('/root/OKXsignal')
from database_spot.validation.validation_utils import load_config, get_db_connection, fetch_all_pairs, fetch_data

# Import all validation functions
from database_spot.validation.validate_completeness import validate_completeness
from database_spot.validation.validate_raw_ohlcv import validate_raw_ohlcv
from database_spot.validation.validate_price_action import validate_price_action
from database_spot.validation.validate_momentum import validate_momentum
from database_spot.validation.validate_volatility import validate_volatility
from database_spot.validation.validate_volume_indicators import validate_volume_indicators
from database_spot.validation.validate_statistical import validate_statistical
from database_spot.validation.validate_pattern_recognition import validate_pattern_recognition
from database_spot.validation.validate_temporal_features import validate_temporal_features
from database_spot.validation.validate_cross_pair_features import validate_cross_pair_features
from database_spot.validation.validate_labels import validate_labels
from database_spot.validation.validate_targets_and_risk import validate_targets_and_risk
from database_spot.validation.validate_data_sanity import validate_data_sanity
from database_spot.validation.validate_data_distributions import validate_data_distributions

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run all validation scripts sequentially and generate a comprehensive report")
    
    parser.add_argument('--rolling-window', type=int, default=450,
                       help='Number of recent candles to validate')
    parser.add_argument('--config', type=str, default=os.path.expanduser("~/OKXsignal/config/config.ini"),
                       help='Path to config.ini file')
    parser.add_argument('--credentials', type=str, default=os.path.expanduser("~/OKXsignal/config/credentials.env"),
                       help='Path to credentials.env file')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Directory for validation reports')
    parser.add_argument('--pairs', type=str,
                       help='Comma-separated list of pairs to validate (default: all)')
    parser.add_argument('--skip', type=str,
                       help='Comma-separated list of validators to skip')
    
    return parser.parse_args()

def run_validator(validator_function, validator_name, df, pair):
    """Run a single validator and return the results"""
    print(f"  Running {validator_name}...")
    start_time = time.time()
    
    try:
        results = validator_function(df, pair)
        duration = time.time() - start_time
        
        # Add duration to results
        results['duration_seconds'] = duration
        
        issues_count = results.get('issues_count', 0)
        if issues_count > 0:
            print(f"  {Fore.YELLOW}Found {issues_count} issues{Style.RESET_ALL} in {duration:.2f} seconds")
        else:
            print(f"  {Fore.GREEN}No issues found{Style.RESET_ALL} in {duration:.2f} seconds")
            
        return results
    except Exception as e:
        print(f"  {Fore.RED}Error running {validator_name}: {e}{Style.RESET_ALL}")
        return {
            'pair': pair,
            'status': 'error',
            'error_message': str(e),
            'duration_seconds': time.time() - start_time
        }

def generate_comprehensive_report(all_results, args):
    """Generate a comprehensive report combining results from all validators"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{args.output_dir}/comprehensive_report_{timestamp}.txt"
    json_filename = f"{args.output_dir}/comprehensive_report_{timestamp}.json"
    
    # Calculate aggregate statistics
    total_pairs = len(all_results)
    pairs_with_issues = {}
    validators_with_issues = {}
    
    for pair, pair_results in all_results.items():
        pair_has_issues = False
        
        for validator, results in pair_results.items():
            # Check if results is a dictionary before using .get()
            if isinstance(results, dict):
                if results.get('status') == 'error':
                    pair_has_issues = True
                    
                    if validator not in validators_with_issues:
                        validators_with_issues[validator] = {'error_count': 0, 'issue_count': 0}
                    validators_with_issues[validator]['error_count'] += 1
                    
                elif results.get('issues_count', 0) > 0:
                    pair_has_issues = True
                    
                    if validator not in validators_with_issues:
                        validators_with_issues[validator] = {'error_count': 0, 'issue_count': 0}
                    validators_with_issues[validator]['issue_count'] += results['issues_count']
            else:
                # Handle case where results is not a dictionary
                pair_has_issues = True
                if validator not in validators_with_issues:
                    validators_with_issues[validator] = {'error_count': 0, 'issue_count': 0}
                validators_with_issues[validator]['error_count'] += 1
    
    # Generate text report
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("OKXsignal Comprehensive Validation Report\n")
        f.write("=======================================\n\n")
        
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pairs analyzed: {total_pairs}\n")
        f.write(f"Pairs with issues: {len(pairs_with_issues)}\n\n")
        
        # Overall status
        if len(pairs_with_issues) == 0:
            f.write("VALIDATION RESULT: PASSED ✓\n\n")
        elif len(pairs_with_issues) < total_pairs * 0.05:
            f.write("VALIDATION RESULT: MINOR ISSUES ⚠\n\n")
        else:
            f.write("VALIDATION RESULT: SIGNIFICANT ISSUES ✗\n\n")
        
        # Validator summary
        f.write("Validator Summary\n")
        f.write("----------------\n")
        
        validator_table = []
        for validator, stats in validators_with_issues.items():
            validator_table.append([
                validator,
                stats['issue_count'],
                stats['error_count']
            ])
        
        f.write(tabulate(
            validator_table, 
            headers=["Validator", "Issue Count", "Error Count"],
            tablefmt="grid"
        ))
        f.write("\n\n")
        
        # Top pairs with issues
        f.write("Top Pairs with Issues\n")
        f.write("-------------------\n")
        
        # Sort pairs by issue count
        top_pairs = sorted(
            pairs_with_issues.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]  # Top 20 pairs
        
        pair_table = []
        for pair, issue_count in top_pairs:
            pair_table.append([pair, issue_count])
        
        f.write(tabulate(
            pair_table,
            headers=["Pair", "Issue Count"],
            tablefmt="grid"
        ))
        f.write("\n\n")
        
        # Individual validator details
        f.write("Validator Details\n")
        f.write("----------------\n")
        
        validators = [
            "Data Completeness", "Raw OHLCV", "Price Action", "Momentum",
            "Volatility", "Volume Indicators", "Statistical", "Pattern Recognition",
            "Temporal Features", "Cross-Pair Features", "Labels", "Targets and Risk",
            "Data Sanity", "Data Distributions"
        ]
        
        for validator in validators:
            f.write(f"\n{validator}:\n")
            
            # Count pairs with issues for this validator
            pairs_with_validator_issues = 0
            total_validator_issues = 0
            
            for pair, pair_results in all_results.items():
                if validator in pair_results:
                    results = pair_results[validator]
                    
                    if results.get('status') == 'error':
                        pairs_with_validator_issues += 1
                    elif results.get('issues_count', 0) > 0:
                        pairs_with_validator_issues += 1
                        total_validator_issues += results['issues_count']
            
            f.write(f"  Pairs with issues: {pairs_with_validator_issues}/{total_pairs}\n")
            f.write(f"  Total issues found: {total_validator_issues}\n")
            
            # Show most common issue types if available
            issue_types = {}
            
            for pair, pair_results in all_results.items():
                if validator in pair_results:
                    results = pair_results[validator]
                    
                    if 'issue_summary' in results:
                        for issue_type, details in results['issue_summary'].items():
                            if issue_type not in issue_types:
                                issue_types[issue_type] = 0
                            if isinstance(details, dict):
                                issue_types[issue_type] += details.get('count', 0)
                            elif isinstance(details, (int, float)):
                                issue_types[issue_type] += details
                            else:
                                issue_types[issue_type] += 1  # fallback if totally unexpected

            
            if issue_types:
                f.write("  Most common issue types:\n")
                
                # Sort issue types by count
                sorted_issues = sorted(
                    issue_types.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # Top 5 issue types
                
                for issue_type, count in sorted_issues:
                    f.write(f"    - {issue_type}: {count}\n")
    
    # Save JSON data
    with open(json_filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_pairs': total_pairs,
            'pairs_with_issues': len(pairs_with_issues),
            'validators_with_issues': validators_with_issues,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\nComprehensive report saved to {report_filename}")
    print(f"JSON data saved to {json_filename}")
    
    return report_filename, json_filename

def main():
    """Main function to run all validators"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config_manager = load_config(args.config, args.credentials)
    
    # Get database connection
    print("Connecting to database...")
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
            return
        
        # Determine validators to skip
        skip_validators = []
        if args.skip:
            skip_validators = [v.strip() for v in args.skip.split(',')]
        
        # Define all validators
        validators = [
            {"name": "Data Completeness", "function": validate_completeness, "skip": "completeness" in skip_validators},
            {"name": "Raw OHLCV", "function": validate_raw_ohlcv, "skip": "raw_ohlcv" in skip_validators},
            {"name": "Price Action", "function": validate_price_action, "skip": "price_action" in skip_validators},
            {"name": "Momentum", "function": validate_momentum, "skip": "momentum" in skip_validators},
            {"name": "Volatility", "function": validate_volatility, "skip": "volatility" in skip_validators},
            {"name": "Volume Indicators", "function": validate_volume_indicators, "skip": "volume" in skip_validators},
            {"name": "Statistical", "function": validate_statistical, "skip": "statistical" in skip_validators},
            {"name": "Pattern Recognition", "function": validate_pattern_recognition, "skip": "pattern" in skip_validators},
            {"name": "Temporal Features", "function": validate_temporal_features, "skip": "temporal" in skip_validators},
            {"name": "Cross-Pair Features", "function": validate_cross_pair_features, "skip": "cross_pair" in skip_validators},
            {"name": "Labels", "function": validate_labels, "skip": "labels" in skip_validators},
            {"name": "Targets and Risk", "function": validate_targets_and_risk, "skip": "targets" in skip_validators},
            {"name": "Data Sanity", "function": validate_data_sanity, "skip": "sanity" in skip_validators},
            {"name": "Data Distributions", "function": validate_data_distributions, "skip": "distributions" in skip_validators}
        ]
        
        # Store all results
        all_results = {}
        
        # Process each pair
        for i, pair in enumerate(pairs):
            print(f"\n[{i+1}/{len(pairs)}] Processing {pair}")
            
            # Fetch data for this pair
            df = fetch_data(conn, pair, args.rolling_window)
            
            if df.empty:
                print(f"  {Fore.YELLOW}No data available for {pair}{Style.RESET_ALL}")
                all_results[pair] = {'status': 'no_data'}
                continue
            
            # Run each validator
            pair_results = {}
            
            for validator in validators:
                if validator["skip"]:
                    print(f"  {Fore.CYAN}Skipping {validator['name']}{Style.RESET_ALL}")
                    continue
                    
                validator_results = run_validator(
                    validator["function"],
                    validator["name"],
                    df,
                    pair
                )
                
                # Store results
                pair_results[validator["name"]] = validator_results
            
            # Store all results for this pair
            all_results[pair] = pair_results
        
        # Generate comprehensive report
        generate_comprehensive_report(all_results, args)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()