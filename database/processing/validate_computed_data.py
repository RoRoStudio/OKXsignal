#!/usr/bin/env python3
"""
Cryptocurrency Data Validation Tool
- Validates the computed features in the candles_1h table
- Checks for data quality issues, gaps, mathematical correctness
- Generates detailed validation reports
"""

# python validate_computed_data.py --mode rolling_validation --rolling-window 450 --batch-size 2 --max-connections 4

import os
import sys
import logging
import argparse
import time
import json
import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO
import psycopg2
import psycopg2.extras
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import configuration
sys.path.append('/root/OKXsignal')
from database.processing.features.config import ConfigManager
from database.processing.features.db_pool import initialize_pool, get_connection, close_all_connections

# Import feature calculators for verification
from database.processing.features.momentum import MomentumFeatures
from database.processing.features.volatility import VolatilityFeatures
from database.processing.features.volume import VolumeFeatures
from database.processing.features.price_action import PriceActionFeatures
from database.processing.features.statistical import StatisticalFeatures
from database.processing.features.pattern import PatternFeatures
from database.processing.features.time import TimeFeatures
from database.processing.features.labels import LabelFeatures
from database.processing.features.cross_pair import CrossPairFeatures

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

# ---------------------------
# Logging Setup
# ---------------------------
def setup_logging(log_dir="logs", log_level="INFO"):
    """Set up application logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = os.path.join(log_dir, f"validate_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(levelname)s] %(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("validate_data")

# ---------------------------
# Signal Handling
# ---------------------------
def setup_signal_handlers():
    """Set up signal handlers for graceful termination"""
    def signal_handler(sig, frame):
        global SHUTDOWN_REQUESTED
        if SHUTDOWN_REQUESTED:
            logging.info("Forced shutdown requested. Exiting immediately.")
            sys.exit(1)
            
        logging.info("Shutdown requested. Finishing current tasks and exiting...")
        SHUTDOWN_REQUESTED = True
        
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)

# ---------------------------
# Data Fetching
# ---------------------------
def fetch_data(db_conn, pair, start_date=None, end_date=None, limit=None):
    """
    Fetch candle data for a specific pair
    
    Args:
        db_conn: Database connection
        pair: Symbol pair (e.g., 'BTC-USDT')
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        limit: Optional limit for number of rows
        
    Returns:
        Pandas DataFrame with candle data
    """
    cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    try:
        query = "SELECT * FROM candles_1h WHERE pair = %s"
        params = [pair]
        
        if start_date:
            query += " AND timestamp_utc >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp_utc <= %s"
            params.append(end_date)
        
        query += " ORDER BY timestamp_utc"
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Check if DataFrame is empty
        if df.empty:
            return df
        
        # Convert timestamp_utc to datetime if exists and not already datetime
        if 'timestamp_utc' in df.columns:
            if not pd.api.types.is_datetime64_dtype(df['timestamp_utc']):
                df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        
        return df
    
    except Exception as e:
        logging.error(f"Error fetching data for {pair}: {str(e)}")
        return pd.DataFrame()
    
    finally:
        if cursor and not cursor.closed:
            cursor.close()

def get_all_pairs(db_conn):
    """Get list of all pairs in the database"""
    cursor = db_conn.cursor()
    
    try:
        cursor.execute("SELECT DISTINCT pair FROM candles_1h")
        pairs = [row[0] for row in cursor.fetchall()]
        return pairs
    
    except Exception as e:
        logging.error(f"Error getting pairs: {str(e)}")
        return []
    
    finally:
        if cursor and not cursor.closed:
            cursor.close()

def get_table_columns(db_conn, table_name='candles_1h'):
    """Get all columns in the table"""
    cursor = db_conn.cursor()
    
    try:
        cursor.execute(f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """)
        columns = [(row[0], row[1]) for row in cursor.fetchall()]
        return columns
    
    except Exception as e:
        logging.error(f"Error getting table columns: {str(e)}")
        return []
    
    finally:
        if cursor and not cursor.closed:
            cursor.close()

# ---------------------------
# Validation Functions
# ---------------------------
def validate_missing_values(df, pair):
    """
    Check for missing values in all columns
    
    Args:
        df: DataFrame to check
        pair: Pair symbol for context
        
    Returns:
        Dictionary with missing value statistics
    """
    if df.empty:
        return {
            "pair": pair,
            "status": "empty_dataframe",
            "total_rows": 0,
            "total_missing_cells": 0,
            "missing_pct_overall": 0,
            "columns_with_missing": {}
        }
    
    # Count missing values per column
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    # Get columns with missing values
    missing_cols = missing[missing > 0]
    
    result = {
        "pair": pair,
        "total_rows": len(df),
        "total_missing_cells": df.isnull().sum().sum(),
        "missing_pct_overall": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        "columns_with_missing": {col: {"count": int(count), "pct": float(missing_pct[col])} 
                               for col, count in missing_cols.items()}
    }
    
    return result

def validate_timestamps(df, pair, candle_interval=3600):
    """
    Check for gaps in timestamps
    
    Args:
        df: DataFrame to check
        pair: Pair symbol for context
        candle_interval: Expected time interval between candles in seconds (default: 1h)
        
    Returns:
        Dictionary with timestamp gap information
    """
    # Check if DataFrame is empty or missing timestamp column
    if df.empty:
        return {"pair": pair, "status": "empty_dataframe"}
    
    if 'timestamp_utc' not in df.columns:
        return {"pair": pair, "status": "missing_timestamp_column"}
    
    if len(df) < 2:
        return {"pair": pair, "status": "insufficient_data"}
    
    # Sort by timestamp
    df = df.sort_values('timestamp_utc')
    
    # Calculate time differences
    try:
        df['time_diff'] = df['timestamp_utc'].diff().dt.total_seconds()
    except Exception as e:
        return {"pair": pair, "status": "error", "error_message": f"Error calculating time differences: {str(e)}"}
    
    # Find gaps (where difference is greater than expected interval)
    expected_diff = candle_interval
    gaps = df[df['time_diff'] > expected_diff * 1.1]  # Allow for 10% margin
    
    result = {
        "pair": pair,
        "total_rows": len(df),
        "expected_interval_seconds": candle_interval,
        "gaps_found": len(gaps),
        "gaps_details": []
    }
    
    # Add details of each gap (limit to 100 for report size)
    if len(gaps) > 0:
        for _, row in gaps.head(100).iterrows():
            try:
                result["gaps_details"].append({
                    "timestamp": row['timestamp_utc'].isoformat(),
                    "previous_timestamp": (row['timestamp_utc'] - timedelta(seconds=row['time_diff'])).isoformat(),
                    "gap_seconds": float(row['time_diff']),
                    "gap_hours": float(row['time_diff'] / 3600)
                })
            except:
                continue
    
    return result

def validate_price_consistency(df, pair):
    """
    Check for price consistency (high >= low, high >= open, high >= close, etc.)
    
    Args:
        df: DataFrame to check
        pair: Pair symbol for context
        
    Returns:
        Dictionary with price consistency information
    """
    # Check for required columns
    required_columns = ['high_1h', 'low_1h', 'open_1h', 'close_1h', 'volume_1h']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return {
            "pair": pair,
            "status": "missing_columns",
            "missing_columns": missing_columns
        }
    
    if df.empty:
        return {"pair": pair, "status": "empty_dataframe"}
    
    # Check for inconsistencies
    inconsistencies = {
        "high_lower_than_low": int(len(df[df['high_1h'] < df['low_1h']])),
        "high_lower_than_open": int(len(df[df['high_1h'] < df['open_1h']])),
        "high_lower_than_close": int(len(df[df['high_1h'] < df['close_1h']])),
        "low_higher_than_open": int(len(df[df['low_1h'] > df['open_1h']])),
        "low_higher_than_close": int(len(df[df['low_1h'] > df['close_1h']])),
        "negative_volume": int(len(df[df['volume_1h'] < 0])),
        "zero_volume": int(len(df[df['volume_1h'] == 0])),
    }
    
    # Check for extreme values
    price_columns = ['open_1h', 'high_1h', 'low_1h', 'close_1h']
    extreme_values = {}
    
    for col in price_columns:
        try:
            # Find potential outliers (3+ standard deviations)
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[abs(df[col] - mean) > 3 * std]
            
            if len(outliers) > 0:
                extreme_values[col] = {
                    "count": int(len(outliers)),
                    "pct": float((len(outliers) / len(df)) * 100),
                    "examples": [float(x) for x in outliers[col].tolist()[:5]]  # Show up to 5 examples
                }
        except Exception as e:
            extreme_values[col] = {"error": str(e)}
    
    result = {
        "pair": pair,
        "total_rows": int(len(df)),
        "inconsistencies": inconsistencies,
        "extreme_values": extreme_values,
        "has_issues": any(v > 0 for v in inconsistencies.values()) or len(extreme_values) > 0
    }
    
    return result

def validate_indicator_ranges(df, pair):
    """
    Check if indicators are within expected ranges
    
    Args:
        df: DataFrame to check
        pair: Pair symbol for context
        
    Returns:
        Dictionary with indicators range validation
    """
    if df.empty:
        return {"pair": pair, "status": "empty_dataframe"}
    
    expected_ranges = {
        "rsi_1h": (0, 100),
        "stoch_k_14": (0, 100),
        "stoch_d_14": (0, 100),
        "williams_r_14": (-100, 0),
        "bollinger_percent_b": (0, 1),
        "money_flow_index_1h": (0, 100),
        "volume_rank_1h": (0, 100),
        "volatility_rank_1h": (0, 100),
        "btc_corr_24h": (-1, 1),
    }
    
    out_of_range = {}
    
    for indicator, (min_val, max_val) in expected_ranges.items():
        if indicator in df.columns:
            try:
                outside_range = df[(df[indicator] < min_val) | (df[indicator] > max_val)]
                
                if len(outside_range) > 0:
                    out_of_range[indicator] = {
                        "count": int(len(outside_range)),
                        "pct": float((len(outside_range) / len(df)) * 100),
                        "expected_range": f"{min_val} to {max_val}",
                        "examples": {
                            "values": [float(x) for x in outside_range[indicator].tolist()[:5]],
                            "timestamps": [ts.isoformat() if isinstance(ts, datetime) else str(ts) 
                                         for ts in outside_range["timestamp_utc"].tolist()[:5]] if "timestamp_utc" in outside_range.columns else [],
                        }
                    }
            except Exception as e:
                out_of_range[indicator] = {"error": str(e)}
    
    result = {
        "pair": pair,
        "total_rows": int(len(df)),
        "out_of_range_indicators": out_of_range,
        "has_issues": len(out_of_range) > 0
    }
    
    return result

def validate_future_returns(df, pair):
    """
    Validate future return calculations
    
    Args:
        df: DataFrame to check
        pair: Pair symbol for context
        
    Returns:
        Dictionary with future return validation results
    """
    if df.empty:
        return {"pair": pair, "status": "empty_dataframe"}
    
    future_return_columns = [col for col in df.columns if col.startswith('future_return_') and col.endswith('_pct')]
    
    if not future_return_columns:
        return {
            "pair": pair,
            "status": "no_future_return_columns"
        }
    
    # Check if required columns exist
    if 'timestamp_utc' not in df.columns or 'close_1h' not in df.columns:
        return {
            "pair": pair,
            "status": "missing_required_columns",
            "missing": [col for col in ['timestamp_utc', 'close_1h'] if col not in df.columns]
        }
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Validate each future return column
    validation_results = {}
    
    for col in future_return_columns:
        try:
            # Extract horizon from column name
            horizon = col.split('_')[2]
            
            # Convert horizon to number of periods
            if horizon == '1h':
                periods = 1
            elif horizon == '4h':
                periods = 4
            elif horizon == '12h':
                periods = 12
            elif horizon == '1d':
                periods = 24
            elif horizon == '3d':
                periods = 72
            elif horizon == '1w':
                periods = 168
            elif horizon == '2w':
                periods = 336
            else:
                validation_results[col] = {"error": f"Unknown horizon: {horizon}"}
                continue
            
            # Check for zeros at the end (most recent candles)
            recent_candles = df.tail(periods)[col]
            zero_count = (recent_candles == 0).sum()
            
            # Check that we have some non-zero future returns
            non_zero_count = (df[col] != 0).sum()
            
            validation_results[col] = {
                "periods": periods,
                "zero_values_at_end": int(zero_count),
                "expected_zeros_at_end": periods,  # We expect recent candles to have zero future returns
                "correct_zeros_at_end": zero_count >= periods * 0.8,  # Allow for some slight variation
                "total_non_zero_values": int(non_zero_count),
                "pct_non_zero": float((non_zero_count / len(df)) * 100) if len(df) > 0 else 0.0
            }
            
            # Add pattern analysis - values should decrease to zero at the end
            if len(df) > periods * 1.5:
                # Check if values transition from non-zero to zero near the end
                last_n = df.tail(periods * 2)[col].values
                has_transition = np.any(last_n[:periods] != 0) and np.all(last_n[-periods//2:] == 0)
                validation_results[col]["has_expected_end_pattern"] = bool(has_transition)
            
        except Exception as e:
            validation_results[col] = {"error": str(e)}
    
    return {
        "pair": pair,
        "future_return_validations": validation_results
    }

def validate_pair(pair, config_manager, validation_mode='rolling_validation', rolling_window=None):
    """
    Validate a single cryptocurrency pair
    
    Args:
        pair: Symbol pair to validate
        config_manager: Configuration manager
        validation_mode: 'rolling_validation' or 'full_validation'
        rolling_window: Number of recent candles to validate
        
    Returns:
        Dictionary with validation results
    """
    # Check if shutdown was requested
    if SHUTDOWN_REQUESTED:
        return {"pair": pair, "status": "skipped_due_to_shutdown"}
    
    try:
        start_time = time.time()
        db_conn = None
        
        try:
            # Get database connection
            db_conn = get_connection()
            
            # Determine date range and limit based on mode
            start_date = None
            end_date = None
            limit = None
            
            if validation_mode == 'rolling_validation' and rolling_window:
                # Get only the most recent candles
                limit = rolling_window
            
            # Fetch data
            df = fetch_data(db_conn, pair, start_date, end_date, limit)
            
            if df.empty:
                return {
                    "pair": pair,
                    "status": "no_data",
                    "duration_seconds": time.time() - start_time
                }
            
            # Run validations
            results = {
                "pair": pair,
                "rows": len(df),
                "timestamp_range": [df['timestamp_utc'].min().isoformat(), df['timestamp_utc'].max().isoformat()] if 'timestamp_utc' in df.columns else ["unknown", "unknown"],
                "missing_values": validate_missing_values(df, pair),
                "timestamp_gaps": validate_timestamps(df, pair),
                "price_consistency": validate_price_consistency(df, pair),
                "indicator_ranges": validate_indicator_ranges(df, pair),
                "future_returns": validate_future_returns(df, pair)
            }
            
            # Calculate overall data quality score
            issues_count = 0
            
            # Count missing values
            issues_count += results["missing_values"]["total_missing_cells"]
            
            # Count timestamp gaps
            issues_count += results["timestamp_gaps"].get("gaps_found", 0)
            
            # Count price inconsistencies
            if results["price_consistency"].get("has_issues", False):
                issues_count += sum(results["price_consistency"]["inconsistencies"].values())
            
            # Count indicator range issues
            if results["indicator_ranges"].get("has_issues", False):
                issues_count += sum(item["count"] for item in results["indicator_ranges"]["out_of_range_indicators"].values())
            
            # Calculate score (0-100, higher is better)
            total_data_points = len(df) * len(df.columns)
            data_quality_score = 100 - min(100, (issues_count / max(1, total_data_points)) * 10000)
            
            results["data_quality_score"] = data_quality_score
            results["duration_seconds"] = time.time() - start_time
            
            return results
        
        finally:
            # Make sure we always return the connection to the pool
            if db_conn:
                db_conn.close()
    
    except Exception as e:
        logging.error(f"Error validating {pair}: {str(e)}")
        return {
            "pair": pair,
            "status": "error",
            "error_message": str(e),
            "duration_seconds": time.time() - start_time
        }

def generate_report(validation_results, output_dir="reports"):
    """
    Generate a comprehensive validation report
    
    Args:
        validation_results: Dictionary with validation results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_path = os.path.join(output_dir, f"validation_report_{timestamp}")
    
    # Generate JSON report with all details
    with open(f"{report_path}.json", 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)  # Use default=str to handle non-serializable objects
    
    # Generate summary report with key findings
    with open(f"{report_path}.txt", 'w') as f:
        f.write("OKXsignal Data Validation Report\n")
        f.write("===============================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Count successful validations
        successful_pairs = sum(1 for result in validation_results['pair_results'].values() 
                             if result.get('status', '') != 'error' and 'data_quality_score' in result)
        
        f.write(f"Total pairs analyzed: {len(validation_results['pair_results'])}\n")
        f.write(f"Successfully validated pairs: {successful_pairs}\n\n")
        
        # Global statistics
        f.write("Global Statistics\n")
        f.write("----------------\n")
        f.write(f"Total rows validated: {validation_results['total_rows']}\n")
        f.write(f"Average data quality score: {validation_results['avg_data_quality_score']:.2f}/100\n")
        f.write(f"Validation duration: {validation_results['duration_seconds']:.2f} seconds\n\n")
        
        # Errors encountered
        error_pairs = [
            pair for pair, result in validation_results['pair_results'].items() 
            if result.get('status', '') == 'error'
        ]
        
        if error_pairs:
            f.write("Pairs with Validation Errors\n")
            f.write("--------------------------\n")
            for pair in sorted(error_pairs):
                error_msg = validation_results['pair_results'][pair].get('error_message', 'Unknown error')
                f.write(f"{pair}: {error_msg}\n")
            f.write("\n")
        
        # Pairs with issues
        problem_pairs = [
            pair for pair, result in validation_results['pair_results'].items() 
            if 'data_quality_score' in result and result['data_quality_score'] < 95
        ]
        
        if problem_pairs:
            f.write("Pairs with Data Quality Issues\n")
            f.write("-----------------------------\n")
            for pair in sorted(problem_pairs):
                score = validation_results['pair_results'][pair].get('data_quality_score', 0)
                f.write(f"{pair}: Score {score:.2f}/100\n")
            f.write("\n")
        
        # Future Return Analysis
        f.write("Future Return Fields Analysis\n")
        f.write("--------------------------\n")
        
        future_return_stats = {}
        for pair, result in validation_results['pair_results'].items():
            if 'future_returns' in result and 'future_return_validations' in result['future_returns']:
                for field, stats in result['future_returns']['future_return_validations'].items():
                    if 'error' not in stats:
                        if field not in future_return_stats:
                            future_return_stats[field] = {
                                'total_pairs': 0,
                                'correct_zeros_at_end_count': 0,
                                'has_expected_end_pattern_count': 0
                            }
                        
                        future_return_stats[field]['total_pairs'] += 1
                        
                        if stats.get('correct_zeros_at_end', False):
                            future_return_stats[field]['correct_zeros_at_end_count'] += 1
                            
                        if stats.get('has_expected_end_pattern', False):
                            future_return_stats[field]['has_expected_end_pattern_count'] += 1
        
        for field, stats in future_return_stats.items():
            f.write(f"\n{field}:\n")
            f.write(f"  - Total pairs with field: {stats['total_pairs']}\n")
            
            if stats['total_pairs'] > 0:
                correct_pct = (stats['correct_zeros_at_end_count'] / stats['total_pairs']) * 100
                pattern_pct = (stats['has_expected_end_pattern_count'] / stats['total_pairs']) * 100
                
                f.write(f"  - Pairs with correct zeros at end: {stats['correct_zeros_at_end_count']} ({correct_pct:.1f}%)\n")
                f.write(f"  - Pairs with expected value pattern: {stats['has_expected_end_pattern_count']} ({pattern_pct:.1f}%)\n")
                
                if correct_pct > 95:
                    f.write("  - VALIDATION: Field appears to be calculated correctly!\n")
                elif correct_pct > 80:
                    f.write("  - VALIDATION: Field is mostly correct but should be checked.\n")
                else:
                    f.write("  - VALIDATION: Field may have issues in calculation!\n")
        
        f.write("\n")
        
        # Common issues
        f.write("Common Issues\n")
        f.write("------------\n")
        
        # Missing values
        total_missing = sum(
            result.get('missing_values', {}).get('total_missing_cells', 0) 
            for result in validation_results['pair_results'].values() 
            if 'missing_values' in result
        )
        
        f.write(f"Total missing values: {total_missing}\n")
        
        # Timestamp gaps
        total_gaps = sum(
            result.get('timestamp_gaps', {}).get('gaps_found', 0) 
            for result in validation_results['pair_results'].values() 
            if 'timestamp_gaps' in result
        )
        
        f.write(f"Total timestamp gaps: {total_gaps}\n")
        
        # Price inconsistencies
        price_inconsistencies = {}
        for result in validation_results['pair_results'].values():
            if 'price_consistency' in result and 'inconsistencies' in result['price_consistency']:
                for issue, count in result['price_consistency']['inconsistencies'].items():
                    price_inconsistencies[issue] = price_inconsistencies.get(issue, 0) + count
        
        if price_inconsistencies:
            f.write("\nPrice inconsistencies:\n")
            for issue, count in price_inconsistencies.items():
                f.write(f"- {issue}: {count}\n")
        
        # Recommendations
        f.write("\nRecommendations\n")
        f.write("--------------\n")
        
        if total_missing > 0:
            f.write("- Address missing values in the dataset\n")
        
        if total_gaps > 0:
            f.write("- Fill timestamp gaps in the data\n")
        
        if price_inconsistencies:
            f.write("- Fix price inconsistencies (high < low, etc.)\n")
        
        if validation_results['avg_data_quality_score'] < 90:
            f.write("- Overall data quality is concerning - review issues in detail\n")
        elif validation_results['avg_data_quality_score'] < 97:
            f.write("- Data quality is acceptable but has room for improvement\n")
        else:
            f.write("- Data quality is good, proceed with model training\n")
    
    return report_path

# ---------------------------
# Main Function
# ---------------------------
def main():
    """Main execution function"""
    # Set up signal handlers for graceful termination
    setup_signal_handlers()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate cryptocurrency data quality')
    parser.add_argument('--mode', choices=['rolling_validation', 'full_validation'], 
                        default='rolling_validation',
                        help='Validation mode')
    parser.add_argument('--config', type=str, help='Path to config.ini file')
    parser.add_argument('--credentials', type=str, help='Path to credentials.env file')
    parser.add_argument('--rolling-window', type=int,
                        help='Number of recent candles to validate in rolling_validation mode (overrides config)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Directory for validation reports')
    parser.add_argument('--pairs', type=str, 
                        help='Comma-separated list of pairs to validate (default: all)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of pairs to validate in parallel')
    parser.add_argument('--max-connections', type=int,
                        help='Maximum database connections (default: batch_size + 2)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir, args.log_level)
    
    # Load configuration
    config_manager = ConfigManager(
        config_path=args.config,
        credentials_path=args.credentials
    )
    
    # Set validation mode
    validation_mode = args.mode
    
    # Set rolling window
    rolling_window = args.rolling_window if args.rolling_window else config_manager.get_rolling_window()
    
    # Set max connections
    max_connections = args.max_connections if args.max_connections else args.batch_size + 2
    
    # Get database connection parameters
    db_params = config_manager.get_db_params()
    
    # Initialize connection pool with appropriate size
    initialize_pool(
        db_params, 
        min_connections=2,
        max_connections=max_connections
    )
    
    logger.info(f"Starting data validation in {validation_mode.upper()} mode")
    
    if validation_mode == 'rolling_validation':
        logger.info(f"Using rolling window of {rolling_window} candles")
    
    start_time = time.time()
    
    try:
        # Get all pairs
        with get_connection() as conn:
            if args.pairs:
                all_pairs = [p.strip() for p in args.pairs.split(',')]
                logger.info(f"Validating {len(all_pairs)} specified pairs")
            else:
                all_pairs = get_all_pairs(conn)
                logger.info(f"Found {len(all_pairs)} pairs to validate")
        
        if not all_pairs:
            logger.error("No pairs found. Exiting.")
            return
        
        # Validate pairs
        validation_results = {
            "validation_mode": validation_mode,
            "rolling_window": rolling_window if validation_mode == 'rolling_validation' else None,
            "timestamp": datetime.now().isoformat(),
            "pair_results": {},
            "total_rows": 0,
            "issues_found": 0
        }
        
        # Lower the batch size to avoid connection pool exhaustion
        effective_batch_size = min(args.batch_size, max_connections - 1)
        if effective_batch_size < args.batch_size:
            logger.warning(f"Reducing batch size from {args.batch_size} to {effective_batch_size} to prevent connection pool exhaustion")
        
        # Validate in parallel with reduced batch size
        with ThreadPoolExecutor(max_workers=effective_batch_size) as executor:
            futures = {}
            
            # Submit all tasks
            for pair in all_pairs:
                if SHUTDOWN_REQUESTED:
                    logger.info("Shutdown requested. Not submitting more validation tasks.")
                    break
                
                future = executor.submit(
                    validate_pair, 
                    pair, 
                    config_manager,
                    validation_mode,
                    rolling_window
                )
                futures[future] = pair
            
            # Show progress
            with tqdm(total=len(futures), desc="Validating pairs") as pbar:
                for future in as_completed(futures):
                    if SHUTDOWN_REQUESTED:
                        pbar.set_description("Shutdown requested, completing current tasks")
                    
                    pair = futures[future]
                    
                    try:
                        result = future.result()
                        validation_results["pair_results"][pair] = result
                        
                        # Update total rows
                        if isinstance(result.get('rows', 0), (int, float)):
                            validation_results["total_rows"] += result['rows']
                        
                        # Log progress
                        pbar.update(1)
                        
                        # Log problems
                        if 'data_quality_score' in result and result['data_quality_score'] < 95:
                            logger.warning(f"{pair}: Data quality score {result['data_quality_score']:.2f}/100")
                        
                    except Exception as e:
                        logger.error(f"Error processing validation result for {pair}: {str(e)}")
                        validation_results["pair_results"][pair] = {
                            "pair": pair,
                            "status": "error",
                            "error_message": str(e)
                        }
                        pbar.update(1)
        
        # Calculate global statistics
        validation_results["duration_seconds"] = time.time() - start_time
        
        # Calculate average data quality score (for pairs that have a score)
        quality_scores = [
            result.get('data_quality_score', 0) 
            for result in validation_results['pair_results'].values()
            if 'data_quality_score' in result
        ]
        
        if quality_scores:
            validation_results["avg_data_quality_score"] = sum(quality_scores) / len(quality_scores)
        else:
            validation_results["avg_data_quality_score"] = 0
        
        # Generate report
        report_path = generate_report(validation_results, args.output_dir)
        
        if SHUTDOWN_REQUESTED:
            logger.info("Validation interrupted but report was generated with partial results")
        else:
            logger.info(f"Validation complete. Reports saved to {report_path}.*")
            
        logger.info(f"Average data quality score: {validation_results['avg_data_quality_score']:.2f}/100")
        logger.info(f"Total duration: {validation_results['duration_seconds']:.2f} seconds")
        
        # Give quick recommendation
        if validation_results['avg_data_quality_score'] < 90:
            logger.warning("Data quality is concerning. Please review the validation report.")
        elif validation_results['avg_data_quality_score'] < 97:
            logger.warning("Data quality is acceptable but has room for improvement.")
        else:
            logger.info("Data quality is good, suitable for model training.")
    
    except Exception as e:
        logger.error(f"Error in validation process: {str(e)}")
    
    finally:
        # Close database connection pool
        try:
            close_all_connections()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")

if __name__ == "__main__":
    main()