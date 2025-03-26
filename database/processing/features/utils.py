#!/usr/bin/env python3
"""
Utility functions for feature computation
"""

import os
import gc
import logging
import threading
import time
from datetime import datetime
import numpy as np
import pandas as pd

# ---------------------------
# Database Utilities
# ---------------------------
def get_database_columns(db_engine, table_name):
    """Get all column names from a specific database table"""
    with db_engine.connect() as conn:
        query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        """
        result = pd.read_sql(query, conn)
        return set(result['column_name'].tolist())

# ---------------------------
# Performance Monitoring
# ---------------------------
class PerformanceMonitor:
    """Class to track and log computation times for performance analysis"""
    
    def __init__(self, log_dir="logs"):
        """Initialize the performance monitor with given log directory"""
        self.log_dir = log_dir
        self.timings = {}
        self.current_pair = None
        self.lock = threading.Lock()
        
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"computing_durations_{timestamp}.log")
        
        # Write header to log file
        with open(self.log_file, 'w') as f:
            f.write("Timestamp,Pair,Operation,Duration(s)\n")
    
    def start_pair(self, pair):
        """Set the current pair being processed"""
        with self.lock:
            self.current_pair = pair
            if pair not in self.timings:
                self.timings[pair] = {
                    "total": 0,
                    "operations": {}
                }
    
    def log_operation(self, operation, duration):
        """Log the duration of a specific operation"""
        if not self.current_pair:
            return
            
        with self.lock:
            if operation not in self.timings[self.current_pair]["operations"]:
                self.timings[self.current_pair]["operations"][operation] = []
                
            self.timings[self.current_pair]["operations"][operation].append(duration)
            
            # Write to log file immediately
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{self.current_pair},{operation},{duration:.6f}\n")
    
    def end_pair(self, total_duration):
        """Log the total processing time for the current pair"""
        if not self.current_pair:
            return
            
        with self.lock:
            self.timings[self.current_pair]["total"] = total_duration
            
            # Write total to log file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{self.current_pair},TOTAL,{total_duration:.6f}\n")
            
            # Reset current pair
            self.current_pair = None
    
    def save_summary(self):
        """Save a summary of all timings to JSON file"""
        import json
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        summary_file = os.path.join(self.log_dir, f"performance_summary_{timestamp}.json")
        
        summary = {
            "pairs_processed": len(self.timings),
            "total_processing_time": sum(data["total"] for data in self.timings.values()),
            "average_pair_time": sum(data["total"] for data in self.timings.values()) / len(self.timings) if self.timings else 0,
            "operation_summaries": {}
        }
        
        # Calculate statistics for each operation
        all_operations = set()
        for pair_data in self.timings.values():
            all_operations.update(pair_data["operations"].keys())
        
        for operation in all_operations:
            operation_times = []
            for pair_data in self.timings.values():
                if operation in pair_data["operations"]:
                    operation_times.extend(pair_data["operations"][operation])
            
            if operation_times:
                summary["operation_summaries"][operation] = {
                    "total_calls": len(operation_times),
                    "average_time": sum(operation_times) / len(operation_times),
                    "min_time": min(operation_times),
                    "max_time": max(operation_times),
                    "total_time": sum(operation_times),
                    "percentage_of_total": (sum(operation_times) / summary["total_processing_time"]) * 100 if summary["total_processing_time"] > 0 else 0
                }
        
        # Sort operations by total time (descending)
        summary["operation_summaries"] = dict(
            sorted(
                summary["operation_summaries"].items(),
                key=lambda x: x[1]["total_time"],
                reverse=True
            )
        )
        
        # Save to file
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save a readable text report
        report_file = os.path.join(self.log_dir, f"performance_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("PERFORMANCE SUMMARY REPORT\n")
            f.write("=========================\n\n")
            f.write(f"Pairs Processed: {summary['pairs_processed']}\n")
            f.write(f"Total Processing Time: {summary['total_processing_time']:.2f} seconds\n")
            f.write(f"Average Time Per Pair: {summary['average_pair_time']:.2f} seconds\n\n")
            
            f.write("OPERATION BREAKDOWN (Sorted by Total Time)\n")
            f.write("----------------------------------------\n")
            f.write(f"{'Operation':<30} {'Total Time (s)':<15} {'Avg Time (s)':<15} {'Calls':<10} {'% of Total':<10}\n")
            f.write("-" * 80 + "\n")
            
            for op, stats in summary["operation_summaries"].items():
                f.write(
                    f"{op[:30]:<30} {stats['total_time']:<15.2f} {stats['average_time']:<15.6f} "
                    f"{stats['total_calls']:<10} {stats['percentage_of_total']:<10.2f}\n"
                )
        
        return summary_file, report_file

# ---------------------------
# Memory Management
# ---------------------------
def clean_memory():
    """Force garbage collection to free memory"""
    gc.collect()

# ---------------------------
# Data Type Conversion
# ---------------------------
def cast_for_sqlalchemy(col_name, val, smallint_columns=None):
    """
    Convert values to appropriate Python types for SQLAlchemy
    
    Args:
        col_name: Name of the column
        val: Value to convert
        smallint_columns: Set of column names that should be smallint
        
    Returns:
        Converted value
    """
    from config import SMALLINT_COLUMNS
    smallint_columns = smallint_columns or SMALLINT_COLUMNS
    
    # Handle null values first
    if val is None or pd.isna(val):
        return None

    # Special handling for boolean columns
    if col_name in ['features_computed', 'targets_computed']:
        if isinstance(val, (int, float, bool, np.bool_)):
            return bool(val)  # Ensure proper boolean type
        else:
            return True if str(val).lower() in ('true', 't', 'yes', 'y', '1') else False

    # Convert numpy types
    if isinstance(val, (np.integer)):
        val = int(val)
    elif isinstance(val, (np.floating)):
        val = float(val)
    elif isinstance(val, (np.datetime64, pd.Timestamp)):
        val = pd.to_datetime(val).to_pydatetime()
    elif isinstance(val, (np.bool_)):
        val = bool(val)

    # Special handling for smallint columns
    if col_name in smallint_columns and val is not None:
        try:
            val = int(round(float(val))) if val is not None else None
        except (ValueError, TypeError):
            logging.warning(f"Failed to convert {col_name} value {val} to int")
            val = None

    # Final type check
    if not isinstance(val, (int, float, bool, datetime, str, type(None))):
        logging.warning(f"Unhandled type {type(val)} for column {col_name}, converting to string")
        val = str(val)

    return val

# ---------------------------
# DataFrame Operations
# ---------------------------
def safe_indicator_assign(df, column_name, indicator_result):
    """
    Safely assign an indicator result to a DataFrame column, handling index misalignment.
    
    Args:
        df: DataFrame to assign to
        column_name: Name of the column to create/update
        indicator_result: Result from a pandas_ta or TA-Lib calculation
    
    Returns:
        DataFrame with the indicator assigned
    """
    # Initialize with zeros
    df[column_name] = 0.0
    
    try:
        # If it's already a DataFrame column or Series
        if isinstance(indicator_result, (pd.Series, pd.DataFrame)):
            # Use DataFrame column if multi-column result
            if isinstance(indicator_result, pd.DataFrame) and indicator_result.shape[1] > 1:
                indicator_result = indicator_result.iloc[:, 0]
            
            # Align and assign
            aligned_result = indicator_result.reindex(df.index, fill_value=0)
            df[column_name] = aligned_result
        
        # Handle numpy array results
        elif isinstance(indicator_result, np.ndarray):
            if len(indicator_result) == len(df):
                df[column_name] = indicator_result
            else:
                # If lengths don't match, attempt to assign where possible
                length = min(len(df), len(indicator_result))
                df.iloc[:length, df.columns.get_loc(column_name)] = indicator_result[:length]
        
        # Handle dict-like results from pandas_ta
        elif hasattr(indicator_result, 'keys') and len(indicator_result) > 0:
            # Get the first key if the result is a dict-like object with multiple columns
            first_key = next(iter(indicator_result))
            result_series = indicator_result[first_key]
            
            # Align and assign
            aligned_result = result_series.reindex(df.index, fill_value=0)
            df[column_name] = aligned_result
    
    except Exception as e:
        logging.warning(f"Error assigning indicator {column_name}: {e}")
    
    # Ensure any NaN values are filled with zeros
    df[column_name] = df[column_name].fillna(0)
    return df

def check_gpu_available():
    """Check if CuPy is available and GPU can be used"""
    try:
        import cupy
        # Basic test to check if GPU is available
        x = cupy.array([1, 2, 3])
        y = x * 2
        cupy.cuda.Stream.null.synchronize()
        return True
    except Exception:
        return False