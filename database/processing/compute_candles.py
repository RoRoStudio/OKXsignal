#!/usr/bin/env python3
"""
Cryptocurrency Technical Feature Computation
- Computes technical indicators and features for OHLCV crypto data
- Optimized for performance using pandas_ta, TA-Lib, and Numba
- Handles multi-timeframe calculations and cross-pair intelligence
"""

import os
import sys
import gc
import io
import math
import time
import signal
import logging
import argparse
import configparser
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import pandas_ta as pta
try:
    import talib
except ImportError:
    print("TA-Lib is not fully installed. Some financial analysis functions may not work.")
from scipy import stats
import statsmodels.api as sm
from numba import jit, njit, prange
from numba.types import float64, int64, boolean
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.pool import QueuePool

try:
    import cupy as cp
    from cupyx.scipy import stats as cp_stats
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    
try:
    import cudf
    import cuxfilter
    from cuml import linear_model as cu_linear_model
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

# ---------------------------
# Constants & Configuration
# ---------------------------
BATCH_SIZE = max(8, os.cpu_count() or 8)  # Default batch size based on CPU cores
MIN_CANDLES_REQUIRED = 200  # Minimum candles needed for reliable calculation
ROLLING_WINDOW = 128  # Default window size for rolling updates

# Default paths
DEFAULT_CONFIG_PATH = "P:/OKXsignal/config/config.ini"
DEFAULT_CREDENTIALS_PATH = "P:/OKXsignal/config/credentials.env"

# Define smallint columns for proper type conversion
SMALLINT_COLUMNS = {
    'performance_rank_btc_1h', 'performance_rank_eth_1h',
    'volatility_rank_1h', 'volume_rank_1h',
    'hour_of_day', 'day_of_week', 'month_of_year',
    'was_profitable_12h', 'is_weekend', 'asian_session',
    'european_session', 'american_session',
    'pattern_doji', 'pattern_engulfing', 'pattern_hammer',
    'pattern_morning_star', 'profit_target_1pct', 'profit_target_2pct'
}

# ---------------------------
# Configuration Management
# ---------------------------
class ConfigManager:
    """Handles loading and managing configuration settings"""
    
    def __init__(self, config_path: str = None, credentials_path: str = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        
        # Load environmental variables
        if os.path.exists(self.credentials_path):
            load_dotenv(dotenv_path=self.credentials_path)
            logging.info(f"Loaded credentials from {self.credentials_path}")
        else:
            logging.warning(f"Credentials file not found: {self.credentials_path}")
            
        # Load config
        self.config = self._load_config()

        # Check GPU availability at initialization
        self.gpu_available = CUPY_AVAILABLE or RAPIDS_AVAILABLE
        if self.gpu_available:
            if CUPY_AVAILABLE:
                logging.info("CuPy is available for GPU acceleration")
            if RAPIDS_AVAILABLE:
                logging.info("RAPIDS is available for GPU acceleration")
    
    # Add this method:
    def use_gpu(self) -> bool:
        """Check if GPU acceleration should be used"""
        if not self.gpu_available:
            return False
            
        if 'GENERAL' in self.config and 'USE_GPU' in self.config['GENERAL']:
            return self.config['GENERAL'].getboolean('USE_GPU')
        return False
    
    def _load_config(self):
        """Load configuration from file"""
        config = configparser.ConfigParser()
        if os.path.exists(self.config_path):
            config.read(self.config_path)
            logging.info(f"Loaded config from {self.config_path}")
        else:
            logging.warning(f"Config file not found: {self.config_path}")
            # Create minimal defaults
            config['DATABASE'] = {
                'DB_HOST': 'localhost',
                'DB_PORT': '5432',
                'DB_NAME': 'okxsignal'
            }
            config['GENERAL'] = {
                'COMPUTE_MODE': 'rolling_update',
                'ROLLING_WINDOW': '128',
                'USE_TALIB': 'True',
                'USE_NUMBA': 'True'
            }
        return config
    
    def get_db_params(self) -> dict:
        """Get database connection parameters"""
        if 'DATABASE' not in self.config:
            raise ValueError("DATABASE section not found in config file")
            
        db_config = self.config['DATABASE']
        
        return {
            'host': db_config.get('DB_HOST', 'localhost'),
            'port': db_config.get('DB_PORT', '5432'),
            'dbname': db_config.get('DB_NAME', 'okxsignal'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
    def get_connection_string(self) -> str:
        """Create a SQLAlchemy connection string from config"""
        db_params = self.get_db_params()
        return (
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@"
            f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
    def get_rolling_window(self) -> int:
        """Get rolling window size from config"""
        if 'GENERAL' in self.config and 'ROLLING_WINDOW' in self.config['GENERAL']:
            return int(self.config['GENERAL']['ROLLING_WINDOW'])
        return ROLLING_WINDOW
        
    def get_compute_mode(self) -> str:
        """Get compute mode from config"""
        if 'GENERAL' in self.config and 'COMPUTE_MODE' in self.config['GENERAL']:
            return self.config['GENERAL']['COMPUTE_MODE']
        return 'rolling_update'
        
    def use_talib(self) -> bool:
        """Check if TA-Lib should be used"""
        if 'GENERAL' in self.config and 'USE_TALIB' in self.config['GENERAL']:
            return self.config['GENERAL'].getboolean('USE_TALIB')
        return True
        
    def use_numba(self) -> bool:
        """Check if Numba should be used for optimization"""
        if 'GENERAL' in self.config and 'USE_NUMBA' in self.config['GENERAL']:
            return self.config['GENERAL'].getboolean('USE_NUMBA')
        return True

# ---------------------------
# Logging Setup
# ---------------------------
def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Set up application logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = os.path.join(log_dir, f"compute_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(levelname)s] %(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("compute_features")

def force_exit_on_ctrl_c():
    """Handles Ctrl-C to forcibly exit all threads"""
    import os
    import signal
    import threading
    
    # Global flag to indicate shutdown is in progress
    is_shutting_down = [False]
    
    def handler(signum, frame):
        if is_shutting_down[0]:
            # If already shutting down and Ctrl+C pressed again, force exit
            print("\nForced exit!")
            os._exit(1)  # Emergency exit
        
        is_shutting_down[0] = True
        print("\nInterrupted. Forcing thread exit... (Press Ctrl+C again to force immediate exit)")
        
        # Forcibly terminate all threads
        for thread in threading.enumerate():
            if thread is not threading.current_thread():
                try:
                    # Set trace function to raise exception
                    import ctypes
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread.ident), 
                        ctypes.py_object(SystemExit)
                    )
                except Exception:
                    pass
        
        # Set a small timeout before forcing exit
        def force_exit():
            print("Shutdown timed out. Forcing exit.")
            os._exit(1)
        
        # Set a timer to force exit after 5 seconds if graceful shutdown fails
        exit_timer = threading.Timer(5.0, force_exit)
        exit_timer.daemon = True
        exit_timer.start()
        
        # Try graceful exit first
        sys.exit(1)

    # Register the handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

# Function to get valid database columns and filter out non-existent columns

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
    
class PerformanceMonitor:
    """Class to track and log computation times for performance analysis"""
    
    def __init__(self, log_dir: str = "logs"):
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
    
    def start_pair(self, pair: str):
        """Set the current pair being processed"""
        with self.lock:
            self.current_pair = pair
            if pair not in self.timings:
                self.timings[pair] = {
                    "total": 0,
                    "operations": {}
                }
    
    def log_operation(self, operation: str, duration: float):
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
    
    def end_pair(self, total_duration: float):
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
# Database Connection Management
# ---------------------------
class DatabaseConnectionManager:
    """Manages database connections and operations"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.engine = self._create_connection_pool()
    
    def _create_connection_pool(self):
        """Create a connection pool for database operations"""
        connection_string = self.config_manager.get_connection_string()
        logging.info(f"Creating database connection pool (hiding credentials)")
        return create_engine(
            connection_string, 
            poolclass=QueuePool,
            pool_size=max(10, BATCH_SIZE),
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            isolation_level="READ COMMITTED"
        )
        
    def get_engine(self):
        """Get the SQLAlchemy engine"""
        return self.engine
        
    def execute_copy_update(self, temp_table_name: str, column_names: List[str], 
                           values: List[List[Any]], update_query: str):
        """
        Execute a COPY operation to a temporary table, then update the main table from it
        """
        if not values:
            logging.warning("No values to update in execute_copy_update")
            return 0
            
        conn = None
        try:
            conn = self.engine.raw_connection()
            cursor = conn.cursor()
            
            # Create temp table with appropriate column types
            create_columns = []
            for col in column_names:
                if col == 'timestamp_utc':
                    col_type = "TIMESTAMP WITH TIME ZONE"
                elif col == 'pair':
                    col_type = "VARCHAR"
                elif col in SMALLINT_COLUMNS:
                    col_type = "SMALLINT"
                else:
                    col_type = "DOUBLE PRECISION"
                    
                create_columns.append(f"{col} {col_type}")
                
            cursor.execute(f"""
                CREATE TEMP TABLE {temp_table_name} (
                    {', '.join(create_columns)}
                ) ON COMMIT DROP;
            """)
            
            # Prepare data for COPY
            csv_data = io.StringIO()
            for row in values:
                # Convert None to empty string for CSV
                csv_row = ','.join(['' if v is None else str(v) for v in row])
                csv_data.write(csv_row + '\n')
                
            csv_data.seek(0)
            
            # Use psycopg2's copy_expert for maximum performance
            cursor.copy_expert(
                f"COPY {temp_table_name} FROM STDIN WITH CSV NULL ''", 
                csv_data
            )
            
            # Execute the update with a transaction
            cursor.execute(update_query.format(temp_table=temp_table_name))
            affected_rows = cursor.rowcount
            conn.commit()
            
            return affected_rows
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Error in execute_copy_update: {e}", exc_info=True)
            raise e
        finally:
            if conn:
                conn.close()
    
    def execute_batch_update(self, query_text, param_lists, batch_size=1000):
        """
        Execute updates in efficient batches
        """
        if not param_lists:
            logging.warning("No params to update in execute_batch_update")
            return 0
            
        conn = None
        updated_rows = 0
        
        try:
            conn = self.engine.raw_connection()
            cursor = conn.cursor()
            
            # Process in batches for better performance
            for i in range(0, len(param_lists), batch_size):
                batch = param_lists[i:i+batch_size]
                
                # Log batch size 
                if i == 0:
                    logging.debug(f"Executing batch update with {len(batch)}/{len(param_lists)} rows")
                
                cursor.executemany(query_text, batch)
                updated_rows += cursor.rowcount
                
            conn.commit()
            return updated_rows
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Error in execute_batch_update: {e}", exc_info=True)
            raise e
        finally:
            if conn:
                conn.close()

# ---------------------------
# Data Type Conversion Helper
# ---------------------------
def cast_for_sqlalchemy(col_name: str, val) -> any:
    """
    Convert values to appropriate Python types for SQLAlchemy
    """
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
    if col_name in SMALLINT_COLUMNS and val is not None:
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

# 2. GPU-accelerated FeatureComputer class

class GPUFeatureComputer(FeatureComputer):
    """Feature computer that uses GPU acceleration when available"""
    
    def __init__(self, use_talib=True, use_numba=True, use_gpu=False):
        super().__init__(use_talib, use_numba)
        self.use_gpu = use_gpu and (CUPY_AVAILABLE or RAPIDS_AVAILABLE)
        
        if self.use_gpu:
            logging.info("GPU acceleration enabled for feature computation")
            
            # Test GPU memory and warm up
            if CUPY_AVAILABLE:
                try:
                    # Simple warm-up calculation
                    a = cp.array([1, 2, 3])
                    b = cp.array([4, 5, 6])
                    c = a + b
                    cp.cuda.Stream.null.synchronize()
                    logging.info("CuPy GPU warm-up successful")
                except Exception as e:
                    logging.warning(f"CuPy GPU warm-up failed: {e}")
                    self.use_gpu = False
    
    def compute_price_action_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute price action features with GPU acceleration if available"""
        # Extract price data
        open_prices = df['open_1h'].values
        high_prices = df['high_1h'].values
        low_prices = df['low_1h'].values
        close_prices = df['close_1h'].values
        
        # Use GPU for body features if available
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                body_features = compute_candle_body_features_gpu(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['candle_body_size'] = body_features[0:len(df)]
                df['upper_shadow'] = body_features[len(df):2*len(df)]
                df['lower_shadow'] = body_features[2*len(df):3*len(df)]
                df['relative_close_position'] = body_features[3*len(df):4*len(df)]
                
                # Clean any NaN values that might have been introduced
                for col in ['candle_body_size', 'upper_shadow', 'lower_shadow', 'relative_close_position']:
                    df[col] = df[col].fillna(0)
                
            except Exception as e:
                logging.warning(f"GPU calculation failed for candle features: {e}")
                # Fall back to CPU implementation
                return super().compute_price_action_features(df, debug_mode)
        else:
            # Use the parent class implementation (CPU)
            return super().compute_price_action_features(df, debug_mode)
        
        # Compute other price action features
        
        # Log returns - using numpy for consistency with CPU version
        df['log_return'] = np.log(df['close_1h'] / df['close_1h'].shift(1)).fillna(0)
        
        # Gap open (compared to previous close)
        df['gap_open'] = (df['open_1h'] / df['close_1h'].shift(1) - 1).fillna(0)
        
        # Price velocity (rate of change over time)
        df['price_velocity'] = df['close_1h'].pct_change(3).fillna(0)
        
        # Price acceleration (change in velocity)
        df['price_acceleration'] = df['price_velocity'].diff(3).fillna(0)
        
        # Previous close percent change
        df['prev_close_change_pct'] = df['close_1h'].pct_change().fillna(0)
        
        return df
    
    def compute_statistical_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute statistical features with GPU acceleration if available"""
        if len(df) < 20:  # Need at least 20 points for statistical features
            return df
            
        # If GPU is not available or not enabled, use CPU implementation
        if not self.use_gpu or not CUPY_AVAILABLE:
            return super().compute_statistical_features(df, debug_mode)
            
        try:
            # Extract price data
            close = df['close_1h']
            
            # Standard deviation of returns - using pandas for consistency
            df['std_dev_returns_20'] = df['log_return'].rolling(window=20).std().fillna(0)
            
            # Skewness and kurtosis - using pandas for consistency
            df['skewness_20'] = df['log_return'].rolling(window=20).apply(
                lambda x: stats.skew(x) if len(x) > 3 else 0, raw=True
            ).fillna(0)
            
            df['kurtosis_20'] = df['log_return'].rolling(window=20).apply(
                lambda x: stats.kurtosis(x) if len(x) > 3 else 0, raw=True
            ).fillna(0)
            
            # Z-score calculation
            ma_20 = close.rolling(window=20).mean()
            
            # Use GPU z-score calculation
            df['z_score_20'] = compute_z_score_gpu(close.values, ma_20.fillna(0).values, 20)
            
            # Hurst Exponent with GPU
            if len(df) >= 100:
                df['hurst_exponent'] = hurst_exponent_gpu(
                    close.replace(0, np.nan).ffill().values, 
                    20
                )
            else:
                df['hurst_exponent'] = 0.5  # Default value for short series
            
            # Shannon Entropy with GPU
            if len(df) >= 20:
                df['shannon_entropy'] = shannon_entropy_gpu(
                    close.replace(0, np.nan).ffill().values, 20
                )
            else:
                df['shannon_entropy'] = 0  # Default value
            
            # Autocorrelation lag 1 - using pandas for consistency
            df['autocorr_1'] = df['log_return'].rolling(window=20).apply(
                lambda x: x.autocorr(1) if len(x) > 1 else 0
            ).fillna(0)
            
            # Estimated slippage and bid-ask spread proxies
            df['estimated_slippage_1h'] = df['high_1h'] - df['low_1h']
            df['bid_ask_spread_1h'] = df['estimated_slippage_1h'] * 0.1  # Rough approximation
            
            return df
            
        except Exception as e:
            logging.warning(f"GPU calculation failed for statistical features: {e}")
            # Fall back to CPU implementation
            return super().compute_statistical_features(df, debug_mode)
        
# ---------------------------
# Numba Optimized Functions
# ---------------------------
@njit(float64[:](float64[:], float64[:], float64[:], float64[:]))
def compute_candle_body_features_numba(open_prices, high_prices, low_prices, close_prices):
    """Compute candle body size, upper shadow, lower shadow using Numba"""
    n = len(open_prices)
    results = np.zeros(4 * n, dtype=np.float64).reshape(4, n)
    
    for i in range(n):
        # Body size
        body_size = abs(close_prices[i] - open_prices[i])
        results[0, i] = body_size
        
        # Upper shadow
        upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
        results[1, i] = upper_shadow
        
        # Lower shadow
        lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
        results[2, i] = lower_shadow
        
        # Relative close position (where close is within high-low range)
        hl_range = high_prices[i] - low_prices[i]
        if hl_range > 0:
            rel_pos = (close_prices[i] - low_prices[i]) / hl_range
            results[3, i] = rel_pos
        else:
            results[3, i] = 0.5  # Default to middle if no range
    
    return results.flatten()

@njit(float64[:](float64[:], float64[:], int64))
def compute_slope_numba(values, timestamps, periods):
    """Compute slope of values over specified periods using Numba"""
    n = len(values)
    slopes = np.zeros(n, dtype=np.float64)
    
    for i in range(periods, n):
        x = timestamps[i-periods:i+1]
        y = values[i-periods:i+1]
        
        # Compute mean of x and y
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Compute slope using covariance / variance
        numerator = 0.0
        denominator = 0.0
        
        for j in range(periods+1):
            x_diff = x[j] - mean_x
            y_diff = y[j] - mean_y
            numerator += x_diff * y_diff
            denominator += x_diff * x_diff
        
        if denominator != 0:
            slopes[i] = numerator / denominator
        else:
            slopes[i] = 0.0
    
    return slopes

@njit(float64[:](float64[:], float64[:], int64))
def compute_z_score_numba(values, ma_values, lookback):
    """Compute z-score using Numba"""
    n = len(values)
    z_scores = np.zeros(n, dtype=np.float64)
    
    for i in range(lookback, n):
        window = values[i-lookback:i]
        std_dev = np.std(window)
        if std_dev > 0:
            z_scores[i] = (values[i] - ma_values[i]) / std_dev
        else:
            z_scores[i] = 0.0
    
    return z_scores

@njit(float64[:](float64[:], int64))
def hurst_exponent_numba(prices, max_lag):
    """Compute Hurst exponent using Numba"""
    n = len(prices)
    hurst_values = np.zeros(n, dtype=np.float64)
    
    # Need at least 100 points for reliable Hurst calculation
    min_window = 100
    
    if n < min_window:
        return hurst_values
    
    for end_idx in range(min_window, n):
        # Take a window of data
        window_size = min(min_window, end_idx)
        price_window = prices[end_idx-window_size:end_idx]
        
        # Returns
        returns = np.diff(np.log(price_window))
        
        if len(returns) < max_lag:
            hurst_values[end_idx] = 0.5  # Default value
            continue
            
        tau = np.arange(1, max_lag+1)
        lagmat = np.zeros(max_lag)
        
        for lag in range(1, max_lag+1):
            # Compute price difference for the lag
            lag_returns = returns[lag:] - returns[:-lag]
            # Root mean square of differences
            lagmat[lag-1] = np.sqrt(np.mean(lag_returns**2))
        
        # Only use valid values
        valid_lags = lagmat > 0
        if np.sum(valid_lags) > 1:
            # Linear regression on log-log scale
            log_lag = np.log(tau[valid_lags])
            log_lagmat = np.log(lagmat[valid_lags])
            
            # Calculate slope
            n_valid = len(log_lag)
            mean_x = np.mean(log_lag)
            mean_y = np.mean(log_lagmat)
            
            numerator = 0.0
            denominator = 0.0
            
            for i in range(n_valid):
                x_diff = log_lag[i] - mean_x
                y_diff = log_lagmat[i] - mean_y
                numerator += x_diff * y_diff
                denominator += x_diff * x_diff
            
            if denominator > 0:
                # Hurst exponent = slope / 2
                hurst_values[end_idx] = numerator / denominator / 2
            else:
                hurst_values[end_idx] = 0.5
        else:
            hurst_values[end_idx] = 0.5
            
    return hurst_values

@njit(float64[:](float64[:], int64))
def shannon_entropy_numba(prices, window_size=20):
    """Compute Shannon Entropy using Numba"""
    n = len(prices)
    entropy_values = np.zeros(n, dtype=np.float64)
    
    # Need at least window_size for calculation
    if n < window_size:
        return entropy_values
        
    for end_idx in range(window_size, n):
        price_window = prices[end_idx-window_size:end_idx]
        returns = np.diff(np.log(price_window))
        
        # Use histogram to estimate probability distribution
        hist, _ = np.histogram(returns, bins=10)
        hist = hist.astype(np.float64)
        
        # Calculate probability of each bin
        total = np.sum(hist)
        if total > 0:
            probs = hist / total
            
            # Calculate entropy
            entropy = 0.0
            for p in probs:
                if p > 0:
                    entropy -= p * np.log(p)
            
            entropy_values[end_idx] = entropy
            
    return entropy_values

# 1. GPU-accelerated versions of Numba functions

def compute_candle_body_features_gpu(open_prices, high_prices, low_prices, close_prices):
    """Compute candle body size, upper shadow, lower shadow using GPU"""
    # Convert to CuPy arrays if they're not already
    if not isinstance(open_prices, cp.ndarray):
        open_prices = cp.array(open_prices)
        high_prices = cp.array(high_prices)
        low_prices = cp.array(low_prices)
        close_prices = cp.array(close_prices)
    
    n = len(open_prices)
    results = cp.zeros(4 * n, dtype=cp.float64).reshape(4, n)
    
    # Body size
    results[0] = cp.abs(close_prices - open_prices)
    
    # Upper shadow
    results[1] = high_prices - cp.maximum(open_prices, close_prices)
    
    # Lower shadow
    results[2] = cp.minimum(open_prices, close_prices) - low_prices
    
    # Relative close position
    hl_range = high_prices - low_prices
    mask = hl_range > 0
    results[3] = cp.where(
        mask,
        (close_prices - low_prices) / hl_range,
        0.5  # Default to middle if no range
    )
    
    # Return as numpy array
    return cp.asnumpy(results.flatten())

def compute_z_score_gpu(values, ma_values, lookback):
    """Compute z-score using GPU"""
    # Convert to CuPy arrays
    values_gpu = cp.array(values)
    ma_values_gpu = cp.array(ma_values)
    
    n = len(values_gpu)
    z_scores = cp.zeros(n, dtype=cp.float64)
    
    # Create rolling windows for standard deviation calculation
    for i in range(lookback, n):
        window = values_gpu[i-lookback:i]
        std_dev = cp.std(window)
        if std_dev > 0:
            z_scores[i] = (values_gpu[i] - ma_values_gpu[i]) / std_dev
    
    # Return as numpy array
    return cp.asnumpy(z_scores)

def hurst_exponent_gpu(prices, max_lag):
    """Compute Hurst exponent using GPU"""
    # Convert to CuPy array
    prices_gpu = cp.array(prices)
    
    n = len(prices_gpu)
    hurst_values = cp.zeros(n, dtype=cp.float64)
    
    # Need at least 100 points for reliable Hurst calculation
    min_window = 100
    
    if n < min_window:
        return cp.asnumpy(hurst_values)
    
    # Log returns
    log_prices = cp.log(prices_gpu)
    
    for end_idx in range(min_window, n):
        # Take a window of data
        window_size = min(min_window, end_idx)
        price_window = prices_gpu[end_idx-window_size:end_idx]
        
        # Calculate returns
        log_window = cp.log(price_window)
        returns = cp.diff(log_window)
        
        if len(returns) < max_lag:
            hurst_values[end_idx] = 0.5  # Default value
            continue
            
        tau = cp.arange(1, max_lag+1)
        lagmat = cp.zeros(max_lag)
        
        # Calculate variance for each lag
        for lag in range(1, max_lag+1):
            if lag < len(returns):
                lag_returns = returns[lag:] - returns[:-lag]
                lagmat[lag-1] = cp.sqrt(cp.mean(lag_returns**2))
            else:
                lagmat[lag-1] = 0
        
        # Filter valid values (avoid log(0))
        valid_mask = lagmat > 0
        if cp.sum(valid_mask) > 1:
            valid_lags = tau[valid_mask]
            valid_lagmat = lagmat[valid_mask]
            
            # Linear regression on log-log scale using polyfit
            log_lag = cp.log(valid_lags)
            log_lagmat = cp.log(valid_lagmat)
            
            # Use CuPy's polyfit for regression
            coeffs = cp.polyfit(log_lag, log_lagmat, 1)
            hurst_values[end_idx] = coeffs[0] / 2
        else:
            hurst_values[end_idx] = 0.5
    
    # Return as numpy array
    return cp.asnumpy(hurst_values)

def shannon_entropy_gpu(prices, window_size=20):
    """Compute Shannon Entropy using GPU"""
    # Convert to CuPy array
    prices_gpu = cp.array(prices)
    
    n = len(prices_gpu)
    entropy_values = cp.zeros(n, dtype=cp.float64)
    
    # Need at least window_size points
    if n < window_size:
        return cp.asnumpy(entropy_values)
    
    for end_idx in range(window_size, n):
        price_window = prices_gpu[end_idx-window_size:end_idx]
        # Calculate log returns
        log_price = cp.log(price_window)
        returns = cp.diff(log_price)
        
        # Use histogramming for probability estimation
        # We'll use 10 bins for the histogram
        hist, _ = cp.histogram(returns, bins=10)
        total = cp.sum(hist)
        
        if total > 0:
            probs = hist / total
            # Calculate entropy, avoiding log(0)
            valid_probs = probs[probs > 0]
            entropy = -cp.sum(valid_probs * cp.log(valid_probs))
            entropy_values[end_idx] = entropy
    
    # Return as numpy array
    return cp.asnumpy(entropy_values)
# ---------------------------
# Feature Computation Classes
# ---------------------------

def safe_indicator_assign(self, df, column_name, indicator_result):
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

class FeatureComputer:
    """Main class for computing all technical features"""
    
    def __init__(self, use_talib=True, use_numba=True):
        self.use_talib = use_talib
        self.use_numba = use_numba

    def safe_indicator_assign(self, df, column_name, indicator_result):
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
            # Handle pandas Series results with potential index mismatch
            if hasattr(indicator_result, 'index'):
                # Reindex to match the DataFrame
                for idx in df.index:
                    if idx in indicator_result.index:
                        df.loc[idx, column_name] = indicator_result.loc[idx]
            # Handle single column results (could be Series or ndarray)
            elif isinstance(indicator_result, (pd.Series, np.ndarray)):
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
                
                # Reindex to match the DataFrame
                for idx in df.index:
                    if idx in result_series.index:
                        df.loc[idx, column_name] = result_series.loc[idx]
        except Exception as e:
            logging.warning(f"Error assigning indicator {column_name}: {e}")
        
        # Ensure any NaN values are filled with zeros
        df[column_name] = df[column_name].fillna(0)
        return df
        
    def compute_all_features(self, df: pd.DataFrame, debug_mode: bool = False, perf_monitor=None) -> pd.DataFrame:
        """Compute all features for the given dataframe with performance tracking"""
        # Ensure dataframe is sorted
        df = df.sort_values('timestamp_utc')
        
        # Compute price action features
        start_time = time.time()
        df = self.compute_price_action_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("price_action_features", time.time() - start_time)
        
        # Compute momentum features
        start_time = time.time()
        df = self.compute_momentum_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("momentum_features", time.time() - start_time)
        
        # Compute volatility features
        start_time = time.time()
        df = self.compute_volatility_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("volatility_features", time.time() - start_time)
        
        # Compute volume features
        start_time = time.time()
        df = self.compute_volume_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("volume_features", time.time() - start_time)
        
        # Compute statistical features
        start_time = time.time()
        df = self.compute_statistical_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("statistical_features", time.time() - start_time)
        
        # Compute pattern features
        start_time = time.time()
        df = self.compute_pattern_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("pattern_features", time.time() - start_time)
        
        # Compute time-based features
        start_time = time.time()
        df = self.compute_time_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("time_features", time.time() - start_time)
        
        # Compute label features
        start_time = time.time()
        df = self.compute_label_features(df, debug_mode)
        if perf_monitor:
            perf_monitor.log_operation("label_features", time.time() - start_time)
        
        # Clean NaN/inf values
        start_time = time.time()
        for col in df.select_dtypes(include=['float64', 'float32']).columns:
            if col not in ['timestamp_utc', 'pair']:
                df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
                
        # Set computation flags as Python booleans (not floats)
        df['features_computed'] = True  # True is a Python boolean
        df['targets_computed'] = True   # True is a Python boolean
        
        if perf_monitor:
            perf_monitor.log_operation("cleanup", time.time() - start_time)
        
        return df
    
    def compute_tsi(self, close, fast=13, slow=25):
        """Custom True Strength Index calculation"""
        # Calculate price changes
        price_change = close.diff()
        
        # Absolute price changes
        abs_price_change = price_change.abs()
        
        # Double smoothed price change
        smooth1 = price_change.ewm(span=fast, adjust=False).mean()
        smooth2 = smooth1.ewm(span=slow, adjust=False).mean()
        
        # Double smoothed absolute price change
        abs_smooth1 = abs_price_change.ewm(span=fast, adjust=False).mean()
        abs_smooth2 = abs_smooth1.ewm(span=slow, adjust=False).mean()
        
        # Avoid division by zero
        abs_smooth2 = abs_smooth2.replace(0, np.nan)
        
        # TSI = 100 * (smooth2 / abs_smooth2)
        tsi = 100 * (smooth2 / abs_smooth2)
        
        return tsi.fillna(0)

    def compute_ppo(self, close, fast=12, slow=26, signal=9):
        """Custom Percentage Price Oscillator calculation"""
        # Calculate EMAs
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        # Avoid division by zero
        ema_slow = ema_slow.replace(0, np.nan)
        
        # Calculate PPO
        ppo = 100 * ((ema_fast - ema_slow) / ema_slow)
        
        return ppo.fillna(0)
        
    def compute_price_action_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute price action features"""
        # Extract price data
        open_prices = df['open_1h'].values
        high_prices = df['high_1h'].values
        low_prices = df['low_1h'].values
        close_prices = df['close_1h'].values
        
        # Use Numba for body features if available
        if self.use_numba:
            body_features = compute_candle_body_features_numba(
                open_prices, high_prices, low_prices, close_prices
            )
            df['candle_body_size'] = body_features[0:len(df)]
            df['upper_shadow'] = body_features[len(df):2*len(df)]
            df['lower_shadow'] = body_features[2*len(df):3*len(df)]
            df['relative_close_position'] = body_features[3*len(df):4*len(df)]
        else:
            # Calculate candle body features directly
            df['candle_body_size'] = abs(df['close_1h'] - df['open_1h'])
            df['upper_shadow'] = df['high_1h'] - df[['open_1h', 'close_1h']].max(axis=1)
            df['lower_shadow'] = df[['open_1h', 'close_1h']].min(axis=1) - df['low_1h']
            
            # Calculate relative position of close within the high-low range
            hl_range = df['high_1h'] - df['low_1h']
            df['relative_close_position'] = np.where(
                hl_range > 0, 
                (df['close_1h'] - df['low_1h']) / hl_range, 
                0.5  # Default to middle if there's no range
            )
        
        # Log returns
        df['log_return'] = np.log(df['close_1h'] / df['close_1h'].shift(1)).fillna(0)
        
        # Gap open (compared to previous close)
        df['gap_open'] = (df['open_1h'] / df['close_1h'].shift(1) - 1).fillna(0)
        
        # Price velocity (rate of change over time)
        df['price_velocity'] = df['close_1h'].pct_change(3).fillna(0)
        
        # Price acceleration (change in velocity)
        df['price_acceleration'] = df['price_velocity'].diff(3).fillna(0)
        
        # Previous close percent change
        df['prev_close_change_pct'] = df['close_1h'].pct_change().fillna(0)
        
        return df
        
    def compute_momentum_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute momentum indicators"""
        if len(df) < 14:  # Need at least 14 points for most indicators
            return df
            
        # Extract price data
        close = df['close_1h']
        high = df['high_1h']
        low = df['low_1h']
        open_prices = df['open_1h']
        
        # RSI - Using pandas_ta
        try:
            rsi_result = pta.rsi(close, length=14)
            self.safe_indicator_assign(df, 'rsi_1h', rsi_result)
        except Exception as e:
            logging.warning(f"Error computing RSI: {e}")
            df['rsi_1h'] = 50  # Default value
        
        # Calculate RSI slope
        rsi_values = df['rsi_1h'].values
        df['rsi_slope_1h'] = np.zeros(len(df))
        if len(df) > 3:
            for i in range(3, len(df)):
                df.loc[i, 'rsi_slope_1h'] = (rsi_values[i] - rsi_values[i-3]) / 3
        
        # MACD - Using pandas_ta
        try:
            macd = pta.macd(close, fast=12, slow=26, signal=9)
            # Store MACD histogram directly (don't store macd_1h itself as it's not in the schema)
            self.safe_indicator_assign(df, 'macd_hist_slope_1h', macd['MACDh_12_26_9'])
            
            # Calculate the slopes that we need directly
            macd_values = macd['MACD_12_26_9'].values
            macd_hist_values = macd['MACDh_12_26_9'].values
            
            df['macd_slope_1h'] = np.zeros(len(df))
            df['macd_hist_slope_1h'] = np.zeros(len(df))
            
            if len(df) > 1:
                try:
                    diff_values = np.diff(macd_values)
                    for i in range(1, len(df)):
                        if i-1 < len(diff_values):
                            df.loc[i, 'macd_slope_1h'] = diff_values[i-1]
                    
                    diff_hist_values = np.diff(macd_hist_values)
                    for i in range(1, len(df)):
                        if i-1 < len(diff_hist_values):
                            df.loc[i, 'macd_hist_slope_1h'] = diff_hist_values[i-1]
                except Exception as e:
                    logging.warning(f"Error computing MACD slopes: {e}")
        except Exception as e:
            logging.warning(f"Error computing MACD: {e}")
            df['macd_slope_1h'] = 0
            df['macd_hist_slope_1h'] = 0
        
        # Stochastic Oscillator
        try:
            if self.use_talib:
                stoch_k, stoch_d = talib.STOCH(high.values, low.values, close.values)
                df['stoch_k_14'] = stoch_k
                df['stoch_d_14'] = stoch_d
            else:
                stoch = pta.stoch(high, low, close, k=14, d=3)
                self.safe_indicator_assign(df, 'stoch_k_14', stoch['STOCHk_14_3_3'])
                self.safe_indicator_assign(df, 'stoch_d_14', stoch['STOCHd_14_3_3'])
        except Exception as e:
            logging.warning(f"Error computing Stochastic Oscillator: {e}")
            df['stoch_k_14'] = 50
            df['stoch_d_14'] = 50
        
        # Williams %R
        try:
            if self.use_talib:
                df['williams_r_14'] = talib.WILLR(high.values, low.values, close.values, timeperiod=14)
            else:
                willr_result = pta.willr(high, low, close, length=14)
                self.safe_indicator_assign(df, 'williams_r_14', willr_result)
        except Exception as e:
            logging.warning(f"Error computing Williams %R: {e}")
            df['williams_r_14'] = -50
        
        # CCI (Commodity Channel Index)
        try:
            if self.use_talib:
                df['cci_14'] = talib.CCI(high.values, low.values, close.values, timeperiod=14)
            else:
                cci_result = pta.cci(high, low, close, length=14)
                self.safe_indicator_assign(df, 'cci_14', cci_result)
        except Exception as e:
            logging.warning(f"Error computing CCI: {e}")
            df['cci_14'] = 0
        
        # ROC (Rate of Change)
        try:
            if self.use_talib:
                df['roc_10'] = talib.ROC(close.values, timeperiod=10)
            else:
                roc_result = pta.roc(close, length=10)
                self.safe_indicator_assign(df, 'roc_10', roc_result)
        except Exception as e:
            logging.warning(f"Error computing ROC: {e}")
            df['roc_10'] = 0
        
        # TSI (True Strength Index) - Using our custom implementation
        try:
            tsi_result = self.compute_tsi(close, fast=13, slow=25)
            df['tsi'] = tsi_result.values
        except Exception as e:
            logging.warning(f"Error computing TSI: {e}")
            df['tsi'] = 0
        
        # Awesome Oscillator
        try:
            ao_result = pta.ao(high, low)
            self.safe_indicator_assign(df, 'awesome_oscillator', ao_result)
        except Exception as e:
            logging.warning(f"Error computing Awesome Oscillator: {e}")
            df['awesome_oscillator'] = 0
        
        # PPO (Percentage Price Oscillator) - Using our custom implementation
        try:
            ppo_result = self.compute_ppo(close)
            df['ppo'] = ppo_result.values
        except Exception as e:
            logging.warning(f"Error computing PPO: {e}")
            df['ppo'] = 0
        
        return df
        
    def compute_volatility_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute volatility indicators with fixed division by zero handling"""
        if len(df) < 20:  # Need at least 20 points for volatility indicators
            return df
            
        # Extract price data
        close = df['close_1h']
        high = df['high_1h']
        low = df['low_1h']
        
        # ATR (Average True Range)
        if self.use_talib:
            df['atr_1h'] = talib.ATR(high.values, low.values, close.values, timeperiod=14)
        else:
            atr_result = pta.atr(high, low, close, length=14)
            self.safe_indicator_assign(df, 'atr_1h', atr_result)
        
        # Normalized ATR (ATR / Close price)
        # Add safety against division by zero
        df['normalized_atr_14'] = np.where(close > 0, df['atr_1h'] / close, 0)
        
        # True Range
        if self.use_talib:
            df['true_range'] = talib.TRANGE(high.values, low.values, close.values)
        else:
            tr_result = pta.true_range(high, low, close.shift(1))
            self.safe_indicator_assign(df, 'true_range', tr_result)
        
        # Bollinger Bands
        if self.use_talib:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close.values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bollinger_width_1h'] = (bb_upper - bb_lower)
            
            # FIXED: Handle division by zero
            bb_diff = bb_upper - bb_lower
            df['bollinger_percent_b'] = np.where(
                bb_diff != 0, 
                (close.values - bb_lower) / bb_diff,
                0.5  # Default to middle if no range
            )
        else:
            bbands = pta.bbands(close, length=20, std=2.0)
            bb_width = bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']
            self.safe_indicator_assign(df, 'bollinger_width_1h', bb_width)
            
            # FIXED: Handle division by zero
            bb_upper = bbands['BBU_20_2.0']
            bb_lower = bbands['BBL_20_2.0']
            bb_diff = bb_upper - bb_lower
            bb_percent = np.where(
                bb_diff > 0,
                (close - bb_lower) / bb_diff,
                0.5  # Default to middle if no range
            )
            df['bollinger_percent_b'] = bb_percent
        
        # Donchian Channels
        dc_high = high.rolling(window=20).max()
        dc_low = low.rolling(window=20).min()
        dc_width = (dc_high - dc_low).fillna(0)
        self.safe_indicator_assign(df, 'donchian_channel_width_1h', dc_width)
        
        # Keltner Channels
        if self.use_talib:
            ema20 = talib.EMA(close.values, timeperiod=20)
            df['keltner_channel_width'] = (
                (ema20 + df['atr_1h'].values * 2) - 
                (ema20 - df['atr_1h'].values * 2)
            )
        else:
            kc = pta.kc(high, low, close, length=20, scalar=2)
            kc_width = kc['KCUe_20_2'] - kc['KCLe_20_2']
            self.safe_indicator_assign(df, 'keltner_channel_width', kc_width)
        
        # Historical Volatility
        hist_vol = close.pct_change().rolling(window=30).std() * np.sqrt(252)
        self.safe_indicator_assign(df, 'historical_vol_30', hist_vol)
        
        # Custom Chaikin Volatility Calculation
        def custom_chaikin_volatility(high_series, low_series, close_series, window=10):
            # Calculate the difference between high and low
            hl_range = high_series - low_series
            
            # Calculate EMA of high-low range
            ema_range = hl_range.ewm(span=window, adjust=False).mean()
            
            # Calculate volatility as percentage change of EMA range
            volatility = ema_range.pct_change()
            
            return volatility.fillna(0)
        
        df['chaikin_volatility'] = custom_chaikin_volatility(high, low, close)
        
        # Calculate volatility rank (will be updated in cross_pair features)
        df['volatility_rank_1h'] = 0
        
        return df
        
    def compute_volume_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute volume-based indicators"""
        if len(df) < 14:  # Need at least 14 points for volume indicators
            return df
            
        # Extract price and volume data
        close = df['close_1h']
        high = df['high_1h']
        low = df['low_1h']
        volume = df['volume_1h']
        
        # Money Flow Index
        try:
            if self.use_talib:
                df['money_flow_index_1h'] = talib.MFI(
                    high.values, low.values, close.values, volume.values, timeperiod=14
                )
            else:
                mfi_result = pta.mfi(high, low, close, volume, length=14)
                self.safe_indicator_assign(df, 'money_flow_index_1h', mfi_result)
        except Exception as e:
            logging.warning(f"Error computing Money Flow Index: {e}")
            df['money_flow_index_1h'] = 50
        
        # OBV (On-Balance Volume)
        try:
            if self.use_talib:
                obv = talib.OBV(close.values, volume.values)
            else:
                obv_result = pta.obv(close, volume)
                self.safe_indicator_assign(df, 'obv_1h', obv_result)
                obv = df['obv_1h'].values
        except Exception as e:
            logging.warning(f"Error computing OBV: {e}")
            obv = np.zeros(len(df))
        
        # OBV slope
        df['obv_slope_1h'] = np.zeros(len(df))
        if len(df) > 3:
            for i in range(3, len(df)):
                df.loc[i, 'obv_slope_1h'] = (obv[i] - obv[i-3]) / 3
        
        # Volume change percentage
        df['volume_change_pct_1h'] = volume.pct_change().fillna(0)
        
        # VWMA (Volume Weighted Moving Average)
        try:
            vwma_result = pta.vwma(close, volume, length=20)
            self.safe_indicator_assign(df, 'vwma_20', vwma_result)
        except Exception as e:
            logging.warning(f"Error computing VWMA: {e}")
            df['vwma_20'] = close
        
        # Chaikin Money Flow
        try:
            cmf_result = pta.cmf(high, low, close, volume)
            self.safe_indicator_assign(df, 'chaikin_money_flow', cmf_result)
        except Exception as e:
            logging.warning(f"Error computing Chaikin Money Flow: {e}")
            df['chaikin_money_flow'] = 0
        
        # Klinger Oscillator
        try:
            kvo_result = pta.kvo(high, low, close, volume)
            self.safe_indicator_assign(df, 'klinger_oscillator', kvo_result['KVO_34_55_13'])
        except Exception as e:
            logging.warning(f"Error computing Klinger Oscillator: {e}")
            df['klinger_oscillator'] = 0
        
        # Volume Oscillator (defined as the difference between fast and slow volume EMAs)
        vol_fast = volume.ewm(span=14).mean()
        vol_slow = volume.ewm(span=28).mean()
        df['volume_oscillator'] = (vol_fast - vol_slow).fillna(0)
        
        # Volume Price Trend (Custom Implementation)
        def volume_price_trend(close, volume):
            """Custom Volume Price Trend (VPT) calculation"""
            vpt = [0]  # Start with 0
            for i in range(1, len(close)):
                percent_change = (close[i] - close[i-1]) / close[i-1]
                vpt.append(vpt[-1] + volume[i] * percent_change)
            return pd.Series(vpt, index=close.index)

        try:
            vpt_result = volume_price_trend(close, volume)
            self.safe_indicator_assign(df, 'volume_price_trend', vpt_result)
        except Exception as e:
            logging.warning(f"Error computing Volume Price Trend: {e}")
            df['volume_price_trend'] = 0
        
        # Volume Zone Oscillator
        def volume_zone_oscillator(close, volume, length=14):
            """
            Volume Zone Oscillator implementation.
            VZO determines if volume is flowing into or out of the security.
            
            Args:
                close: Series of closing prices
                volume: Series of volume data
                length: Period for calculations, default 14
                
            Returns:
                Series containing VZO values
            """
            # Make sure the indexes match
            close = close.copy()
            volume = volume.copy()
            
            # Determine price direction
            price_up = close > close.shift(1)
            
            # Categorize volume based on price direction
            vol_up = volume.copy()
            vol_up[~price_up] = 0
            
            vol_down = volume.copy()
            vol_down[price_up] = 0
            
            # Calculate EMAs
            ema_vol = volume.ewm(span=length, adjust=False).mean()
            ema_vol_up = vol_up.ewm(span=length, adjust=False).mean()
            ema_vol_down = vol_down.ewm(span=length, adjust=False).mean()
            
            # Calculate VZO
            vzo = 100 * (ema_vol_up - ema_vol_down) / ema_vol
            
            return vzo.fillna(0)
        
        try:
            vzo_result = volume_zone_oscillator(close, volume)
            df['volume_zone_oscillator'] = vzo_result
        except Exception as e:
            logging.warning(f"Error computing Volume Zone Oscillator: {e}")
            df['volume_zone_oscillator'] = 0
        
        # Volume Price Confirmation Indicator (close direction matches volume direction)
        close_dir = np.sign(close.diff()).fillna(0)
        vol_dir = np.sign(volume.diff()).fillna(0)
        df['volume_price_confirmation'] = (close_dir == vol_dir).astype(int)
        
        # Volume rank (will be updated in cross_pair features)
        df['volume_rank_1h'] = 0
        df['prev_volume_rank'] = 0
        
        return df
        
    def compute_statistical_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute statistical and microstructure features with fixed ffill() calls"""
        if len(df) < 20:  # Need at least 20 points for statistical features
            return df
            
        # Extract price data
        close = df['close_1h']
        
        # Standard deviation of returns
        df['std_dev_returns_20'] = df['log_return'].rolling(window=20).std().fillna(0)
        
        # Skewness and kurtosis
        df['skewness_20'] = df['log_return'].rolling(window=20).apply(
            lambda x: stats.skew(x) if len(x) > 3 else 0, raw=True
        ).fillna(0)
        
        df['kurtosis_20'] = df['log_return'].rolling(window=20).apply(
            lambda x: stats.kurtosis(x) if len(x) > 3 else 0, raw=True
        ).fillna(0)
        
        # Z-score
        ma_20 = close.rolling(window=20).mean()
        
        if self.use_numba:
            df['z_score_20'] = compute_z_score_numba(close.values, ma_20.fillna(0).values, 20)
        else:
            std_20 = close.rolling(window=20).std()
            df['z_score_20'] = ((close - ma_20) / std_20).fillna(0)
        
        # Hurst Exponent (measure of long-term memory in time series)
        if len(df) >= 100:
            if self.use_numba:
                # FIXED: Using ffill() instead of fillna(method='ffill')
                df['hurst_exponent'] = hurst_exponent_numba(
                    close.replace(0, np.nan).ffill().values, 
                    20  # Explicitly provide the max_lag parameter
                )
            else:
                # Calculate standard (non-numba) Hurst exponent for each window
                def hurst_window(window):
                    if len(window) < 100 or np.any(window <= 0):
                        return 0.5
                    try:
                        # Returns
                        returns = np.diff(np.log(window))
                        # Lags to calculate
                        lags = range(2, 20)
                        tau = []; lag_var = []
                        
                        for lag in lags:
                            # Price difference for the lag
                            lag_returns = returns[lag:] - returns[:-lag]
                            # Variance
                            lag_var.append(np.std(lag_returns))
                            tau.append(lag)
                        
                        # Regression
                        m = np.polyfit(np.log(tau), np.log(lag_var), 1)
                        return m[0] / 2
                    except:
                        return 0.5
                
                # Apply rolling window for Hurst calculation
                window_size = 100
                df['hurst_exponent'] = 0.5  # Default value
                for i in range(window_size, len(df)):
                    window = close.iloc[i-window_size:i].values
                    df.loc[df.index[i], 'hurst_exponent'] = hurst_window(window)
        else:
            df['hurst_exponent'] = 0.5  # Default value for short series
        
        # Shannon Entropy (measure of randomness/predictability)
        if self.use_numba and len(df) >= 20:
            df['shannon_entropy'] = shannon_entropy_numba(
                close.replace(0, np.nan).ffill().values, 20
            )
        else:
            df['shannon_entropy'] = 0  # Default value
            
            # Compute entropy for each window
            if len(df) >= 20:
                def entropy(window):
                    if len(window) < 2:
                        return 0
                    returns = np.diff(np.log(window))
                    hist, _ = np.histogram(returns, bins=10)
                    probs = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
                    return -np.sum(probs * np.log(probs + 1e-10))
                
                for i in range(20, len(df)):
                    window = close.iloc[i-20:i].values
                    if np.any(window <= 0):
                        df.loc[df.index[i], 'shannon_entropy'] = 0
                    else:
                        df.loc[df.index[i], 'shannon_entropy'] = entropy(window)
        
        # Autocorrelation lag 1
        df['autocorr_1'] = df['log_return'].rolling(window=20).apply(
            lambda x: x.autocorr(1) if len(x) > 1 else 0
        ).fillna(0)
        
        # Estimated slippage and bid-ask spread proxies
        df['estimated_slippage_1h'] = df['high_1h'] - df['low_1h']
        df['bid_ask_spread_1h'] = df['estimated_slippage_1h'] * 0.1  # Rough approximation
        
        return df
        
    def compute_pattern_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute candlestick patterns with fixed ffill() calls"""
        if len(df) < 5:  # Need at least a few candles for patterns
            return df
            
        # Extract price data
        open_prices = df['open_1h']
        high = df['high_1h']
        low = df['low_1h']
        close = df['close_1h']
        
        # Use TA-Lib for patterns if available
        if self.use_talib:
            # Doji pattern
            df['pattern_doji'] = talib.CDLDOJI(open_prices.values, high.values, low.values, close.values)
            
            # Engulfing pattern
            df['pattern_engulfing'] = talib.CDLENGULFING(open_prices.values, high.values, low.values, close.values)
            
            # Hammer pattern
            df['pattern_hammer'] = talib.CDLHAMMER(open_prices.values, high.values, low.values, close.values)
            
            # Morning star pattern
            df['pattern_morning_star'] = talib.CDLMORNINGSTAR(open_prices.values, high.values, low.values, close.values)
        else:
            # Implement simplified pattern detection without TA-Lib
            
            # Doji: open and close are almost equal
            body_size = abs(close - open_prices)
            total_size = high - low
            df['pattern_doji'] = ((body_size / total_size) < 0.1).astype(int)
            
            # Engulfing: current candle's body completely engulfs previous candle's body
            prev_body_low = np.minimum(open_prices.shift(1), close.shift(1))
            prev_body_high = np.maximum(open_prices.shift(1), close.shift(1))
            curr_body_low = np.minimum(open_prices, close)
            curr_body_high = np.maximum(open_prices, close)
            
            bullish_engulfing = (close > open_prices) & (curr_body_low < prev_body_low) & (curr_body_high > prev_body_high)
            bearish_engulfing = (close < open_prices) & (curr_body_low < prev_body_low) & (curr_body_high > prev_body_high)
            df['pattern_engulfing'] = (bullish_engulfing | bearish_engulfing).astype(int)
            
            # Hammer: small body at the top, long lower shadow
            body_size = abs(close - open_prices)
            upper_shadow = high - np.maximum(open_prices, close)
            lower_shadow = np.minimum(open_prices, close) - low
            
            df['pattern_hammer'] = (
                (body_size < 0.3 * total_size) & 
                (lower_shadow > 2 * body_size) & 
                (upper_shadow < 0.1 * total_size)
            ).astype(int)
            
            # Morning star: simplified
            df['pattern_morning_star'] = 0  # Stub for compatibility
        
        # Convert all to int for small int columns
        for col in ['pattern_doji', 'pattern_engulfing', 'pattern_hammer', 'pattern_morning_star']:
            df[col] = df[col].astype(int)
        
        # Simple support and resistance levels (using rolling max/min)
        df['resistance_level'] = high.rolling(20).max().ffill()
        df['support_level'] = low.rolling(20).min().ffill()
        
        return df
        
    def compute_time_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute time-based features"""
        # Extract timestamp
        timestamps = pd.to_datetime(df['timestamp_utc'])
        
        # Extract hour, day, month
        df['hour_of_day'] = timestamps.dt.hour
        df['day_of_week'] = timestamps.dt.dayofweek
        df['month_of_year'] = timestamps.dt.month
        
        # Weekend indicator
        df['is_weekend'] = ((df['day_of_week'] >= 5) | (df['day_of_week'] == 4) & (df['hour_of_day'] >= 21)).astype(int)
        
        # Trading sessions (approximate)
        df['asian_session'] = (
            ((df['hour_of_day'] >= 0) & (df['hour_of_day'] < 8))
        ).astype(int)
        
        df['european_session'] = (
            ((df['hour_of_day'] >= 8) & (df['hour_of_day'] < 16))
        ).astype(int)
        
        df['american_session'] = (
            ((df['hour_of_day'] >= 14) & (df['hour_of_day'] < 22))
        ).astype(int)
        
        return df
        
    def compute_multi_timeframe_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute features for 4h and 1d timeframes"""
        if len(df) < 24:  # Need at least a day of data
            return df
            
        # Make sure timestamp is properly set
        df_copy = df.copy()
        df_copy['timestamp_utc'] = pd.to_datetime(df_copy['timestamp_utc'])
        
        # Calculate 4h timeframe features
        df = self.resample_and_compute('4h', df, df_copy, debug_mode)
        
        # Calculate 1d timeframe features
        df = self.resample_and_compute('1d', df, df_copy, debug_mode)
        
        return df
        
    def resample_and_compute(self, timeframe: str, original_df: pd.DataFrame, 
                        df_with_timestamp: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Resample to the given timeframe and compute features with fixed ffill() calls"""
        if debug_mode:
            start_time = time.time()
            
        tf_label = '4h' if timeframe == '4h' else '1d'
        resample_rule = timeframe
        
        required_points = {
            '4h': 24,   # 4h window = 1 day
            '1d': 30    # 1d window = ~1 month
        }
        min_points = required_points.get(resample_rule, 20)
        
        # Create a copy of the dataframe to avoid modifying the original
        result_df = original_df.copy()
        
        # Define expected column names for this timeframe
        expected_columns = [
            f'rsi_{tf_label}', f'rsi_slope_{tf_label}', f'macd_slope_{tf_label}',
            f'macd_hist_slope_{tf_label}', f'atr_{tf_label}', f'bollinger_width_{tf_label}',
            f'donchian_channel_width_{tf_label}', f'money_flow_index_{tf_label}', 
            f'obv_slope_{tf_label}', f'volume_change_pct_{tf_label}'
        ]
        
        # Initialize all expected columns with default value 0
        for col_name in expected_columns:
            result_df[col_name] = 0.0
        
        if len(df_with_timestamp) < min_points:
            if 'pair' in df_with_timestamp.columns and not df_with_timestamp.empty:
                logging.warning(
                    f"Skipping {df_with_timestamp['pair'].iloc[0]} {tf_label} features: only {len(df_with_timestamp)} rows (need >= {min_points})"
                )
            # Return the dataframe with initialized columns
            return result_df
            
        # Set timestamp as index for resampling
        df_with_ts = df_with_timestamp.copy()
        df_with_ts.set_index('timestamp_utc', inplace=True)
        
        # Perform resampling
        resampled = pd.DataFrame()
        resampled[f'open_{tf_label}'] = df_with_ts['open_1h'].resample(resample_rule).first()
        resampled[f'high_{tf_label}'] = df_with_ts['high_1h'].resample(resample_rule).max()
        resampled[f'low_{tf_label}'] = df_with_ts['low_1h'].resample(resample_rule).min()
        resampled[f'close_{tf_label}'] = df_with_ts['close_1h'].resample(resample_rule).last()
        resampled[f'volume_{tf_label}'] = df_with_ts['volume_1h'].resample(resample_rule).sum()
        
        # Drop rows with missing values
        resampled.dropna(inplace=True)
        
        if len(resampled) < 5:  # Need minimum data for indicators
            logging.warning(f"Not enough resampled data for {tf_label}, using default values")
            return result_df
        
        if debug_mode:
            logging.debug(f"{tf_label} resampling completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # Compute indicators on resampled data
        # RSI
        resampled[f'rsi_{tf_label}'] = pta.rsi(resampled[f'close_{tf_label}'], length=14).fillna(50)
        
        # RSI slope
        resampled[f'rsi_slope_{tf_label}'] = resampled[f'rsi_{tf_label}'].diff(2) / 2
        
        # MACD
        macd = pta.macd(resampled[f'close_{tf_label}'], fast=12, slow=26, signal=9)
        resampled[f'macd_{tf_label}'] = macd['MACD_12_26_9'].fillna(0)
        resampled[f'macd_signal_{tf_label}'] = macd['MACDs_12_26_9'].fillna(0)
        resampled[f'macd_slope_{tf_label}'] = resampled[f'macd_{tf_label}'].diff().fillna(0)
        resampled[f'macd_hist_{tf_label}'] = macd['MACDh_12_26_9'].fillna(0)
        resampled[f'macd_hist_slope_{tf_label}'] = resampled[f'macd_hist_{tf_label}'].diff().fillna(0)
        
        # ATR
        resampled[f'atr_{tf_label}'] = pta.atr(
            resampled[f'high_{tf_label}'],
            resampled[f'low_{tf_label}'],
            resampled[f'close_{tf_label}'],
            length=14
        ).fillna(0)
        
        # Bollinger Bands
        bbands = pta.bbands(resampled[f'close_{tf_label}'], length=20, std=2.0)
        resampled[f'bollinger_width_{tf_label}'] = (
            bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']
        ).fillna(0)
        
        # Donchian Channels
        high_series = resampled[f'high_{tf_label}']
        low_series = resampled[f'low_{tf_label}']
        resampled[f'donchian_high_{tf_label}'] = high_series.rolling(window=20).max().ffill().fillna(high_series)
        resampled[f'donchian_low_{tf_label}'] = low_series.rolling(window=20).min().ffill().fillna(low_series)
        resampled[f'donchian_channel_width_{tf_label}'] = resampled[f'donchian_high_{tf_label}'] - resampled[f'donchian_low_{tf_label}']
        
        # MFI
        resampled[f'money_flow_index_{tf_label}'] = pta.mfi(
            resampled[f'high_{tf_label}'],
            resampled[f'low_{tf_label}'],
            resampled[f'close_{tf_label}'],
            resampled[f'volume_{tf_label}'],
            length=14
        ).fillna(50)
        
        # OBV and slope
        resampled[f'obv_{tf_label}'] = pta.obv(
            resampled[f'close_{tf_label}'], 
            resampled[f'volume_{tf_label}']
        ).fillna(0)
        resampled[f'obv_slope_{tf_label}'] = resampled[f'obv_{tf_label}'].diff(2) / 2
        
        # Volume change
        resampled[f'volume_change_pct_{tf_label}'] = resampled[f'volume_{tf_label}'].pct_change().fillna(0)
        
        # Clean up any NaN/Inf values
        for col in resampled.columns:
            resampled[col] = resampled[col].replace([np.inf, -np.inf], 0).fillna(0)
            
        if debug_mode:
            logging.debug(f"{tf_label} feature calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # Map resampled values back to original timeframe using forward fill
        for col in expected_columns:
            if col in resampled.columns:
                # Create a Series with resampled index
                temp_series = pd.Series(resampled[col].values, index=resampled.index)
                
                # Reindex to all timestamps in original dataframe
                temp_series = temp_series.reindex(
                    pd.date_range(
                        start=min(df_with_timestamp['timestamp_utc']),
                        end=max(df_with_timestamp['timestamp_utc']),
                        freq='1H'
                    ),
                    method='ffill'
                )
                
                # Map back to original dataframe
                for idx, row in df_with_timestamp.iterrows():
                    timestamp = row['timestamp_utc']
                    if timestamp in temp_series.index:
                        result_df.loc[idx, col] = temp_series[timestamp]
                        
        if debug_mode:
            logging.debug(f"{tf_label} data mapping completed in {time.time() - start_time:.3f}s")
            
        return result_df
        
    def compute_label_features(self, df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute label/target features"""
        if len(df) < 5:
            return df
            
        # Extract price data
        close = df['close_1h'].values
        high = df['high_1h'].values
        low = df['low_1h'].values
        
        # Calculate future returns for different time horizons
        horizons = {
            '1h': 1,
            '4h': 4,
            '12h': 12,
            '1d': 24,
            '3d': 72,
            '1w': 168,
            '2w': 336
        }
        
        for horizon_name, shift in horizons.items():
            col_name = f'future_return_{horizon_name}_pct'
            
            # Use vectorized operations for speed
            future_return = np.zeros(len(close))
            
            if len(close) > shift:
                divider = np.maximum(close[:-shift], np.ones(len(close) - shift) * 1e-8)
                future_return[:-shift] = (close[shift:] - close[:-shift]) / divider
                
            df[col_name] = future_return
        
        # Max future return calculation
        future_max_return = np.zeros(len(close))
        for i in range(len(close) - 1):
            end_idx = min(i + 25, len(high))  # 24h = 24 candles in 1h timeframe
            if i + 1 < end_idx:
                max_high = np.max(high[i+1:end_idx])
                future_max_return[i] = (max_high - close[i]) / max(close[i], 1e-8)
                
        df['future_max_return_24h_pct'] = future_max_return
        
        # Max future drawdown calculation
        future_max_drawdown = np.zeros(len(close))
        for i in range(len(close) - 1):
            end_idx = min(i + 13, len(low))  # 12h = 12 candles in 1h timeframe
            if i + 1 < end_idx:
                min_low = np.min(low[i+1:end_idx])
                future_max_drawdown[i] = (min_low - close[i]) / max(close[i], 1e-8)
                
        df['future_max_drawdown_12h_pct'] = future_max_drawdown
        
        # Was profitable
        df['was_profitable_12h'] = (df['future_return_12h_pct'] > 0).astype(int)
        
        # Risk-adjusted return (return / max drawdown)
        df['future_risk_adj_return_12h'] = (
            df['future_return_12h_pct'] / 
            np.abs(df['future_max_drawdown_12h_pct'].replace(0, 1e-8))
        ).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Target hit indicators (simple profit targets)
        df['profit_target_1pct'] = (df['future_max_return_24h_pct'] > 0.01).astype(int)
        df['profit_target_2pct'] = (df['future_max_return_24h_pct'] > 0.02).astype(int)
        
        # BTC correlation (for non-BTC pairs)
        df['btc_corr_24h'] = 0.0  # Default, will be computed in cross-pair features
        
        return df
    
class CrossPairFeatureComputer:
    """Compute features that require data across multiple pairs"""
    
    def compute_cross_pair_features(self, latest_df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """Compute cross-pair metrics relative to BTC-USDT and ETH-USDT"""
        if len(latest_df) == 0:
            logging.warning("Empty DataFrame passed to compute_cross_pair_features")
            return latest_df
            
        # Initialize columns with 0 to guarantee they exist
        latest_df['volume_rank_1h'] = 0
        latest_df['volatility_rank_1h'] = 0
        latest_df['performance_rank_btc_1h'] = 0
        latest_df['performance_rank_eth_1h'] = 0
        latest_df['btc_corr_24h'] = 0.0
        
        # Volume rank
        if 'volume_1h' in latest_df.columns:
            latest_df['volume_rank_1h'] = latest_df['volume_1h'].rank(pct=True) * 100
            latest_df['prev_volume_rank'] = latest_df['volume_rank_1h'].shift(1).fillna(0)
            
        # Volatility rank
        if 'atr_1h' in latest_df.columns:
            latest_df['volatility_rank_1h'] = latest_df['atr_1h'].rank(pct=True) * 100

        # Bitcoin correlation (for pairs that aren't BTC)
        btc_pair = latest_df[latest_df['pair'] == 'BTC-USDT']
        if not btc_pair.empty and len(latest_df) > 24:  # Need at least 24h of data
            # Get BTC returns
            btc_returns = btc_pair['log_return'].values
            if len(btc_returns) >= 24:
                # For each non-BTC pair, calculate correlation
                for pair_name in latest_df['pair'].unique():
                    if pair_name != 'BTC-USDT':
                        pair_data = latest_df[latest_df['pair'] == pair_name]
                        pair_returns = pair_data['log_return'].values
                        
                        # Only compute if we have enough data
                        if len(pair_returns) >= 24:
                            # Calculate rolling correlation
                            for i in range(24, len(pair_returns)):
                                btc_window = btc_returns[i-24:i]
                                pair_window = pair_returns[i-24:i]
                                
                                try:
                                    corr = np.corrcoef(btc_window, pair_window)[0, 1]
                                    latest_df.loc[
                                        (latest_df['pair'] == pair_name) & 
                                        (latest_df.index == pair_data.index[i]), 
                                        'btc_corr_24h'
                                    ] = corr
                                except:
                                    pass  # Keep as 0 if correlation fails

        # Performance rank relative to BTC
        btc_row = latest_df[latest_df['pair'] == 'BTC-USDT']
        if not btc_row.empty and 'future_return_1h_pct' in latest_df.columns:
            btc_return = btc_row['future_return_1h_pct'].values[0]
            if not pd.isna(btc_return) and abs(btc_return) > 1e-9:
                latest_df['performance_rank_btc_1h'] = (
                    (latest_df['future_return_1h_pct'] - btc_return) / abs(btc_return)
                ).rank(pct=True) * 100

        # Performance rank relative to ETH
        eth_row = latest_df[latest_df['pair'] == 'ETH-USDT']
        if not eth_row.empty and 'future_return_1h_pct' in latest_df.columns:
            eth_return = eth_row['future_return_1h_pct'].values[0]
            if not pd.isna(eth_return) and abs(eth_return) > 1e-9:
                latest_df['performance_rank_eth_1h'] = (
                    (latest_df['future_return_1h_pct'] - eth_return) / abs(eth_return)
                ).rank(pct=True) * 100

        # Fill NaN values with 0
        for col in ['volume_rank_1h', 'volatility_rank_1h', 'performance_rank_btc_1h', 
                    'performance_rank_eth_1h', 'btc_corr_24h', 'prev_volume_rank']:
            latest_df[col] = latest_df[col].fillna(0)

        return latest_df

# ---------------------------
# Single-Pair Processing Logic
# ---------------------------
def fetch_data(pair: str, db_conn, rolling_window: int = None) -> pd.DataFrame:
    """Fetch candles_1h for the given pair, limited to rolling_window if specified"""
    if rolling_window:
        # Fetch with extra padding for indicators
        lookback_padding = 300
        query = """
            SELECT *
            FROM candles_1h
            WHERE pair = :pair
            ORDER BY timestamp_utc DESC
            LIMIT :limit
        """
        df = pd.read_sql(
            text(query), 
            db_conn, 
            params={"pair": pair, "limit": rolling_window + lookback_padding}
        )
        return df.sort_values('timestamp_utc')
    else:
        # Fetch all data
        query = """
            SELECT *
            FROM candles_1h
            WHERE pair = :pair
            ORDER BY timestamp_utc ASC
        """
        return pd.read_sql(text(query), db_conn, params={"pair": pair})

def process_pair(pair: str, db_manager, rolling_window: int, config_manager=None, debug_mode: bool = False, perf_monitor=None) -> None:
    """Process a single pair, compute features, and update the database with optional GPU acceleration"""
    start_process = time.time()
    logging.info(f"Computing features for {pair}")
    updated_rows = 0

    try:
        # Create feature computer instance with GPU support if available
        use_talib = True if config_manager is None else config_manager.use_talib()
        use_numba = True if config_manager is None else config_manager.use_numba()
        use_gpu = False if config_manager is None else config_manager.use_gpu()
        
        # Use GPU feature computer if enabled
        if use_gpu:
            feature_computer = GPUFeatureComputer(use_talib=use_talib, use_numba=use_numba, use_gpu=use_gpu)
        else:
            feature_computer = FeatureComputer(use_talib=use_talib, use_numba=use_numba)
        
        # Create a fresh connection for this pair
        with db_manager.get_engine().connect() as db_conn:
            # Track database connection time
            start_db_connect = time.time()
            db_columns = get_database_columns(db_manager.get_engine(), 'candles_1h')
            if perf_monitor:
                perf_monitor.log_operation("db_get_columns", time.time() - start_db_connect)
            
            # Fetch data with timing
            start_fetch = time.time()
            df = fetch_data(pair, db_conn, rolling_window + 300)
            if perf_monitor:
                perf_monitor.log_operation("fetch_data", time.time() - start_fetch)
            
            if debug_mode:
                logging.debug(f"{pair}: Fetched data in {time.time() - start_process:.3f}s, {len(df)} rows")
                
            row_count = len(df)

            if df.empty or row_count < MIN_CANDLES_REQUIRED:
                logging.warning(
                    f"Skipping {pair}: only {row_count} candles, need >= {MIN_CANDLES_REQUIRED}"
                )
                return
                
            # Free memory
            gc.collect()

            # Compute all features with timing
            start_compute = time.time()
            df = feature_computer.compute_all_features(df, debug_mode, perf_monitor)
            if perf_monitor:
                perf_monitor.log_operation("compute_all_features_total", time.time() - start_compute)
            
            # Take only the newest rows for updating
            start_prep = time.time()
            df_to_update = df.iloc[-rolling_window:].copy() if len(df) > rolling_window else df
            
            if debug_mode:
                logging.debug(f"{pair}: Selected {len(df_to_update)}/{len(df)} rows for update")
                start_time = time.time()
                
            # Define columns to update (all computed features that exist in the database)
            reserved_columns = ['id', 'pair', 'timestamp_utc', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 
                              'volume_1h', 'quote_volume_1h', 'taker_buy_base_1h']
            
            # Filter out columns that don't exist in the database
            columns_for_update = [col for col in df_to_update.columns 
                               if col not in reserved_columns and col in db_columns]
            
            if debug_mode:
                # Log columns being skipped because they're not in the database
                skipped_columns = [col for col in df_to_update.columns 
                                if col not in reserved_columns and col not in db_columns]
                if skipped_columns:
                    logging.debug(f"{pair}: Skipping columns not in database: {', '.join(skipped_columns)}")

            # Verify all columns exist and have correct types
            for col in columns_for_update:
                if col not in df_to_update.columns:
                    # Add missing column with appropriate type
                    col_type = int if col in SMALLINT_COLUMNS else float
                    df_to_update[col] = np.zeros(len(df_to_update), dtype=col_type)
                else:
                    # Handle NaN and type conversion
                    df_to_update[col] = df_to_update[col].replace([np.inf, -np.inf], 0).fillna(0)
                    
                    # Ensure correct type
                    col_type = int if col in SMALLINT_COLUMNS else float
                    try:
                        df_to_update[col] = df_to_update[col].astype(col_type)
                    except Exception as e:
                        logging.warning(f"Error converting {col} to {col_type}: {e}")
                        df_to_update[col] = 0 if col_type == int else 0.0

            # Skip update if no valid columns to update
            if not columns_for_update:
                logging.warning(f"{pair}: No valid columns to update in database")
                return
            
            # Execute database update with timing
            start_update = time.time()

            # Prepare dynamic update query
            update_query = """
            UPDATE candles_1h
            SET """ + ",\n".join([f"{col} = %s" for col in columns_for_update]) + """
            WHERE pair = %s AND timestamp_utc = %s;
            """

            # Use batch update
            try:
                # Prepare parameter lists
                param_lists = []
                
                for _, row_data in df_to_update.iterrows():
                    # Add all feature columns
                    param_values = []
                    for col in columns_for_update:
                        raw_val = row_data.get(col, 0)
                        safe_val = cast_for_sqlalchemy(col, raw_val)
                        param_values.append(safe_val)
                        
                    # Add WHERE clause params
                    param_values.append(row_data['pair'])
                    param_values.append(row_data['timestamp_utc'])
                    
                    param_lists.append(tuple(param_values))
                
                # Execute batch update
                updated_rows = db_manager.execute_batch_update(
                    update_query, 
                    param_lists, 
                    batch_size=100
                )
                
                if debug_mode:
                    logging.debug(f"{pair}: Database update completed in {time.time() - start_time:.3f}s")
                
            except Exception as batch_error:
                logging.error(f"Batch update error for {pair}: {batch_error}", exc_info=True)
                raise

            if perf_monitor:
                perf_monitor.log_operation("database_update", time.time() - start_update)

    except Exception as e:
        logging.error(f"Error processing {pair}: {e}", exc_info=True)
        raise
    finally:
        total_time = time.time() - start_process
        logging.info(f"{pair}: Updated {updated_rows}/{len(df_to_update) if 'df_to_update' in locals() else 0} rows in {total_time:.2f}s")

# ---------------------------
# Main Function
# ---------------------------
def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute technical indicators for crypto data')
    parser.add_argument('--mode', choices=['rolling_update', 'full_backfill'], 
                       help='Processing mode (overrides config)')
    parser.add_argument('--config', type=str, help='Path to config.ini file')
    parser.add_argument('--credentials', type=str, help='Path to credentials.env file')
    parser.add_argument('--rolling-window', type=int,
                       help='Number of recent candles to process in rolling_update mode (overrides config)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for log files')
    parser.add_argument('--batch-size', type=int,
                       help='Number of parallel workers (overrides default)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed timing')
    parser.add_argument('--pairs', type=str, help='Comma-separated list of pairs to process (default: all)')
    parser.add_argument('--no-talib', action='store_true', help='Disable TA-Lib (use pandas_ta only)')
    parser.add_argument('--no-numba', action='store_true', help='Disable Numba optimizations')
    
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration (if available)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir, args.log_level)

    # Set up performance monitoring
    perf_monitor = PerformanceMonitor(args.log_dir)
    
    # Track runtime
    start_time_global = datetime.now()
    
    # Enable force exit for CTRL+C
    force_exit_on_ctrl_c()
    
    # Load configuration
    config_path = args.config if args.config else DEFAULT_CONFIG_PATH
    credentials_path = args.credentials if args.credentials else DEFAULT_CREDENTIALS_PATH
    
    config_manager = ConfigManager(
        config_path=config_path,
        credentials_path=credentials_path
    )
    
    # Override config with command line arguments
    if args.no_talib:
        config_manager.config['GENERAL']['USE_TALIB'] = 'False'
    if args.no_numba:
        config_manager.config['GENERAL']['USE_NUMBA'] = 'False'
    if args.use_gpu:
        config_manager.config['GENERAL']['USE_GPU'] = 'True'
    if args.no_gpu:
        config_manager.config['GENERAL']['USE_GPU'] = 'False'
    
    # Set processing mode
    mode = args.mode if args.mode else config_manager.get_compute_mode()
    
    # Set rolling window
    rolling_window = args.rolling_window if args.rolling_window else config_manager.get_rolling_window()
    
    # Set batch size
    batch_size = args.batch_size if args.batch_size else min(24, max(8, int(os.cpu_count() * 1.5)))
    
    # Set up database connection
    db_manager = DatabaseConnectionManager(config_manager)
    
    # Create runtime log path
    runtime_log_path = os.path.join(args.log_dir, "runtime_stats.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    
    logger.info(f"Starting compute_candles.py in {mode.upper()} mode with rolling_window={rolling_window}")
    logger.info(f"Using {batch_size} parallel workers")
    logger.info(f"TA-Lib: {'enabled' if config_manager.use_talib() else 'disabled'}, "
               f"Numba: {'enabled' if config_manager.use_numba() else 'disabled'}, "
               f"GPU: {'enabled' if config_manager.use_gpu() else 'disabled'}")
    
    if args.debug:
        logger.info("Debug mode enabled - detailed timing information will be logged")

    try:
        # Test connection
        with db_manager.get_engine().connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
    except Exception as e:
        logger.critical(f"Unable to start due to database connection issue: {e}")
        sys.exit(1)

    try:
        with db_manager.get_engine().connect() as conn:
            # Get pairs to process
            if args.pairs:
                all_pairs = [p.strip() for p in args.pairs.split(',')]
                logger.info(f"Processing {len(all_pairs)} specified pairs")
            else:
                all_pairs = pd.read_sql(text("SELECT DISTINCT pair FROM candles_1h"), conn)['pair'].tolist()
                logger.info(f"Found {len(all_pairs)} pairs")
    except Exception as e:
        logger.critical(f"Error loading pair list: {e}")
        sys.exit(1)

    if not all_pairs:
        logger.warning("No pairs found in candles_1h. Exiting early.")
        sys.exit()

    if mode == "rolling_update":
        logger.info(f"Rolling update mode: computing last {rolling_window} rows per pair")

        def process_pair_thread(pair):
            """Wrapper function for thread pool to handle a single pair"""
            try:
                # Start timing for this pair
                perf_monitor.start_pair(pair)
                start_time = time.time()
                
                # Process the pair
                process_pair(pair, db_manager, rolling_window, config_manager, args.debug, perf_monitor)
                
                # End timing for this pair
                total_time = time.time() - start_time
                perf_monitor.end_pair(total_time)
                
            except KeyboardInterrupt:
                # Handle keyboard interrupt
                logger.info(f"Processing of {pair} interrupted by user")
                return
            except Exception as e:
                logger.error(f"Thread error for pair {pair}: {e}", exc_info=True)

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(process_pair_thread, pair): pair for pair in all_pairs}
            completed = 0
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    future.result()
                    completed += 1
                    if completed % 25 == 0 or completed == len(all_pairs):
                        elapsed = (datetime.now() - start_time_global).total_seconds()
                        pairs_per_second = completed / elapsed if elapsed > 0 else 0
                        remaining = (len(all_pairs) - completed) / pairs_per_second if pairs_per_second > 0 else 0
                        logger.info(f"Progress: {completed}/{len(all_pairs)} pairs processed "
                                    f"({pairs_per_second:.2f} pairs/sec, ~{remaining:.0f}s remaining)")
                except Exception as e:
                    logger.error(f"Error processing pair {pair}: {e}")

            # Save performance summary at the end
            summary_file, report_file = perf_monitor.save_summary()
            logger.info(f"Performance summary saved to {summary_file}")
            logger.info(f"Performance report saved to {report_file}")

        # Compute cross-pair features after all individual pairs are done
        logger.info("Computing cross-pair features")
        try:
            with db_manager.get_engine().connect() as conn:
                # Get most recent candles
                latest_df = pd.read_sql(
                    text("""
                    SELECT * FROM candles_1h 
                    WHERE timestamp_utc >= (SELECT MAX(timestamp_utc) FROM candles_1h) - INTERVAL '24 hours'
                    """), 
                    conn
                )
                
                if not latest_df.empty:
                    # Compute cross-pair features
                    cross_computer = CrossPairFeatureComputer()
                    latest_df = cross_computer.compute_cross_pair_features(latest_df, args.debug)
                    
                    # Update database with cross-pair features
                    cross_update_query = """
                    UPDATE candles_1h
                    SET 
                        performance_rank_btc_1h = %s,
                        performance_rank_eth_1h = %s,
                        volume_rank_1h = %s,
                        volatility_rank_1h = %s,
                        btc_corr_24h = %s,
                        prev_volume_rank = %s
                    WHERE pair = %s AND timestamp_utc = %s;
                    """
                    
                    # Prepare parameters
                    cross_params = []
                    for _, row in latest_df.iterrows():
                        cross_params.append((
                            int(round(row['performance_rank_btc_1h'])),
                            int(round(row['performance_rank_eth_1h'])),
                            int(round(row['volume_rank_1h'])),
                            int(round(row['volatility_rank_1h'])),
                            float(row['btc_corr_24h']),
                            float(row['prev_volume_rank']),
                            row['pair'],
                            row['timestamp_utc']
                        ))
                    
                    # Execute batch update
                    if cross_params:
                        updated = db_manager.execute_batch_update(cross_update_query, cross_params)
                        logger.info(f"Updated {updated} rows with cross-pair features")
        except Exception as e:
            logger.error(f"Error computing cross-pair features: {e}", exc_info=True)

        logger.info("Rolling update mode completed.")

    # Runtime logging
    end_time = datetime.now()
    duration = (end_time - start_time_global).total_seconds()
    with open(runtime_log_path, "a") as f:
        f.write(f"[{end_time}] compute_candles.py ({mode}) completed in {duration:.2f} seconds\n")
    
    logger.info(f"Total runtime: {duration:.2f} seconds")

    # Runtime logging
    end_time = datetime.now()
    duration = (end_time - start_time_global).total_seconds()
    with open(runtime_log_path, "a") as f:
        f.write(f"[{end_time}] compute_candles.py ({mode}) completed in {duration:.2f} seconds\n")
    
    logger.info(f"Total runtime: {duration:.2f} seconds")

if __name__ == '__main__':
    main()