import os
import configparser
import sys
import math
import signal
import argparse
import logging
import time
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from ta.momentum import RSIIndicator
from ta.trend import MACD, PSARIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator

# ---------------------------
# Load Configuration & Constants
# ---------------------------
BATCH_SIZE = os.cpu_count() or 8  # parallel processes
MIN_CANDLES_REQUIRED = 388
ROLLING_WINDOW = 128  # Default from config

# Default paths for configuration files
DEFAULT_CONFIG_PATH = "P:/OKXsignal/config/config.ini"
DEFAULT_CREDENTIALS_PATH = "P:/OKXsignal/config/credentials.env"

# Smallint columns in candles_1h that must be cast to int
SMALLINT_COLUMNS = {
    'supertrend_direction_1h',
    'supertrend_direction_4h',
    'supertrend_direction_1d',
    'performance_rank_btc_1h',
    'performance_rank_eth_1h',
    'volatility_rank_1h',
    'volume_rank_1h',
    'hour_of_day',
    'day_of_week',
    'was_profitable_12h'
}

# DataFrame column types for proper typing
COLUMN_TYPES = {
    'rsi_1h': float, 'rsi_slope_1h': float, 'macd_slope_1h': float, 'macd_hist_slope_1h': float,
    'atr_1h': float, 'bollinger_width_1h': float, 'donchian_channel_width_1h': float,
    'supertrend_direction_1h': int, 'parabolic_sar_1h': float, 'money_flow_index_1h': float,
    'obv_slope_1h': float, 'volume_change_pct_1h': float, 'estimated_slippage_1h': float,
    'bid_ask_spread_1h': float, 'hour_of_day': int, 'day_of_week': int,
    'rsi_4h': float, 'rsi_slope_4h': float, 'macd_slope_4h': float, 'macd_hist_slope_4h': float,
    'atr_4h': float, 'bollinger_width_4h': float, 'donchian_channel_width_4h': float,
    'supertrend_direction_4h': int, 'money_flow_index_4h': float, 'obv_slope_4h': float,
    'volume_change_pct_4h': float, 'rsi_1d': float, 'rsi_slope_1d': float, 'macd_slope_1d': float,
    'macd_hist_slope_1d': float, 'atr_1d': float, 'bollinger_width_1d': float,
    'donchian_channel_width_1d': float, 'supertrend_direction_1d': int, 'money_flow_index_1d': float,
    'obv_slope_1d': float, 'volume_change_pct_1d': float, 'performance_rank_btc_1h': float,
    'performance_rank_eth_1h': float, 'volatility_rank_1h': float, 'volume_rank_1h': float,
    'was_profitable_12h': int, 'prev_close_change_pct': float, 'prev_volume_rank': float,
    'future_max_return_24h_pct': float, 'future_max_drawdown_12h_pct': float
}

# ---------------------------
# Enhanced Configuration Management
# ---------------------------
class ConfigManager:
    def __init__(self, config_path: str = None, credentials_path: str = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        
        # Load env variables
        if os.path.exists(self.credentials_path):
            load_dotenv(dotenv_path=self.credentials_path)
            logging.info(f"Loaded credentials from {self.credentials_path}")
        else:
            logging.warning(f"Credentials file not found: {self.credentials_path}")
            
        # Load config
        self.config = self._load_config()
    
    def _load_config(self):
        config = configparser.ConfigParser()
        if os.path.exists(self.config_path):
            config.read(self.config_path)
            logging.info(f"Loaded config from {self.config_path}")
        else:
            logging.warning(f"Config file not found: {self.config_path}")
            # Create a minimal default config to prevent errors
            config['DATABASE'] = {
                'DB_HOST': 'localhost',
                'DB_PORT': '5432',
                'DB_NAME': 'okxsignal'
            }
            config['GENERAL'] = {
                'COMPUTE_MODE': 'rolling_update',
                'ROLLING_WINDOW': '128'
            }
        return config
    
    def get_db_params(self):
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
        
    def get_connection_string(self):
        """Create a SQLAlchemy connection string from config"""
        db_params = self.get_db_params()
        return (
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@"
            f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
    def get_rolling_window(self):
        """Get rolling window size from config"""
        if 'GENERAL' in self.config and 'ROLLING_WINDOW' in self.config['GENERAL']:
            return int(self.config['GENERAL']['ROLLING_WINDOW'])
        return ROLLING_WINDOW  # Default fallback
        
    def get_compute_mode(self):
        """Get compute mode from config"""
        if 'GENERAL' in self.config and 'COMPUTE_MODE' in self.config['GENERAL']:
            return self.config['GENERAL']['COMPUTE_MODE']
        return 'rolling_update'  # Default fallback

# ---------------------------
# Enhanced Logging
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
    return logging.getLogger("compute_candles")

def force_exit_on_ctrl_c():
    """
    Allows Ctrl-C to forcibly exit all threads.
    """
    import threading
    import ctypes
    
    def handler(signum, frame):
        print("\nInterrupted. Forcing thread exit...")
        for t in threading.enumerate():
            if t is not threading.main_thread():
                try:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(t.ident), ctypes.py_object(SystemExit))
                except Exception:
                    pass
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

# ---------------------------
# Database Connection
# ---------------------------
class DatabaseConnectionManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.engine = self._create_connection_pool()
    
    def _create_connection_pool(self):
        connection_string = self.config_manager.get_connection_string()
        logging.info(f"Creating database connection pool (hiding credentials)")
        return create_engine(
            connection_string, 
            poolclass=QueuePool,
            pool_size=max(10, BATCH_SIZE),  # Ensure enough connections for all threads
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,  # Test connection before using
            isolation_level="READ COMMITTED"  # Good balance for read performance
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
    
    def execute_batch_update(self, query_text, param_lists, batch_size=1000):
        """
        Execute updates in efficient batches
        """
        conn = None
        updated_rows = 0
        
        try:
            conn = self.engine.raw_connection()
            cursor = conn.cursor()
            
            # Process in batches
            for i in range(0, len(param_lists), batch_size):
                batch = param_lists[i:i+batch_size]
                cursor.executemany(query_text, batch)
                updated_rows += cursor.rowcount
                
            conn.commit()
            return updated_rows
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

# ---------------------------
# Casting Helper
# ---------------------------
def cast_for_sqlalchemy(col_name: str, val) -> any:
    """
    Convert `val` into a Python scalar suitable for SQLAlchemy param binding.
    - Convert numpy types to standard Python (float, int).
    - Convert datetime64 or pd.Timestamp to Python datetime.
    - Convert NaN/NaT to None.
    - Force columns in SMALLINT_COLUMNS to int if not None.
    """
    # Handle null values first
    if val is None or pd.isna(val):
        return None

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

# ---------------------------
# Feature Computation Logic
# ---------------------------
def compute_1h_features(df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """
    Compute 1h-based indicators on the DataFrame using optimized methods.
    """
    try:
        if debug_mode:
            start_time = time.time()
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Convert timestamps and sort
        result_df['timestamp_utc'] = pd.to_datetime(result_df['timestamp_utc'], utc=True)
        result_df = result_df.sort_values('timestamp_utc')

        # Extract price and volume columns into numpy arrays for faster calculations
        close = result_df['close_1h'].values
        high = result_df['high_1h'].values
        low = result_df['low_1h'].values
        open_prices = result_df['open_1h'].values
        volume = result_df['volume_1h'].values
        
        # RSI using the TA library
        result_df['rsi_1h'] = RSIIndicator(pd.Series(close), window=14, fillna=True).rsi().values
        
        # Fast slope calculation using numpy
        rsi_values = result_df['rsi_1h'].values
        rsi_slope = np.zeros(len(result_df))
        if len(result_df) >= 3:
            rsi_slope[3:] = (rsi_values[3:] - rsi_values[:-3]) / 3
        result_df.loc[:, 'rsi_slope_1h'] = rsi_slope
        
        if debug_mode:
            logging.debug(f"RSI calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # MACD - optimized
        macd_obj = MACD(pd.Series(close), window_slow=26, window_fast=12, window_sign=9, fillna=True)
        macd_line = macd_obj.macd().values
        macd_signal = macd_obj.macd_signal().values
        macd_hist = macd_obj.macd_diff().values
        
        # Store MACD values
        result_df['macd_1h'] = macd_line
        result_df['macd_signal_1h'] = macd_signal
        
        # Calculate slopes using numpy diff
        macd_slope = np.zeros(len(result_df))
        if len(macd_line) > 1:
            macd_slope[1:] = np.diff(macd_line)
        result_df.loc[:, 'macd_slope_1h'] = macd_slope
        
        result_df['macd_hist_1h'] = macd_hist
        macd_hist_slope = np.zeros(len(result_df))
        if len(macd_hist) > 1:
            macd_hist_slope[1:] = np.diff(macd_hist)
        result_df.loc[:, 'macd_hist_slope_1h'] = macd_hist_slope
        
        if debug_mode:
            logging.debug(f"MACD calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # ATR using vectorized operations
        result_df['atr_1h'] = AverageTrueRange(
            pd.Series(high), pd.Series(low), pd.Series(close), window=14, fillna=True
        ).average_true_range().values
        
        if debug_mode:
            logging.debug(f"ATR calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # Bollinger Bands
        bb = BollingerBands(pd.Series(close), window=20, window_dev=2, fillna=True)
        upper = bb.bollinger_hband().values
        lower = bb.bollinger_lband().values
        result_df['bollinger_width_1h'] = upper - lower
        
        if debug_mode:
            logging.debug(f"Bollinger calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # Donchian Channel - vectorized
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(20).max().values
        donchian_low = low_series.rolling(20).min().values
        
        # Fill NaN values with appropriate defaults
        for i in range(len(donchian_high)):
            if pd.isna(donchian_high[i]) and i > 0:
                donchian_high[i] = donchian_high[i-1] if not pd.isna(donchian_high[i-1]) else high[i]
            elif pd.isna(donchian_high[i]):
                donchian_high[i] = high[i]
                
            if pd.isna(donchian_low[i]) and i > 0:
                donchian_low[i] = donchian_low[i-1] if not pd.isna(donchian_low[i-1]) else low[i]
            elif pd.isna(donchian_low[i]):
                donchian_low[i] = low[i]
                
        result_df['donchian_high_1h'] = donchian_high
        result_df['donchian_low_1h'] = donchian_low
        result_df['donchian_channel_width_1h'] = donchian_high - donchian_low
        
        if debug_mode:
            logging.debug(f"Donchian calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
        
        # Supertrend placeholder - use 0 instead of NaN
        result_df['supertrend_direction_1h'] = 0
        
        # Parabolic SAR
        result_df['parabolic_sar_1h'] = PSARIndicator(
            pd.Series(high), pd.Series(low), pd.Series(close), step=0.02, max_step=0.2, fillna=True
        ).psar().values
        
        if debug_mode:
            logging.debug(f"PSAR calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # MFI
        result_df['money_flow_index_1h'] = MFIIndicator(
            pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume), 
            window=14, fillna=True
        ).money_flow_index().values
        
        if debug_mode:
            logging.debug(f"MFI calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # OBV - efficient implementation
        obv = np.zeros(len(close))
        
        # First value
        obv[0] = volume[0]
        
        # Loop through with minimal operations
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        result_df['obv_1h'] = obv
        
        # Calculate OBV slope efficiently
        obv_slope = np.zeros(len(result_df))
        if len(obv) >= 3:
            obv_slope[3:] = (obv[3:] - obv[:-3]) / 3
        result_df.loc[:, 'obv_slope_1h'] = obv_slope
        
        if debug_mode:
            logging.debug(f"OBV calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # Other metrics using efficient calculation
        volume_change_pct = np.zeros(len(volume))
        if len(volume) > 1:
            nonzero_indices = np.where(volume[:-1] > 0)[0] + 1
            if len(nonzero_indices) > 0:
                volume_change_pct[nonzero_indices] = (
                    (volume[nonzero_indices] - volume[nonzero_indices-1]) / 
                    volume[nonzero_indices-1]
                )
        result_df.loc[:, 'volume_change_pct_1h'] = volume_change_pct
        
        # Simple calculations
        result_df['estimated_slippage_1h'] = high - low
        result_df['bid_ask_spread_1h'] = close - open_prices
        
        # Time-based features
        result_df['hour_of_day'] = result_df['timestamp_utc'].dt.hour
        result_df['day_of_week'] = result_df['timestamp_utc'].dt.weekday
        
        # For prediction targets - was profitable 12h later
        was_profitable = np.zeros(len(close), dtype=int)
        shift = min(12, len(close)-1)
        if shift > 0:
            was_profitable[:-shift] = (close[shift:] > close[:-shift]).astype(int)
        result_df.loc[:, 'was_profitable_12h'] = was_profitable
        
        # Previous close change percent
        prev_close_change_pct = np.zeros(len(close))
        if len(close) > 1:
            nonzero_indices = np.where(close[:-1] > 0)[0] + 1
            if len(nonzero_indices) > 0:
                prev_close_change_pct[nonzero_indices] = (
                    (close[nonzero_indices] - close[nonzero_indices-1]) / 
                    close[nonzero_indices-1]
                )
        result_df.loc[:, 'prev_close_change_pct'] = prev_close_change_pct
        
        # Previous volume rank
        result_df['prev_volume_rank'] = pd.Series(volume).rank(pct=True).shift(1).fillna(0).values * 100
        
        if debug_mode:
            logging.debug(f"Additional calculations completed in {time.time() - start_time:.3f}s")

        # Clean up NaN/Inf values
        for col in result_df.select_dtypes(include=['float64', 'float32']).columns:
            if col not in ['timestamp_utc', 'pair']:
                result_df[col] = result_df[col].replace([np.inf, -np.inf], 0).fillna(0)

        return result_df
    except Exception as e:
        logging.error(f"Error in compute_1h_features: {e}", exc_info=True)
        raise

def compute_multi_tf_features(df: pd.DataFrame, tf_label: str, rule: str, debug_mode: bool = False) -> pd.DataFrame:
    """
    Resample from 1h to 4h or 1d, then compute relevant multi-timeframe indicators.
    Fixed version that avoids NoneType errors and improves performance.
    """
    try:
        if debug_mode:
            start_time = time.time()
            
        required_points = {
            '4h': 24,   # 4h window = 1 day
            '1d': 30    # 1d window = ~1 month
        }
        min_points = required_points.get(rule, 20)
        
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Define expected column names for this timeframe
        expected_columns = [
            f'rsi_{tf_label}', f'rsi_slope_{tf_label}', f'macd_slope_{tf_label}',
            f'macd_hist_slope_{tf_label}', f'atr_{tf_label}', f'bollinger_width_{tf_label}',
            f'donchian_channel_width_{tf_label}', f'supertrend_direction_{tf_label}',
            f'money_flow_index_{tf_label}', f'obv_slope_{tf_label}', f'volume_change_pct_{tf_label}'
        ]
        
        # Initialize all expected columns with default value 0
        for col_name in expected_columns:
            col_type = float if col_name != f'supertrend_direction_{tf_label}' else int
            result_df[col_name] = np.zeros(len(result_df), dtype=col_type)

        if len(df) < min_points:
            if 'pair' in df.columns and not df.empty:
                logging.warning(
                    f"Skipping {df['pair'].iloc[0]} {tf_label} features: only {len(df)} rows (need >= {min_points})"
                )
            # Return the dataframe with initialized columns
            return result_df

        # Use timestamps for resampling
        df_with_ts = result_df.copy()
        df_with_ts['timestamp_utc'] = pd.to_datetime(df_with_ts['timestamp_utc'], utc=True)
        df_with_ts.set_index('timestamp_utc', inplace=True)
        
        # Get OHLCV columns
        high = df_with_ts['high_1h']
        low = df_with_ts['low_1h']
        close = df_with_ts['close_1h']
        open_prices = df_with_ts['open_1h']
        volume = df_with_ts['volume_1h']
        
        # Perform resampling
        resampled = pd.DataFrame()
        resampled[f'open_{tf_label}'] = open_prices.resample(rule).first()
        resampled[f'high_{tf_label}'] = high.resample(rule).max()
        resampled[f'low_{tf_label}'] = low.resample(rule).min()
        resampled[f'close_{tf_label}'] = close.resample(rule).last()
        resampled[f'volume_{tf_label}'] = volume.resample(rule).sum()
        
        # Drop rows with missing values
        resampled = resampled.dropna()
        
        if len(resampled) < 5:  # Need minimum data for indicators
            logging.warning(f"Not enough resampled data for {tf_label}, using default values")
            return result_df
        
        if debug_mode:
            logging.debug(f"{tf_label} resampling completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # Compute technical indicators on resampled data
        # RSI
        resampled[f'rsi_{tf_label}'] = RSIIndicator(
            resampled[f'close_{tf_label}'], window=14, fillna=True
        ).rsi()
        
        # RSI slope - use diff instead of polyfit for speed
        resampled[f'rsi_slope_{tf_label}'] = resampled[f'rsi_{tf_label}'].diff(2) / 2
        
        if debug_mode:
            logging.debug(f"{tf_label} RSI calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # MACD - simplified calculation
        macd = MACD(
            resampled[f'close_{tf_label}'], 
            window_slow=26, window_fast=12, window_sign=9, 
            fillna=True
        )
        resampled[f'macd_{tf_label}'] = macd.macd()
        resampled[f'macd_signal_{tf_label}'] = macd.macd_signal()
        resampled[f'macd_slope_{tf_label}'] = resampled[f'macd_{tf_label}'].diff()
        resampled[f'macd_hist_{tf_label}'] = macd.macd_diff()
        resampled[f'macd_hist_slope_{tf_label}'] = resampled[f'macd_hist_{tf_label}'].diff()
        
        if debug_mode:
            logging.debug(f"{tf_label} MACD calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # ATR
        resampled[f'atr_{tf_label}'] = AverageTrueRange(
            resampled[f'high_{tf_label}'],
            resampled[f'low_{tf_label}'],
            resampled[f'close_{tf_label}'],
            window=14, fillna=True
        ).average_true_range()
        
        if debug_mode:
            logging.debug(f"{tf_label} ATR calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # Bollinger Bands
        bb = BollingerBands(
            resampled[f'close_{tf_label}'], 
            window=20, window_dev=2, fillna=True
        )
        resampled[f'bollinger_width_{tf_label}'] = bb.bollinger_hband() - bb.bollinger_lband()
        
        if debug_mode:
            logging.debug(f"{tf_label} Bollinger calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # Donchian Channels - use ffill instead of deprecated fillna with method
        donchian_high = resampled[f'high_{tf_label}'].rolling(20).max()
        donchian_high = donchian_high.ffill().fillna(resampled[f'high_{tf_label}'].iloc[0] if len(resampled) > 0 else 0)
        
        donchian_low = resampled[f'low_{tf_label}'].rolling(20).min()
        donchian_low = donchian_low.ffill().fillna(resampled[f'low_{tf_label}'].iloc[0] if len(resampled) > 0 else 0)
        
        resampled[f'donchian_high_{tf_label}'] = donchian_high
        resampled[f'donchian_low_{tf_label}'] = donchian_low
        resampled[f'donchian_channel_width_{tf_label}'] = donchian_high - donchian_low
        
        if debug_mode:
            logging.debug(f"{tf_label} Donchian calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # Supertrend (placeholder) - explicitly cast to int 
        resampled[f'supertrend_direction_{tf_label}'] = 0
        
        # MFI
        resampled[f'money_flow_index_{tf_label}'] = MFIIndicator(
            resampled[f'high_{tf_label}'],
            resampled[f'low_{tf_label}'],
            resampled[f'close_{tf_label}'],
            resampled[f'volume_{tf_label}'],
            window=14, fillna=True
        ).money_flow_index()
        
        if debug_mode:
            logging.debug(f"{tf_label} MFI calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
            
        # OBV - vectorized implementation
        obv_values = np.zeros(len(resampled))
        close_arr = resampled[f'close_{tf_label}'].values
        volume_arr = resampled[f'volume_{tf_label}'].values
        
        # First value
        obv_values[0] = volume_arr[0]
        
        # Compute OBV values efficiently
        for i in range(1, len(resampled)):
            if close_arr[i] > close_arr[i-1]:
                obv_values[i] = obv_values[i-1] + volume_arr[i]
            elif close_arr[i] < close_arr[i-1]:
                obv_values[i] = obv_values[i-1] - volume_arr[i]
            else:
                obv_values[i] = obv_values[i-1]
                
        resampled[f'obv_{tf_label}'] = obv_values
        resampled[f'obv_slope_{tf_label}'] = resampled[f'obv_{tf_label}'].diff(2) / 2
        
        # Volume change
        resampled[f'volume_change_pct_{tf_label}'] = resampled[f'volume_{tf_label}'].pct_change().fillna(0)
        
        if debug_mode:
            logging.debug(f"{tf_label} OBV calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()

        # Clean up any NaN/Inf values
        for col in resampled.columns:
            resampled[col] = resampled[col].replace([np.inf, -np.inf], 0).fillna(0)

        # Create time-based mapping for merging values back
        # Convert to numpy arrays for faster processing
        result_timestamps = pd.DatetimeIndex(result_df['timestamp_utc']).to_numpy()
        
        # For each column of interest, create a mapping array
        for col in expected_columns:
            if col in resampled.columns:
                col_type = float if col != f'supertrend_direction_{tf_label}' else int
                values_array = np.zeros(len(result_df), dtype=col_type)
                
                # Get resampled timestamps as numpy
                resampled_ts = np.array(resampled.index)
                
                # For each resampled period, find all rows in the original df that fall into that period
                for i, ts in enumerate(resampled_ts):
                    # Define period start/end
                    period_start = ts
                    period_end = ts + pd.Timedelta(rule)
                    
                    # Find indices in original data that fall into this period
                    mask = (result_timestamps >= period_start) & (result_timestamps < period_end)
                    
                    # Set all matching rows to the resampled value
                    if np.any(mask):
                        val = resampled[col].iloc[i]
                        
                        # Convert to proper type for smallint columns
                        if col == f'supertrend_direction_{tf_label}':
                            val = int(val)
                        else:
                            val = float(val)
                            
                        values_array[mask] = val
                
                # Assign the entire array at once to the result dataframe
                result_df[col] = values_array
        
        if debug_mode:
            logging.debug(f"{tf_label} data mapping completed in {time.time() - start_time:.3f}s")
            
        return result_df
        
    except Exception as e:
        logging.error(f"Error in compute_multi_tf_features for {tf_label}: {e}", exc_info=True)
        # Return df with placeholder columns to avoid errors
        for col_name in expected_columns:
            df[col_name] = 0
        return df

# ---------------------------
# Label Computation Logic
# ---------------------------
def compute_labels(df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """
    Compute forward returns and related label columns using optimized methods.
    """
    try:
        if debug_mode:
            start_time = time.time()
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        result_df = result_df.sort_values('timestamp_utc')
        
        # Get price arrays for calculation
        close = result_df['close_1h'].values
        high = result_df['high_1h'].values
        low = result_df['low_1h'].values
        
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
        
        for horizon, shift in horizons.items():
            col_name = f'future_return_{horizon}_pct'
            
            # Use vectorized NumPy operations for speed
            future_return = np.zeros(len(close))
            
            # Only calculate for rows where we have future data
            if len(close) > shift:
                divider = np.maximum(close[:-shift], np.ones_like(close[:-shift]) * 1e-8)
                future_return[:-shift] = (close[shift:] - close[:-shift]) / divider
            
            result_df[col_name] = future_return
        
        if debug_mode:
            logging.debug(f"Future returns calculation completed in {time.time() - start_time:.3f}s")
            start_time = time.time()
        
        # Max future return calculation using a more accurate method
        future_max_return = np.zeros(len(close))
        
        for i in range(len(close) - 1):
            end_idx = min(i + 25, len(high))
            if i + 1 < end_idx:
                max_high = np.max(high[i+1:end_idx])
                future_max_return[i] = (max_high - close[i]) / max(close[i], 1e-8)
        
        result_df['future_max_return_24h_pct'] = future_max_return
        
        # Max future drawdown calculation with proper handling
        future_max_drawdown = np.zeros(len(close))
        
        for i in range(len(close) - 1):
            end_idx = min(i + 13, len(low))
            if i + 1 < end_idx:
                min_low = np.min(low[i+1:end_idx])
                future_max_drawdown[i] = (min_low - close[i]) / max(close[i], 1e-8)
        
        result_df['future_max_drawdown_12h_pct'] = future_max_drawdown
        
        if debug_mode:
            logging.debug(f"Max returns/drawdowns calculation completed in {time.time() - start_time:.3f}s")
        
        # Make sure we don't have NaN or Inf values
        for col in result_df.columns:
            if col.startswith('future_') and col.endswith('_pct'):
                result_df[col] = result_df[col].replace([np.inf, -np.inf], 0).fillna(0)
                
        result_df['targets_computed'] = True
        
        return result_df
    except Exception as e:
        logging.error(f"Error in compute_labels: {e}", exc_info=True)
        raise

# ---------------------------
# Cross-Pair Intelligence
# ---------------------------
def compute_cross_pair_features(latest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-pair metrics relative to BTC-USDT and ETH-USDT.
    """
    try:
        if len(latest_df) == 0:
            logging.warning("Empty DataFrame passed to compute_cross_pair_features")
            return latest_df
            
        # Initialize columns with 0 to guarantee they exist
        latest_df['volume_rank_1h'] = 0
        latest_df['volatility_rank_1h'] = 0
        latest_df['performance_rank_btc_1h'] = 0
        latest_df['performance_rank_eth_1h'] = 0
        
        # Fill values where possible
        if 'volume_1h' in latest_df.columns:
            latest_df['volume_rank_1h'] = latest_df['volume_1h'].rank(pct=True) * 100
            
        if 'atr_1h' in latest_df.columns:
            latest_df['volatility_rank_1h'] = latest_df['atr_1h'].rank(pct=True) * 100

        # Check for BTC data
        btc_row = latest_df[latest_df['pair'] == 'BTC-USDT']
        if not btc_row.empty and 'future_return_1h_pct' in latest_df.columns:
            btc_return = btc_row['future_return_1h_pct'].values[0]
            if not pd.isna(btc_return) and abs(btc_return) > 1e-9:
                latest_df['performance_rank_btc_1h'] = (
                    (latest_df['future_return_1h_pct'] - btc_return) / abs(btc_return)
                ).rank(pct=True) * 100

        # Check for ETH data
        eth_row = latest_df[latest_df['pair'] == 'ETH-USDT']
        if not eth_row.empty and 'future_return_1h_pct' in latest_df.columns:
            eth_return = eth_row['future_return_1h_pct'].values[0]
            if not pd.isna(eth_return) and abs(eth_return) > 1e-9:
                latest_df['performance_rank_eth_1h'] = (
                    (latest_df['future_return_1h_pct'] - eth_return) / abs(eth_return)
                ).rank(pct=True) * 100

        # Fill NaN values with 0
        for col in ['volume_rank_1h', 'volatility_rank_1h', 'performance_rank_btc_1h', 'performance_rank_eth_1h']:
            latest_df[col] = latest_df[col].fillna(0)

        return latest_df
    except Exception as e:
        logging.error(f"Error in compute_cross_pair_features: {e}", exc_info=True)
        raise

# ---------------------------
# Single-Pair Processing
# ---------------------------
def fetch_data(pair: str, db_conn, rolling_window: int = None) -> pd.DataFrame:
    """
    Fetch candles_1h for the given pair, limited to rolling_window if specified.
    """
    if rolling_window:
        # Fetch only the rolling window + additional data needed for lookback calculations
        # We need extra history for proper indicator calculations
        lookback_padding = 300  # Extra candles for indicators that need history
        query = """
            SELECT *
            FROM candles_1h
            WHERE pair = :pair
            ORDER BY timestamp_utc DESC
            LIMIT :limit
        """
        df = pd.read_sql(text(query), db_conn, params={"pair": pair, "limit": rolling_window + lookback_padding})
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

def process_pair(pair: str, db_manager, rolling_window: int, debug_mode: bool = False) -> None:
    """
    For the given pair, compute 1h, multi-timeframe features, labels, etc.
    Then update the database efficiently with only the rolling window rows.
    """
    start_process = time.time()
    logging.info(f"Computing features for {pair}")
    updated_rows = 0

    try:
        # Create a fresh connection for each pair
        with db_manager.get_engine().connect() as db_conn:
            # Fetch data with rolling window + padding for calculations
            df = fetch_data(pair, db_conn, rolling_window + 300)
            
            if debug_mode:
                logging.debug(f"{pair}: Fetched data in {time.time() - start_process:.3f}s, {len(df)} rows")
                start_time = time.time()
                
            row_count = len(df)

            if df.empty or row_count < MIN_CANDLES_REQUIRED:
                logging.warning(
                    f"Skipping {pair}: only {row_count} candles, need >= {MIN_CANDLES_REQUIRED}"
                )
                return
                
            # Enable garbage collection to reduce memory usage
            gc.collect()

            # Compute features with optimized code
            df = compute_1h_features(df, debug_mode)
            
            if debug_mode:
                logging.debug(f"{pair}: Computed 1h features in {time.time() - start_time:.3f}s")
                start_time = time.time()
                
            df = compute_labels(df, debug_mode)
            
            if debug_mode:
                logging.debug(f"{pair}: Computed labels in {time.time() - start_time:.3f}s")
                start_time = time.time()
                
            df = compute_multi_tf_features(df, '4h', '4h', debug_mode)
            
            if debug_mode:
                logging.debug(f"{pair}: Computed 4h features in {time.time() - start_time:.3f}s")
                start_time = time.time()
                
            df = compute_multi_tf_features(df, '1d', '1d', debug_mode)
            
            if debug_mode:
                logging.debug(f"{pair}: Computed 1d features in {time.time() - start_time:.3f}s")
                start_time = time.time()
                
            # Compute cross-pair features if we have the relevant pairs
            if pair in ['BTC-USDT', 'ETH-USDT'] or df['pair'].iloc[0] in ['BTC-USDT', 'ETH-USDT']:
                df = compute_cross_pair_features(df)
                
                if debug_mode:
                    logging.debug(f"{pair}: Computed cross-pair features in {time.time() - start_time:.3f}s")
                    start_time = time.time()
                
            df['features_computed'] = True
            df['targets_computed'] = True

            # Take only the newest 'rolling_window' rows for updating
            df_to_update = df.iloc[-rolling_window:].copy() if len(df) > rolling_window else df
            
            if debug_mode:
                logging.debug(f"{pair}: Selected {len(df_to_update)}/{len(df)} rows for update")
                start_time = time.time()
                
            # The columns we want to update
            columns_for_update = [
                'rsi_1h', 'rsi_slope_1h', 'macd_slope_1h', 'macd_hist_slope_1h', 'atr_1h',
                'bollinger_width_1h', 'donchian_channel_width_1h', 'supertrend_direction_1h',
                'parabolic_sar_1h', 'money_flow_index_1h', 'obv_slope_1h', 'volume_change_pct_1h',
                'estimated_slippage_1h', 'bid_ask_spread_1h', 'hour_of_day', 'day_of_week',
                'rsi_4h', 'rsi_slope_4h', 'macd_slope_4h', 'macd_hist_slope_4h', 'atr_4h',
                'bollinger_width_4h', 'donchian_channel_width_4h', 'supertrend_direction_4h',
                'money_flow_index_4h', 'obv_slope_4h', 'volume_change_pct_4h',
                'rsi_1d', 'rsi_slope_1d', 'macd_slope_1d', 'macd_hist_slope_1d', 'atr_1d',
                'bollinger_width_1d', 'donchian_channel_width_1d', 'supertrend_direction_1d',
                'money_flow_index_1d', 'obv_slope_1d', 'volume_change_pct_1d',
                'performance_rank_btc_1h', 'performance_rank_eth_1h', 'volume_rank_1h',
                'volatility_rank_1h', 'was_profitable_12h', 'prev_close_change_pct', 'prev_volume_rank',
                'future_max_return_24h_pct', 'future_max_drawdown_12h_pct'
            ]

            # Verify that all columns exist and have correct types
            for col in columns_for_update:
                if col not in df_to_update.columns:
                    # Add missing column with appropriate type
                    col_type = int if col in SMALLINT_COLUMNS else float
                    df_to_update[col] = np.zeros(len(df_to_update), dtype=col_type)
                else:
                    # Handle NaN values before type conversion
                    df_to_update[col] = df_to_update[col].replace([np.inf, -np.inf], 0).fillna(0)
                    
                    # Ensure correct type for each column
                    col_type = int if col in SMALLINT_COLUMNS else float
                    try:
                        df_to_update[col] = df_to_update[col].astype(col_type)
                    except Exception as e:
                        logging.warning(f"Error converting {col} to {col_type}: {e}")
                        # If conversion failed, set to default values
                        df_to_update[col] = 0 if col_type == int else 0.0

            # Clean up NaN/Inf values before update
            for col in df_to_update.columns:
                if col not in ['timestamp_utc', 'pair', 'id']:
                    df_to_update[col] = df_to_update[col].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Prepare update query
            update_query = """
            UPDATE candles_1h
            SET
                rsi_1h = %s,
                rsi_slope_1h = %s,
                macd_slope_1h = %s,
                macd_hist_slope_1h = %s,
                atr_1h = %s,
                bollinger_width_1h = %s,
                donchian_channel_width_1h = %s,
                supertrend_direction_1h = %s,
                parabolic_sar_1h = %s,
                money_flow_index_1h = %s,
                obv_slope_1h = %s,
                volume_change_pct_1h = %s,
                estimated_slippage_1h = %s,
                bid_ask_spread_1h = %s,
                hour_of_day = %s,
                day_of_week = %s,
                rsi_4h = %s,
                rsi_slope_4h = %s,
                macd_slope_4h = %s,
                macd_hist_slope_4h = %s,
                atr_4h = %s,
                bollinger_width_4h = %s,
                donchian_channel_width_4h = %s,
                supertrend_direction_4h = %s,
                money_flow_index_4h = %s,
                obv_slope_4h = %s,
                volume_change_pct_4h = %s,
                rsi_1d = %s,
                rsi_slope_1d = %s,
                macd_slope_1d = %s,
                macd_hist_slope_1d = %s,
                atr_1d = %s,
                bollinger_width_1d = %s,
                donchian_channel_width_1d = %s,
                supertrend_direction_1d = %s,
                money_flow_index_1d = %s,
                obv_slope_1d = %s,
                volume_change_pct_1d = %s,
                performance_rank_btc_1h = %s,
                performance_rank_eth_1h = %s,
                volume_rank_1h = %s,
                volatility_rank_1h = %s,
                was_profitable_12h = %s,
                prev_close_change_pct = %s,
                prev_volume_rank = %s,
                future_max_return_24h_pct = %s,
                future_max_drawdown_12h_pct = %s,
                features_computed = TRUE,
                targets_computed = TRUE
            WHERE pair = %s AND timestamp_utc = %s;
            """

            # Use optimized batch update
            try:
                # Prepare parameter lists for batch update
                param_lists = []
                
                for _, row_data in df_to_update.iterrows():
                    # Build param list
                    param_values = []
                    
                    # Add all feature columns
                    for col in columns_for_update:
                        raw_val = row_data.get(col, 0)  # Default to 0 if column missing
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

    except Exception as e:
        logging.error(f"Error processing {pair}: {e}", exc_info=True)
        raise
    finally:
        total_time = time.time() - start_process
        logging.info(f"{pair}: Updated {updated_rows}/{len(df_to_update) if 'df_to_update' in locals() else 0} rows in {total_time:.2f}s")

# ---------------------------
# Main
# ---------------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute technical indicators for OKX data')
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
    
    args = parser.parse_args()
    
    # Set up logging early
    logger = setup_logging(args.log_dir, args.log_level)
    
    # Setup runtime tracking
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
    
    # Set processing mode from config or command line
    mode = args.mode if args.mode else config_manager.get_compute_mode()
    
    # Get rolling window from config or command line
    rolling_window = args.rolling_window if args.rolling_window else config_manager.get_rolling_window()
    
    # Set batch size from command line or use default (cpu count + 50% for optimal resource use)
    batch_size = args.batch_size if args.batch_size else min(24, max(8, int(os.cpu_count() * 1.5)))
    
    # Set up database connection
    db_manager = DatabaseConnectionManager(config_manager)
    
    # Create runtime log path
    runtime_log_path = os.path.join(args.log_dir, "runtime_stats.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    
    logger.info(f"Starting compute_candles.py in {mode.upper()} mode with rolling_window={rolling_window}")
    logger.info(f"Using {batch_size} parallel workers")
    
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
            # If specific pairs are requested, use them; otherwise get all pairs
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
            try:
                process_pair(pair, db_manager, rolling_window, args.debug)
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

        logger.info("Rolling update mode completed.")
        
    elif mode == "full_backfill":
        logger.info("Full backfill mode: fetching all candles per pair and computing everything...")

        all_rows = []
        skipped = 0
        columns_for_update = [
            'pair', 'timestamp_utc',
            'rsi_1h', 'rsi_slope_1h', 'macd_slope_1h', 'macd_hist_slope_1h', 'atr_1h',
            'bollinger_width_1h', 'donchian_channel_width_1h', 'supertrend_direction_1h',
            'parabolic_sar_1h', 'money_flow_index_1h', 'obv_slope_1h', 'volume_change_pct_1h',
            'estimated_slippage_1h', 'bid_ask_spread_1h', 'hour_of_day', 'day_of_week',
            'rsi_4h', 'rsi_slope_4h', 'macd_hist_slope_4h', 'macd_slope_4h', 'atr_4h',
            'bollinger_width_4h', 'donchian_channel_width_4h', 'supertrend_direction_4h',
            'money_flow_index_4h', 'obv_slope_4h', 'volume_change_pct_4h',
            'rsi_1d', 'rsi_slope_1d', 'macd_hist_slope_1d', 'macd_slope_1d', 'atr_1d',
            'bollinger_width_1d', 'donchian_channel_width_1d', 'supertrend_direction_1d',
            'money_flow_index_1d', 'obv_slope_1d', 'volume_change_pct_1d',
            'performance_rank_btc_1h', 'performance_rank_eth_1h',
            'volume_rank_1h', 'volatility_rank_1h',
            'was_profitable_12h', 'prev_close_change_pct', 'prev_volume_rank',
            'future_max_return_24h_pct', 'future_max_drawdown_12h_pct'
        ]

        def compute_and_collect(pair: str) -> list:
            """
            Processes a single pair, returns a list of rows for bulk update in the full_backfill mode.
            """
            batch_rows = []
            try:
                # Create a new connection for thread safety
                connection_string = config_manager.get_connection_string()
                local_engine = create_engine(connection_string)
                
                with local_engine.connect() as conn:
                    df = pd.read_sql(
                        text("SELECT * FROM candles_1h WHERE pair = :pair ORDER BY timestamp_utc ASC"),
                        conn, params={"pair": pair}
                    )
                
                if df.empty or len(df) < MIN_CANDLES_REQUIRED:
                    logger.warning(f"Skipping {pair} (full_backfill): only {len(df)} candles; need >= {MIN_CANDLES_REQUIRED}")
                    return batch_rows

                # Enable garbage collection to reduce memory usage
                gc.collect()
                
                # Compute all features
                df = compute_1h_features(df, args.debug)
                df = compute_labels(df, args.debug)
                df = compute_multi_tf_features(df, '4h', '4h', args.debug)
                df = compute_multi_tf_features(df, '1d', '1d', args.debug)
                
                # Compute cross-pair on pairs that need it
                if pair in ['BTC-USDT', 'ETH-USDT'] or df['pair'].iloc[0] in ['BTC-USDT', 'ETH-USDT']:
                    df = compute_cross_pair_features(df)
                    
                df['features_computed'] = True
                df['targets_computed'] = True

                # Ensure proper types for all columns
                for col in columns_for_update[2:]:  # Skip pair, timestamp_utc
                    if col in SMALLINT_COLUMNS:
                        # Handle NaN values before conversion
                        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
                        df[col] = df[col].astype(int)
                    else:
                        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
                        df[col] = df[col].astype(float)

                # Cross-pair for last rolling_window
                latest = compute_cross_pair_features(df.tail(rolling_window).copy())
                
                # Only merge columns that actually exist in latest
                cross_pair_cols = [
                    'pair', 'timestamp_utc',
                    'performance_rank_btc_1h', 'performance_rank_eth_1h',
                    'volume_rank_1h', 'volatility_rank_1h'
                ]
                
                # Filter to columns that exist
                merge_cols = ['pair', 'timestamp_utc'] + [
                    col for col in cross_pair_cols[2:] 
                    if col in latest.columns
                ]
                
                if len(merge_cols) > 2:  # Only merge if we have columns beyond pair/timestamp
                    df = df.merge(
                        latest[merge_cols],
                        on=['pair', 'timestamp_utc'], how='left'
                    )

                # Clean up NaN/Inf values before update
                for col in df.columns:
                    if col not in ['timestamp_utc', 'pair', 'id']:
                        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

                # Prepare rows for bulk update
                for _, row_data in df.iterrows():
                    row_to_insert = []
                    for col in columns_for_update:
                        raw_val = row_data.get(col, 0)  # Default to 0 if column missing
                        safe_val = cast_for_sqlalchemy(col, raw_val)
                        row_to_insert.append(safe_val)
                    batch_rows.append(row_to_insert)

            except Exception as pair_error:
                logger.error(f"Error in {pair} during full_backfill: {pair_error}", exc_info=True)
                return batch_rows

            return batch_rows

        logger.info(f"Running full backfill with {batch_size} parallel workers...")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures_map = {executor.submit(compute_and_collect, pair): pair for pair in all_pairs}
            completed = 0
            for future in as_completed(futures_map):
                pair_str = futures_map[future]
                try:
                    result = future.result()
                    all_rows.extend(result)
                    completed += 1
                    if completed % 25 == 0 or completed == len(all_pairs):
                        elapsed = (datetime.now() - start_time_global).total_seconds()
                        pairs_per_second = completed / elapsed if elapsed > 0 else 0
                        remaining = (len(all_pairs) - completed) / pairs_per_second if pairs_per_second > 0 else 0
                        logger.info(f"Progress: {completed}/{len(all_pairs)} pairs processed "
                                    f"({pairs_per_second:.2f} pairs/sec, ~{remaining:.0f}s remaining)")
                except Exception as e:
                    logger.error(f"Future error for pair {pair_str}: {e}")
                    skipped += 1

        rows_written = len(all_rows)
        pairs_processed = len(set(row[0] for row in all_rows if row and len(row) > 0))  # row[0] is 'pair'

        if rows_written == 0:
            logger.critical("No rows were collected for update. Aborting write.")
        else:
            logger.info(f"Writing {rows_written} rows across {pairs_processed} pairs to DB...")

            # Use our own implementation for bulk update
            update_query = """
            UPDATE candles_1h AS c SET
            """ + ",\n".join([
                f"{col} = t.{col}" for col in columns_for_update[2:]  # skip pair, timestamp_utc
            ]) + """
            , features_computed = TRUE,
              targets_computed = TRUE
            FROM {temp_table} t
            WHERE c.pair = t.pair AND c.timestamp_utc = t.timestamp_utc;
            """
            
            db_manager.execute_copy_update(
                temp_table_name="temp_full_backfill",
                column_names=columns_for_update,
                values=all_rows,
                update_query=update_query
            )

            logger.info(
                f"Full backfill complete: {rows_written} rows updated for {pairs_processed} pairs (skipped {skipped})."
            )

    # Runtime logging
    end_time = datetime.now()
    duration = (end_time - start_time_global).total_seconds()
    with open(runtime_log_path, "a") as f:
        f.write(f"[{end_time}] compute_candles.py ({mode}) completed in {duration:.2f} seconds\n")
    
    logger.info(f"Total runtime: {duration:.2f} seconds")

if __name__ == '__main__':
    main()