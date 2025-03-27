#!/usr/bin/env python3
"""
Cryptocurrency Technical Feature Computation
- High-performance implementation using NumPy, Numba, and CuPy
- Processes cryptocurrency OHLCV data to compute technical features
- Optimized for maximum throughput and minimum memory usage
"""
import os
import sys
import gc
import logging
import argparse
import signal
import time
import threading
import psutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# Import database connection pool
from database.processing.features.db_pool import (
    initialize_connection_pool,
    get_db_connection,
    get_thread_connection,
    release_thread_connection,
    get_pool_status
)

# Import configuration
from database.processing.features.config import ConfigManager, SMALLINT_COLUMNS, MIN_CANDLES_REQUIRED

# Import performance monitoring
from database.processing.features.performance_monitor import PerformanceMonitor

# Import optimized feature processor
from database.processing.features.optimized.feature_processor import OptimizedFeatureProcessor

# Import database utilities
from database.processing.features.db_operations import (
    fetch_data_numpy,
    batch_update_features,
    bulk_copy_update,
    get_database_columns
)

# ---------------------------
# Logging Setup
# ---------------------------
def setup_logging(log_dir="logs", log_level="INFO"):
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
# ---------------------------
# Memory Optimization
# ---------------------------
def configure_memory_settings():
    """Configure optimal memory settings for performance"""
    # NumPy settings
    np.set_printoptions(threshold=100, precision=4, suppress=True)
    
    # Adjust Numba settings
    try:
        from numba import config as numba_config
        # Use threadsafe mode for better parallelism
        numba_config.THREADING_LAYER = 'threadsafe'
        # Limit number of threads to avoid oversubscription
        numba_config.NUMBA_NUM_THREADS = min(os.cpu_count(), 16)
    except ImportError:
        pass
    
    # Configure CuPy if available
    try:
        import cupy as cp
        # Create a memory pool to reduce allocation overhead
        memory_pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(memory_pool.malloc)
        
        # Use pinned memory for faster CPU-GPU transfers
        pinned_memory_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
        
        # Log GPU memory info
        device = cp.cuda.Device(0)
        mem_info = device.mem_info
        free_memory = mem_info[0] / (1024**3)
        total_memory = mem_info[1] / (1024**3)
        
        logging.info(f"GPU memory: {free_memory:.2f} GB free / {total_memory:.2f} GB total")
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"Error configuring GPU memory: {e}")
    
    # Report available system memory
    vm = psutil.virtual_memory()
    logging.info(f"System memory: {vm.available/(1024**3):.2f} GB available / {vm.total/(1024**3):.2f} GB total")

def setup_error_reporting(log_dir="logs"):
    """Set up detailed error reporting"""
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create an error log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
    
    # Set up error handler
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter('[%(levelname)s] %(asctime)s | %(message)s')
    error_handler.setFormatter(error_formatter)
    
    # Add handler to root logger
    logging.getLogger().addHandler(error_handler)
    
    # Set up sys.excepthook for uncaught exceptions
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Call original excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = handle_uncaught_exception
    
    return error_log_file

# Add error context manager for better tracking
class ErrorContext:
    """Context manager for tracking operations and reporting errors with context"""
    
    def __init__(self, operation, pair=None):
        self.operation = operation
        self.pair = pair
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Error occurred
            pair_info = f" for {self.pair}" if self.pair else ""
            logging.error(f"Error in {self.operation}{pair_info}: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))
        
        # Log duration regardless of error
        duration = time.time() - self.start_time
        logging.debug(f"{self.operation} completed in {duration:.3f}s")
        
        # Don't suppress the exception
        return False
# ---------------------------
# Pair Processing
# ---------------------------
def process_pair(pair, rolling_window, config_manager, debug_mode=False, perf_monitor=None):
    """
    Process a single pair, compute features, and update the database
    
    Args:
        pair: Symbol pair (e.g., 'BTC-USDT')
        rolling_window: Number of recent candles to process
        config_manager: Configuration manager
        debug_mode: Whether to log debug info
        perf_monitor: Performance monitor
        
    Returns:
        Number of rows updated
    """
    start_process = time.time()
    logging.info(f"Computing features for {pair}")
    updated_rows = 0
    
    try:
        # Start performance monitoring for this pair
        if perf_monitor:
            perf_monitor.start_pair(pair)
        
        # Get a database connection for this thread
        db_conn = get_thread_connection()
        
        # Create feature processor with GPU acceleration if enabled
        feature_processor = OptimizedFeatureProcessor(
            use_numba=config_manager.use_numba(),
            use_gpu=config_manager.use_gpu()
        )
        
        # Get database columns
        start_db_connect = time.time()
        db_columns = get_database_columns(db_conn, 'candles_1h')
        if perf_monitor:
            perf_monitor.log_operation("db_get_columns", time.time() - start_db_connect)
        
        # Fetch data using optimized method
        start_fetch = time.time()
        price_data = fetch_data_numpy(db_conn, pair, rolling_window + 300)
        if perf_monitor:
            perf_monitor.log_operation("fetch_data", time.time() - start_fetch)
        
        if not price_data:
            logging.warning(f"No data found for {pair}")
            return 0
            
        row_count = len(price_data['closes'])
        
        if debug_mode:
            logging.debug(f"{pair}: Fetched {row_count} rows in {time.time() - start_fetch:.3f}s")
        
        if row_count < MIN_CANDLES_REQUIRED:
            logging.warning(f"Skipping {pair}: only {row_count} candles, need >= {MIN_CANDLES_REQUIRED}")
            return 0
        
        # Determine enabled feature groups
        enabled_features = {
            feature_name.lower() for feature_name 
            in ['price_action', 'momentum', 'volatility', 'volume', 
                'statistical', 'pattern', 'time', 'labels']
            if config_manager.is_feature_enabled(feature_name)
        }
        
        # Free memory
        gc.collect()
        
        # Process features
        start_compute = time.time()
        feature_results = feature_processor.process_features(
            price_data, enabled_features, perf_monitor
        )
        
        if perf_monitor:
            perf_monitor.log_operation("compute_features_total", time.time() - start_compute)
        
        # Take only the newest rows for updating
        if row_count > rolling_window:
            # Slice arrays to get only the newest rows
            start_idx = row_count - rolling_window
            for key in feature_results:
                feature_results[key] = feature_results[key][start_idx:]
                
            # Also slice timestamps
            update_timestamps = price_data['raw_timestamps'][start_idx:]
        else:
            update_timestamps = price_data['raw_timestamps']
        
        # Define columns to update (all computed features that exist in the database)
        reserved_columns = {'id', 'pair', 'timestamp_utc', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 
                          'volume_1h', 'quote_volume_1h', 'taker_buy_base_1h'}
        
        # Filter out columns that don't exist in the database
        columns_for_update = [
            col for col in feature_results.keys() 
            if col in db_columns and col not in reserved_columns
        ]
        
        if debug_mode:
            logging.debug(f"{pair}: Updating {len(columns_for_update)} columns for {len(update_timestamps)} rows")
        
        # Update database
        start_update = time.time()
        
        if columns_for_update:
            # Use bulk copy method for maximum performance
            try:
                updated_rows = bulk_copy_update(
                    db_conn, pair, update_timestamps, feature_results, columns_for_update
                )
            except Exception as e:
                logging.warning(f"Bulk copy failed for {pair}, falling back to batch update: {e}")
                # Fall back to batch update
                updated_rows = batch_update_features(
                    db_conn, pair, update_timestamps, feature_results, columns_for_update
                )
        
        if perf_monitor:
            perf_monitor.log_operation("database_update", time.time() - start_update)
    
    except Exception as e:
        logging.error(f"Error processing {pair}: {e}", exc_info=True)
        # Don't re-raise the exception - just log it and return
        return 0
    finally:
        # Release the connection back to the pool
        try:
            release_thread_connection(db_conn)
        except Exception as e:
            logging.warning(f"Error releasing connection for {pair}: {e}")
            
        total_time = time.time() - start_process
        logging.info(f"{pair}: Updated {updated_rows}/{len(update_timestamps) if 'update_timestamps' in locals() else 0} rows in {total_time:.2f}s")
        
        # End performance monitoring for this pair
        if perf_monitor:
            try:
                perf_monitor.end_pair(total_time)
            except Exception as e:
                logging.debug(f"Error ending performance monitoring for {pair}: {e}")
        
        # Clear memory
        gc.collect()
        
        return updated_rows

def process_pair_thread(pair, rolling_window, config_manager, debug_mode=False, perf_monitor=None):
    """Wrapper function for thread pool to handle a single pair"""
    try:
        # Set current pair for performance tracking
        if perf_monitor:
            perf_monitor.start_pair(pair)
            
        start_time = time.time()
        
        # Process the pair
        result = process_pair(pair, rolling_window, config_manager, debug_mode, perf_monitor)
        
        # End timing for this pair
        total_time = time.time() - start_time
        if perf_monitor:
            try:
                perf_monitor.end_pair(total_time)
            except Exception as e:
                logging.debug(f"Error ending performance monitoring for {pair}: {e}")
            
        return result
        
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        logging.info(f"Processing of {pair} interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Thread error for pair {pair}: {e}", exc_info=True)
        # Make sure to end performance monitoring even on error
        if perf_monitor:
            try:
                perf_monitor.end_pair(0)  # Use 0 duration for error case
            except Exception:
                pass
        return 0

# ---------------------------
# Cross-Pair Features
# ---------------------------
def compute_cross_pair_features(db_conn, config_manager, debug_mode=False, perf_monitor=None):
    """
    Compute features that require data across multiple pairs
    
    Args:
        db_conn: Database connection
        config_manager: Configuration manager
        debug_mode: Whether to log debug info
        perf_monitor: Performance monitor
        
    Returns:
        Number of rows updated
    """
    start_time = time.time()
    logging.info("Computing cross-pair features")
    updated_rows = 0
    
    # Track this operation under a special "CROSS_PAIR" context
    if perf_monitor:
        perf_monitor.start_pair("CROSS_PAIR")
    
    try:
        # Only compute if cross_pair features are enabled
        if not config_manager.is_feature_enabled('cross_pair'):
            logging.info("Cross-pair features disabled, skipping")
            return 0
            
        # Get most recent candles
        cursor = db_conn.cursor()
        
        query = """
        SELECT pair, timestamp_utc, close_1h, volume_1h, atr_1h, future_return_1h_pct, log_return
        FROM candles_1h 
        WHERE timestamp_utc >= (SELECT MAX(timestamp_utc) FROM candles_1h) - INTERVAL '24 hours'
        ORDER BY timestamp_utc, pair
        """
        cursor.execute(query)
        
        rows = cursor.fetchall()
        cursor.close()
        
        if not rows:
            logging.warning("No recent data found for cross-pair features")
            return 0
            
        # Group by timestamp
        timestamps = {}
        for row in rows:
            pair, ts, close, volume, atr, future_return, log_return = row
            
            if ts not in timestamps:
                timestamps[ts] = {
                    'pairs': [],
                    'volumes': [],
                    'atrs': [],
                    'future_returns': [],
                    'log_returns': {},
                    'btc_returns': []
                }
                
            timestamps[ts]['pairs'].append(pair)
            timestamps[ts]['volumes'].append(volume if volume is not None else 0)
            timestamps[ts]['atrs'].append(atr if atr is not None else 0)
            timestamps[ts]['future_returns'].append(future_return if future_return is not None else 0)
            
            # Store log returns by pair for correlation
            timestamps[ts]['log_returns'][pair] = log_return if log_return is not None else 0
            
            # Save BTC returns separately for correlation
            if pair == 'BTC-USDT':
                timestamps[ts]['btc_returns'].append(log_return if log_return is not None else 0)
        
        # Prepare update data
        updates = []
        
        # Process each timestamp
        for ts, data in timestamps.items():
            pairs = data['pairs']
            volumes = data['volumes']
            atrs = data['atrs']
            future_returns = data['future_returns']
            log_returns = data['log_returns']
            
            # Skip if no data or missing BTC
            if not pairs or 'BTC-USDT' not in log_returns:
                continue
                
            # Compute volume ranks
            vol_ranks = np.zeros(len(volumes))
            
            if volumes and any(v > 0 for v in volumes):
                # Convert to numpy array for ranking
                vol_array = np.array(volumes)
                # Sort indices (ascending)
                sorted_indices = np.argsort(vol_array)
                # Assign ranks (higher volume = higher rank)
                for i, idx in enumerate(sorted_indices):
                    vol_ranks[idx] = int(100 * i / (len(sorted_indices) - 1)) if len(sorted_indices) > 1 else 50
            
            # Compute volatility ranks
            atr_ranks = np.zeros(len(atrs))
            
            if atrs and any(a > 0 for a in atrs):
                # Convert to numpy array for ranking
                atr_array = np.array(atrs)
                # Sort indices (ascending)
                sorted_indices = np.argsort(atr_array)
                # Assign ranks (higher ATR = higher rank)
                for i, idx in enumerate(sorted_indices):
                    atr_ranks[idx] = int(100 * i / (len(sorted_indices) - 1)) if len(sorted_indices) > 1 else 50
            
            # Compute performance ranks relative to BTC
            btc_return = log_returns.get('BTC-USDT', 0)
            perf_ranks_btc = np.zeros(len(future_returns))
            
            if btc_return != 0 and future_returns and any(fr != 0 for fr in future_returns):
                # Compute relative performance
                rel_perf = np.array([(fr - btc_return) / abs(btc_return) if btc_return != 0 else 0 
                                   for fr in future_returns])
                # Compute percentile rank
                perf_ranks_btc = np.zeros_like(rel_perf)
                # Sort indices (ascending)
                sorted_indices = np.argsort(rel_perf)
                # Assign ranks (higher relative perf = higher rank)
                for i, idx in enumerate(sorted_indices):
                    perf_ranks_btc[idx] = int(100 * i / (len(sorted_indices) - 1)) if len(sorted_indices) > 1 else 50
            
            # Compute performance ranks relative to ETH
            eth_return = log_returns.get('ETH-USDT', 0)
            perf_ranks_eth = np.zeros(len(future_returns))
            
            if eth_return != 0 and future_returns and any(fr != 0 for fr in future_returns):
                # Compute relative performance
                rel_perf = np.array([(fr - eth_return) / abs(eth_return) if eth_return != 0 else 0 
                                   for fr in future_returns])
                # Compute percentile rank
                perf_ranks_eth = np.zeros_like(rel_perf)
                # Sort indices (ascending)
                sorted_indices = np.argsort(rel_perf)
                # Assign ranks (higher relative perf = higher rank)
                for i, idx in enumerate(sorted_indices):
                    perf_ranks_eth[idx] = int(100 * i / (len(sorted_indices) - 1)) if len(sorted_indices) > 1 else 50
            
            # Compute BTC correlation
            btc_corr = np.zeros(len(pairs))
            
            # Get correlation window data
            correlation_window = 24  # 24 hours
            
            cursor = db_conn.cursor()
            
            # Get historical data for correlation
            window_query = """
            SELECT pair, timestamp_utc, log_return
            FROM candles_1h 
            WHERE timestamp_utc >= %s - INTERVAL '24 hours'
              AND timestamp_utc <= %s
              AND pair IN ('BTC-USDT', %s)
            ORDER BY pair, timestamp_utc
            """
            
            for i, pair in enumerate(pairs):
                if pair == 'BTC-USDT':
                    btc_corr[i] = 1.0  # BTC correlation with itself is 1
                    continue
                    
                cursor.execute(window_query, (ts, ts, pair))
                corr_rows = cursor.fetchall()
                
                # Group by pair
                pair_returns = {}
                for row in corr_rows:
                    p, t, ret = row
                    if p not in pair_returns:
                        pair_returns[p] = {}
                    pair_returns[p][t] = ret if ret is not None else 0
                
                # Get common timestamps
                if 'BTC-USDT' in pair_returns and pair in pair_returns:
                    btc_times = set(pair_returns['BTC-USDT'].keys())
                    pair_times = set(pair_returns[pair].keys())
                    common_times = sorted(btc_times.intersection(pair_times))
                    
                    if len(common_times) >= 12:  # Need at least 12 points for correlation
                        # Create arrays for correlation
                        btc_vals = np.array([pair_returns['BTC-USDT'][t] for t in common_times])
                        pair_vals = np.array([pair_returns[pair][t] for t in common_times])
                        
                        # Compute correlation
                        if np.std(btc_vals) > 0 and np.std(pair_vals) > 0:
                            corr = np.corrcoef(btc_vals, pair_vals)[0, 1]
                            btc_corr[i] = corr
            
            cursor.close()
            
            # Prepare update parameters
            for i, pair in enumerate(pairs):
                updates.append((
                    int(round(perf_ranks_btc[i])),
                    int(round(perf_ranks_eth[i])),
                    int(round(vol_ranks[i])),
                    int(round(atr_ranks[i] if i < len(atr_ranks) else 0)),
                    float(btc_corr[i]),
                    float(vol_ranks[i-1] if i > 0 else 0),  # Previous volume rank
                    pair,
                    ts
                ))
        
        # Execute updates
        if updates:
            cursor = db_conn.cursor()
            
            update_query = """
            UPDATE candles_1h
            SET 
                performance_rank_btc_1h = %s,
                performance_rank_eth_1h = %s,
                volume_rank_1h = %s,
                volatility_rank_1h = %s,
                btc_corr_24h = %s,
                prev_volume_rank = %s
            WHERE pair = %s AND timestamp_utc = %s
            """
            
            psycopg2.extras.execute_batch(
                cursor, 
                update_query, 
                updates,
                page_size=1000
            )
            
            updated_rows = cursor.rowcount
            db_conn.commit()
            cursor.close()
            
            logging.info(f"Updated {updated_rows} rows with cross-pair features")
        
    except Exception as e:
        logging.error(f"Error computing cross-pair features: {e}", exc_info=True)
        if 'cursor' in locals() and not cursor.closed:
            cursor.close()
        # Rollback on error
        try:
            db_conn.rollback()
        except Exception:
            pass
    finally:
        if perf_monitor:
            perf_monitor.log_operation("cross_pair_features", time.time() - start_time)
            try:
                perf_monitor.end_pair(time.time() - start_time)
            except Exception as e:
                logging.debug(f"Error ending CROSS_PAIR performance monitoring: {e}")
    
    return updated_rows

def calibrate_batch_size(config_manager, initial_pairs=10):
    """Calibrate the optimal batch size based on test runs"""
    logging.info("Calibrating optimal batch size...")
    
    # Get a small sample of pairs
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pair FROM candles_1h LIMIT %s", (initial_pairs,))
        test_pairs = [row[0] for row in cursor.fetchall()]
        cursor.close()
    
    if not test_pairs:
        logging.warning("No pairs found for calibration")
        return None
    
    # Testing with different batch sizes
    test_sizes = [4, 8, 12, 16, 24, 32]
    results = {}
    
    # Create a dummy performance monitor that doesn't log to files
    class DummyMonitor:
        def start_pair(self, pair): pass
        def log_operation(self, op, duration): pass
        def end_pair(self, duration): pass
    
    dummy_monitor = DummyMonitor()
    
    for batch_size in test_sizes:
        if batch_size > len(test_pairs):
            continue
            
        logging.info(f"Testing batch size {batch_size}...")
        
        # Time the processing
        start_time = time.time()
        
        # Process with current batch size
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(
                    process_pair_thread, 
                    pair, 10, config_manager, False, dummy_monitor
                ): pair for pair in test_pairs[:batch_size]
            }
            
            for future in as_completed(futures):
                # Just wait for completion
                pair = futures[future]
                try:
                    future.result()
                except Exception:
                    pass
        
        # Record time
        duration = time.time() - start_time
        if duration > 0:
            pairs_per_second = len(test_pairs[:batch_size]) / duration
            results[batch_size] = pairs_per_second
            logging.info(f"Batch size {batch_size}: {pairs_per_second:.2f} pairs/sec")
        
        # Force cleanup
        gc.collect()
    
    # Find optimal batch size
    if results:
        optimal_size = max(results, key=results.get)
        logging.info(f"Optimal batch size determined: {optimal_size} ({results[optimal_size]:.2f} pairs/sec)")
        return optimal_size
    
    return None

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
    parser.add_argument('--max-connections', type=int,
                       help='Maximum database connections')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed timing')
    parser.add_argument('--pairs', type=str, help='Comma-separated list of pairs to process (default: all)')
    parser.add_argument('--no-numba', action='store_true', help='Disable Numba optimizations')
    parser.add_argument('--no-calibration', action='store_true', help='Skip auto-calibration of processing parameters')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration (if available)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--disable-features', type=str, 
                       help='Comma-separated list of feature groups to disable (e.g., momentum,pattern)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir, args.log_level)

    # Set up performance monitoring
    perf_monitor = PerformanceMonitor(args.log_dir)
    
    # Track runtime
    start_time_global = datetime.now()
    
    # Enable force exit for CTRL+C
    force_exit_on_ctrl_c()
    
    # Configure memory settings
    configure_memory_settings()
    
    # Load configuration
    config_manager = ConfigManager(
        config_path=args.config,
        credentials_path=args.credentials
    )

    # Enable GPU if available and not explicitly disabled
    if config_manager.gpu_available and not args.no_gpu:
        if 'GENERAL' not in config_manager.config:
            config_manager.config['GENERAL'] = {}
        config_manager.config['GENERAL']['USE_GPU'] = 'True'
        logger.info("GPU capabilities detected - enabling GPU acceleration")

    # Override config with command line arguments
    if args.no_numba:
        config_manager.config['GENERAL']['USE_NUMBA'] = 'False'
    if args.use_gpu:
        config_manager.config['GENERAL']['USE_GPU'] = 'True'
    if args.no_gpu:
        config_manager.config['GENERAL']['USE_GPU'] = 'False'
    if args.batch_size:
        config_manager.config['GENERAL']['BATCH_SIZE'] = str(args.batch_size)
    
    # Handle disabled features
    if args.disable_features:
        disabled_features = [f.strip().upper() for f in args.disable_features.split(',')]
        if 'FEATURES' not in config_manager.config:
            config_manager.config['FEATURES'] = {}
        for feature in disabled_features:
            config_manager.config['FEATURES'][feature] = 'False'
            logger.info(f"Disabled feature group: {feature}")
    
    # Set processing mode
    mode = args.mode if args.mode else config_manager.get_compute_mode()
    
    # Set rolling window
    rolling_window = args.rolling_window if args.rolling_window else config_manager.get_rolling_window()
    
    # Set batch size - potentially auto-calibrate later
    batch_size = args.batch_size if args.batch_size else config_manager.get_batch_size()
    
    # Set max connections for the pool
    max_connections = args.max_connections if hasattr(args, 'max_connections') and args.max_connections is not None else batch_size + 5
    
    # Initialize the connection pool
    db_params = config_manager.get_db_params()
    initialize_connection_pool(db_params, min_connections=2, max_connections=max_connections)
    
    # Create runtime log path
    runtime_log_path = os.path.join(args.log_dir, "runtime_stats.log")
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    
    logger.info(f"Starting compute_features.py in {mode.upper()} mode with rolling_window={rolling_window}")
    logger.info(f"Using {batch_size} parallel workers with max {max_connections} database connections")
    logger.info(f"Numba: {'enabled' if config_manager.use_numba() else 'disabled'}, "
               f"GPU: {'enabled' if config_manager.use_gpu() else 'disabled'}")
    
    if args.debug:
        logger.info("Debug mode enabled - detailed timing information will be logged")

    try:
        # Test connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            logger.info("Database connection test successful")
    except Exception as e:
        logger.critical(f"Unable to start due to database connection issue: {e}")
        sys.exit(1)

    try:
        # Get pairs to process
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if args.pairs:
                all_pairs = [p.strip() for p in args.pairs.split(',')]
                logger.info(f"Processing {len(all_pairs)} specified pairs")
            else:
                cursor.execute("SELECT DISTINCT pair FROM candles_1h")
                all_pairs = [row[0] for row in cursor.fetchall()]
                logger.info(f"Found {len(all_pairs)} pairs")
                
            cursor.close()
    except Exception as e:
        logger.critical(f"Error loading pair list: {e}")
        sys.exit(1)

    if not all_pairs:
        logger.warning("No pairs found in candles_1h. Exiting early.")
        sys.exit()

    if mode == "rolling_update":
        logger.info(f"Rolling update mode: computing last {rolling_window} rows per pair")
        
        # Group pairs by complexity for balanced processing
        def estimate_pair_complexity(pair):
            if any(major in pair for major in ['BTC', 'ETH']):
                return 2  # Higher complexity
            return 1      # Standard complexity
                
        # Sort pairs by complexity (process simpler pairs first)
        sorted_pairs = sorted(all_pairs, key=estimate_pair_complexity)

        # Auto-calibrate batch size if not specified and calibration not disabled
        if not args.batch_size and not args.no_calibration:
            optimal_batch_size = calibrate_batch_size(config_manager)
            if optimal_batch_size:
                batch_size = optimal_batch_size
                logger.info(f"Using auto-calibrated batch size: {batch_size}")
        
        # Split pairs into batches
        batches = []
        batch = []
        batch_complexity = 0
        max_batch_complexity = batch_size * 1.5
        
        for pair in sorted_pairs:
            pair_complexity = estimate_pair_complexity(pair)
            
            if batch_complexity + pair_complexity > max_batch_complexity and batch:
                batches.append(batch)
                batch = [pair]
                batch_complexity = pair_complexity
            else:
                batch.append(pair)
                batch_complexity += pair_complexity
                
        if batch:
            batches.append(batch)
            
        logger.info(f"Grouped {len(all_pairs)} pairs into {len(batches)} balanced batches")

        # Process each batch with controlled parallelism
        completed = 0
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} pairs")
            
            # Use a thread pool size that won't exhaust connections
            # Allow 2 extra connections for overhead/monitoring
            max_pool_size = max_connections - 2
            thread_pool_size = min(batch_size, len(batch), max_pool_size)
            logger.info(f"Using thread pool size of {thread_pool_size} for this batch")
            
            with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
                futures = {
                    executor.submit(
                        process_pair_thread, 
                        pair, rolling_window, 
                        config_manager, args.debug, perf_monitor
                    ): pair for pair in batch
                }
                
                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        rows_updated = future.result()
                        completed += 1
                        
                        # Log pool status every 10 pairs
                        if completed % 10 == 0:
                            pool_status = get_pool_status()
                            logger.info(f"Connection pool status: {pool_status}")
                        
                        if completed % 25 == 0 or completed == len(all_pairs):
                            elapsed = (datetime.now() - start_time_global).total_seconds()
                            pairs_per_second = completed / elapsed if elapsed > 0 else 0
                            remaining = (len(all_pairs) - completed) / pairs_per_second if pairs_per_second > 0 else 0
                            logger.info(f"Progress: {completed}/{len(all_pairs)} pairs processed "
                                        f"({pairs_per_second:.2f} pairs/sec, ~{remaining:.0f}s remaining)")
                    except Exception as e:
                        logger.error(f"Error processing pair {pair}: {e}")
            
            # Log pool status after each batch
            pool_status = get_pool_status()
            logger.info(f"Connection pool status after batch: {pool_status}")
            
            # Cleanup between batches
            gc.collect()

        # Compute cross-pair features after all individual pairs are done
        if config_manager.is_feature_enabled('cross_pair'):
            with get_db_connection() as conn:
                compute_cross_pair_features(conn, config_manager, args.debug, perf_monitor)

        # Save performance summary at the end
        summary_file, report_file = perf_monitor.save_summary()
        logger.info(f"Performance summary saved to {summary_file}")
        logger.info(f"Performance report saved to {report_file}")
        
        logger.info("Rolling update mode completed.")

    elif mode == "full_backfill":
        logger.info("Full backfill mode: computing all historical data")
        logger.warning("Full backfill mode is not fully implemented yet")
        # TODO: Implement full backfill if needed

    # Runtime logging
    end_time = datetime.now()
    duration = (end_time - start_time_global).total_seconds()
    with open(runtime_log_path, "a") as f:
        f.write(f"[{end_time}] compute_features.py ({mode}) completed in {duration:.2f} seconds\n")
    
    logger.info(f"Total runtime: {duration:.2f} seconds")

if __name__ == "__main__":
    main()