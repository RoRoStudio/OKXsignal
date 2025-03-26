#!/usr/bin/env python3
"""
Test script for the feature computation system
- Runs feature computation on sample data and verifies outputs
- Can be used for benchmarking and validating feature calculations
"""

import os
import sys
import time
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path for imports to work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up to database directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import configuration manager
from database.processing.features.config import ConfigManager

# Import feature classes
from database.processing.features.price_action import PriceActionFeatures
from database.processing.features.momentum import MomentumFeatures
from database.processing.features.volatility import VolatilityFeatures
from database.processing.features.volume import VolumeFeatures
from database.processing.features.statistical import StatisticalFeatures
from database.processing.features.pattern import PatternFeatures
from database.processing.features.time import TimeFeatures
from database.processing.features.multi_timeframe import MultiTimeframeFeatures
from database.processing.features.cross_pair import CrossPairFeatures
from database.processing.features.labels import LabelFeatures

# Import utilities
from database.processing.features.utils import PerformanceMonitor

def setup_logging(log_level="INFO"):
    """Set up logging for the test script"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(levelname)s] %(asctime)s | %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("test_features")

def generate_test_data(rows=1000, pairs=["BTC-USDT"]):
    """
    Generate synthetic OHLCV data for testing
    
    Args:
        rows: Number of rows to generate per pair
        pairs: List of pairs to generate data for
        
    Returns:
        DataFrame with synthetic data
    """
    all_data = []
    
    for pair in pairs:
        # Generate timestamps
        now = pd.Timestamp.now().floor('H')
        timestamps = [now - pd.Timedelta(hours=i) for i in range(rows)]
        timestamps.reverse()  # Oldest first
        
        # Start with a base price
        base_price = 100.0 if "BTC" not in pair else 30000.0
        
        # Generate prices using random walk
        np.random.seed(42)  # For reproducibility
        
        # Random walk parameters
        volatility = 0.02
        drift = 0.001
        
        # Generate log returns
        log_returns = np.random.normal(drift, volatility, rows)
        
        # Convert to price series
        close_prices = base_price * np.exp(np.cumsum(log_returns))
        
        # Generate other prices around close
        high_prices = close_prices * np.random.uniform(1.0, 1.05, rows)
        low_prices = close_prices * np.random.uniform(0.95, 1.0, rows)
        open_prices = low_prices + np.random.uniform(0, 1, rows) * (high_prices - low_prices)
        
        # Generate volume
        volumes = np.random.lognormal(mean=10, sigma=1, size=rows)
        
        # Create DataFrame
        df = pd.DataFrame({
            'pair': pair,
            'timestamp_utc': timestamps,
            'open_1h': open_prices,
            'high_1h': high_prices,
            'low_1h': low_prices,
            'close_1h': close_prices,
            'volume_1h': volumes,
            'quote_volume_1h': volumes * close_prices,
            'taker_buy_base_1h': volumes * np.random.uniform(0.4, 0.6, rows)
        })
        
        all_data.append(df)
    
    # Combine all pairs
    return pd.concat(all_data, ignore_index=True)

def test_single_feature_group(feature_class, feature_name, test_data, use_numba=True, use_gpu=False):
    """
    Test a single feature group
    
    Args:
        feature_class: Feature computer class
        feature_name: Name of the feature group
        test_data: Test data DataFrame
        use_numba: Whether to use Numba
        use_gpu: Whether to use GPU
        
    Returns:
        Tuple of (DataFrame with features, computation time)
    """
    # Create feature computer
    computer = feature_class(use_numba=use_numba, use_gpu=use_gpu)
    
    # Time the computation
    start_time = time.time()
    result_df = computer.compute_features(test_data, debug_mode=True)
    compute_time = time.time() - start_time
    
    print(f"{feature_name} features computed in {compute_time:.4f} seconds")
    
    # Get new columns added
    original_cols = set(test_data.columns)
    new_cols = set(result_df.columns) - original_cols
    
    print(f"  Added {len(new_cols)} columns: {', '.join(sorted(new_cols))}")
    
    # Check for NaN or inf values
    has_bad_values = False
    for col in new_cols:
        if col in result_df.columns:
            nan_count = result_df[col].isna().sum()
            inf_count = np.isinf(result_df[col]).sum()
            
            if nan_count > 0 or inf_count > 0:
                print(f"  WARNING: Column {col} has {nan_count} NaN and {inf_count} inf values")
                has_bad_values = True
    
    if not has_bad_values:
        print("  All values are valid (no NaN or inf)")
    
    return result_df, compute_time

def run_tests(use_numba=True, use_gpu=False, pairs=None):
    """
    Run tests for all feature groups
    
    Args:
        use_numba: Whether to use Numba
        use_gpu: Whether to use GPU
        pairs: List of pairs to test with
    """
    logger = setup_logging()
    
    # Default pairs if none provided
    pairs = pairs or ["BTC-USDT", "ETH-USDT"]
    
    logger.info(f"Generating test data for {len(pairs)} pairs")
    test_data = generate_test_data(rows=500, pairs=pairs)
    logger.info(f"Generated {len(test_data)} rows of test data")
    
    # Create performance monitor for tracking
    perf_monitor = PerformanceMonitor(log_dir="logs")
    
    # Dict of feature computers to test
    feature_tests = {
        "Price Action": PriceActionFeatures,
        "Momentum": MomentumFeatures,
        "Volatility": VolatilityFeatures,
        "Volume": VolumeFeatures,
        "Statistical": StatisticalFeatures,
        "Pattern": PatternFeatures,
        "Time": TimeFeatures,
        "Multi-Timeframe": MultiTimeframeFeatures,
        "Labels": LabelFeatures
    }
    
    # Track total time and features computed
    total_time = 0
    all_features_df = test_data.copy()
    
    # Test each feature group
    for feature_name, feature_class in feature_tests.items():
        try:
            logger.info(f"Testing {feature_name} features")
            perf_monitor.start_pair(feature_name)
            result_df, compute_time = test_single_feature_group(
                feature_class, feature_name, all_features_df, use_numba, use_gpu
            )
            
            # Update accumulated DataFrame and time
            all_features_df = result_df
            total_time += compute_time
            perf_monitor.end_pair(compute_time)
            
        except Exception as e:
            logger.error(f"Error testing {feature_name} features: {e}")
    
    # Test cross-pair features separately (needs multiple pairs)
    if len(pairs) > 1:
        try:
            logger.info("Testing Cross-Pair features")
            perf_monitor.start_pair("Cross-Pair")
            cross_computer = CrossPairFeatures(use_numba=use_numba, use_gpu=use_gpu)
            
            start_time = time.time()
            all_features_df = cross_computer.compute_features(all_features_df, debug_mode=True)
            cross_time = time.time() - start_time
            
            logger.info(f"Cross-Pair features computed in {cross_time:.4f} seconds")
            total_time += cross_time
            perf_monitor.end_pair(cross_time)
            
        except Exception as e:
            logger.error(f"Error testing Cross-Pair features: {e}")
    
    # Save performance summary
    summary_file, report_file = perf_monitor.save_summary()
    
    # Log total computation time
    logger.info(f"All features computed in {total_time:.4f} seconds")
    logger.info(f"Final DataFrame has {len(all_features_df.columns)} columns")
    logger.info(f"Performance summary saved to {summary_file}")
    logger.info(f"Performance report saved to {report_file}")
    
    return all_features_df

def compare_implementations():
    """Compare different implementations (Numpy, Numba, GPU)"""
    logger = setup_logging()
    
    # Test pairs
    pairs = ["BTC-USDT", "ETH-USDT"]
    
    # Generate test data once
    test_data = generate_test_data(rows=1000, pairs=pairs)
    
    # Run tests with different implementations
    implementations = [
        ("NumPy (no optimization)", False, False),
        ("Numba", True, False)
    ]
    
    # Check if GPU is available
    try:
        import cupy
        cupy.array([1, 2, 3])  # Test if GPU works
        implementations.append(("GPU (CuPy)", True, True))
    except (ImportError, Exception):
        logger.warning("CuPy/GPU not available, skipping GPU tests")
    
    # Run and time each implementation
    results = {}
    
    for name, use_numba, use_gpu in implementations:
        logger.info(f"Testing {name} implementation")
        
        start_time = time.time()
        # Create feature computers
        price_action = PriceActionFeatures(use_numba=use_numba, use_gpu=use_gpu)
        statistical = StatisticalFeatures(use_numba=use_numba, use_gpu=use_gpu)
        labels = LabelFeatures(use_numba=use_numba, use_gpu=use_gpu)
        
        # Test computationally intensive features
        test_data_copy = test_data.copy()
        
        # Compute features sequentially
        df1 = price_action.compute_features(test_data_copy)
        df2 = statistical.compute_features(df1)
        df3 = labels.compute_features(df2)
        
        total_time = time.time() - start_time
        results[name] = total_time
        
        logger.info(f"{name} implementation: {total_time:.4f} seconds")
    
    # Print comparison
    logger.info("Performance Comparison:")
    baseline = results.get("NumPy (no optimization)", 1.0)
    
    for name, time_taken in results.items():
        speedup = baseline / time_taken if time_taken > 0 else float('inf')
        logger.info(f"  {name}: {time_taken:.4f}s (Speedup: {speedup:.2f}x)")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test feature computation system')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark comparing different implementations')
    parser.add_argument('--no-numba', action='store_true',
                       help='Disable Numba optimization')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--pairs', type=str, default="BTC-USDT,ETH-USDT",
                       help='Comma-separated list of pairs to test with')
    
    args = parser.parse_args()
    
    if args.benchmark:
        compare_implementations()
    else:
        pairs = [p.strip() for p in args.pairs.split(',')]
        run_tests(
            use_numba=(not args.no_numba),
            use_gpu=args.use_gpu,
            pairs=pairs
        )