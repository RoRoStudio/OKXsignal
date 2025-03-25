"""
Test script for verifying cryptocurrency technical feature computation optimizations
"""
import os
import logging
import time
from datetime import datetime
import sys
sys.path.append("P:/OKXsignal/database/processing")



# Import your main script functions
from compute_candles import main, FeatureComputer, GPUFeatureComputer, PerformanceMonitor

def test_fixes():
    """Test the fixes for deprecated methods and warnings"""
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple test dataframe
    import pandas as pd
    import numpy as np
    
    # Create sample data
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='1H')
    df = pd.DataFrame({
        'timestamp_utc': dates,
        'open_1h': np.random.normal(100, 10, n),
        'high_1h': np.random.normal(105, 10, n),
        'low_1h': np.random.normal(95, 10, n),
        'close_1h': np.random.normal(102, 10, n),
        'volume_1h': np.random.normal(1000, 100, n),
    })
    
    # Make sure high is actually highest and low is lowest
    for i in range(n):
        values = [df.loc[i, 'open_1h'], df.loc[i, 'close_1h'], df.loc[i, 'high_1h'], df.loc[i, 'low_1h']]
        df.loc[i, 'high_1h'] = max(values)
        df.loc[i, 'low_1h'] = min(values)
    
    # Create a feature computer
    fc = FeatureComputer(use_talib=False, use_numba=True)
    
    # Test the pattern features function (fixed ffill)
    df_result = fc.compute_pattern_features(df, debug_mode=True)
    logging.info("Pattern features computed successfully")
    
    # Test the volatility features function (fixed division by zero)
    df_result = fc.compute_volatility_features(df, debug_mode=True)
    logging.info("Volatility features computed successfully")
    
    # Test the statistical features function (fixed ffill)
    df_result = fc.compute_statistical_features(df, debug_mode=True)
    logging.info("Statistical features computed successfully")
    
    return True

def test_performance_monitoring():
    """Test the performance monitoring functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Create performance monitor
    perf_monitor = PerformanceMonitor(log_dir="logs")
    
    # Test timing for a sample operation
    perf_monitor.start_pair("TEST-PAIR")
    
    # Simulate some operations
    start_time = time.time()
    time.sleep(0.1)  # Simulate operation
    perf_monitor.log_operation("test_operation_1", time.time() - start_time)
    
    start_time = time.time()
    time.sleep(0.2)  # Simulate operation
    perf_monitor.log_operation("test_operation_2", time.time() - start_time)
    
    # End pair timing
    perf_monitor.end_pair(0.3)
    
    # Save summary
    summary_file, report_file = perf_monitor.save_summary()
    logging.info(f"Performance summary saved to {summary_file}")
    
    return os.path.exists(summary_file) and os.path.exists(report_file)

def test_gpu_acceleration():
    """Test GPU acceleration if available"""
    logging.basicConfig(level=logging.INFO)
    
    # Check if GPU support is available
    try:
        import cupy as cp
        CUPY_AVAILABLE = True
    except ImportError:
        CUPY_AVAILABLE = False
        logging.warning("CuPy not available, skipping GPU test")
        return None
    
    if not CUPY_AVAILABLE:
        return None
    
    # Create a simple test dataframe
    import pandas as pd
    import numpy as np
    
    # Create sample data
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='1H')
    df = pd.DataFrame({
        'timestamp_utc': dates,
        'open_1h': np.random.normal(100, 10, n),
        'high_1h': np.random.normal(105, 10, n),
        'low_1h': np.random.normal(95, 10, n),
        'close_1h': np.random.normal(102, 10, n),
        'volume_1h': np.random.normal(1000, 100, n),
    })
    
    # Make sure high is actually highest and low is lowest
    for i in range(n):
        values = [df.loc[i, 'open_1h'], df.loc[i, 'close_1h'], df.loc[i, 'high_1h'], df.loc[i, 'low_1h']]
        df.loc[i, 'high_1h'] = max(values)
        df.loc[i, 'low_1h'] = min(values)
    
    # Create a GPU feature computer
    gpu_fc = GPUFeatureComputer(use_talib=False, use_numba=True, use_gpu=True)
    
    # Create a CPU feature computer for comparison
    cpu_fc = FeatureComputer(use_talib=False, use_numba=True)
    
    # Time GPU computation
    start_time = time.time()
    df_gpu = gpu_fc.compute_statistical_features(df.copy(), debug_mode=True)
    gpu_time = time.time() - start_time
    
    # Time CPU computation
    start_time = time.time()
    df_cpu = cpu_fc.compute_statistical_features(df.copy(), debug_mode=True)
    cpu_time = time.time() - start_time
    
    logging.info(f"GPU computation time: {gpu_time:.4f}s")
    logging.info(f"CPU computation time: {cpu_time:.4f}s")
    logging.info(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Verify results are similar
    for col in ['hurst_exponent', 'shannon_entropy', 'z_score_20']:
        if col in df_gpu.columns and col in df_cpu.columns:
            diff = np.abs(df_gpu[col] - df_cpu[col]).mean()
            logging.info(f"Average difference for {col}: {diff}")
    
    return {
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'speedup': cpu_time/gpu_time
    }

if __name__ == "__main__":
    print("Testing fixes for deprecated methods and warnings...")
    test_fixes()
    
    print("\nTesting performance monitoring...")
    test_performance_monitoring()
    
    print("\nTesting GPU acceleration (if available)...")
    test_gpu_acceleration()
    
    print("\nAll tests completed.")