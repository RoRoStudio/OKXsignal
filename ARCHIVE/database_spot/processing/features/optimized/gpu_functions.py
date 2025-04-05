#!/usr/bin/env python3
"""
GPU-accelerated functions for feature computation using CuPy
"""

import numpy as np
import logging

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. GPU acceleration will not be used.")

# -------------------------------
# Core GPU Functions
# -------------------------------
def initialize_gpu():
    """Initialize GPU and configure memory pool"""
    if not CUPY_AVAILABLE:
        return False
        
    try:
        # Create memory pool with unified management
        memory_pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(memory_pool.malloc)
        
        # Set pinned memory for faster transfers
        pinned_memory_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
        
        # Get device info
        device = cp.cuda.Device()
        device_id = device.id
        
        # Get memory info
        mem_info = device.mem_info
        total_memory = mem_info[1]/1024**3
        
        # Try to get device name if possible
        device_name = f"Device #{device_id}"
        try:
            device_props = cp.cuda.runtime.getDeviceProperties(device_id)
            if hasattr(device_props, 'name'):
                device_name = device_props.name
        except:
            pass
        
        logging.info(f"Initialized GPU: {device_name}, Memory: {total_memory:.2f} GB")
        
        # Test basic operations to ensure GPU is working
        test_array = cp.array([1, 2, 3])
        test_result = cp.sum(test_array)
        cp.cuda.Stream.null.synchronize()  # Ensure operation completes
        
        return True
    except Exception as e:
        logging.error(f"GPU initialization failed: {e}")
        return False
    
def is_gpu_available():
    """Check if GPU is available"""
    if not CUPY_AVAILABLE:
        return False
        
    try:
        # Test GPU with a small array
        a = cp.array([1, 2, 3])
        b = a * 2
        cp.cuda.Stream.null.synchronize()
        
        # Log device info
        device = cp.cuda.Device()
        mem_info = device.mem_info
        free_memory = mem_info[0] / (1024**3)  # in GB
        total_memory = mem_info[1] / (1024**3)  # in GB
        
        # Get device ID
        device_id = device.id
        
        logging.info(f"GPU is available: Device #{device_id}, "
                    f"Free: {free_memory:.2f} GB / {total_memory:.2f} GB")
        return True
    except Exception as e:
        logging.warning(f"GPU test failed: {e}")
        return False

# -------------------------------
# Price Action Features
# -------------------------------

def compute_candle_body_features_gpu(open_prices, high_prices, low_prices, close_prices):
    """
    Compute candle body size, upper shadow, lower shadow using GPU
    
    Args:
        open_prices: Array of open prices
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        
    Returns:
        Array with candle body features (flattened)
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    # Initialize GPU if not already done
    initialize_gpu()
        
    # Convert to CuPy arrays if they're not already
    try:
        # Use asarray with explicit types
        open_prices_gpu = cp.asarray(open_prices, dtype=cp.float64)
        high_prices_gpu = cp.asarray(high_prices, dtype=cp.float64)
        low_prices_gpu = cp.asarray(low_prices, dtype=cp.float64)
        close_prices_gpu = cp.asarray(close_prices, dtype=cp.float64)
        
        # Synchronize to ensure data is moved to GPU
        cp.cuda.Stream.null.synchronize()
        
        n = len(open_prices_gpu)
        results = cp.zeros(4 * n, dtype=cp.float64).reshape(4, n)
        
        # Body size
        results[0] = cp.abs(close_prices_gpu - open_prices_gpu)
        
        # Upper shadow
        results[1] = high_prices_gpu - cp.maximum(open_prices_gpu, close_prices_gpu)
        
        # Lower shadow
        results[2] = cp.minimum(open_prices_gpu, close_prices_gpu) - low_prices_gpu
        
        # Relative close position
        hl_range = high_prices_gpu - low_prices_gpu
        mask = hl_range > 0
        results[3] = cp.where(
            mask,
            (close_prices_gpu - low_prices_gpu) / hl_range,
            0.5  # Default to middle if no range
        )
        
        # Make sure computation is complete before returning
        cp.cuda.Stream.null.synchronize()
        
        # Return as numpy array
        return cp.asnumpy(results.flatten())
    except Exception as e:
        logging.error(f"GPU calculation failed: {e}")
        # Fall back to CPU implementation
        raise

# -------------------------------
# Batch Feature Computation
# -------------------------------

def compute_batch_features_gpu(price_data):
    """
    Batch compute multiple features using GPU
    
    Args:
        price_data: Dictionary with price arrays
        
    Returns:
        Dictionary with computed features
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Make sure GPU is initialized
        if not initialize_gpu():
            logging.warning("Failed to initialize GPU for batch computation")
            return {}
        
        # Extract arrays
        opens = price_data['opens']
        highs = price_data['highs']
        lows = price_data['lows']
        closes = price_data['closes']
        volumes = price_data.get('volumes', np.zeros_like(closes))
        
        # Safety check for array sizes before moving to GPU
        if any(arr is None or len(arr) == 0 for arr in [opens, highs, lows, closes]):
            return {}
            
        # Check if arrays are compatible sizes
        array_len = len(closes)
        if any(len(arr) != array_len for arr in [opens, highs, lows]):
            return {}
            
        # Move data to GPU with explicit copy to ensure data is properly transferred
        cp_opens = cp.asarray(opens, dtype=cp.float64)
        cp_highs = cp.asarray(highs, dtype=cp.float64)
        cp_lows = cp.asarray(lows, dtype=cp.float64)
        cp_closes = cp.asarray(closes, dtype=cp.float64)
        cp_volumes = cp.asarray(volumes, dtype=cp.float64)
        
        # Synchronize to ensure data is moved to GPU before operations
        cp.cuda.Stream.null.synchronize()
        
        # Results dictionary
        results = {}
        
        # Compute more features in one GPU batch
        n = len(cp_closes)
        
        # 1. Price action features
        # Body size
        body_size = cp.abs(cp_closes - cp_opens)
        results['body_size'] = cp.asnumpy(body_size)
        results['candle_body_size'] = cp.asnumpy(body_size)
        
        # Shadows
        upper_shadow = cp_highs - cp.maximum(cp_opens, cp_closes)
        lower_shadow = cp.minimum(cp_opens, cp_closes) - cp_lows
        results['upper_shadow'] = cp.asnumpy(upper_shadow)
        results['lower_shadow'] = cp.asnumpy(lower_shadow)
        
        # Relative close position
        hl_range = cp_highs - cp_lows
        mask = hl_range > 0
        rel_pos = cp.where(
            mask,
            (cp_closes - cp_lows) / hl_range,
            0.5  # Default to middle if no range
        )
        results['relative_close_position'] = cp.asnumpy(rel_pos)
        
        # 2. Log returns
        log_returns = cp.zeros_like(cp_closes)
        if n > 1:
            # Make sure we don't have zeros or negatives
            safe_closes = cp.maximum(cp_closes, 1e-8)
            # Avoid division by zero
            divisor = cp.maximum(cp.roll(safe_closes, 1), 1e-8)
            # Skip first element which would divide by undefined "previous" 
            log_returns[1:] = cp.log(safe_closes[1:] / divisor[1:])
        
        results['log_return'] = cp.asnumpy(log_returns)
        
        # 3. Price change percentage
        price_change = cp.zeros_like(cp_closes)
        if n > 1:
            safe_closes = cp.maximum(cp_closes, 1e-8)
            divisor = cp.maximum(cp.roll(safe_closes, 1), 1e-8)
            price_change[1:] = (safe_closes[1:] / divisor[1:]) - 1.0
            
        results['prev_close_change_pct'] = cp.asnumpy(price_change)
        
        # 4. Gap open (versus previous close)
        gap_open = cp.zeros_like(cp_closes)
        if n > 1:
            safe_closes = cp.maximum(cp_closes, 1e-8)
            divisor = cp.maximum(cp.roll(safe_closes, 1), 1e-8)
            gap_open[1:] = (cp_opens[1:] / divisor[1:]) - 1.0
            
        results['gap_open'] = cp.asnumpy(gap_open)
        
        # 5. Price velocity and acceleration
        price_velocity = cp.zeros_like(cp_closes)
        price_accel = cp.zeros_like(cp_closes)
        
        if n > 3:
            safe_closes = cp.maximum(cp_closes, 1e-8)
            rolled_closes = cp.roll(safe_closes, 3)
            
            # Skip first 3 elements
            valid_idx = cp.arange(3, n)
            
            # Compute velocity for valid indices
            divisor = cp.maximum(rolled_closes[valid_idx], 1e-8)
            price_velocity[valid_idx] = (safe_closes[valid_idx] / divisor) - 1.0
            
        if n > 6:
            # Compute acceleration (velocity change)
            rolled_velocity = cp.roll(price_velocity, 3)
            
            # Skip first 6 elements
            valid_idx = cp.arange(6, n)
            
            # Compute acceleration for valid indices
            price_accel[valid_idx] = price_velocity[valid_idx] - rolled_velocity[valid_idx]
            
        results['price_velocity'] = cp.asnumpy(price_velocity)
        results['price_acceleration'] = cp.asnumpy(price_accel)
        
        # Synchronize before returning to ensure all computations complete
        cp.cuda.Stream.null.synchronize()
        
        # Clean up GPU memory explicitly
        del cp_opens, cp_highs, cp_lows, cp_closes, cp_volumes
        del body_size, upper_shadow, lower_shadow, hl_range, mask, rel_pos
        del log_returns, price_change, gap_open, price_velocity, price_accel
        cp.get_default_memory_pool().free_all_blocks()
        
        return results
    except Exception as e:
        logging.error(f"GPU batch calculation failed: {e}")
        # Clean up GPU memory on error
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        # Return empty dictionary on failure
        return {}

# -------------------------------
# Statistical Features
# -------------------------------

def compute_z_score_gpu(values, ma_values, lookback):
    """
    Compute z-score using GPU
    
    Args:
        values: Array of values
        ma_values: Array of moving average values
        lookback: Lookback period for standard deviation
        
    Returns:
        Array of z-scores
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Convert to CuPy arrays
        values_gpu = cp.asarray(values)
        ma_values_gpu = cp.asarray(ma_values)
        
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
    except Exception as e:
        logging.error(f"GPU z-score calculation failed: {e}")
        raise

# -------------------------------
# Future Return Calculations
# -------------------------------

def compute_future_return_gpu(close, shift):
    """
    Compute future returns for a specific time horizon using GPU
    
    Args:
        close: Array of closing prices
        shift: Number of periods to shift for future returns
        
    Returns:
        Array of future returns
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Convert to CuPy array
        close_gpu = cp.asarray(close)
        
        n = len(close_gpu)
        future_return = cp.zeros(n, dtype=cp.float64)
        
        if n > shift:
            # Calculate future returns
            future_close = close_gpu[shift:]
            current_close = close_gpu[:-shift]
            
            # Calculate returns, handling division by zero
            # Create a mask for non-zero denominators
            mask = current_close > 0
            
            # Use CuPy's masked operations
            future_return_slice = cp.zeros(n - shift, dtype=cp.float64)
            future_return_slice[mask] = (future_close[mask] - current_close[mask]) / current_close[mask]
            
            # Assign to output array
            future_return[:n-shift] = future_return_slice
        
        # Return as numpy array
        return cp.asnumpy(future_return)
    except Exception as e:
        logging.error(f"GPU future return calculation failed: {e}")
        raise

def compute_max_future_return_gpu(close, high, window):
    """
    Compute maximum future return within a window using GPU
    
    Args:
        close: Array of closing prices
        high: Array of high prices
        window: Window size for max return calculation
        
    Returns:
        Array of maximum future returns
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Convert to CuPy arrays
        close_gpu = cp.asarray(close)
        high_gpu = cp.asarray(high)
        
        n = len(close_gpu)
        max_future_return = cp.zeros(n, dtype=cp.float64)
        
        # This operation is harder to vectorize fully on GPU, so use a loop
        for i in range(n - 1):
            end_idx = min(i + window, n)
            if i + 1 < end_idx:
                max_high = cp.max(high_gpu[i+1:end_idx])
                # Handle division by zero
                if close_gpu[i] > 0:
                    max_future_return[i] = (max_high - close_gpu[i]) / close_gpu[i]
        
        # Return as numpy array
        return cp.asnumpy(max_future_return)
    except Exception as e:
        logging.error(f"GPU max future return calculation failed: {e}")
        raise

def compute_max_future_drawdown_gpu(close, low, window):
    """
    Compute maximum future drawdown within a window using GPU
    
    Args:
        close: Array of closing prices
        low: Array of low prices
        window: Window size for drawdown calculation
        
    Returns:
        Array of maximum future drawdowns
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Convert to CuPy arrays
        close_gpu = cp.asarray(close)
        low_gpu = cp.asarray(low)
        
        n = len(close_gpu)
        max_future_drawdown = cp.zeros(n, dtype=cp.float64)
        
        # This operation is harder to vectorize fully on GPU, so use a loop
        for i in range(n - 1):
            end_idx = min(i + window, n)
            if i + 1 < end_idx:
                min_low = cp.min(low_gpu[i+1:end_idx])
                # Handle division by zero
                if close_gpu[i] > 0:
                    max_future_drawdown[i] = (min_low - close_gpu[i]) / close_gpu[i]
        
        # Return as numpy array
        return cp.asnumpy(max_future_drawdown)
    except Exception as e:
        logging.error(f"GPU max future drawdown calculation failed: {e}")
        raise

# -------------------------------
# Batch Feature Computation
# -------------------------------

def compute_batch_features_gpu(price_data):
    """
    Batch compute multiple features using GPU
    
    Args:
        price_data: Dictionary with price arrays
        
    Returns:
        Dictionary with computed features
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Extract arrays
        opens = price_data['opens']
        highs = price_data['highs']
        lows = price_data['lows']
        closes = price_data['closes']
        volumes = price_data.get('volumes')
        
        # Safety check for array sizes before moving to GPU
        if any(arr is None or len(arr) == 0 for arr in [opens, highs, lows, closes]):
            return {}
            
        # Check if arrays are compatible sizes
        array_len = len(closes)
        if any(len(arr) != array_len for arr in [opens, highs, lows]):
            return {}
            
        # Move data to GPU with explicit copy to ensure data is properly transferred
        cp_opens = cp.asarray(opens, dtype=cp.float64)
        cp_highs = cp.asarray(highs, dtype=cp.float64)
        cp_lows = cp.asarray(lows, dtype=cp.float64)
        cp_closes = cp.asarray(closes, dtype=cp.float64)
        
        # Synchronize to ensure data is moved to GPU before operations
        cp.cuda.Stream.null.synchronize()
        
        # Results dictionary
        results = {}
        
        # Compute price action features
        # Body size
        results['body_size'] = cp.asnumpy(cp.abs(cp_closes - cp_opens))
        results['candle_body_size'] = results['body_size']
        
        # Shadows
        results['upper_shadow'] = cp.asnumpy(cp_highs - cp.maximum(cp_opens, cp_closes))
        results['lower_shadow'] = cp.asnumpy(cp.minimum(cp_opens, cp_closes) - cp_lows)
        
        # Relative close position
        hl_range = cp_highs - cp_lows
        mask = hl_range > 0
        rel_pos = cp.where(
            mask,
            (cp_closes - cp_lows) / hl_range,
            0.5  # Default to middle if no range
        )
        results['relative_close_position'] = cp.asnumpy(rel_pos)
        
        # Log returns
        n = len(cp_closes)
        log_returns = cp.zeros_like(cp_closes)
        if n > 1:
            # Avoid division by zero
            safe_closes = cp.maximum(cp_closes[:-1], 1e-8)
            log_returns[1:] = cp.log(cp_closes[1:] / safe_closes)
        
        results['log_return'] = cp.asnumpy(log_returns)
        
        # Clean up GPU memory explicitly
        del cp_opens, cp_highs, cp_lows, cp_closes, hl_range, mask, rel_pos, log_returns
        cp.get_default_memory_pool().free_all_blocks()
        
        # Return results dictionary
        return results
    except Exception as e:
        logging.error(f"GPU calculation failed: {e}")
        # Clean up GPU memory on error
        cp.get_default_memory_pool().free_all_blocks()
        # Return empty dictionary on failure
        return {}