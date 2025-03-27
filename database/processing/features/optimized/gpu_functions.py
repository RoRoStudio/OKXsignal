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
        
        # Get device info safely - some CuPy versions don't have the 'name' attribute
        device_info = f"Device {device.id}"
        try:
            # Try to get more detailed information if available
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            if hasattr(props, 'name'):
                device_info = props.name
        except:
            pass
            
        logging.info(f"GPU is available: {device_info}, "
                    f"Free: {free_memory:.2f} GB / {total_memory:.2f} GB")
        return True
    except Exception as e:
        logging.warning(f"GPU test failed: {e}")
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
        
        # Get device properties in a safer way
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
        
    # Convert to CuPy arrays if they're not already
    try:
        open_prices_gpu = cp.asarray(open_prices)
        high_prices_gpu = cp.asarray(high_prices)
        low_prices_gpu = cp.asarray(low_prices)
        close_prices_gpu = cp.asarray(close_prices)
        
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
        
        # Return as numpy array
        return cp.asnumpy(results.flatten())
    except Exception as e:
        logging.error(f"GPU calculation failed: {e}")
        # Fall back to CPU implementation
        raise

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

def hurst_exponent_gpu(prices, max_lag):
    """
    Compute Hurst exponent using GPU
    
    Args:
        prices: Array of prices
        max_lag: Maximum lag for calculation
        
    Returns:
        Array of Hurst exponents
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Convert to CuPy array
        prices_gpu = cp.asarray(prices)
        
        n = len(prices_gpu)
        hurst_values = cp.zeros(n, dtype=cp.float64)
        
        # Need at least 100 points for reliable Hurst calculation
        min_window = 100
        
        if n < min_window:
            return cp.asnumpy(hurst_values)
        
        # Log returns
        # Handle zeros and negative values
        safe_prices = cp.maximum(prices_gpu, 1e-8)
        log_prices = cp.log(safe_prices)
        
        for end_idx in range(min_window, n):
            # Take a window of data
            window_size = min(min_window, end_idx)
            log_window = log_prices[end_idx-window_size:end_idx]
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
                
                # Linear regression on log-log scale
                log_lag = cp.log(valid_lags)
                log_lagmat = cp.log(valid_lagmat)
                
                # Calculate regression using polyfit
                if cp.isnan(log_lag).any() or cp.isnan(log_lagmat).any():
                    hurst_values[end_idx] = 0.5
                else:
                    try:
                        coeffs = cp.polyfit(log_lag, log_lagmat, 1)
                        hurst_values[end_idx] = coeffs[0] / 2
                    except:
                        hurst_values[end_idx] = 0.5
            else:
                hurst_values[end_idx] = 0.5
        
        # Return as numpy array
        return cp.asnumpy(hurst_values)
    except Exception as e:
        logging.error(f"GPU Hurst calculation failed: {e}")
        raise

def shannon_entropy_gpu(prices, window_size=20):
    """
    Compute Shannon Entropy using GPU
    
    Args:
        prices: Array of prices
        window_size: Window size for entropy calculation
        
    Returns:
        Array of entropy values
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
        
    try:
        # Convert to CuPy array
        prices_gpu = cp.asarray(prices)
        
        n = len(prices_gpu)
        entropy_values = cp.zeros(n, dtype=cp.float64)
        
        # Need at least window_size points
        if n < window_size:
            return cp.asnumpy(entropy_values)
        
        # Handle zeros and negative values
        safe_prices = cp.maximum(prices_gpu, 1e-8)
        
        for end_idx in range(window_size, n):
            price_window = safe_prices[end_idx-window_size:end_idx]
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
    except Exception as e:
        logging.error(f"GPU Entropy calculation failed: {e}")
        raise

# -------------------------------
# Future Returns
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
        
        # Move data to GPU
        cp_opens = cp.asarray(opens)
        cp_highs = cp.asarray(highs)
        cp_lows = cp.asarray(lows)
        cp_closes = cp.asarray(closes)
        
        # Results dictionary
        results = {}
        
        # Compute price action features
        # Body size
        results['body_size'] = cp.asnumpy(cp.abs(cp_closes - cp_opens))
        
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
            mask = cp_closes[:-1] > 0
            indices = cp.where(mask)[0]
            log_returns[indices+1] = cp.log(cp_closes[indices+1] / cp_closes[indices])
        
        results['log_return'] = cp.asnumpy(log_returns)
        
        # Add more features as needed...
        
        # Return results dictionary
        return results
    except Exception as e:
        logging.error(f"Batch GPU computation failed: {e}")
        # Return empty dictionary on failure
        return {}