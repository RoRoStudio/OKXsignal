#!/usr/bin/env python3
"""
NumPy and Numba-optimized functions for feature computation
"""

import numpy as np
from numba import njit, prange, float64, int64, boolean

# -------------------------------
# Core Numerical Functions
# -------------------------------

@njit(parallel=True)
def moving_average_numba(array, window):
    """
    Compute moving average using Numba
    
    Args:
        array: Input array
        window: Window size
        
    Returns:
        Array with moving averages
    """
    n = len(array)
    result = np.zeros_like(array)
    
    # Initialize with NaN for proper handling
    result[:window-1] = np.nan
    
    # Compute first window directly
    result[window-1] = np.mean(array[:window])
    
    # Use a more efficient algorithm for the rest
    # Each new value is: prev_avg + (new - removed)/window
    for i in range(window, n):
        result[i] = result[i-1] + (array[i] - array[i-window]) / window
        
    # Fill NaN values with backward filling
    for i in range(window-2, -1, -1):
        result[i] = result[i+1]
        
    return result

@njit
def moving_std_numba(array, window):
    """
    Compute moving standard deviation using Numba
    
    Args:
        array: Input array
        window: Window size
        
    Returns:
        Array with moving standard deviations
    """
    n = len(array)
    result = np.zeros_like(array)
    
    for i in range(window-1, n):
        window_slice = array[i-window+1:i+1]
        result[i] = np.std(window_slice)
        
    # Fill beginning with first valid value
    if window-1 < n:
        fill_value = result[window-1]
        for i in range(window-1):
            result[i] = fill_value
            
    return result

@njit
def ewma_numba(array, span):
    """
    Compute exponential weighted moving average using Numba
    
    Args:
        array: Input array
        span: Span parameter, analogous to pandas EWM
        
    Returns:
        Array with exponential moving averages
    """
    n = len(array)
    result = np.zeros_like(array)
    
    # Define alpha (smoothing factor)
    alpha = 2.0 / (span + 1.0)
    
    # Initialize with first value
    if n > 0:
        result[0] = array[0]
        
        # Apply EWM formula iteratively
        for i in range(1, n):
            result[i] = alpha * array[i] + (1 - alpha) * result[i-1]
            
    return result

# -------------------------------
# Price Action Features
# -------------------------------

@njit(float64[:](float64[:], float64[:], float64[:], float64[:]))
def compute_candle_body_features_numba(open_prices, high_prices, low_prices, close_prices):
    """
    Compute candle body size, upper shadow, lower shadow using Numba
    
    Args:
        open_prices: Array of open prices
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        
    Returns:
        Array with candle body features (flattened)
    """
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

@njit
def compute_price_features_numba(opens, highs, lows, closes):
    """
    Compute price action features using Numba
    
    Args:
        opens: Array of open prices
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        
    Returns:
        Dictionary with price action features
    """
    n = len(closes)
    
    # Pre-allocate result arrays
    log_return = np.zeros(n, dtype=np.float64)
    gap_open = np.zeros(n, dtype=np.float64)
    price_velocity = np.zeros(n, dtype=np.float64)
    price_accel = np.zeros(n, dtype=np.float64)
    prev_close_change = np.zeros(n, dtype=np.float64)
    
    # Compute features
    for i in range(1, n):
        # Log return
        if closes[i-1] > 0:
            log_return[i] = np.log(closes[i] / closes[i-1])
        
        # Gap open
        if closes[i-1] > 0:
            gap_open[i] = (opens[i] / closes[i-1]) - 1.0
        
        # Previous close change
        if closes[i-1] > 0:
            prev_close_change[i] = (closes[i] / closes[i-1]) - 1.0
    
    # Price velocity (3-period)
    for i in range(3, n):
        if closes[i-3] > 0:
            price_velocity[i] = (closes[i] / closes[i-3]) - 1.0
    
    # Price acceleration
    for i in range(6, n):
        price_accel[i] = price_velocity[i] - price_velocity[i-3]
    
    # Return as dictionary for easier access
    return {
        'log_return': log_return,
        'gap_open': gap_open,
        'price_velocity': price_velocity,
        'price_acceleration': price_accel,
        'prev_close_change_pct': prev_close_change
    }

# -------------------------------
# Momentum Features
# -------------------------------

@njit
def compute_rsi_numba(closes, length=14):
    """
    Compute RSI using Numba
    
    Args:
        closes: Array of closing prices
        length: RSI period
        
    Returns:
        Array with RSI values
    """
    n = len(closes)
    rsi = np.zeros(n, dtype=np.float64)
    
    # Default value for beginning
    rsi[:length] = 50.0
    
    if n <= length:
        return rsi
    
    # Calculate price changes
    changes = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        changes[i] = closes[i] - closes[i-1]
    
    # Separate gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if changes[i] > 0:
            gains[i] = changes[i]
        else:
            losses[i] = -changes[i]
    
    # Initial averages
    avg_gain = np.sum(gains[1:length+1]) / length
    avg_loss = np.sum(losses[1:length+1]) / length
    
    # Calculate RSI
    if avg_loss == 0:
        rsi[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[length] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate remaining RSI values
    for i in range(length+1, n):
        avg_gain = ((avg_gain * (length-1)) + gains[i]) / length
        avg_loss = ((avg_loss * (length-1)) + losses[i]) / length
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@njit
def compute_macd_numba(closes, fast=12, slow=26, signal=9):
    """
    Compute MACD using Numba
    
    Args:
        closes: Array of closing prices
        fast: Fast period
        slow: Slow period
        signal: Signal period
        
    Returns:
        Dictionary with MACD line, signal and histogram
    """
    n = len(closes)
    
    # Pre-allocate arrays
    macd_line = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    histogram = np.zeros(n, dtype=np.float64)
    
    # Compute EMAs
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    alpha_signal = 2.0 / (signal + 1.0)
    
    # Fast EMA
    ema_fast = np.zeros(n, dtype=np.float64)
    ema_fast[0] = closes[0]
    for i in range(1, n):
        ema_fast[i] = closes[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
    
    # Slow EMA
    ema_slow = np.zeros(n, dtype=np.float64)
    ema_slow[0] = closes[0]
    for i in range(1, n):
        ema_slow[i] = closes[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
    
    # MACD Line
    for i in range(n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal Line (EMA of MACD Line)
    signal_line[0] = macd_line[0]
    for i in range(1, n):
        signal_line[i] = macd_line[i] * alpha_signal + signal_line[i-1] * (1 - alpha_signal)
    
    # Histogram
    for i in range(n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    # Return as dictionary
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }

# -------------------------------
# Statistical & Complex Features
# -------------------------------

@njit(float64[:](float64[:], float64[:], int64))
def compute_z_score_numba(values, ma_values, lookback):
    """
    Compute z-score using Numba
    
    Args:
        values: Array of values
        ma_values: Array of moving average values
        lookback: Lookback period for standard deviation
        
    Returns:
        Array of z-scores
    """
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
    """
    Compute Hurst exponent using Numba
    
    Args:
        prices: Array of prices
        max_lag: Maximum lag for calculation
        
    Returns:
        Array of Hurst exponents
    """
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
        
        # Calculate log returns
        log_prices = np.log(np.maximum(price_window, 1e-8))
        returns = np.diff(log_prices)
        
        if len(returns) < max_lag:
            hurst_values[end_idx] = 0.5  # Default value
            continue
            
        tau = np.arange(1, max_lag+1)
        lagmat = np.zeros(max_lag)
        
        for lag in range(1, max_lag+1):
            # Skip if lag is too large
            if lag >= len(returns):
                continue
                
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
    """
    Compute Shannon Entropy using Numba
    
    Args:
        prices: Array of prices
        window_size: Window size for entropy calculation
        
    Returns:
        Array of entropy values
    """
    n = len(prices)
    entropy_values = np.zeros(n, dtype=np.float64)
    
    # Need at least window_size for calculation
    if n < window_size:
        return entropy_values
        
    for end_idx in range(window_size, n):
        price_window = prices[end_idx-window_size:end_idx]
        
        # Calculate log returns, handling zeros and negatives
        safe_prices = np.maximum(price_window, 1e-8)
        log_prices = np.log(safe_prices)
        returns = np.diff(log_prices)
        
        # Use histogram to estimate probability distribution
        # Create a fixed number of bins for returns
        hist = np.zeros(10, dtype=np.float64)
        
        # Determine min and max for binning
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        # Avoid division by zero for uniform returns
        bin_width = max(max_return - min_return, 1e-8)
        
        # Count returns in each bin
        for ret in returns:
            # Calculate bin index
            bin_idx = min(9, max(0, int(((ret - min_return) / bin_width) * 10)))
            hist[bin_idx] += 1
        
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

# -------------------------------
# Future Return Calculations
# -------------------------------

@njit(float64[:](float64[:], int64))
def compute_future_return_numba(close, shift):
    """
    Compute future returns for a specific time horizon
    
    Args:
        close: Array of closing prices
        shift: Number of periods to shift for future returns
        
    Returns:
        Array of future returns
    """
    n = len(close)
    future_return = np.zeros(n, dtype=np.float64)
    
    if n > shift:
        for i in range(n - shift):
            # Avoid division by zero
            if close[i] > 0:
                future_return[i] = (close[i + shift] - close[i]) / close[i]
            else:
                future_return[i] = 0.0
    
    return future_return

@njit(float64[:](float64[:], float64[:], int64))
def compute_max_future_return_numba(close, high, window):
    """
    Compute maximum future return within a window
    
    Args:
        close: Array of closing prices
        high: Array of high prices
        window: Window size for max return calculation
        
    Returns:
        Array of maximum future returns
    """
    n = len(close)
    max_future_return = np.zeros(n, dtype=np.float64)
    
    for i in range(n - 1):
        end_idx = min(i + window, n)
        if i + 1 < end_idx:
            max_high = np.max(high[i+1:end_idx])
            # Avoid division by zero
            if close[i] > 0:
                max_future_return[i] = (max_high - close[i]) / close[i]
            else:
                max_future_return[i] = 0.0
                
    return max_future_return

@njit(float64[:](float64[:], float64[:], int64))
def compute_max_future_drawdown_numba(close, low, window):
    """
    Compute maximum future drawdown within a window
    
    Args:
        close: Array of closing prices
        low: Array of low prices
        window: Window size for drawdown calculation
        
    Returns:
        Array of maximum future drawdowns
    """
    n = len(close)
    max_future_drawdown = np.zeros(n, dtype=np.float64)
    
    for i in range(n - 1):
        end_idx = min(i + window, n)
        if i + 1 < end_idx:
            min_low = np.min(low[i+1:end_idx])
            # Avoid division by zero
            if close[i] > 0:
                max_future_drawdown[i] = (min_low - close[i]) / close[i]
            else:
                max_future_drawdown[i] = 0.0
                
    return max_future_drawdown

@njit
def resample_ohlcv_numba(timestamps, opens, highs, lows, closes, volumes, period):
    """
    Optimized OHLCV resampling using Numba
    
    Args:
        timestamps: Array of timestamps as integers (e.g. Unix timestamps)
        opens, highs, lows, closes, volumes: Price and volume arrays
        period: Period in hours (4 for 4h, 24 for 1d)
        
    Returns:
        Tuple of resampled arrays (timestamps, opens, highs, lows, closes, volumes)
    """
    n = len(timestamps)
    if n == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64)
        )
    
    # Convert timestamps to period indices
    # Assuming timestamps are already in ascending order
    hour_timestamps = timestamps // 3600  # Convert to hours
    indices = hour_timestamps // period
    
    # Get unique indices (each representing one new period)
    unique_indices = np.unique(indices)
    m = len(unique_indices)
    
    # Prepare output arrays
    out_timestamps = np.zeros(m, dtype=np.int64)
    out_opens = np.zeros(m, dtype=np.float64)
    out_highs = np.zeros(m, dtype=np.float64)
    out_lows = np.zeros(m, dtype=np.float64)
    out_closes = np.zeros(m, dtype=np.float64)
    out_volumes = np.zeros(m, dtype=np.float64)
    
    # Fill outputs
    for i in range(m):
        idx = unique_indices[i]
        mask = (indices == idx)
        
        if np.any(mask):
            # First timestamp in the group
            out_timestamps[i] = timestamps[np.argmax(mask)]
            
            # First open in the group
            out_opens[i] = opens[np.argmax(mask)]
            
            # Maximum high in the group
            out_highs[i] = np.max(highs[mask])
            
            # Minimum low in the group
            out_lows[i] = np.min(lows[mask])
            
            # Last close in the group
            out_closes[i] = closes[np.where(mask)[0][-1]]
            
            # Sum of volumes in the group
            out_volumes[i] = np.sum(volumes[mask])
    
    return out_timestamps, out_opens, out_highs, out_lows, out_closes, out_volumes