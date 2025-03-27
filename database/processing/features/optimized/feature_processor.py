#!/usr/bin/env python3
"""
High-performance feature processor using NumPy, Numba, and CuPy
- Provides optimized implementation of the feature computation
- Handles both CPU and GPU modes
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import optimized feature functions
from database.processing.features.optimized import (
    compute_candle_body_features_numba,
    compute_price_features_numba,
    compute_rsi_numba,
    compute_macd_numba,
    compute_z_score_numba,
    hurst_exponent_numba,
    shannon_entropy_numba,
    compute_future_return_numba,
    compute_max_future_return_numba,
    compute_max_future_drawdown_numba,
    resample_ohlcv_numba,
    GPU_AVAILABLE
)

# Import GPU functions if available
if GPU_AVAILABLE:
    from database.processing.features.optimized import (
        initialize_gpu,
        is_gpu_available,
        compute_candle_body_features_gpu,
        compute_z_score_gpu,
        hurst_exponent_gpu,
        shannon_entropy_gpu,
        compute_future_return_gpu,
        compute_max_future_return_gpu,
        compute_max_future_drawdown_gpu,
        compute_batch_features_gpu
    )

# Import feature parameters
from database.processing.features.config import (
    PRICE_ACTION_PARAMS,
    MOMENTUM_PARAMS,
    VOLATILITY_PARAMS,
    VOLUME_PARAMS,
    STATISTICAL_PARAMS,
    PATTERN_PARAMS,
    TIME_PARAMS,
    MULTI_TIMEFRAME_PARAMS,
    LABEL_PARAMS,
    CROSS_PAIR_PARAMS
)

class OptimizedFeatureProcessor:
    """High-performance feature processor using NumPy, Numba, and CuPy"""
    
    def __init__(self, use_numba=True, use_gpu=False):
        """
        Initialize feature processor
        
        Args:
            use_numba: Whether to use Numba JIT
            use_gpu: Whether to use GPU acceleration
        """
        self.use_numba = use_numba
        
        # Check if GPU was already initialized
        self.gpu_available = GPU_AVAILABLE and is_gpu_available() if GPU_AVAILABLE else False
        self.use_gpu = use_gpu and self.gpu_available
        
        # Don't re-initialize GPU if it's already been initialized
        if self.use_gpu:
            self.gpu_initialized = True  # Assume GPU is already initialized
            logging.debug("Using pre-initialized GPU for computation")
        else:
            self.gpu_initialized = False
            if use_gpu and not self.gpu_available:
                logging.debug("GPU acceleration requested but not available")
                
        # Log initialization
        logging.debug(f"Optimized feature processor initialized: "
                    f"Numba: {'enabled' if use_numba else 'disabled'}, "
                    f"GPU: {'enabled' if self.use_gpu else 'disabled'}")
                     
    def process_features(self, price_data, enabled_features=None, perf_monitor=None):
        """
        Process all features for the given price data
        
        Args:
            price_data: Dictionary with OHLCV data as NumPy arrays
            enabled_features: Set of enabled feature groups
            perf_monitor: Optional performance monitor
            
        Returns:
            Dictionary with computed features
        """
        # Default to all features enabled
        if enabled_features is None:
            enabled_features = {
                'price_action', 'momentum', 'volatility', 'volume', 
                'statistical', 'pattern', 'time', 'labels'
            }
            
        # Extract price arrays
        opens = price_data['opens']
        highs = price_data['highs']
        lows = price_data['lows']
        closes = price_data['closes']
        volumes = price_data.get('volumes', np.zeros_like(closes))
        timestamps = price_data.get('timestamps', np.arange(len(closes)))
        
        # Results dictionary
        results = {}
        
        # Try GPU batch computation first if enabled
        if self.use_gpu and 'price_action' in enabled_features:
            gpu_batch_start = time.time()
            try:
                # Compute multiple features at once using GPU
                gpu_results = compute_batch_features_gpu(price_data)
                if gpu_results:
                    results.update(gpu_results)
                    if perf_monitor:
                        perf_monitor.log_operation("gpu_batch_compute", time.time() - gpu_batch_start)
                    logging.info(f"GPU batch computation successful: {len(gpu_results)} features")
            except Exception as e:
                logging.warning(f"GPU batch computation failed, falling back to individual: {e}")
        
        # Process individual feature groups
        if 'price_action' in enabled_features and not all(k in results for k in ['body_size', 'upper_shadow']):
            self._compute_price_action(opens, highs, lows, closes, results, perf_monitor)
            
        if 'momentum' in enabled_features:
            self._compute_momentum(closes, highs, lows, results, perf_monitor)
            
        if 'volatility' in enabled_features:
            self._compute_volatility(closes, highs, lows, results, perf_monitor)
            
        if 'volume' in enabled_features:
            self._compute_volume(closes, highs, lows, volumes, results, perf_monitor)
            
        if 'statistical' in enabled_features:
            # Make sure we have log_return calculated
            if 'log_return' not in results:
                # Calculate log returns
                log_return = np.zeros_like(closes)
                log_return[1:] = np.log(np.maximum(closes[1:] / np.maximum(closes[:-1], 1e-8), 1e-8))
                results['log_return'] = log_return
                
            # Pass highs and lows for estimating slippage and spread
            self._compute_statistical(closes, results['log_return'], results, perf_monitor)
            
            # Add estimated slippage and spread here since they weren't added in _compute_statistical
            results['estimated_slippage_1h'] = highs - lows
            results['bid_ask_spread_1h'] = results['estimated_slippage_1h'] * 0.1  # Approximation
            
        if 'pattern' in enabled_features:
            self._compute_pattern(opens, highs, lows, closes, results, perf_monitor)
            
        if 'time' in enabled_features:
            self._compute_time(timestamps, results, perf_monitor)
            
        if 'labels' in enabled_features:
            self._compute_labels(closes, highs, lows, results, perf_monitor)
            
        # Post-processing - ensure all arrays are NumPy arrays (not CuPy)
        for key in results:
            if not isinstance(results[key], np.ndarray):
                results[key] = np.array(results[key])
                
        # Add completion flags
        results['features_computed'] = np.ones(len(closes), dtype=np.bool_)
        results['targets_computed'] = np.ones(len(closes), dtype=np.bool_)
        
        return results
        
    def _compute_price_action(self, opens, highs, lows, closes, results, perf_monitor=None):
        """Compute price action features"""
        start_time = time.time()
        
        try:
            # Compute candle body features
            if self.use_gpu:
                try:
                    body_features = compute_candle_body_features_gpu(opens, highs, lows, closes)
                    results['body_size'] = body_features[:len(closes)]
                    results['upper_shadow'] = body_features[len(closes):2*len(closes)]
                    results['lower_shadow'] = body_features[2*len(closes):3*len(closes)]
                    results['relative_close_position'] = body_features[3*len(closes):4*len(closes)]
                    logging.debug("Used GPU for candle body features")
                except Exception as e:
                    logging.warning(f"GPU calculation failed for candle features: {e}")
                    self.use_gpu = False  # Disable GPU for future calls
            
            # Fall back to Numba if GPU failed or not enabled
            if 'body_size' not in results and self.use_numba:
                body_features = compute_candle_body_features_numba(opens, highs, lows, closes)
                results['body_size'] = body_features[:len(closes)]
                results['upper_shadow'] = body_features[len(closes):2*len(closes)]
                results['lower_shadow'] = body_features[2*len(closes):3*len(closes)]
                results['relative_close_position'] = body_features[3*len(closes):4*len(closes)]
                logging.debug("Used Numba for candle body features")
            
            # Compute other price features
            if self.use_numba:
                price_features = compute_price_features_numba(opens, highs, lows, closes)
                results.update(price_features)
            else:
                # Standard NumPy implementation as fallback
                n = len(closes)
                
                # Log returns
                log_return = np.zeros(n, dtype=np.float64)
                log_return[1:] = np.log(np.maximum(closes[1:] / np.maximum(closes[:-1], 1e-8), 1e-8))
                results['log_return'] = log_return
                
                # Gap open
                gap_open = np.zeros(n, dtype=np.float64)
                gap_open[1:] = (opens[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0
                results['gap_open'] = gap_open
                
                # Previous close change
                prev_close_change = np.zeros(n, dtype=np.float64)
                prev_close_change[1:] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0
                results['prev_close_change_pct'] = prev_close_change
                
                # Price velocity and acceleration
                price_velocity = np.zeros(n, dtype=np.float64)
                price_accel = np.zeros(n, dtype=np.float64)
                
                for i in range(3, n):
                    price_velocity[i] = (closes[i] / np.maximum(closes[i-3], 1e-8)) - 1.0
                
                for i in range(6, n):
                    price_accel[i] = price_velocity[i] - price_velocity[i-3]
                    
                results['price_velocity'] = price_velocity
                results['price_acceleration'] = price_accel
        except Exception as e:
            logging.error(f"Error computing price action features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("price_action_features", time.time() - start_time)
            
    def _compute_momentum(self, closes, highs, lows, results, perf_monitor=None):
        """Compute momentum features"""
        start_time = time.time()
        
        try:
            # RSI using Numba
            if self.use_numba:
                rsi = compute_rsi_numba(closes, MOMENTUM_PARAMS['rsi_length'])
                results['rsi_1h'] = rsi
                
                # RSI slope
                rsi_slope = np.zeros_like(rsi)
                rsi_slope[3:] = (rsi[3:] - rsi[:-3]) / 3
                results['rsi_slope_1h'] = rsi_slope
                
                # MACD
                macd_result = compute_macd_numba(
                    closes, 
                    MOMENTUM_PARAMS['macd_fast'],
                    MOMENTUM_PARAMS['macd_slow'],
                    MOMENTUM_PARAMS['macd_signal']
                )
                
                # Store MACD slopes
                macd_slope = np.zeros_like(closes)
                macd_slope[1:] = np.diff(macd_result['macd_line'])
                results['macd_slope_1h'] = macd_slope
                
                macd_hist_slope = np.zeros_like(closes)
                macd_hist_slope[1:] = np.diff(macd_result['histogram'])
                results['macd_hist_slope_1h'] = macd_hist_slope
            else:
                # Standard NumPy implementation as fallback
                # RSI
                n = len(closes)
                rsi_length = MOMENTUM_PARAMS['rsi_length']
                
                # Calculate price changes
                changes = np.zeros(n)
                changes[1:] = closes[1:] - closes[:-1]
                
                # Separate gains and losses
                gains = np.where(changes > 0, changes, 0)
                losses = np.where(changes < 0, -changes, 0)
                
                # Calculate average gains and losses
                avg_gain = np.zeros(n)
                avg_loss = np.zeros(n)
                
                # First average
                if n > rsi_length:
                    avg_gain[rsi_length] = np.mean(gains[1:rsi_length+1])
                    avg_loss[rsi_length] = np.mean(losses[1:rsi_length+1])
                    
                    # Rest of averages
                    for i in range(rsi_length+1, n):
                        avg_gain[i] = (avg_gain[i-1] * (rsi_length-1) + gains[i]) / rsi_length
                        avg_loss[i] = (avg_loss[i-1] * (rsi_length-1) + losses[i]) / rsi_length
                
                # Calculate RSI
                rsi = np.zeros(n)
                rsi[:rsi_length] = 50.0  # Default value
                
                # Avoid division by zero
                nonzero_indices = avg_loss[rsi_length:] > 0
                zero_indices = avg_loss[rsi_length:] == 0
                
                rs = np.zeros(n - rsi_length)
                rs[nonzero_indices] = avg_gain[rsi_length:][nonzero_indices] / avg_loss[rsi_length:][nonzero_indices]
                
                rsi[rsi_length:][nonzero_indices] = 100.0 - (100.0 / (1.0 + rs[nonzero_indices]))
                rsi[rsi_length:][zero_indices] = 100.0
                
                results['rsi_1h'] = rsi
                
                # Calculate RSI slope
                rsi_slope = np.zeros(n)
                rsi_slope[3:] = (rsi[3:] - rsi[:-3]) / 3
                results['rsi_slope_1h'] = rsi_slope
                
                # MACD would be implemented here
                # Skipping for brevity
            
            # Additional momentum indicators would be computed here
            # Stochastic, Williams %R, etc.
            
        except Exception as e:
            logging.error(f"Error computing momentum features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("momentum_features", time.time() - start_time)
    
    def _compute_volatility(self, closes, highs, lows, results, perf_monitor=None):
        """Compute volatility features"""
        start_time = time.time()
        
        try:
            n = len(closes)
            atr_length = VOLATILITY_PARAMS['atr_length']
            
            # ATR calculation
            tr = np.zeros(n)
            
            # First true range is high-low
            if n > 0:
                tr[0] = highs[0] - lows[0]
            
            # Rest of true ranges
            for i in range(1, n):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr[i] = max(tr1, tr2, tr3)
            
            # ATR calculation
            atr = np.zeros(n)
            
            # First ATR is first TR
            if n > atr_length:
                atr[atr_length-1] = np.mean(tr[:atr_length])
                
                # Rest use smoothing
                for i in range(atr_length, n):
                    atr[i] = (atr[i-1] * (atr_length-1) + tr[i]) / atr_length
            
            results['atr_1h'] = atr
            results['true_range'] = tr
            
            # Normalized ATR
            normalized_atr = np.zeros(n)
            positive_close = closes > 0
            normalized_atr[positive_close] = atr[positive_close] / closes[positive_close]
            results['normalized_atr_14'] = normalized_atr
            
            # Bollinger Bands
            window = VOLATILITY_PARAMS['bb_length']
            std_dev_multiplier = VOLATILITY_PARAMS['bb_std']
            
            # Calculate SMA
            sma = np.zeros(n)
            
            if n >= window:
                # First SMA
                sma[window-1] = np.mean(closes[:window])
                
                # Rest use sliding window optimization
                for i in range(window, n):
                    sma[i] = sma[i-1] + (closes[i] - closes[i-window]) / window
            
            # Calculate standard deviation
            std_dev = np.zeros(n)
            
            for i in range(window-1, n):
                std_dev[i] = np.std(closes[i-window+1:i+1])
            
            # Calculate Bollinger Bands
            upper_band = sma + (std_dev * std_dev_multiplier)
            lower_band = sma - (std_dev * std_dev_multiplier)
            
            # Bollinger Band width
            bollinger_width = upper_band - lower_band
            results['bollinger_width_1h'] = bollinger_width
            
            # Bollinger Percent B
            band_range = upper_band - lower_band
            percent_b = np.zeros(n)
            nonzero_range = band_range > 0
            percent_b[nonzero_range] = (closes[nonzero_range] - lower_band[nonzero_range]) / band_range[nonzero_range]
            percent_b[~nonzero_range] = 0.5
            results['bollinger_percent_b'] = percent_b
            
            # Donchian Channels
            dc_length = VOLATILITY_PARAMS['donchian_length']
            dc_high = np.zeros(n)
            dc_low = np.zeros(n)
            
            for i in range(dc_length-1, n):
                dc_high[i] = np.max(highs[i-dc_length+1:i+1])
                dc_low[i] = np.min(lows[i-dc_length+1:i+1])
            
            # Fill the beginning values
            for i in range(dc_length-1):
                dc_high[i] = dc_high[dc_length-1]
                dc_low[i] = dc_low[dc_length-1]
            
            # Donchian Channel width
            dc_width = dc_high - dc_low
            results['donchian_channel_width_1h'] = dc_width
            
            # Keltner Channels
            kc_length = VOLATILITY_PARAMS['kc_length']
            kc_scalar = VOLATILITY_PARAMS['kc_scalar']
            
            # EMA calculation for Keltner
            ema = np.zeros(n)
            if n > 0:
                ema[0] = closes[0]
                alpha = 2.0 / (kc_length + 1.0)
                
                for i in range(1, n):
                    ema[i] = alpha * closes[i] + (1 - alpha) * ema[i-1]
            
            # Calculate Keltner Channel width
            kc_width = (ema + atr * kc_scalar) - (ema - atr * kc_scalar)
            results['keltner_channel_width'] = kc_width
            
            # Historical Volatility
            hist_vol_length = VOLATILITY_PARAMS['historical_vol_length']
            returns = np.zeros(n)
            returns[1:] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1
            
            hist_vol = np.zeros(n)
            for i in range(hist_vol_length, n):
                hist_vol[i] = np.std(returns[i-hist_vol_length+1:i+1]) * np.sqrt(252)
            
            results['historical_vol_30'] = hist_vol
            
            # Chaikin Volatility
            cv_length = VOLATILITY_PARAMS['chaikin_volatility_length']
            hl_range = highs - lows
            
            # Calculate EMA of high-low range
            ema_range = np.zeros(n)
            if n > 0:
                ema_range[0] = hl_range[0]
                alpha = 2.0 / (cv_length + 1.0)
                
                for i in range(1, n):
                    ema_range[i] = alpha * hl_range[i] + (1 - alpha) * ema_range[i-1]
            
            # Calculate Chaikin Volatility
            chaikin_vol = np.zeros(n)
            chaikin_vol[1:] = (ema_range[1:] / np.maximum(ema_range[:-1], 1e-8)) - 1
            results['chaikin_volatility'] = chaikin_vol
            
            # Initialize volatility rank (will be updated in cross-pair)
            results['volatility_rank_1h'] = np.zeros(n, dtype=np.int32)
            
        except Exception as e:
            logging.error(f"Error computing volatility features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("volatility_features", time.time() - start_time)
            
    def _compute_volume(self, closes, highs, lows, volumes, results, perf_monitor=None):
        """Compute volume features"""
        start_time = time.time()
        
        try:
            n = len(closes)
            
            # Money Flow Index
            mfi_length = VOLUME_PARAMS['mfi_length']
            
            # Calculate typical price
            typical_price = (highs + lows + closes) / 3
            
            # Calculate money flow
            money_flow = typical_price * volumes
            
            # Separate positive and negative money flow
            pos_flow = np.zeros(n)
            neg_flow = np.zeros(n)
            
            for i in range(1, n):
                if typical_price[i] > typical_price[i-1]:
                    pos_flow[i] = money_flow[i]
                else:
                    neg_flow[i] = money_flow[i]
            
            # Calculate money flow ratio and MFI
            pos_sum = np.zeros(n)
            neg_sum = np.zeros(n)
            
            for i in range(mfi_length, n):
                pos_sum[i] = np.sum(pos_flow[i-mfi_length+1:i+1])
                neg_sum[i] = np.sum(neg_flow[i-mfi_length+1:i+1])
            
            # Calculate MFI
            mfi = np.zeros(n)
            mfi[:] = 50.0  # Default value
            
            # Avoid division by zero
            nonzero_indices = neg_sum[mfi_length:] > 0
            zero_indices = neg_sum[mfi_length:] == 0
            
            mfr = np.zeros(n - mfi_length)
            mfr[nonzero_indices] = pos_sum[mfi_length:][nonzero_indices] / neg_sum[mfi_length:][nonzero_indices]
            
            mfi[mfi_length:][nonzero_indices] = 100.0 - (100.0 / (1.0 + mfr[nonzero_indices]))
            mfi[mfi_length:][zero_indices] = 100.0
            
            results['money_flow_index_1h'] = mfi
            
            # OBV (On-Balance Volume)
            obv = np.zeros(n)
            
            # First OBV is first volume
            if n > 0:
                obv[0] = volumes[0]
                
                # Calculate rest of OBV
                for i in range(1, n):
                    if closes[i] > closes[i-1]:
                        obv[i] = obv[i-1] + volumes[i]
                    elif closes[i] < closes[i-1]:
                        obv[i] = obv[i-1] - volumes[i]
                    else:
                        obv[i] = obv[i-1]
            
            results['obv_1h'] = obv
            
            # OBV slope
            obv_slope = np.zeros(n)
            obv_slope[3:] = (obv[3:] - obv[:-3]) / 3
            results['obv_slope_1h'] = obv_slope
            
            # Volume change percentage
            vol_change = np.zeros(n)
            vol_change[1:] = (volumes[1:] / np.maximum(volumes[:-1], 1e-8)) - 1
            results['volume_change_pct_1h'] = vol_change
            
            # VWMA (Volume Weighted Moving Average)
            vwma_length = VOLUME_PARAMS['vwma_length']
            vwma = np.zeros(n)
            
            for i in range(vwma_length-1, n):
                sum_vol = np.sum(volumes[i-vwma_length+1:i+1])
                if sum_vol > 0:
                    vwma[i] = np.sum(closes[i-vwma_length+1:i+1] * volumes[i-vwma_length+1:i+1]) / sum_vol
                else:
                    vwma[i] = closes[i]
            
            # Fill the beginning values
            for i in range(vwma_length-1):
                vwma[i] = closes[i]
                
            results['vwma_20'] = vwma
            
            # Additional volume indicators would go here
            # Chaikin Money Flow, Klinger Oscillator, etc.
            
            # Initialize volume ranks (will be updated in cross-pair)
            results['volume_rank_1h'] = np.zeros(n, dtype=np.int32)
            results['prev_volume_rank'] = np.zeros(n)
            
        except Exception as e:
            logging.error(f"Error computing volume features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("volume_features", time.time() - start_time)
            
    def _compute_statistical(self, closes, log_returns, results, perf_monitor=None):
        """Compute statistical features"""
        start_time = time.time()
        
        try:
            n = len(closes)
            window_size = STATISTICAL_PARAMS['window_size']
            
            # Standard deviation of returns
            std_dev_returns = np.zeros(n, dtype=np.float64)
            
            for i in range(window_size, n):
                std_dev_returns[i] = np.std(log_returns[i-window_size+1:i+1])
            
            results['std_dev_returns_20'] = std_dev_returns
            
            # Skewness
            skewness = np.zeros(n, dtype=np.float64)
            
            for i in range(window_size, n):
                window = log_returns[i-window_size+1:i+1]
                
                # Need at least 3 points for skewness
                if len(window) >= 3:
                    mean = np.mean(window)
                    std = np.std(window)
                    
                    if std > 0:
                        skew_sum = np.sum(((window - mean) / std) ** 3)
                        skewness[i] = skew_sum / len(window)
            
            results['skewness_20'] = skewness
            
            # Kurtosis
            kurtosis = np.zeros(n, dtype=np.float64)
            
            for i in range(window_size, n):
                window = log_returns[i-window_size+1:i+1]
                
                # Need at least 4 points for kurtosis
                if len(window) >= 4:
                    mean = np.mean(window)
                    std = np.std(window)
                    
                    if std > 0:
                        kurt_sum = np.sum(((window - mean) / std) ** 4)
                        kurtosis[i] = kurt_sum / len(window) - 3  # Excess kurtosis
            
            results['kurtosis_20'] = kurtosis
            
            # Z-score
            z_score_length = STATISTICAL_PARAMS['z_score_length']
            
            # Calculate moving average
            ma = np.zeros(n)
            
            if n >= z_score_length:
                # First moving average
                ma[z_score_length-1] = np.mean(closes[:z_score_length])
                
                # Rest use sliding window optimization
                for i in range(z_score_length, n):
                    ma[i] = ma[i-1] + (closes[i] - closes[i-z_score_length]) / z_score_length
            
            # Use optimized z-score calculation if available
            if self.use_gpu:
                try:
                    results['z_score_20'] = compute_z_score_gpu(closes, ma, z_score_length)
                    logging.debug("Used GPU for z-score calculation")
                except Exception as e:
                    logging.warning(f"GPU calculation failed for z-score: {e}")
                    self.use_gpu = False
            
            if 'z_score_20' not in results and self.use_numba:
                try:
                    results['z_score_20'] = compute_z_score_numba(closes, ma, z_score_length)
                    logging.debug("Used Numba for z-score calculation")
                except Exception as e:
                    logging.warning(f"Numba calculation failed for z-score: {e}")
                
            if 'z_score_20' not in results:
                # Standard numpy implementation
                z_score = np.zeros(n)
                
                for i in range(z_score_length, n):
                    window = closes[i-z_score_length+1:i+1]
                    std = np.std(window)
                    
                    if std > 0:
                        z_score[i] = (closes[i] - ma[i]) / std
                
                results['z_score_20'] = z_score
            
            # Hurst exponent
            hurst_window = STATISTICAL_PARAMS['hurst_window']
            hurst_max_lag = STATISTICAL_PARAMS['hurst_max_lag']
            
            # Calculate Hurst exponent with optimized functions if possible
            if len(closes) >= hurst_window:
                # Make sure prices are positive
                safe_prices = np.maximum(closes, 1e-8)
                
                if self.use_gpu:
                    try:
                        results['hurst_exponent'] = hurst_exponent_gpu(safe_prices, hurst_max_lag)
                        logging.debug("Used GPU for Hurst exponent calculation")
                    except Exception as e:
                        logging.warning(f"GPU calculation failed for Hurst exponent: {e}")
                        self.use_gpu = False
                
                if 'hurst_exponent' not in results and self.use_numba:
                    try:
                        results['hurst_exponent'] = hurst_exponent_numba(safe_prices, hurst_max_lag)
                        logging.debug("Used Numba for Hurst exponent calculation")
                    except Exception as e:
                        logging.warning(f"Numba calculation failed for Hurst exponent: {e}")
                
                if 'hurst_exponent' not in results:
                    # Default to 0.5 - too complex for standard fallback implementation
                    results['hurst_exponent'] = np.ones(n) * 0.5
            else:
                results['hurst_exponent'] = np.ones(n) * 0.5
            
            # Shannon Entropy
            entropy_window = STATISTICAL_PARAMS['entropy_window']
            
            if len(closes) >= entropy_window:
                # Make sure prices are positive
                safe_prices = np.maximum(closes, 1e-8)
                
                if self.use_gpu:
                    try:
                        results['shannon_entropy'] = shannon_entropy_gpu(safe_prices, entropy_window)
                        logging.debug("Used GPU for Shannon entropy calculation")
                    except Exception as e:
                        logging.warning(f"GPU calculation failed for Shannon entropy: {e}")
                        self.use_gpu = False
                
                if 'shannon_entropy' not in results and self.use_numba:
                    try:
                        results['shannon_entropy'] = shannon_entropy_numba(safe_prices, entropy_window)
                        logging.debug("Used Numba for Shannon entropy calculation")
                    except Exception as e:
                        logging.warning(f"Numba calculation failed for Shannon entropy: {e}")
                
                if 'shannon_entropy' not in results:
                    # Default to 0 - too complex for standard fallback implementation
                    results['shannon_entropy'] = np.zeros(n)
            else:
                results['shannon_entropy'] = np.zeros(n)
            
            # Autocorrelation lag 1
            autocorr_lag = STATISTICAL_PARAMS['autocorr_lag']
            autocorr = np.zeros(n)
            
            for i in range(window_size, n):
                window = log_returns[i-window_size+1:i+1]
                
                if len(window) > autocorr_lag:
                    # Calculate mean and variance
                    mean = np.mean(window)
                    var = np.var(window)
                    
                    if var > 0:
                        # Calculate autocovariance
                        autocovariance = 0
                        for j in range(autocorr_lag, len(window)):
                            autocovariance += (window[j] - mean) * (window[j-autocorr_lag] - mean)
                        
                        autocovariance /= (len(window) - autocorr_lag)
                        autocorr[i] = autocovariance / var
            
            results['autocorr_1'] = autocorr
            
            # Estimated slippage and bid-ask spread proxies
            # Remove references to highs and lows that aren't passed to this function
            results['estimated_slippage_1h'] = np.zeros_like(closes)
            results['bid_ask_spread_1h'] = np.zeros_like(closes)
            
        except Exception as e:
            logging.error(f"Error computing statistical features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("statistical_features", time.time() - start_time)
            
    def _compute_pattern(self, opens, highs, lows, closes, results, perf_monitor=None):
        """Compute pattern features"""
        start_time = time.time()
        
        try:
            n = len(closes)
            
            # Doji pattern
            doji_threshold = PATTERN_PARAMS['doji_threshold']
            body_size = np.abs(closes - opens)
            total_size = highs - lows
            
            # Avoid division by zero
            total_size_safe = np.maximum(total_size, 1e-8)
            
            doji = np.zeros(n, dtype=np.int32)
            doji = np.where(body_size / total_size_safe < doji_threshold, 1, 0)
            results['pattern_doji'] = doji
            
            # Engulfing pattern
            engulfing = np.zeros(n, dtype=np.int32)
            
            for i in range(1, n):
                prev_body_low = min(opens[i-1], closes[i-1])
                prev_body_high = max(opens[i-1], closes[i-1])
                curr_body_low = min(opens[i], closes[i])
                curr_body_high = max(opens[i], closes[i])
                
                # Bullish or bearish engulfing
                if ((closes[i] > opens[i]) and 
                    (curr_body_low < prev_body_low) and 
                    (curr_body_high > prev_body_high)):
                    engulfing[i] = 1
                elif ((closes[i] < opens[i]) and 
                      (curr_body_low < prev_body_low) and 
                      (curr_body_high > prev_body_high)):
                    engulfing[i] = 1
            
            results['pattern_engulfing'] = engulfing
            
            # Hammer pattern
            hammer_body_threshold = PATTERN_PARAMS['hammer_body_threshold']
            hammer_shadow_threshold = PATTERN_PARAMS['hammer_shadow_threshold']
            
            hammer = np.zeros(n, dtype=np.int32)
            
            for i in range(n):
                upper_shadow = highs[i] - max(opens[i], closes[i])
                lower_shadow = min(opens[i], closes[i]) - lows[i]
                
                # Avoid division by zero
                body_size_safe = max(body_size[i], 1e-8)
                
                if ((body_size[i] < hammer_body_threshold * total_size[i]) and 
                    (lower_shadow > hammer_shadow_threshold * body_size[i]) and 
                    (upper_shadow < 0.1 * total_size[i])):
                    hammer[i] = 1
            
            results['pattern_hammer'] = hammer
            
            # Morning star - simplified
            morning_star = np.zeros(n, dtype=np.int32)
            
            # Need at least 3 candles for morning star
            for i in range(2, n):
                # Day 1 - bearish with large body
                day1_bearish = closes[i-2] < opens[i-2]
                day1_large_body = body_size[i-2] > 0.5 * total_size[i-2]
                
                # Day 2 - small body
                day2_small_body = body_size[i-1] < 0.3 * body_size[i-2]
                day2_gap_down = highs[i-1] < closes[i-2]
                
                # Day 3 - bullish closing above day 1 midpoint
                day3_bullish = closes[i] > opens[i]
                day3_close_above = closes[i] > (opens[i-2] + closes[i-2]) / 2
                
                if (day1_bearish and day1_large_body and 
                    day2_small_body and day2_gap_down and 
                    day3_bullish and day3_close_above):
                    morning_star[i] = 1
            
            results['pattern_morning_star'] = morning_star
            
            # Support and resistance levels
            sr_length = PATTERN_PARAMS['support_resistance_length']
            
            resistance = np.zeros(n)
            support = np.zeros(n)
            
            for i in range(sr_length, n):
                resistance[i] = np.max(highs[i-sr_length:i])
                support[i] = np.min(lows[i-sr_length:i])
            
            # Fill the beginning values
            for i in range(sr_length):
                if i > 0:
                    resistance[i] = resistance[i-1]
                    support[i] = support[i-1]
                else:
                    resistance[i] = highs[i]
                    support[i] = lows[i]
            
            results['resistance_level'] = resistance
            results['support_level'] = support
            
        except Exception as e:
            logging.error(f"Error computing pattern features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("pattern_features", time.time() - start_time)
            
    def _compute_time(self, timestamps, results, perf_monitor=None):
        """Compute time-based features"""
        start_time = time.time()
        
        try:
            n = len(timestamps)
            
            # Convert timestamps to datetime if necessary
            if isinstance(timestamps[0], (int, float, np.int64, np.float64)):
                # Convert Unix timestamps to datetime64
                dt_timestamps = pd.to_datetime(timestamps, unit='s')
            else:
                dt_timestamps = pd.to_datetime(timestamps)
            
            # Extract time components
            hour = np.array([ts.hour for ts in dt_timestamps], dtype=np.int32)
            day = np.array([ts.dayofweek for ts in dt_timestamps], dtype=np.int32)
            month = np.array([ts.month for ts in dt_timestamps], dtype=np.int32)
            
            # Store time features
            results['hour_of_day'] = hour
            results['day_of_week'] = day
            results['month_of_year'] = month
            
            # Weekend indicator
            is_weekend = np.zeros(n, dtype=np.int32)
            
            # Friday after 9 PM or Saturday/Sunday
            is_weekend = np.where(
                ((day == 4) & (hour >= 21)) | (day >= 5),
                1, 0
            )
            results['is_weekend'] = is_weekend
            
            # Trading sessions
            asian_start = TIME_PARAMS['asian_session_start']
            asian_end = TIME_PARAMS['asian_session_end']
            euro_start = TIME_PARAMS['european_session_start']
            euro_end = TIME_PARAMS['european_session_end']
            amer_start = TIME_PARAMS['american_session_start']
            amer_end = TIME_PARAMS['american_session_end']
            
            # Asian session
            asian_session = np.zeros(n, dtype=np.int32)
            asian_session = np.where(
                (hour >= asian_start) & (hour < asian_end),
                1, 0
            )
            results['asian_session'] = asian_session
            
            # European session
            euro_session = np.zeros(n, dtype=np.int32)
            euro_session = np.where(
                (hour >= euro_start) & (hour < euro_end),
                1, 0
            )
            results['european_session'] = euro_session
            
            # American session
            amer_session = np.zeros(n, dtype=np.int32)
            amer_session = np.where(
                (hour >= amer_start) & (hour < amer_end),
                1, 0
            )
            results['american_session'] = amer_session
            
        except Exception as e:
            logging.error(f"Error computing time features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("time_features", time.time() - start_time)
            
    def _compute_labels(self, closes, highs, lows, results, perf_monitor=None):
        """Compute label/target features"""
        start_time = time.time()
        
        try:
            n = len(closes)
            horizons = LABEL_PARAMS['horizons']
            
            # Calculate future returns for different horizons
            for horizon_name, shift in horizons.items():
                col_name = f'future_return_{horizon_name}_pct'
                
                if self.use_gpu:
                    try:
                        results[col_name] = compute_future_return_gpu(closes, shift)
                        logging.debug(f"Used GPU for {col_name}")
                    except Exception as e:
                        logging.warning(f"GPU calculation failed for {col_name}: {e}")
                        self.use_gpu = False
                
                if col_name not in results and self.use_numba:
                    try:
                        results[col_name] = compute_future_return_numba(closes, shift)
                        logging.debug(f"Used Numba for {col_name}")
                    except Exception as e:
                        logging.warning(f"Numba calculation failed for {col_name}: {e}")
                
                if col_name not in results:
                    # Standard NumPy implementation
                    future_return = np.zeros(n)
                    
                    for i in range(n - shift):
                        if closes[i] > 0:
                            future_return[i] = (closes[i + shift] - closes[i]) / closes[i]
                    
                    results[col_name] = future_return
            
            # Max future return calculation
            max_return_window = LABEL_PARAMS['max_return_window']
            
            if self.use_gpu:
                try:
                    results['future_max_return_24h_pct'] = compute_max_future_return_gpu(
                        closes, highs, max_return_window
                    )
                    logging.debug("Used GPU for max future return")
                except Exception as e:
                    logging.warning(f"GPU calculation failed for max future return: {e}")
                    self.use_gpu = False
            
            if 'future_max_return_24h_pct' not in results and self.use_numba:
                try:
                    results['future_max_return_24h_pct'] = compute_max_future_return_numba(
                        closes, highs, max_return_window
                    )
                    logging.debug("Used Numba for max future return")
                except Exception as e:
                    logging.warning(f"Numba calculation failed for max future return: {e}")
            
            if 'future_max_return_24h_pct' not in results:
                # Standard NumPy implementation
                max_future_return = np.zeros(n)
                
                for i in range(n - 1):
                    end_idx = min(i + max_return_window, n)
                    if i + 1 < end_idx:
                        max_high = np.max(highs[i+1:end_idx])
                        if closes[i] > 0:
                            max_future_return[i] = (max_high - closes[i]) / closes[i]
                
                results['future_max_return_24h_pct'] = max_future_return
            
            # Max future drawdown calculation
            max_drawdown_window = LABEL_PARAMS['max_drawdown_window']
            
            if self.use_gpu:
                try:
                    results['future_max_drawdown_12h_pct'] = compute_max_future_drawdown_gpu(
                        closes, lows, max_drawdown_window
                    )
                    logging.debug("Used GPU for max future drawdown")
                except Exception as e:
                    logging.warning(f"GPU calculation failed for max future drawdown: {e}")
                    self.use_gpu = False
            
            if 'future_max_drawdown_12h_pct' not in results and self.use_numba:
                try:
                    results['future_max_drawdown_12h_pct'] = compute_max_future_drawdown_numba(
                        closes, lows, max_drawdown_window
                    )
                    logging.debug("Used Numba for max future drawdown")
                except Exception as e:
                    logging.warning(f"Numba calculation failed for max future drawdown: {e}")
            
            if 'future_max_drawdown_12h_pct' not in results:
                # Standard NumPy implementation
                max_future_drawdown = np.zeros(n)
                
                for i in range(n - 1):
                    end_idx = min(i + max_drawdown_window, n)
                    if i + 1 < end_idx:
                        min_low = np.min(lows[i+1:end_idx])
                        if closes[i] > 0:
                            max_future_drawdown[i] = (min_low - closes[i]) / closes[i]
                
                results['future_max_drawdown_12h_pct'] = max_future_drawdown
            
            # Was profitable (12h)
            was_profitable = np.zeros(n, dtype=np.int32)
            was_profitable = np.where(results['future_return_12h_pct'] > 0, 1, 0)
            results['was_profitable_12h'] = was_profitable
            
            # Risk-adjusted return
            # Avoid division by zero in drawdown
            safe_drawdown = np.maximum(np.abs(results['future_max_drawdown_12h_pct']), 1e-8)
            risk_adj_return = results['future_return_12h_pct'] / safe_drawdown
            results['future_risk_adj_return_12h'] = risk_adj_return
            
            # Profit target indicators
            profit_target_1 = np.zeros(n, dtype=np.int32)
            profit_target_1 = np.where(results['future_max_return_24h_pct'] > 0.01, 1, 0)
            results['profit_target_1pct'] = profit_target_1
            
            profit_target_2 = np.zeros(n, dtype=np.int32)
            profit_target_2 = np.where(results['future_max_return_24h_pct'] > 0.02, 1, 0)
            results['profit_target_2pct'] = profit_target_2
            
            # BTC correlation (will be set in cross-pair features)
            results['btc_corr_24h'] = np.zeros(n)
            
        except Exception as e:
            logging.error(f"Error computing label features: {e}")
            
        if perf_monitor:
            perf_monitor.log_operation("label_features", time.time() - start_time)