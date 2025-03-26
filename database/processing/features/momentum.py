#!/usr/bin/env python3
"""
Momentum Features
- Computes technical indicators related to price momentum
"""

import logging
import time
import numpy as np
import pandas as pd

from database.processing.features.base import BaseFeatureComputer
from database.processing.features.config import MOMENTUM_PARAMS

class MomentumFeatures(BaseFeatureComputer):
    """Compute momentum-based technical indicators"""
    
    def __init__(self, use_numba=True, use_gpu=False):
        """
        Initialize momentum features computer
        
        Args:
            use_numba: Whether to use Numba optimizations
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__(use_numba, use_gpu)
        
    def compute_tsi(self, close, fast=13, slow=25):
        """
        Custom True Strength Index calculation
        
        Args:
            close: Series of closing prices
            fast: Fast period
            slow: Slow period
            
        Returns:
            Series with TSI values
        """
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
        """
        Custom Percentage Price Oscillator calculation
        
        Args:
            close: Series of closing prices
            fast: Fast period
            slow: Slow period
            signal: Signal period
            
        Returns:
            Series with PPO values
        """
        # Calculate EMAs
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        # Avoid division by zero
        ema_slow = ema_slow.replace(0, np.nan)
        
        # Calculate PPO
        ppo = 100 * ((ema_fast - ema_slow) / ema_slow)
        
        return ppo.fillna(0)
        
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute momentum features
        
        Args:
            df: DataFrame with price data
            params: Parameters for momentum features (optional)
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with momentum features
        """
        start_time = time.time()
        self._debug_log("Computing momentum features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or MOMENTUM_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) < 14:  # Need at least 14 points for most indicators
            self._debug_log("Not enough data for momentum features", debug_mode)
            return df
            
        # Extract price data
        close = df['close_1h']
        high = df['high_1h']
        low = df['low_1h']
        open_prices = df['open_1h']
        
        # RSI
        try:
            # Simple numpy implementation
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=params['rsi_length']).mean()
            avg_loss = loss.rolling(window=params['rsi_length']).mean()
            
            # For values beyond the initial window
            for i in range(params['rsi_length'], len(delta)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (params['rsi_length']-1) + gain.iloc[i]) / params['rsi_length']
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (params['rsi_length']-1) + loss.iloc[i]) / params['rsi_length']
            
            # Avoid division by zero
            rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
            rsi = 100 - (100 / (1 + rs))
            
            df['rsi_1h'] = rsi
            
            # Calculate RSI slope
            df['rsi_slope_1h'] = df['rsi_1h'].diff(3) / 3
            
        except Exception as e:
            return self._handle_exceptions(df, 'rsi_1h', 50, "RSI", e)
        
        # MACD
        try:
            ema_fast = close.ewm(span=params['macd_fast'], adjust=False).mean()
            ema_slow = close.ewm(span=params['macd_slow'], adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=params['macd_signal'], adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            # Slopes rather than raw values
            df['macd_slope_1h'] = macd_line.diff().fillna(0)
            df['macd_hist_slope_1h'] = macd_hist.diff().fillna(0)
            
        except Exception as e:
            df['macd_slope_1h'] = 0
            df['macd_hist_slope_1h'] = 0
            logging.warning(f"Error computing MACD: {e}")
        
        # Stochastic Oscillator
        try:
            rolling_low = low.rolling(window=params['stoch_k']).min()
            rolling_high = high.rolling(window=params['stoch_k']).max()
            
            # Avoid division by zero
            high_low_range = rolling_high - rolling_low
            high_low_range = high_low_range.replace(0, np.nan)
            
            k = 100 * ((close - rolling_low) / high_low_range)
            k = k.fillna(50)
            
            d = k.rolling(window=params['stoch_d']).mean().fillna(50)
            
            df['stoch_k_14'] = k
            df['stoch_d_14'] = d
            
        except Exception as e:
            return self._handle_exceptions(df, 'stoch_k_14', 50, "Stochastic", e)
        
        # Williams %R
        try:
            highest_high = high.rolling(window=14).max()
            lowest_low = low.rolling(window=14).min()
            
            # Avoid division by zero
            range_hl = highest_high - lowest_low
            range_hl = range_hl.replace(0, np.nan)
            
            wr = -100 * ((highest_high - close) / range_hl)
            df['williams_r_14'] = wr.fillna(-50)
            
        except Exception as e:
            return self._handle_exceptions(df, 'williams_r_14', -50, "Williams %R", e)
        
        # CCI (Commodity Channel Index)
        try:
            typical_price = (high + low + close) / 3
            mean_dev = np.abs(typical_price - typical_price.rolling(window=params['cci_length']).mean())
            mean_dev = mean_dev.rolling(window=params['cci_length']).mean()
            
            # Avoid division by zero
            mean_dev = mean_dev.replace(0, np.nan)
            
            cci = (typical_price - typical_price.rolling(window=params['cci_length']).mean()) / (0.015 * mean_dev)
            df['cci_14'] = cci.fillna(0)
            
        except Exception as e:
            return self._handle_exceptions(df, 'cci_14', 0, "CCI", e)
        
        # ROC (Rate of Change)
        try:
            roc = 100 * (close / close.shift(params['roc_length']) - 1)
            df['roc_10'] = roc.fillna(0)
            
        except Exception as e:
            return self._handle_exceptions(df, 'roc_10', 0, "ROC", e)
        
        # TSI (True Strength Index)
        try:
            tsi = self.compute_tsi(close, params['tsi_fast'], params['tsi_slow'])
            df['tsi'] = tsi
            
        except Exception as e:
            return self._handle_exceptions(df, 'tsi', 0, "TSI", e)
        
        # Awesome Oscillator
        try:
            median_price = (high + low) / 2
            ao = (median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean())
            df['awesome_oscillator'] = ao.fillna(0)
            
        except Exception as e:
            return self._handle_exceptions(df, 'awesome_oscillator', 0, "Awesome Oscillator", e)
        
        # PPO (Percentage Price Oscillator)
        try:
            ppo = self.compute_ppo(close, params['ppo_fast'], params['ppo_slow'])
            df['ppo'] = ppo
            
        except Exception as e:
            return self._handle_exceptions(df, 'ppo', 0, "PPO", e)
        
        # Clean any NaN values
        df = self._clean_dataframe(df)
        
        self._log_performance("momentum_features", start_time, perf_monitor)
        return df