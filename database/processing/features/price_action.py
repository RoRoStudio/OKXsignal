#!/usr/bin/env python3
"""
Price Action Features
- Computes candle-based features and price movement metrics
"""

import logging
import time
import numpy as np
import pandas as pd

from .base import BaseFeatureComputer
from .optimized.numba_functions import compute_candle_body_features_numba
from .optimized.gpu_functions import compute_candle_body_features_gpu

class PriceActionFeatures(BaseFeatureComputer):
    """Compute price action features including candle characteristics"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute price action features
        
        Args:
            df: DataFrame with price data
            params: Parameters for feature calculation (optional)
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with price action features
        """
        start_time = time.time()
        self._debug_log("Computing price action features...", debug_mode)
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        # Extract price data
        open_prices = df['open_1h'].values
        high_prices = df['high_1h'].values
        low_prices = df['low_1h'].values
        close_prices = df['close_1h'].values
        
        # Use GPU if available and enabled
        if self.use_gpu:
            try:
                body_features = compute_candle_body_features_gpu(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['candle_body_size'] = body_features[0:len(df)]
                df['upper_shadow'] = body_features[len(df):2*len(df)]
                df['lower_shadow'] = body_features[2*len(df):3*len(df)]
                df['relative_close_position'] = body_features[3*len(df):4*len(df)]
            except Exception as e:
                logging.warning(f"GPU calculation failed for candle features: {e}")
                # Fall back to Numba or numpy
                self.use_gpu = False
        
        # Use Numba if enabled
        if not self.use_gpu and self.use_numba:
            try:
                body_features = compute_candle_body_features_numba(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['candle_body_size'] = body_features[0:len(df)]
                df['upper_shadow'] = body_features[len(df):2*len(df)]
                df['lower_shadow'] = body_features[2*len(df):3*len(df)]
                df['relative_close_position'] = body_features[3*len(df):4*len(df)]
            except Exception as e:
                logging.warning(f"Numba calculation failed for candle features: {e}")
                # Fall back to numpy
                self.use_numba = False
        
        # Use NumPy if GPU and Numba are not available or failed
        if not self.use_gpu and not self.use_numba:
            # Calculate candle body features directly
            df['candle_body_size'] = np.abs(df['close_1h'] - df['open_1h'])
            df['upper_shadow'] = df['high_1h'] - np.maximum(df['open_1h'], df['close_1h'])
            df['lower_shadow'] = np.minimum(df['open_1h'], df['close_1h']) - df['low_1h']
            
            # Calculate relative position of close within the high-low range
            hl_range = df['high_1h'] - df['low_1h']
            df['relative_close_position'] = np.where(
                hl_range > 0, 
                (df['close_1h'] - df['low_1h']) / hl_range, 
                0.5  # Default to middle if there's no range
            )
        
        # Calculate log returns
        df['log_return'] = np.log(df['close_1h'] / df['close_1h'].shift(1)).fillna(0)
        
        # Gap open (compared to previous close)
        df['gap_open'] = (df['open_1h'] / df['close_1h'].shift(1) - 1).fillna(0)
        
        # Price velocity (rate of change over time)
        df['price_velocity'] = df['close_1h'].pct_change(3).fillna(0)
        
        # Price acceleration (change in velocity)
        df['price_acceleration'] = df['price_velocity'].diff(3).fillna(0)
        
        # Previous close percent change
        df['prev_close_change_pct'] = df['close_1h'].pct_change().fillna(0)
        
        # Clean any NaN values
        df = self._clean_dataframe(df)
        
        self._log_performance("price_action_features", start_time, perf_monitor)
        return df