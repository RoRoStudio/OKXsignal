#!/usr/bin/env python3
"""
Volatility Features
- Computes technical indicators related to price volatility
"""

import logging
import time
import numpy as np
import pandas as pd

from database.processing.features.base import BaseFeatureComputer
from database.processing.features.config import VOLATILITY_PARAMS

class VolatilityFeatures(BaseFeatureComputer):
    """Compute volatility-based technical indicators"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute volatility features
        
        Args:
            df: DataFrame with price data
            params: Parameters for volatility features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with volatility features
        """
        start_time = time.time()
        self._debug_log("Computing volatility features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or VOLATILITY_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) < 20:  # Need at least 20 points for volatility indicators
            self._debug_log("Not enough data for volatility features", debug_mode)
            return df
            
        # Extract price data
        close = df['close_1h']
        high = df['high_1h']
        low = df['low_1h']
        
        # ATR (Average True Range)
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            
            # True Range is the max of the three
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # Calculate ATR as the rolling average of True Range
            atr = tr.rolling(window=params['atr_length']).mean()
            df['atr_1h'] = atr.fillna(0)
            
            # Store the True Range as well
            df['true_range'] = tr.fillna(0)
            
            # Normalized ATR (ATR / Close price)
            # Add safety against division by zero
            df['normalized_atr_14'] = np.where(close > 0, df['atr_1h'] / close, 0)
            
        except Exception as e:
            self._handle_exceptions(df, 'atr_1h', 0, "ATR", e)
            df['true_range'] = 0
            df['normalized_atr_14'] = 0
        
        # Bollinger Bands
        try:
            # Calculate middle band (SMA)
            middle_band = close.rolling(window=params['bb_length']).mean()
            
            # Calculate standard deviation
            std_dev = close.rolling(window=params['bb_length']).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * params['bb_std'])
            lower_band = middle_band - (std_dev * params['bb_std'])
            
            # Bollinger Band Width
            df['bollinger_width_1h'] = (upper_band - lower_band).fillna(0)
            
            # Bollinger Percent B
            # Safe division: (close - lower) / (upper - lower)
            band_diff = upper_band - lower_band
            # Handle division by zero
            bb_percent_b = np.where(
                band_diff > 0,
                (close - lower_band) / band_diff,
                0.5  # Default to middle if no range
            )
            df['bollinger_percent_b'] = pd.Series(bb_percent_b, index=close.index).fillna(0.5)
            
        except Exception as e:
            self._handle_exceptions(df, 'bollinger_width_1h', 0, "Bollinger Bands", e)
            df['bollinger_percent_b'] = 0.5
        
        # Donchian Channels
        try:
            dc_high = high.rolling(window=params['donchian_length']).max()
            dc_low = low.rolling(window=params['donchian_length']).min()
            dc_width = (dc_high - dc_low).fillna(0)
            df['donchian_channel_width_1h'] = dc_width
            
        except Exception as e:
            self._handle_exceptions(df, 'donchian_channel_width_1h', 0, "Donchian Channels", e)
        
        # Keltner Channels
        try:
            # Calculate EMA
            ema20 = close.ewm(span=params['kc_length'], adjust=False).mean()
            
            # Calculate Keltner Channel width
            keltner_width = ((ema20 + df['atr_1h'] * params['kc_scalar']) - 
                               (ema20 - df['atr_1h'] * params['kc_scalar']))
            df['keltner_channel_width'] = keltner_width.fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'keltner_channel_width', 0, "Keltner Channels", e)
        
        # Historical Volatility
        try:
            hist_vol = close.pct_change().rolling(window=params['historical_vol_length']).std() * np.sqrt(252)
            df['historical_vol_30'] = hist_vol.fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'historical_vol_30', 0, "Historical Volatility", e)
        
        # Custom Chaikin Volatility Calculation
        try:
            # Calculate the difference between high and low
            hl_range = high - low
            
            # Calculate EMA of high-low range
            ema_range = hl_range.ewm(span=params['chaikin_volatility_length'], adjust=False).mean()
            
            # Calculate volatility as percentage change of EMA range
            volatility = ema_range.pct_change()
            df['chaikin_volatility'] = volatility.fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'chaikin_volatility', 0, "Chaikin Volatility", e)
        
        # Initialize volatility rank (will be updated in cross_pair features)
        df['volatility_rank_1h'] = 0
        
        # Clean any NaN values
        df = self._clean_dataframe(df)
        
        self._log_performance("volatility_features", start_time, perf_monitor)
        return df