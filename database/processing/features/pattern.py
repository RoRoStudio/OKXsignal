#!/usr/bin/env python3
"""
Pattern Features
- Computes candlestick patterns and chart patterns
"""

import logging
import time
import numpy as np
import pandas as pd

from database.processing.features.base import BaseFeatureComputer
from database.processing.features.config import PATTERN_PARAMS

class PatternFeatures(BaseFeatureComputer):
    """Compute candlestick pattern features"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute pattern features
        
        Args:
            df: DataFrame with price data
            params: Parameters for pattern features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with pattern features
        """
        start_time = time.time()
        self._debug_log("Computing pattern features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or PATTERN_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) < 5:  # Need at least a few candles for patterns
            self._debug_log("Not enough data for pattern features", debug_mode)
            return df
            
        # Extract price data
        open_prices = df['open_1h']
        high = df['high_1h']
        low = df['low_1h']
        close = df['close_1h']
        
        # Implement simplified pattern detection
        try:
            # Calculate body size and total candle size for use in multiple patterns
            body_size = np.abs(close - open_prices)
            total_size = high - low
            
            # Doji: open and close are almost equal
            df['pattern_doji'] = ((body_size / total_size.replace(0, np.inf)) < params['doji_threshold']).astype(int)
            
            # Engulfing: current candle's body completely engulfs previous candle's body
            prev_body_low = np.minimum(open_prices.shift(1), close.shift(1))
            prev_body_high = np.maximum(open_prices.shift(1), close.shift(1))
            curr_body_low = np.minimum(open_prices, close)
            curr_body_high = np.maximum(open_prices, close)
            
            bullish_engulfing = (close > open_prices) & (curr_body_low < prev_body_low) & (curr_body_high > prev_body_high)
            bearish_engulfing = (close < open_prices) & (curr_body_low < prev_body_low) & (curr_body_high > prev_body_high)
            df['pattern_engulfing'] = (bullish_engulfing | bearish_engulfing).astype(int)
            
            # Hammer: small body at the top, long lower shadow
            upper_shadow = high - np.maximum(open_prices, close)
            lower_shadow = np.minimum(open_prices, close) - low
            
            # Fix division by zero in body_size
            body_size_safe = body_size.replace(0, np.inf)
            
            df['pattern_hammer'] = (
                (body_size < params['hammer_body_threshold'] * total_size.replace(0, np.inf)) & 
                (lower_shadow > params['hammer_shadow_threshold'] * body_size_safe) & 
                (upper_shadow < 0.1 * total_size.replace(0, np.inf))
            ).astype(int)
            
            # Morning star: simplified version
            # Day 1: Bearish candle with large body
            # Day 2: Small body with gap down
            # Day 3: Bullish candle that closes above the midpoint of Day 1
            
            # Day 1 conditions
            bearish_day1 = (close.shift(2) < open_prices.shift(2)) & (body_size.shift(2) > 0.5 * total_size.shift(2).replace(0, np.inf))
            
            # Day 2 conditions
            small_body_day2 = body_size.shift(1) < 0.3 * body_size.shift(2).replace(0, np.inf)
            gap_down = high.shift(1) < close.shift(2)
            
            # Day 3 conditions
            bullish_day3 = close > open_prices
            closes_above_midpoint = close > (open_prices.shift(2) + close.shift(2)) / 2
            
            # Combined pattern
            df['pattern_morning_star'] = (
                bearish_day1 & small_body_day2 & gap_down & bullish_day3 & closes_above_midpoint
            ).astype(int)
            
        except Exception as e:
            logging.warning(f"Error computing candlestick patterns: {e}")
            df['pattern_doji'] = 0
            df['pattern_engulfing'] = 0
            df['pattern_hammer'] = 0
            df['pattern_morning_star'] = 0
        
        # Convert all to int for smallint columns (ensure integer values)
        for col in ['pattern_doji', 'pattern_engulfing', 'pattern_hammer', 'pattern_morning_star']:
            df[col] = df[col].astype(int)
        
        # Simple support and resistance levels (using rolling max/min)
        try:
            df['resistance_level'] = high.rolling(params['support_resistance_length']).max().ffill().fillna(high)
            df['support_level'] = low.rolling(params['support_resistance_length']).min().ffill().fillna(low)
        except Exception as e:
            logging.warning(f"Error computing support/resistance levels: {e}")
            df['resistance_level'] = high
            df['support_level'] = low
        
        # Clean any NaN values
        df = self._clean_dataframe(df)
        
        self._log_performance("pattern_features", start_time, perf_monitor)
        return df