#!/usr/bin/env python3
"""
Multi-Timeframe Features
- Computes features for higher timeframes (4h, 1d) from 1h data
"""

import logging
import time
import numpy as np
import pandas as pd

from .base import BaseFeatureComputer
from .config import MULTI_TIMEFRAME_PARAMS

class MultiTimeframeFeatures(BaseFeatureComputer):
    """Compute multi-timeframe features"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute multi-timeframe features
        
        Args:
            df: DataFrame with price/volume data
            params: Parameters for multi-timeframe features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with multi-timeframe features
        """
        start_time = time.time()
        self._debug_log("Computing multi-timeframe features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or MULTI_TIMEFRAME_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) < 24:  # Need at least a day of data for 4h timeframe
            self._debug_log("Not enough data for multi-timeframe features", debug_mode)
            return df
            
        # Calculate 4h timeframe features
        df = self.resample_and_compute('4h', df, params, debug_mode, perf_monitor)
        
        # Calculate 1d timeframe features
        df = self.resample_and_compute('1d', df, params, debug_mode, perf_monitor)
        
        self._log_performance("multi_timeframe_features_total", start_time, perf_monitor)
        return df
        
    def resample_and_compute(self, timeframe, df, params, debug_mode=False, perf_monitor=None):
        """
        Resample to the given timeframe and compute features
        
        Args:
            timeframe: Timeframe to resample to ('4h' or '1d')
            df: DataFrame with price/volume data
            params: Parameters for resampling
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with features for the specified timeframe
        """
        resample_start = time.time()
        
        tf_label = timeframe  # '4h' or '1d'
        resample_rule = params['resample_rules'].get(timeframe, timeframe)
        min_points = params['min_points'].get(timeframe, 20)
        
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Define expected column names for this timeframe
        expected_columns = [
            f'rsi_{tf_label}', f'rsi_slope_{tf_label}', f'macd_slope_{tf_label}',
            f'macd_hist_slope_{tf_label}', f'atr_{tf_label}', f'bollinger_width_{tf_label}',
            f'donchian_channel_width_{tf_label}', f'money_flow_index_{tf_label}', 
            f'obv_slope_{tf_label}', f'volume_change_pct_{tf_label}'
        ]
        
        # Initialize all expected columns with default value 0
        for col_name in expected_columns:
            result_df[col_name] = 0.0
        
        if len(df) < min_points:
            self._debug_log(f"Not enough data for {tf_label} features (need >= {min_points})", debug_mode)
            return result_df
            
        try:
            # Make a temporary copy with the timestamp as index for resampling
            df_with_ts = df.copy()
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df_with_ts['timestamp_utc']):
                df_with_ts['timestamp_utc'] = pd.to_datetime(df_with_ts['timestamp_utc'])
                
            # Set timestamp as index for resampling
            df_with_ts = df_with_ts.set_index('timestamp_utc')
            
            # Perform resampling
            resampled = pd.DataFrame()
            resampled[f'open_{tf_label}'] = df_with_ts['open_1h'].resample(resample_rule).first()
            resampled[f'high_{tf_label}'] = df_with_ts['high_1h'].resample(resample_rule).max()
            resampled[f'low_{tf_label}'] = df_with_ts['low_1h'].resample(resample_rule).min()
            resampled[f'close_{tf_label}'] = df_with_ts['close_1h'].resample(resample_rule).last()
            resampled[f'volume_{tf_label}'] = df_with_ts['volume_1h'].resample(resample_rule).sum()
            
            # Drop rows with missing values
            resampled.dropna(inplace=True)
            
            if len(resampled) < 5:  # Need minimum data for indicators
                self._debug_log(f"Not enough resampled data for {tf_label} features", debug_mode)
                return result_df
                
            self._debug_log(f"{tf_label} resampling completed with {len(resampled)} rows", debug_mode)
            
            # Compute indicators on resampled data
            # RSI
            delta = resampled[f'close_{tf_label}'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # For values beyond the initial window
            for i in range(14, len(delta)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
            
            # Avoid division by zero
            rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
            rsi = 100 - (100 / (1 + rs))
            
            resampled[f'rsi_{tf_label}'] = pd.Series(rsi, index=resampled.index).fillna(50)
            
            # RSI slope
            resampled[f'rsi_slope_{tf_label}'] = resampled[f'rsi_{tf_label}'].diff(2) / 2
            
            # MACD
            ema_fast = resampled[f'close_{tf_label}'].ewm(span=12, adjust=False).mean()
            ema_slow = resampled[f'close_{tf_label}'].ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            resampled[f'macd_{tf_label}'] = macd_line.fillna(0)
            resampled[f'macd_signal_{tf_label}'] = signal_line.fillna(0)
            resampled[f'macd_slope_{tf_label}'] = macd_line.diff().fillna(0)
            resampled[f'macd_hist_{tf_label}'] = macd_hist.fillna(0)
            resampled[f'macd_hist_slope_{tf_label}'] = macd_hist.diff().fillna(0)
            
            # ATR
            high_series = resampled[f'high_{tf_label}']
            low_series = resampled[f'low_{tf_label}']
            close_series = resampled[f'close_{tf_label}']
            
            # Calculate True Range
            tr1 = high_series - low_series
            tr2 = np.abs(high_series - close_series.shift(1))
            tr3 = np.abs(low_series - close_series.shift(1))
            
            # True Range is the max of the three
            tr = pd.DataFrame({
                'tr1': tr1, 
                'tr2': tr2, 
                'tr3': tr3
            }).max(axis=1)
            
            # Calculate ATR as the rolling average of True Range
            resampled[f'atr_{tf_label}'] = tr.rolling(window=14).mean().fillna(0)
            
            # Bollinger Bands
            middle_band = resampled[f'close_{tf_label}'].rolling(window=20).mean()
            std_dev = resampled[f'close_{tf_label}'].rolling(window=20).std()
            
            upper_band = middle_band + (std_dev * 2)
            lower_band = middle_band - (std_dev * 2)
            
            resampled[f'bollinger_width_{tf_label}'] = (upper_band - lower_band).fillna(0)
            
            # Donchian Channels
            resampled[f'donchian_high_{tf_label}'] = high_series.rolling(window=20).max().fillna(high_series)
            resampled[f'donchian_low_{tf_label}'] = low_series.rolling(window=20).min().fillna(low_series)
            resampled[f'donchian_channel_width_{tf_label}'] = (
                resampled[f'donchian_high_{tf_label}'] - resampled[f'donchian_low_{tf_label}']
            ).fillna(0)
            
            # MFI
            typical_price = (high_series + low_series + close_series) / 3
            money_flow = typical_price * resampled[f'volume_{tf_label}']
            
            pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
            neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
            
            pos_sum = pd.Series(pos_flow).rolling(window=14).sum()
            neg_sum = pd.Series(neg_flow).rolling(window=14).sum()
            
            # Avoid division by zero
            mfr = np.where(neg_sum > 0, pos_sum / neg_sum, 100)
            mfi = 100 - (100 / (1 + mfr))
            
            resampled[f'money_flow_index_{tf_label}'] = pd.Series(mfi, index=resampled.index).fillna(50)
            
            # OBV and slope
            obv = np.zeros(len(resampled))
            
            for i in range(1, len(resampled)):
                if close_series.iloc[i] > close_series.iloc[i-1]:
                    obv[i] = obv[i-1] + resampled[f'volume_{tf_label}'].iloc[i]
                elif close_series.iloc[i] < close_series.iloc[i-1]:
                    obv[i] = obv[i-1] - resampled[f'volume_{tf_label}'].iloc[i]
                else:
                    obv[i] = obv[i-1]
                    
            resampled[f'obv_{tf_label}'] = obv
            resampled[f'obv_slope_{tf_label}'] = pd.Series(obv).diff(2) / 2
            
            # Volume change
            resampled[f'volume_change_pct_{tf_label}'] = resampled[f'volume_{tf_label}'].pct_change().fillna(0)
            
            # Clean up any NaN/Inf values
            for col in resampled.columns:
                resampled[col] = resampled[col].replace([np.inf, -np.inf], 0).fillna(0)
                
            # Map resampled values back to original timeframe using forward fill
            all_timestamps = df_with_ts.index
            
            # Create a mapping dictionary for efficiency
            resampled_dict = {}
            for col in expected_columns:
                if col in resampled.columns:
                    # Use efficient reindex with forward fill
                    temp_series = resampled[col].reindex(
                        all_timestamps, 
                        method='ffill'
                    )
                    resampled_dict[col] = temp_series
            
            # Update the result dataframe
            for col, series in resampled_dict.items():
                result_df[col] = series.values
                
        except Exception as e:
            logging.warning(f"Error computing {tf_label} features: {e}")
            # Keep the initialized zero values for all expected columns
        
        self._log_performance(f"compute_{tf_label}_features", time.time() - resample_start, perf_monitor)
        return result_df