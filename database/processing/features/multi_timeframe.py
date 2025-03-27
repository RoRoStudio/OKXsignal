#!/usr/bin/env python3
"""
Multi-Timeframe Features
- Computes features for higher timeframes (4h, 1d) from 1h data
- Enhanced error handling and diagnostic capabilities
- Improved resampling and mapping logic
"""

import logging
import time
import numpy as np
import pandas as pd
import traceback

from database.processing.features.base import BaseFeatureComputer
from database.processing.features.config import MULTI_TIMEFRAME_PARAMS

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
        
        # Check data quality first
        if 'timestamp_utc' not in df.columns:
            logging.warning("Missing timestamp_utc column in input data, skipping multi-timeframe features")
            return df
            
        # Ensure there are enough data points
        if len(df) < 24:  # Need at least a day of data for 4h timeframe
            self._debug_log(f"Not enough data points ({len(df)}) for multi-timeframe features, need at least 24", debug_mode)
            return df
            
        # Calculate 4h timeframe features
        try:
            df = self.resample_and_compute('4h', df, params, debug_mode, perf_monitor)
        except Exception as e:
            logging.error(f"Error computing 4h features: {e}")
            if debug_mode:
                logging.error(traceback.format_exc())
        
        # Calculate 1d timeframe features
        try:
            df = self.resample_and_compute('1d', df, params, debug_mode, perf_monitor)
        except Exception as e:
            logging.error(f"Error computing 1d features: {e}")
            if debug_mode:
                logging.error(traceback.format_exc())
        
        # Ensure all multi-timeframe columns are properly typed as floats
        self._ensure_feature_types(df)
        
        # FIX: Use a reasonable duration for performance monitoring
        duration = time.time() - start_time
        if perf_monitor:
            # Add sanity check for timing
            if duration > 1000:  # If over 1000 seconds, likely a bug
                logging.warning(f"Unusually long duration for multi_timeframe_features_total: {duration}s, capping at 1000s")
                duration = 1000.0
            perf_monitor.log_operation("multi_timeframe_features_total", duration)
            
        return df
    
    def _ensure_feature_types(self, df):
        """Ensure all multi-timeframe feature columns are properly typed as floats"""
        multi_tf_columns = [col for col in df.columns if ('_4h' in col or '_1d' in col)]
        
        for col in multi_tf_columns:
            # Convert to float
            try:
                df[col] = df[col].astype(float)
            except Exception:
                # If conversion fails, replace with 0.0
                logging.warning(f"Failed to convert {col} to float, replacing with 0.0")
                df[col] = 0.0
                
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
        
        # Define all timeframe-specific column names we need to compute
        expected_columns = [
            f'rsi_{tf_label}', f'rsi_slope_{tf_label}', f'macd_slope_{tf_label}',
            f'macd_hist_slope_{tf_label}', f'atr_{tf_label}', f'bollinger_width_{tf_label}',
            f'donchian_channel_width_{tf_label}', f'money_flow_index_{tf_label}', 
            f'obv_slope_{tf_label}', f'volume_change_pct_{tf_label}'
        ]
        
        # Initialize all expected columns with default value 0.0
        for col_name in expected_columns:
            result_df[col_name] = 0.0
        
        # Check if we have enough data based on requested timeframe
        if len(df) < min_points:
            self._debug_log(f"Not enough data for {tf_label} features (need >= {min_points})", debug_mode)
            return result_df
            
        try:
            # FIX: Add improved error handling and diagnostics
            self._debug_log(f"Starting computation for {tf_label} features with {len(df)} rows", debug_mode)
            
            # Make a temporary copy with the timestamp as index for resampling
            df_with_ts = df.copy()
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df_with_ts['timestamp_utc']):
                df_with_ts['timestamp_utc'] = pd.to_datetime(df_with_ts['timestamp_utc'])
            
            # Verify timestamps are valid and handle any issues
            invalid_timestamps = df_with_ts['timestamp_utc'].isnull().sum()
            if invalid_timestamps > 0:
                self._debug_log(f"Found {invalid_timestamps} invalid timestamps for {tf_label}", debug_mode)
                # Filter out invalid timestamps
                df_with_ts = df_with_ts[df_with_ts['timestamp_utc'].notnull()]
                
            # Sort by timestamp to ensure correct order for resampling
            df_with_ts = df_with_ts.sort_values('timestamp_utc')
                
            # Set timestamp as index for resampling
            df_with_ts = df_with_ts.set_index('timestamp_utc')
            
            # Verify data quality before resampling
            for col in ['open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h']:
                if col not in df_with_ts.columns:
                    raise ValueError(f"Required column {col} missing for {tf_label} resampling")
                if df_with_ts[col].isnull().any():
                    # Log warning and fill NaN values
                    null_count = df_with_ts[col].isnull().sum()
                    self._debug_log(f"Found {null_count} null values in {col} for {tf_label}", debug_mode)
                    # Fill NaN values appropriately
                    if col == 'volume_1h':
                        df_with_ts[col] = df_with_ts[col].fillna(0)
                    else:
                        df_with_ts[col] = df_with_ts[col].fillna(method='ffill').fillna(method='bfill')
            
            # Log info about input data
            self._debug_log(f"Resampling {len(df_with_ts)} rows from 1h to {tf_label}", debug_mode)
            self._debug_log(f"Timestamp range: {df_with_ts.index.min()} to {df_with_ts.index.max()}", debug_mode)
            
            # Perform resampling
            resampled = pd.DataFrame()
            resampled[f'open_{tf_label}'] = df_with_ts['open_1h'].resample(resample_rule).first()
            resampled[f'high_{tf_label}'] = df_with_ts['high_1h'].resample(resample_rule).max()
            resampled[f'low_{tf_label}'] = df_with_ts['low_1h'].resample(resample_rule).min()
            resampled[f'close_{tf_label}'] = df_with_ts['close_1h'].resample(resample_rule).last()
            resampled[f'volume_{tf_label}'] = df_with_ts['volume_1h'].resample(resample_rule).sum()
            
            # Fill forward any missing values (incomplete periods)
            resampled = resampled.fillna(method='ffill')
            
            # Check for NaN values after resampling
            null_counts = resampled.isnull().sum()
            if null_counts.sum() > 0:
                self._debug_log(f"Null values after {tf_label} resampling: {null_counts.to_dict()}", debug_mode)
                # Fill remaining NaN values
                resampled = resampled.fillna(method='bfill').fillna(0)
            
            # Make sure we have enough data after resampling
            if len(resampled) < 20:  # Need at least 20 points for most indicators
                self._debug_log(f"Not enough resampled data ({len(resampled)} rows) for {tf_label} features", debug_mode)
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
            
            # Calculate RS with proper handling of zero avg_loss
            rs = np.zeros(len(avg_gain))
            for i in range(len(avg_gain)):
                if avg_loss.iloc[i] == 0:
                    rs[i] = 100  # If avg_loss is zero, RS is max
                else:
                    rs[i] = avg_gain.iloc[i] / avg_loss.iloc[i]
            
            # Calculate RSI
            rsi = np.zeros(len(rs))
            for i in range(len(rs)):
                rsi[i] = 100 - (100 / (1 + rs[i]))
            
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
            tr2 = np.abs(high_series - close_series.shift(1).fillna(high_series))
            tr3 = np.abs(low_series - close_series.shift(1).fillna(low_series))
            
            # True Range is the max of the three
            tr = pd.DataFrame({
                'tr1': tr1, 
                'tr2': tr2, 
                'tr3': tr3
            }).max(axis=1)
            
            # Calculate ATR as the rolling average of True Range
            atr = tr.rolling(window=14, min_periods=1).mean().fillna(tr)
            resampled[f'atr_{tf_label}'] = atr
            
            # Bollinger Bands
            rolling_mean = resampled[f'close_{tf_label}'].rolling(window=20, min_periods=1).mean()
            rolling_std = resampled[f'close_{tf_label}'].rolling(window=20, min_periods=1).std()
            
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            
            width = upper_band - lower_band
            resampled[f'bollinger_width_{tf_label}'] = width.fillna(0)
            
            # Donchian Channels
            rolling_high = high_series.rolling(window=20, min_periods=1).max()
            rolling_low = low_series.rolling(window=20, min_periods=1).min()
            
            resampled[f'donchian_high_{tf_label}'] = rolling_high.fillna(high_series)
            resampled[f'donchian_low_{tf_label}'] = rolling_low.fillna(low_series)
            resampled[f'donchian_channel_width_{tf_label}'] = (
                resampled[f'donchian_high_{tf_label}'] - resampled[f'donchian_low_{tf_label}']
            )
            
            # MFI (Money Flow Index)
            typical_price = (high_series + low_series + close_series) / 3
            money_flow = typical_price * resampled[f'volume_{tf_label}']
            
            # Calculate positive and negative money flow
            pos_flow = np.zeros(len(money_flow))
            neg_flow = np.zeros(len(money_flow))
            
            # First value has no previous value to compare
            pos_flow[0] = money_flow.iloc[0]
            
            for i in range(1, len(money_flow)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    pos_flow[i] = money_flow.iloc[i]
                else:
                    neg_flow[i] = money_flow.iloc[i]
            
            # Calculate money flow ratio
            pos_sum = pd.Series(pos_flow).rolling(window=14, min_periods=1).sum()
            neg_sum = pd.Series(neg_flow).rolling(window=14, min_periods=1).sum()
            
            # Calculate MFI
            mfi = np.zeros(len(pos_sum))
            
            for i in range(len(pos_sum)):
                if neg_sum.iloc[i] == 0:
                    mfi[i] = 100  # All money flow is positive
                else:
                    money_ratio = pos_sum.iloc[i] / neg_sum.iloc[i]
                    mfi[i] = 100 - (100 / (1 + money_ratio))
            
            resampled[f'money_flow_index_{tf_label}'] = pd.Series(mfi, index=resampled.index).fillna(50)
            
            # OBV and slope
            obv = np.zeros(len(resampled))
            
            # First OBV is first volume
            obv[0] = resampled[f'volume_{tf_label}'].iloc[0]
            
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
            vol_change = np.zeros(len(resampled))
            
            for i in range(1, len(resampled)):
                if resampled[f'volume_{tf_label}'].iloc[i-1] > 0:
                    vol_change[i] = (resampled[f'volume_{tf_label}'].iloc[i] / resampled[f'volume_{tf_label}'].iloc[i-1]) - 1
            
            resampled[f'volume_change_pct_{tf_label}'] = vol_change
            
            # Clean up any NaN/Inf values
            for col in resampled.columns:
                resampled[col] = resampled[col].replace([np.inf, -np.inf], 0).fillna(0)
                
            # Check for any remaining NaN or inf values
            null_counts = resampled.isnull().sum()
            if null_counts.sum() > 0:
                self._debug_log(f"Null values remain after cleaning {tf_label} features: {null_counts.to_dict()}", debug_mode)
                # Force all values to be valid
                resampled = resampled.fillna(0).replace([np.inf, -np.inf], 0)
            
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
                    ).fillna(0)  # Fill any remaining NaN values
                    resampled_dict[col] = temp_series
            
            # Update the result dataframe
            for col, series in resampled_dict.items():
                result_df[col] = series.values
                
            # Log successful computation
            self._debug_log(f"Successfully computed {len(expected_columns)} features for {tf_label}", debug_mode)
            
        except Exception as e:
            logging.error(f"Error computing {tf_label} features: {e}")
            if debug_mode:
                logging.error(traceback.format_exc())
            # Keep the initialized zero values for all expected columns
        
        # FIX: Use a reasonable duration for performance monitoring
        duration = time.time() - resample_start
        if perf_monitor:
            # Add sanity check for timing
            if duration > 1000:  # If over 1000 seconds, likely a bug
                logging.warning(f"Unusually long duration for compute_{tf_label}_features: {duration}s, capping at 1000s")
                duration = 1000.0
            perf_monitor.log_operation(f"compute_{tf_label}_features", duration)
        
        # Validate all expected columns are present and numeric
        for col in expected_columns:
            if col not in result_df.columns:
                logging.warning(f"Expected column {col} missing after {tf_label} feature computation")
                result_df[col] = 0.0
            else:
                # Force column to be numeric
                try:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
                except Exception as e:
                    logging.warning(f"Error converting {col} to numeric: {e}")
                    result_df[col] = 0.0
        
        return result_df