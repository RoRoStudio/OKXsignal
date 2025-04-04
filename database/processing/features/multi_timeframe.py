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
            
        # Process 4h and 1d timeframes in parallel
        result_df = df.copy()
        
        # Prepare DataFrame with timestamp as index for resampling
        df_with_ts = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_with_ts['timestamp_utc']):
            df_with_ts['timestamp_utc'] = pd.to_datetime(df_with_ts['timestamp_utc'])
        
        # Sort for consistency
        df_with_ts = df_with_ts.sort_values('timestamp_utc')
        
        # Set timestamp as index
        df_with_ts = df_with_ts.set_index('timestamp_utc')
        
        # Define mapping for OHLCV aggregation
        ohlcv_columns = {
            'open_1h': 'first',
            'high_1h': 'max',
            'low_1h': 'min',
            'close_1h': 'last',
            'volume_1h': 'sum'
        }
        
        # Start with 4h timeframe computation
        start_4h = time.time()
        try:
            # Resample to 4h
            resampled_4h = df_with_ts[list(ohlcv_columns.keys())].resample('4H').agg(ohlcv_columns)
            
            # Rename columns for clarity
            resampled_4h.columns = [col.replace('1h', '4h') for col in resampled_4h.columns]
            
            # Compute all indicators on resampled data
            if len(resampled_4h) >= 10:  # Need at least 10 points for indicators
                # 1. RSI
                close = resampled_4h['close_4h']
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # For values beyond the initial window
                for i in range(14, len(delta)):
                    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
                    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
                
                rs = np.zeros(len(avg_gain))
                rs_mask = avg_loss > 0
                rs[rs_mask] = avg_gain[rs_mask] / avg_loss[rs_mask]
                rs[~rs_mask] = 100
                
                rsi = 100 - (100 / (1 + rs))
                resampled_4h['rsi_4h'] = pd.Series(rsi, index=resampled_4h.index).fillna(50)
                
                # RSI slope
                resampled_4h['rsi_slope_4h'] = resampled_4h['rsi_4h'].diff(2) / 2
                
                # 2. MACD
                ema_fast = close.ewm(span=12, adjust=False).mean()
                ema_slow = close.ewm(span=26, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = macd_line - signal_line
                
                resampled_4h['macd_slope_4h'] = macd_line.diff().fillna(0)
                resampled_4h['macd_hist_slope_4h'] = macd_hist.diff().fillna(0)
                
                # 3. ATR
                high = resampled_4h['high_4h']
                low = resampled_4h['low_4h']
                
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                
                # Use exponential moving average for smoother ATR
                resampled_4h['atr_4h'] = tr.ewm(span=14, adjust=False).mean().fillna(tr)
                
                # 4. Bollinger Bands
                rolling_mean = close.rolling(window=20).mean()
                rolling_std = close.rolling(window=20).std()
                
                upper_band = rolling_mean + (rolling_std * 2)
                lower_band = rolling_mean - (rolling_std * 2)
                
                resampled_4h['bollinger_width_4h'] = (upper_band - lower_band).fillna(0)
                
                # 5. Donchian Channel
                resampled_4h['donchian_channel_width_4h'] = (
                    high.rolling(window=20).max() - low.rolling(window=20).min()
                ).fillna(0)
                
                # 6. MFI
                typical_price = (high + low + close) / 3
                
                money_flow = typical_price * resampled_4h['volume_4h']
                
                # Get positive and negative money flow
                pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
                neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
                
                # Calculate money flow ratio
                pos_sum = pd.Series(pos_flow).rolling(window=14).sum()
                neg_sum = pd.Series(neg_flow).rolling(window=14).sum()
                
                # Avoid division by zero
                mfr = np.zeros(len(pos_sum))
                mask = neg_sum > 0
                mfr[mask] = pos_sum[mask] / neg_sum[mask]
                mfr[~mask] = 100
                
                # Calculate Money Flow Index
                mfi = 100 - (100 / (1 + mfr))
                resampled_4h['money_flow_index_4h'] = pd.Series(mfi, index=resampled_4h.index).fillna(50)
                
                # 7. OBV and slope
                obv = np.zeros(len(resampled_4h))
                
                for i in range(1, len(resampled_4h)):
                    if close.iloc[i] > close.iloc[i-1]:
                        obv[i] = obv[i-1] + resampled_4h['volume_4h'].iloc[i]
                    elif close.iloc[i] < close.iloc[i-1]:
                        obv[i] = obv[i-1] - resampled_4h['volume_4h'].iloc[i]
                    else:
                        obv[i] = obv[i-1]
                
                resampled_4h['obv_slope_4h'] = pd.Series(obv).diff(2) / 2
                
                # 8. Volume change
                resampled_4h['volume_change_pct_4h'] = resampled_4h['volume_4h'].pct_change().fillna(0)
                
                # Ensure all columns are proper floats
                for col in resampled_4h.columns:
                    if col not in ohlcv_columns and col != 'timestamp_utc':
                        resampled_4h[col] = resampled_4h[col].astype(float)
                
                # Map back to original timeframe
                for col in resampled_4h.columns:
                    if col not in ohlcv_columns and col != 'timestamp_utc':
                        # Forward fill resampled values to 1h data
                        series_4h = resampled_4h[col]
                        result_df[col] = series_4h.reindex(
                            df_with_ts.index, method='ffill'
                        ).fillna(0).values
            
            if perf_monitor:
                perf_monitor.log_operation("compute_4h_features", time.time() - start_4h)
                
        except Exception as e:
            logging.error(f"Error computing 4h features: {e}")
            if debug_mode:
                logging.error(traceback.format_exc())
                
        # Similar pattern for 1d features
        start_1d = time.time()
        try:
            # Resample to 1d
            resampled_1d = df_with_ts[list(ohlcv_columns.keys())].resample('1D').agg(ohlcv_columns)
            
            # Rename columns for clarity
            resampled_1d.columns = [col.replace('1h', '1d') for col in resampled_1d.columns]
            
            # Compute indicators
            if len(resampled_1d) >= 7:  # Need at least 7 points for daily indicators
                # 1. RSI
                close = resampled_1d['close_1d']
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # For values beyond the initial window
                for i in range(14, len(delta)):
                    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
                    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
                
                rs = np.zeros(len(avg_gain))
                rs_mask = avg_loss > 0
                rs[rs_mask] = avg_gain[rs_mask] / avg_loss[rs_mask]
                rs[~rs_mask] = 100
                
                rsi = 100 - (100 / (1 + rs))
                resampled_1d['rsi_1d'] = pd.Series(rsi, index=resampled_1d.index).fillna(50)
                
                # RSI slope
                resampled_1d['rsi_slope_1d'] = resampled_1d['rsi_1d'].diff(1).fillna(0)
                
                # 2. MACD
                ema_fast = close.ewm(span=12, adjust=False).mean()
                ema_slow = close.ewm(span=26, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = macd_line - signal_line
                
                resampled_1d['macd_slope_1d'] = macd_line.diff().fillna(0)
                resampled_1d['macd_hist_slope_1d'] = macd_hist.diff().fillna(0)
                
                # 3. ATR
                high = resampled_1d['high_1d']
                low = resampled_1d['low_1d']
                
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                
                # Use exponential moving average for smoother ATR
                resampled_1d['atr_1d'] = tr.ewm(span=14, adjust=False).mean().fillna(tr)
                
                # 4. Bollinger Bands
                rolling_mean = close.rolling(window=20, min_periods=1).mean()
                rolling_std = close.rolling(window=20, min_periods=1).std()
                
                upper_band = rolling_mean + (rolling_std * 2)
                lower_band = rolling_mean - (rolling_std * 2)
                
                resampled_1d['bollinger_width_1d'] = (upper_band - lower_band).fillna(0)
                
                # 5. Donchian Channel
                resampled_1d['donchian_channel_width_1d'] = (
                    high.rolling(window=20, min_periods=1).max() - 
                    low.rolling(window=20, min_periods=1).min()
                ).fillna(0)
                
                # 6. MFI
                typical_price = (high + low + close) / 3
                
                money_flow = typical_price * resampled_1d['volume_1d']
                
                # Get positive and negative money flow
                money_flow_direction = np.sign(typical_price.diff().fillna(0))
                pos_flow = money_flow * (money_flow_direction > 0)
                neg_flow = money_flow * (money_flow_direction < 0)
                
                # Calculate money flow ratio
                pos_sum = pos_flow.rolling(window=14, min_periods=1).sum()
                neg_sum = neg_flow.rolling(window=14, min_periods=1).sum()
                
                # Avoid division by zero
                mfi = np.where(
                    neg_sum > 0,
                    100 - (100 / (1 + (pos_sum / neg_sum))),
                    100
                )
                
                resampled_1d['money_flow_index_1d'] = pd.Series(mfi, index=resampled_1d.index).fillna(50)
                
                # 7. OBV slope
                obv = np.zeros(len(resampled_1d))
                
                # First OBV is just the volume
                if len(resampled_1d) > 0:
                    obv[0] = resampled_1d['volume_1d'].iloc[0]
                    
                    # Calculate rest of OBV
                    for i in range(1, len(resampled_1d)):
                        if close.iloc[i] > close.iloc[i-1]:
                            obv[i] = obv[i-1] + resampled_1d['volume_1d'].iloc[i]
                        elif close.iloc[i] < close.iloc[i-1]:
                            obv[i] = obv[i-1] - resampled_1d['volume_1d'].iloc[i]
                        else:
                            obv[i] = obv[i-1]
                
                # Calculate OBV slope
                obv_series = pd.Series(obv, index=resampled_1d.index)
                resampled_1d['obv_slope_1d'] = obv_series.diff().fillna(0)
                
                # 8. Volume change
                resampled_1d['volume_change_pct_1d'] = resampled_1d['volume_1d'].pct_change().fillna(0)
                
                # Ensure all columns are proper floats
                for col in resampled_1d.columns:
                    if col not in ohlcv_columns and col != 'timestamp_utc':
                        resampled_1d[col] = resampled_1d[col].astype(float)
                
                # Map back to original timeframe
                for col in resampled_1d.columns:
                    if col not in ohlcv_columns and col != 'timestamp_utc':
                        # Forward fill resampled values to 1h data
                        series_1d = resampled_1d[col]
                        result_df[col] = series_1d.reindex(
                            df_with_ts.index, method='ffill'
                        ).fillna(0).values
                        
            if perf_monitor:
                perf_monitor.log_operation("compute_1d_features", time.time() - start_1d)
                
        except Exception as e:
            logging.error(f"Error computing 1d features: {e}")
            if debug_mode:
                logging.error(traceback.format_exc())
        
        # Ensure all multi-timeframe columns are properly typed as floats
        multi_tf_columns = [col for col in result_df.columns if ('_4h' in col or '_1d' in col)]
        
        for col in multi_tf_columns:
            # Convert to float
            try:
                result_df[col] = result_df[col].astype(float)
            except Exception:
                # If conversion fails, replace with 0.0
                logging.warning(f"Failed to convert {col} to float, replacing with 0.0")
                result_df[col] = 0.0
                
        # Log total duration
        duration = time.time() - start_time
        if perf_monitor:
            # Add sanity check for timing
            if duration > 1000:  # If over 1000 seconds, likely a bug
                logging.warning(f"Unusually long duration for multi_timeframe_features_total: {duration}s, capping at 1000s")
                duration = 1000.0
            perf_monitor.log_operation("multi_timeframe_features_total", duration)
            
        return result_df