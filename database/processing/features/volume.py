#!/usr/bin/env python3
"""
Volume Features
- Computes technical indicators related to trading volume
"""

import logging
import time
import numpy as np
import pandas as pd

from .base import BaseFeatureComputer
from .config import VOLUME_PARAMS

class VolumeFeatures(BaseFeatureComputer):
    """Compute volume-based technical indicators"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute volume features
        
        Args:
            df: DataFrame with price/volume data
            params: Parameters for volume features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with volume features
        """
        start_time = time.time()
        self._debug_log("Computing volume features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or VOLUME_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) < 14:  # Need at least 14 points for volume indicators
            self._debug_log("Not enough data for volume features", debug_mode)
            return df
            
        # Extract price and volume data
        close = df['close_1h']
        high = df['high_1h']
        low = df['low_1h']
        volume = df['volume_1h']
        
        # Money Flow Index
        try:
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate money flow
            money_flow = typical_price * volume
            
            # Get positive and negative money flow
            pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
            neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
            
            # Calculate money flow ratio
            pos_sum = pd.Series(pos_flow).rolling(window=params['mfi_length']).sum()
            neg_sum = pd.Series(neg_flow).rolling(window=params['mfi_length']).sum()
            
            # Avoid division by zero
            mfr = np.where(neg_sum != 0, pos_sum / neg_sum, 100)
            
            # Calculate Money Flow Index
            mfi = 100 - (100 / (1 + mfr))
            df['money_flow_index_1h'] = pd.Series(mfi, index=close.index).fillna(50)
            
        except Exception as e:
            self._handle_exceptions(df, 'money_flow_index_1h', 50, "Money Flow Index", e)
        
        # OBV (On-Balance Volume)
        try:
            # Initialize OBV with first volume value
            obv = np.zeros(len(df))
            
            # Calculate OBV for subsequent data points
            for i in range(1, len(df)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv[i] = obv[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv[i] = obv[i-1] - volume.iloc[i]
                else:
                    obv[i] = obv[i-1]
            
            df['obv_1h'] = obv
            
            # OBV slope
            df['obv_slope_1h'] = pd.Series(obv).diff(3) / 3
            
        except Exception as e:
            self._handle_exceptions(df, 'obv_1h', 0, "OBV", e)
            df['obv_slope_1h'] = 0
        
        # Volume change percentage
        df['volume_change_pct_1h'] = volume.pct_change().fillna(0)
        
        # VWMA (Volume Weighted Moving Average)
        try:
            vwma = (close * volume).rolling(window=params['vwma_length']).sum() / \
                   volume.rolling(window=params['vwma_length']).sum()
            df['vwma_20'] = vwma.fillna(close)
            
        except Exception as e:
            self._handle_exceptions(df, 'vwma_20', close, "VWMA", e)
        
        # Chaikin Money Flow
        try:
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
            money_flow_volume = money_flow_multiplier * volume
            
            cmf = money_flow_volume.rolling(window=params['cmf_length']).sum() / \
                  volume.rolling(window=params['cmf_length']).sum()
            df['chaikin_money_flow'] = cmf.fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'chaikin_money_flow', 0, "Chaikin Money Flow", e)
        
        # Klinger Oscillator
        try:
            # Calculate trend direction
            dm = np.where(
                (high + low + close) > (high.shift(1) + low.shift(1) + close.shift(1)),
                1,
                -1
            )
            
            # Calculate volume force
            vf = volume * dm * abs(
                2 * ((close - low) - (high - close)) / (high - low)
            )
            vf = pd.Series(vf, index=close.index).replace([np.inf, -np.inf], 0).fillna(0)
            
            # Calculate EMAs of volume force
            ema_fast = vf.ewm(span=params['kvo_fast'], adjust=False).mean()
            ema_slow = vf.ewm(span=params['kvo_slow'], adjust=False).mean()
            
            # Klinger Oscillator
            kvo = ema_fast - ema_slow
            df['klinger_oscillator'] = kvo.fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'klinger_oscillator', 0, "Klinger Oscillator", e)
        
        # Volume Oscillator (defined as the difference between fast and slow volume EMAs)
        try:
            vol_fast = volume.ewm(span=params['vol_fast']).mean()
            vol_slow = volume.ewm(span=params['vol_slow']).mean()
            df['volume_oscillator'] = (vol_fast - vol_slow).fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'volume_oscillator', 0, "Volume Oscillator", e)
        
        # Volume Price Trend
        try:
            # Calculate VPT
            vpt = np.zeros(len(df))
            
            for i in range(1, len(df)):
                close_prev = close.iloc[i-1]
                if close_prev > 0:  # Avoid division by zero
                    percent_change = (close.iloc[i] - close_prev) / close_prev
                    vpt[i] = vpt[i-1] + volume.iloc[i] * percent_change
            
            df['volume_price_trend'] = vpt
            
        except Exception as e:
            self._handle_exceptions(df, 'volume_price_trend', 0, "Volume Price Trend", e)
        
        # Volume Zone Oscillator
        try:
            # Determine price direction
            price_up = close > close.shift(1)
            
            # Categorize volume based on price direction
            vol_up = volume.copy()
            vol_up.loc[~price_up] = 0
            
            vol_down = volume.copy()
            vol_down.loc[price_up] = 0
            
            # Calculate EMAs
            ema_vol = volume.ewm(span=params['vzo_length'], adjust=False).mean()
            ema_vol_up = vol_up.ewm(span=params['vzo_length'], adjust=False).mean()
            ema_vol_down = vol_down.ewm(span=params['vzo_length'], adjust=False).mean()
            
            # Calculate VZO - avoid division by zero
            vzo = np.where(
                ema_vol > 0,
                100 * (ema_vol_up - ema_vol_down) / ema_vol,
                0
            )
            df['volume_zone_oscillator'] = pd.Series(vzo, index=close.index).fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'volume_zone_oscillator', 0, "Volume Zone Oscillator", e)
        
        # Volume Price Confirmation Indicator (close direction matches volume direction)
        try:
            close_dir = np.sign(close.diff()).fillna(0)
            vol_dir = np.sign(volume.diff()).fillna(0)
            df['volume_price_confirmation'] = (close_dir == vol_dir).astype(int)
            
        except Exception as e:
            self._handle_exceptions(df, 'volume_price_confirmation', 0, "Volume Price Confirmation", e)
        
        # Volume rank (will be updated in cross_pair features)
        df['volume_rank_1h'] = 0
        df['prev_volume_rank'] = 0
        
        # Clean any NaN values
        df = self._clean_dataframe(df)
        
        self._log_performance("volume_features", start_time, perf_monitor)
        return df