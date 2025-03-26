#!/usr/bin/env python3
"""
Label Features
- Computes target features like future returns and profit targets
"""

import logging
import time
import numpy as np
import pandas as pd

from .base import BaseFeatureComputer
from .config import LABEL_PARAMS
from .optimized.numba_functions import (
    compute_future_return_numba,
    compute_max_future_return_numba,
    compute_max_future_drawdown_numba
)
from .optimized.gpu_functions import (
    compute_future_return_gpu,
    compute_max_future_return_gpu,
    compute_max_future_drawdown_gpu
)

class LabelFeatures(BaseFeatureComputer):
    """Compute label/target features for supervised learning"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute label features
        
        Args:
            df: DataFrame with price data
            params: Parameters for label features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with label features
        """
        start_time = time.time()
        self._debug_log("Computing label features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or LABEL_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) < 5:  # Need at least a few candles for future returns
            self._debug_log("Not enough data for label features", debug_mode)
            return df
            
        # Extract price data
        close = df['close_1h'].values
        high = df['high_1h'].values
        low = df['low_1h'].values
        
        # Calculate future returns for different time horizons
        horizons = params['horizons']
        
        # Choose appropriate computation method (GPU, Numba, or standard)
        try:
            for horizon_name, shift in horizons.items():
                col_name = f'future_return_{horizon_name}_pct'
                
                if self.use_gpu:
                    try:
                        df[col_name] = compute_future_return_gpu(close, shift)
                    except Exception as e:
                        logging.warning(f"GPU calculation failed for {col_name}: {e}")
                        self.use_gpu = False
                        
                if not self.use_gpu and self.use_numba:
                    try:
                        df[col_name] = compute_future_return_numba(close, shift)
                    except Exception as e:
                        logging.warning(f"Numba calculation failed for {col_name}: {e}")
                        self.use_numba = False
                        
                if not self.use_gpu and not self.use_numba:
                    # Use standard numpy vectorized operations
                    future_return = np.zeros(len(close))
                    
                    if len(close) > shift:
                        # Avoid division by zero
                        valid_indices = np.where(close[:-shift] > 0)[0]
                        future_return[valid_indices] = (close[valid_indices + shift] - close[valid_indices]) / close[valid_indices]
                        
                    df[col_name] = future_return
                    
        except Exception as e:
            logging.warning(f"Error computing future returns: {e}")
            # Initialize with zeros if computation fails
            for horizon_name in horizons.keys():
                df[f'future_return_{horizon_name}_pct'] = 0.0
        
        # Max future return calculation
        try:
            if self.use_gpu:
                try:
                    df['future_max_return_24h_pct'] = compute_max_future_return_gpu(
                        close, high, params['max_return_window']
                    )
                except Exception as e:
                    logging.warning(f"GPU calculation failed for max future return: {e}")
                    self.use_gpu = False
                    
            if not self.use_gpu and self.use_numba:
                try:
                    df['future_max_return_24h_pct'] = compute_max_future_return_numba(
                        close, high, params['max_return_window']
                    )
                except Exception as e:
                    logging.warning(f"Numba calculation failed for max future return: {e}")
                    self.use_numba = False
                    
            if not self.use_gpu and not self.use_numba:
                # Standard calculation
                max_future_return = np.zeros(len(close))
                
                for i in range(len(close) - 1):
                    end_idx = min(i + params['max_return_window'], len(high))
                    if i + 1 < end_idx:
                        max_high = np.max(high[i+1:end_idx])
                        # Avoid division by zero
                        if close[i] > 0:
                            max_future_return[i] = (max_high - close[i]) / close[i]
                            
                df['future_max_return_24h_pct'] = max_future_return
                
        except Exception as e:
            self._handle_exceptions(df, 'future_max_return_24h_pct', 0, "Max Future Return", e)
        
        # Max future drawdown calculation
        try:
            if self.use_gpu:
                try:
                    df['future_max_drawdown_12h_pct'] = compute_max_future_drawdown_gpu(
                        close, low, params['max_drawdown_window']
                    )
                except Exception as e:
                    logging.warning(f"GPU calculation failed for max future drawdown: {e}")
                    self.use_gpu = False
                    
            if not self.use_gpu and self.use_numba:
                try:
                    df['future_max_drawdown_12h_pct'] = compute_max_future_drawdown_numba(
                        close, low, params['max_drawdown_window']
                    )
                except Exception as e:
                    logging.warning(f"Numba calculation failed for max future drawdown: {e}")
                    self.use_numba = False
                    
            if not self.use_gpu and not self.use_numba:
                # Standard calculation
                max_future_drawdown = np.zeros(len(close))
                
                for i in range(len(close) - 1):
                    end_idx = min(i + params['max_drawdown_window'], len(low))
                    if i + 1 < end_idx:
                        min_low = np.min(low[i+1:end_idx])
                        # Avoid division by zero
                        if close[i] > 0:
                            max_future_drawdown[i] = (min_low - close[i]) / close[i]
                            
                df['future_max_drawdown_12h_pct'] = max_future_drawdown
                
        except Exception as e:
            self._handle_exceptions(df, 'future_max_drawdown_12h_pct', 0, "Max Future Drawdown", e)
        
        # Was profitable
        df['was_profitable_12h'] = (df['future_return_12h_pct'] > 0).astype(int)
        
        # Risk-adjusted return (return / max drawdown)
        try:
            # Avoid division by zero
            safe_drawdown = np.abs(df['future_max_drawdown_12h_pct'].replace(0, 1e-8))
            df['future_risk_adj_return_12h'] = (df['future_return_12h_pct'] / safe_drawdown)
            
            # Handle infinite and NaN values
            df['future_risk_adj_return_12h'] = df['future_risk_adj_return_12h'].replace([np.inf, -np.inf], 0).fillna(0)
            
        except Exception as e:
            self._handle_exceptions(df, 'future_risk_adj_return_12h', 0, "Risk-Adjusted Return", e)
        
        # Target hit indicators (simple profit targets)
        df['profit_target_1pct'] = (df['future_max_return_24h_pct'] > 0.01).astype(int)
        df['profit_target_2pct'] = (df['future_max_return_24h_pct'] > 0.02).astype(int)
        
        # BTC correlation (will be updated in cross-pair features)
        df['btc_corr_24h'] = 0.0
        
        # Set computation flags
        df['features_computed'] = True
        df['targets_computed'] = True
        
        # Clean any NaN values
        df = self._clean_dataframe(df)
        
        self._log_performance("label_features", time.time() - start_time, perf_monitor)
        return df