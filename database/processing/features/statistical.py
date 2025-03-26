#!/usr/bin/env python3
"""
Statistical Features
- Computes statistical and microstructure features
"""

import logging
import time
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseFeatureComputer
from .config import STATISTICAL_PARAMS
from .optimized.numba_functions import compute_z_score_numba, hurst_exponent_numba, shannon_entropy_numba
from .optimized.gpu_functions import compute_z_score_gpu, hurst_exponent_gpu, shannon_entropy_gpu

class StatisticalFeatures(BaseFeatureComputer):
    """Compute statistical and microstructure features"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute statistical features
        
        Args:
            df: DataFrame with price data
            params: Parameters for statistical features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with statistical features
        """
        start_time = time.time()
        self._debug_log("Computing statistical features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or STATISTICAL_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) < params['window_size']:  # Need at least window_size points
            self._debug_log("Not enough data for statistical features", debug_mode)
            return df
            
        # Extract price data
        close = df['close_1h']
        
        # Standard deviation of returns
        df['std_dev_returns_20'] = df['log_return'].rolling(window=params['window_size']).std().fillna(0)
        
        # Skewness and kurtosis
        try:
            df['skewness_20'] = df['log_return'].rolling(window=params['window_size']).apply(
                lambda x: stats.skew(x) if len(x) > 3 else 0, raw=True
            ).fillna(0)
            
            df['kurtosis_20'] = df['log_return'].rolling(window=params['window_size']).apply(
                lambda x: stats.kurtosis(x) if len(x) > 3 else 0, raw=True
            ).fillna(0)
        except Exception as e:
            self._handle_exceptions(df, 'skewness_20', 0, "Skewness/Kurtosis", e)
            df['kurtosis_20'] = 0
        
        # Z-score
        try:
            ma_20 = close.rolling(window=params['z_score_length']).mean().fillna(0).values
            
            if self.use_gpu:
                try:
                    df['z_score_20'] = compute_z_score_gpu(close.values, ma_20, params['z_score_length'])
                except Exception as e:
                    logging.warning(f"GPU z-score calculation failed: {e}")
                    self.use_gpu = False
                    
            if not self.use_gpu and self.use_numba:
                try:
                    df['z_score_20'] = compute_z_score_numba(close.values, ma_20, params['z_score_length'])
                except Exception as e:
                    logging.warning(f"Numba z-score calculation failed: {e}")
                    self.use_numba = False
                    
            if not self.use_gpu and not self.use_numba:
                # Fallback to standard calculation
                std_20 = close.rolling(window=params['z_score_length']).std()
                z_score = ((close - ma_20) / std_20).fillna(0)
                z_score = z_score.replace([np.inf, -np.inf], 0)
                df['z_score_20'] = z_score
                
        except Exception as e:
            self._handle_exceptions(df, 'z_score_20', 0, "Z-score", e)
        
        # Hurst Exponent (measure of long-term memory in time series)
        if len(df) >= params['hurst_window']:
            try:
                # Handle zero or negative prices safely
                safe_close = close.replace(0, np.nan).fillna(method='ffill').fillna(close.mean())
                
                if self.use_gpu:
                    try:
                        df['hurst_exponent'] = hurst_exponent_gpu(
                            safe_close.values, 
                            params['hurst_max_lag']
                        )
                    except Exception as e:
                        logging.warning(f"GPU Hurst calculation failed: {e}")
                        self.use_gpu = False
                        
                if not self.use_gpu and self.use_numba:
                    try:
                        df['hurst_exponent'] = hurst_exponent_numba(
                            safe_close.values, 
                            params['hurst_max_lag']
                        )
                    except Exception as e:
                        logging.warning(f"Numba Hurst calculation failed: {e}")
                        self.use_numba = False
                        
                if not self.use_gpu and not self.use_numba:
                    # Default value
                    df['hurst_exponent'] = 0.5
                    
                    # Manual calculation is very computationally intensive,
                    # so we'll set a default value and only compute for specific cases
                    if debug_mode:
                        logging.debug("Skipping manual Hurst computation for performance reasons")
                    
            except Exception as e:
                self._handle_exceptions(df, 'hurst_exponent', 0.5, "Hurst Exponent", e)
        else:
            df['hurst_exponent'] = 0.5  # Default value for short series
        
        # Shannon Entropy (measure of randomness/predictability)
        if len(df) >= params['entropy_window']:
            try:
                # Handle zero or negative prices safely
                safe_close = close.replace(0, np.nan).fillna(method='ffill').fillna(close.mean())
                
                if self.use_gpu:
                    try:
                        df['shannon_entropy'] = shannon_entropy_gpu(
                            safe_close.values,
                            params['entropy_window']
                        )
                    except Exception as e:
                        logging.warning(f"GPU Entropy calculation failed: {e}")
                        self.use_gpu = False
                        
                if not self.use_gpu and self.use_numba:
                    try:
                        df['shannon_entropy'] = shannon_entropy_numba(
                            safe_close.values,
                            params['entropy_window']
                        )
                    except Exception as e:
                        logging.warning(f"Numba Entropy calculation failed: {e}")
                        self.use_numba = False
                        
                if not self.use_gpu and not self.use_numba:
                    # Calculate entropy using standard numpy
                    df['shannon_entropy'] = 0  # Default value
                    
                    # Manual calculation is very computationally intensive,
                    # so we'll set a default value and only compute for specific cases
                    if debug_mode:
                        logging.debug("Skipping manual Entropy computation for performance reasons")
                    
            except Exception as e:
                self._handle_exceptions(df, 'shannon_entropy', 0, "Shannon Entropy", e)
        else:
            df['shannon_entropy'] = 0  # Default value
        
        # Autocorrelation lag 1
        try:
            df['autocorr_1'] = df['log_return'].rolling(window=params['window_size']).apply(
                lambda x: pd.Series(x).autocorr(lag=params['autocorr_lag']) if len(x) > params['autocorr_lag'] else 0
            ).fillna(0)
        except Exception as e:
            self._handle_exceptions(df, 'autocorr_1', 0, "Autocorrelation", e)
        
        # Estimated slippage and bid-ask spread proxies
        df['estimated_slippage_1h'] = df['high_1h'] - df['low_1h']
        df['bid_ask_spread_1h'] = df['estimated_slippage_1h'] * 0.1  # Rough approximation
        
        # Clean any NaN values
        df = self._clean_dataframe(df)
        
        self._log_performance("statistical_features", start_time, perf_monitor)
        return df