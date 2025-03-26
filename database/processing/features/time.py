#!/usr/bin/env python3
"""
Time Features
- Computes time-based features like hour of day, day of week, etc.
"""

import logging
import time
import numpy as np
import pandas as pd

from .base import BaseFeatureComputer
from .config import TIME_PARAMS

class TimeFeatures(BaseFeatureComputer):
    """Compute time-based features"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute time features
        
        Args:
            df: DataFrame with timestamp data
            params: Parameters for time features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with time features
        """
        start_time = time.time()
        self._debug_log("Computing time features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or TIME_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        try:
            # Extract timestamp
            timestamps = pd.to_datetime(df['timestamp_utc'])
            
            # Extract hour, day, month
            df['hour_of_day'] = timestamps.dt.hour
            df['day_of_week'] = timestamps.dt.dayofweek
            df['month_of_year'] = timestamps.dt.month
            
            # Weekend indicator (Friday after 9 PM or Saturday or Sunday)
            df['is_weekend'] = (
                ((df['day_of_week'] == 4) & (df['hour_of_day'] >= 21)) | 
                (df['day_of_week'] >= 5)
            ).astype(int)
            
            # Trading sessions (approximate)
            df['asian_session'] = (
                (df['hour_of_day'] >= params['asian_session_start']) & 
                (df['hour_of_day'] < params['asian_session_end'])
            ).astype(int)
            
            df['european_session'] = (
                (df['hour_of_day'] >= params['european_session_start']) & 
                (df['hour_of_day'] < params['european_session_end'])
            ).astype(int)
            
            df['american_session'] = (
                (df['hour_of_day'] >= params['american_session_start']) & 
                (df['hour_of_day'] < params['american_session_end'])
            ).astype(int)
            
        except Exception as e:
            logging.warning(f"Error computing time features: {e}")
            
            # Set default values if extraction fails
            for col in ['hour_of_day', 'day_of_week', 'month_of_year']:
                df[col] = 0
                
            for col in ['is_weekend', 'asian_session', 'european_session', 'american_session']:
                df[col] = 0
        
        # Convert integer columns to ensure they're integers
        int_columns = ['hour_of_day', 'day_of_week', 'month_of_year', 
                      'is_weekend', 'asian_session', 'european_session', 'american_session']
        
        for col in int_columns:
            df[col] = df[col].astype(int)
        
        self._log_performance("time_features", start_time, perf_monitor)
        return df