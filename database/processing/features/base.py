#!/usr/bin/env python3
"""
Base Feature Computer class
- Provides foundation for all feature computers
"""

import logging
import time
import numpy as np
import pandas as pd

from .utils import safe_indicator_assign, check_gpu_available

class BaseFeatureComputer:
    """Base class for all feature computers with common utilities"""
    
    def __init__(self, use_numba=True, use_gpu=False):
        """
        Initialize base feature computer
        
        Args:
            use_numba: Whether to use Numba JIT for optimization
            use_gpu: Whether to use GPU acceleration
        """
        self.use_numba = use_numba
        self.use_gpu = use_gpu and check_gpu_available()
        
        if self.use_gpu:
            import cupy
            logging.info("GPU acceleration enabled for feature computation")
            
    def safe_assign(self, df, column_name, indicator_result):
        """
        Safely assign an indicator result to a DataFrame column, handling index misalignment.
        
        Args:
            df: DataFrame to assign to
            column_name: Name of the column to create/update
            indicator_result: Result from calculation
        
        Returns:
            DataFrame with the indicator assigned
        """
        return safe_indicator_assign(df, column_name, indicator_result)

    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Main method to compute features, to be implemented by subclasses
        
        Args:
            df: DataFrame with price/volume data
            params: Parameters for feature calculation
            debug_mode: Whether to log detailed debug information
            perf_monitor: Performance monitor for tracking computation time
            
        Returns:
            DataFrame with computed features
        """
        raise NotImplementedError("This method should be implemented by subclasses")
        
    def _log_performance(self, operation, start_time, perf_monitor):
        """
        Log performance data if a performance monitor is available
        
        Args:
            operation: Name of the operation
            start_time: Start time of the operation
            perf_monitor: Performance monitor object
        """
        if perf_monitor:
            duration = time.time() - start_time
            perf_monitor.log_operation(operation, duration)
            
    def _debug_log(self, message, debug_mode):
        """
        Log debug message if debug_mode is enabled
        
        Args:
            message: Message to log
            debug_mode: Whether debug mode is enabled
        """
        if debug_mode:
            logging.debug(message)

    def _handle_exceptions(self, df, column_name, default_value, operation, e):
        """
        Handle exceptions in feature computation
        
        Args:
            df: DataFrame to assign to
            column_name: Name of the column
            default_value: Default value to use
            operation: Name of the operation for logging
            e: Exception object
            
        Returns:
            DataFrame with default value assigned
        """
        logging.warning(f"Error computing {operation}: {e}")
        df[column_name] = default_value
        return df
        
    def _clean_dataframe(self, df, columns=None):
        """
        Clean NaN/inf values in the DataFrame
        
        Args:
            df: DataFrame to clean
            columns: List of columns to clean, if None, clean all float columns
            
        Returns:
            Cleaned DataFrame
        """
        columns = columns or df.select_dtypes(include=['float64', 'float32']).columns
        
        for col in columns:
            if col not in ['timestamp_utc', 'pair']:
                df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
                
        return df