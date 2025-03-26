#!/usr/bin/env python3
"""
Cross-Pair Features
- Computes features across multiple pairs
"""

import logging
import time
import numpy as np
import pandas as pd

from database.processing.features.base import BaseFeatureComputer
from database.processing.features.config import CROSS_PAIR_PARAMS

class CrossPairFeatures(BaseFeatureComputer):
    """Compute cross-pair features across multiple cryptocurrency pairs"""
    
    def compute_features(self, df, params=None, debug_mode=False, perf_monitor=None):
        """
        Compute cross-pair features
        
        Args:
            df: DataFrame with data from multiple pairs
            params: Parameters for cross-pair features
            debug_mode: Whether to log debug info
            perf_monitor: Performance monitor
            
        Returns:
            DataFrame with cross-pair features
        """
        start_time = time.time()
        self._debug_log("Computing cross-pair features...", debug_mode)
        
        # Use provided parameters or defaults
        params = params or CROSS_PAIR_PARAMS
        
        # Make sure we have a copy to avoid modifying the original
        df = df.copy()
        
        if len(df) == 0:
            logging.warning("Empty DataFrame passed to compute_cross_pair_features")
            return df
            
        # Initialize columns with 0 to guarantee they exist
        df['volume_rank_1h'] = 0
        df['volatility_rank_1h'] = 0
        df['performance_rank_btc_1h'] = 0
        df['performance_rank_eth_1h'] = 0
        df['btc_corr_24h'] = 0.0
        df['prev_volume_rank'] = 0
        
        # Volume rank
        if 'volume_1h' in df.columns:
            self._debug_log("Computing volume rank", debug_mode)
            try:
                # Group by timestamp to calculate rank within each timeframe
                df = df.sort_values(['timestamp_utc', 'volume_1h'], ascending=[True, False])
                df['volume_rank_1h'] = df.groupby('timestamp_utc')['volume_1h'].rank(pct=True) * 100
                
                # Previous volume rank
                df['prev_volume_rank'] = df.groupby('pair')['volume_rank_1h'].shift(1).fillna(0)
            except Exception as e:
                logging.warning(f"Error computing volume rank: {e}")
            
        # Volatility rank
        if 'atr_1h' in df.columns:
            self._debug_log("Computing volatility rank", debug_mode)
            try:
                # Group by timestamp to calculate rank within each timeframe
                df = df.sort_values(['timestamp_utc', 'atr_1h'], ascending=[True, False])
                df['volatility_rank_1h'] = df.groupby('timestamp_utc')['atr_1h'].rank(pct=True) * 100
            except Exception as e:
                logging.warning(f"Error computing volatility rank: {e}")

        # Bitcoin correlation (for pairs that aren't BTC)
        self._debug_log("Computing BTC correlation", debug_mode)
        try:
            btc_pair = df[df['pair'] == 'BTC-USDT']
            if not btc_pair.empty and len(df) > params['correlation_window']:
                # Get BTC returns
                btc_returns = btc_pair['log_return'].values
                btc_times = btc_pair['timestamp_utc'].values
                
                if len(btc_returns) >= params['correlation_window']:
                    # For each non-BTC pair, calculate correlation
                    for pair_name in df['pair'].unique():
                        if pair_name != 'BTC-USDT':
                            pair_data = df[df['pair'] == pair_name]
                            pair_returns = pair_data['log_return'].values
                            pair_times = pair_data['timestamp_utc'].values
                            
                            # Only compute if we have enough data
                            if len(pair_returns) >= params['correlation_window']:
                                # Create a mapping of timestamps to indices for efficient lookup
                                btc_time_to_idx = {t: i for i, t in enumerate(btc_times)}
                                
                                # Calculate rolling correlation for each point
                                for i in range(params['correlation_window'], len(pair_returns)):
                                    # Get the current timestamp
                                    curr_time = pair_times[i]
                                    
                                    # Find the corresponding index in BTC data
                                    if curr_time in btc_time_to_idx:
                                        btc_idx = btc_time_to_idx[curr_time]
                                        
                                        # Ensure we have enough data for both
                                        if (btc_idx >= params['correlation_window'] and 
                                            i >= params['correlation_window']):
                                            
                                            # Get the rolling windows
                                            btc_window = btc_returns[btc_idx-params['correlation_window']:btc_idx]
                                            pair_window = pair_returns[i-params['correlation_window']:i]
                                            
                                            # Calculate correlation
                                            if len(btc_window) == len(pair_window):
                                                try:
                                                    corr = np.corrcoef(btc_window, pair_window)[0, 1]
                                                    df.loc[
                                                        (df['pair'] == pair_name) & 
                                                        (df['timestamp_utc'] == curr_time), 
                                                        'btc_corr_24h'
                                                    ] = corr
                                                except Exception as e:
                                                    logging.warning(f"Error calculating correlation: {e}")
        except Exception as e:
            logging.warning(f"Error computing BTC correlation: {e}")

        # Performance rank relative to BTC
        self._debug_log("Computing performance rank relative to BTC", debug_mode)
        try:
            # Group by timestamp
            for timestamp in df['timestamp_utc'].unique():
                timestamp_df = df[df['timestamp_utc'] == timestamp]
                
                btc_row = timestamp_df[timestamp_df['pair'] == 'BTC-USDT']
                if not btc_row.empty and 'future_return_1h_pct' in df.columns:
                    btc_return = btc_row['future_return_1h_pct'].values[0]
                    
                    if not pd.isna(btc_return) and abs(btc_return) > 1e-9:
                        # Calculate relative performance
                        rel_perf = (timestamp_df['future_return_1h_pct'] - btc_return) / abs(btc_return)
                        
                        # Rank the relative performance
                        perf_rank = rel_perf.rank(pct=True) * 100
                        
                        # Update the original dataframe
                        df.loc[df['timestamp_utc'] == timestamp, 'performance_rank_btc_1h'] = perf_rank.values
        except Exception as e:
            logging.warning(f"Error computing BTC performance rank: {e}")

        # Performance rank relative to ETH
        self._debug_log("Computing performance rank relative to ETH", debug_mode)
        try:
            # Group by timestamp
            for timestamp in df['timestamp_utc'].unique():
                timestamp_df = df[df['timestamp_utc'] == timestamp]
                
                eth_row = timestamp_df[timestamp_df['pair'] == 'ETH-USDT']
                if not eth_row.empty and 'future_return_1h_pct' in df.columns:
                    eth_return = eth_row['future_return_1h_pct'].values[0]
                    
                    if not pd.isna(eth_return) and abs(eth_return) > 1e-9:
                        # Calculate relative performance
                        rel_perf = (timestamp_df['future_return_1h_pct'] - eth_return) / abs(eth_return)
                        
                        # Rank the relative performance
                        perf_rank = rel_perf.rank(pct=True) * 100
                        
                        # Update the original dataframe
                        df.loc[df['timestamp_utc'] == timestamp, 'performance_rank_eth_1h'] = perf_rank.values
        except Exception as e:
            logging.warning(f"Error computing ETH performance rank: {e}")

        # Fill NaN values with 0
        for col in ['volume_rank_1h', 'volatility_rank_1h', 'performance_rank_btc_1h', 
                   'performance_rank_eth_1h', 'btc_corr_24h', 'prev_volume_rank']:
            df[col] = df[col].fillna(0)
            
        # Convert rank columns to integers (smallint in DB)
        for col in ['volume_rank_1h', 'volatility_rank_1h', 'performance_rank_btc_1h', 
                   'performance_rank_eth_1h']:
            df[col] = df[col].astype(int)

        self._log_performance("cross_pair_features", time.time() - start_time, perf_monitor)
        return df