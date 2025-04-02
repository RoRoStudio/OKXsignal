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
                for timestamp in df['timestamp_utc'].unique():
                    mask = df['timestamp_utc'] == timestamp
                    timestamp_df = df.loc[mask]
                    pairs = timestamp_df.index.tolist()
                    
                    if len(pairs) > 1:
                        # Get volume values for this timestamp
                        volumes = timestamp_df['volume_1h'].values
                        
                        if any(v > 0 for v in volumes):
                            # Initialize ranks array
                            vol_ranks = np.zeros(len(volumes))
                            
                            # Get the sorted indices (ascending order)
                            sorted_indices = np.argsort(volumes)
                            
                            # Assign ranks from 0 to 100 based on position in sorted array
                            for i in range(len(sorted_indices)):
                                vol_ranks[sorted_indices[i]] = (i * 100) / (len(sorted_indices) - 1) if len(sorted_indices) > 1 else 50
                            
                            # Handle ties by averaging ranks
                            unique_values, unique_indices = np.unique(volumes, return_inverse=True)
                            for i, val in enumerate(unique_values):
                                # Find all indices with this value
                                mask_indices = unique_indices == i
                                if np.sum(mask_indices) > 1:  # If there are ties
                                    # Calculate average rank for these tied values
                                    avg_rank = np.mean(vol_ranks[mask_indices])
                                    # Assign the average rank to all tied values
                                    vol_ranks[mask_indices] = avg_rank
                            
                            # Set ranks in the dataframe
                            df.loc[mask, 'volume_rank_1h'] = vol_ranks.round().astype(int)
                        else:
                            # All volumes are 0, assign default ranks
                            df.loc[mask, 'volume_rank_1h'] = 50
                    else:
                        # Default value for a single pair
                        df.loc[mask, 'volume_rank_1h'] = 50
                    
                    # Previous volume rank
                    df['prev_volume_rank'] = df.groupby('pair')['volume_rank_1h'].shift(1).fillna(0)
                
            except Exception as e:
                logging.warning(f"Error computing volume rank: {e}")
            
        # Volatility rank
        if 'atr_1h' in df.columns:
            self._debug_log("Computing volatility rank", debug_mode)
            try:
                # Group by timestamp to calculate rank within each timeframe
                for timestamp in df['timestamp_utc'].unique():
                    mask = df['timestamp_utc'] == timestamp
                    timestamp_df = df.loc[mask]
                    
                    if len(timestamp_df) > 1:
                        # Get ATR values for this timestamp
                        atr_values = timestamp_df['atr_1h'].values
                        
                        if any(a > 0 for a in atr_values):
                            # Initialize ranks array
                            atr_ranks = np.zeros(len(atr_values))
                            
                            # Get the sorted indices (ascending order)
                            sorted_indices = np.argsort(atr_values)
                            
                            # Assign ranks from 0 to 100 based on position in sorted array
                            for i in range(len(sorted_indices)):
                                atr_ranks[sorted_indices[i]] = (i * 100) / (len(sorted_indices) - 1) if len(sorted_indices) > 1 else 50
                            
                            # Handle ties by averaging ranks
                            unique_values, unique_indices = np.unique(atr_values, return_inverse=True)
                            for i, val in enumerate(unique_values):
                                # Find all indices with this value
                                mask_indices = unique_indices == i
                                if np.sum(mask_indices) > 1:  # If there are ties
                                    # Calculate average rank for these tied values
                                    avg_rank = np.mean(atr_ranks[mask_indices])
                                    # Assign the average rank to all tied values
                                    atr_ranks[mask_indices] = avg_rank
                            
                            # Set ranks in the dataframe
                            df.loc[mask, 'volatility_rank_1h'] = atr_ranks.round().astype(int)
                        else:
                            # All ATR values are 0, assign default ranks
                            df.loc[mask, 'volatility_rank_1h'] = 50
                    else:
                        # Default value for a single pair
                        df.loc[mask, 'volatility_rank_1h'] = 50
                        
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
                        # Make sure BTC has a correlation of 1.0 with itself
                        df.loc[df['pair'] == 'BTC-USDT', 'btc_corr_24h'] = 1.0
        except Exception as e:
            logging.warning(f"Error computing BTC correlation: {e}")

        # Performance rank relative to BTC
        self._debug_log("Computing performance rank relative to BTC", debug_mode)
        try:
            # Group by timestamp
            for timestamp in df['timestamp_utc'].unique():
                mask = df['timestamp_utc'] == timestamp
                timestamp_df = df.loc[mask]
                
                btc_row = timestamp_df[timestamp_df['pair'] == 'BTC-USDT']
                if not btc_row.empty and 'future_return_1h_pct' in df.columns:
                    btc_return = btc_row['future_return_1h_pct'].values[0]
                    
                    if not pd.isna(btc_return) and abs(btc_return) > 1e-9:
                        # Calculate relative performance
                        pairs_count = len(timestamp_df)
                        if pairs_count > 1:
                            # Get relative performance values
                            rel_perf = (timestamp_df['future_return_1h_pct'] - btc_return) / abs(btc_return)
                            rel_perf_values = rel_perf.values
                            
                            # Initialize ranks array
                            perf_ranks = np.zeros(pairs_count)
                            
                            # Get the sorted indices (ascending order)
                            sorted_indices = np.argsort(rel_perf_values)
                            
                            # Assign ranks from 0 to 100 based on position in sorted array
                            for i in range(len(sorted_indices)):
                                perf_ranks[sorted_indices[i]] = (i * 100) / (len(sorted_indices) - 1) if len(sorted_indices) > 1 else 50
                            
                            # Handle ties by averaging ranks
                            unique_values, unique_indices = np.unique(rel_perf_values, return_inverse=True)
                            for i, val in enumerate(unique_values):
                                # Find all indices with this value
                                mask_indices = unique_indices == i
                                if np.sum(mask_indices) > 1:  # If there are ties
                                    # Calculate average rank for these tied values
                                    avg_rank = np.mean(perf_ranks[mask_indices])
                                    # Assign the average rank to all tied values
                                    perf_ranks[mask_indices] = avg_rank
                            
                            # Update the original dataframe with calculated ranks
                            df.loc[mask, 'performance_rank_btc_1h'] = perf_ranks.round().astype(int)
        except Exception as e:
            logging.warning(f"Error computing BTC performance rank: {e}")

        # Performance rank relative to ETH
        self._debug_log("Computing performance rank relative to ETH", debug_mode)
        try:
            # Group by timestamp
            for timestamp in df['timestamp_utc'].unique():
                mask = df['timestamp_utc'] == timestamp
                timestamp_df = df.loc[mask]
                
                eth_row = timestamp_df[timestamp_df['pair'] == 'ETH-USDT']
                if not eth_row.empty and 'future_return_1h_pct' in df.columns:
                    eth_return = eth_row['future_return_1h_pct'].values[0]
                    
                    if not pd.isna(eth_return) and abs(eth_return) > 1e-9:
                        # Calculate relative performance
                        pairs_count = len(timestamp_df)
                        if pairs_count > 1:
                            # Get relative performance values
                            rel_perf = (timestamp_df['future_return_1h_pct'] - eth_return) / abs(eth_return)
                            rel_perf_values = rel_perf.values
                            
                            # Initialize ranks array
                            perf_ranks = np.zeros(pairs_count)
                            
                            # Get the sorted indices (ascending order)
                            sorted_indices = np.argsort(rel_perf_values)
                            
                            # Assign ranks from 0 to 100 based on position in sorted array
                            for i in range(len(sorted_indices)):
                                perf_ranks[sorted_indices[i]] = (i * 100) / (len(sorted_indices) - 1) if len(sorted_indices) > 1 else 50
                            
                            # Handle ties by averaging ranks
                            unique_values, unique_indices = np.unique(rel_perf_values, return_inverse=True)
                            for i, val in enumerate(unique_values):
                                # Find all indices with this value
                                mask_indices = unique_indices == i
                                if np.sum(mask_indices) > 1:  # If there are ties
                                    # Calculate average rank for these tied values
                                    avg_rank = np.mean(perf_ranks[mask_indices])
                                    # Assign the average rank to all tied values
                                    perf_ranks[mask_indices] = avg_rank
                            
                            # Update the original dataframe with calculated ranks
                            df.loc[mask, 'performance_rank_eth_1h'] = perf_ranks.round().astype(int)
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