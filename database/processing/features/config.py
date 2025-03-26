#!/usr/bin/env python3
"""
Feature computation configuration 
- Contains parameters and settings for all feature calculations
"""

import os
import logging
import configparser
from pathlib import Path
from dotenv import load_dotenv
import psutil

# ---------------------------
# Constants & Default Values
# ---------------------------
# Automatically determine optimal batch size based on CPU cores
CPU_CORES = os.cpu_count() or 8
BATCH_SIZE = max(8, min(24, CPU_CORES * 2))
MIN_CANDLES_REQUIRED = 200  # Minimum candles needed for reliable calculation
ROLLING_WINDOW = 128  # Default window size for rolling updates

# Default paths
DEFAULT_CONFIG_PATH = os.path.expanduser("~/OKXsignal/config/config.ini")
DEFAULT_CREDENTIALS_PATH = os.path.expanduser("~/OKXsignal/config/credentials.env")

# Define smallint columns for proper type conversion
SMALLINT_COLUMNS = {
    'performance_rank_btc_1h', 'performance_rank_eth_1h',
    'volatility_rank_1h', 'volume_rank_1h',
    'hour_of_day', 'day_of_week', 'month_of_year',
    'was_profitable_12h', 'is_weekend', 'asian_session',
    'european_session', 'american_session',
    'pattern_doji', 'pattern_engulfing', 'pattern_hammer',
    'pattern_morning_star', 'profit_target_1pct', 'profit_target_2pct'
}

# ---------------------------
# Feature Parameters
# ---------------------------

# Price action features
PRICE_ACTION_PARAMS = {
    'use_numba': True,
    'use_gpu': False
}

# Momentum features
MOMENTUM_PARAMS = {
    'rsi_length': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'stoch_k': 14,
    'stoch_d': 3,
    'cci_length': 14,
    'roc_length': 10,
    'tsi_fast': 13,
    'tsi_slow': 25,
    'awesome_oscillator_fast': 5,
    'awesome_oscillator_slow': 34,
    'ppo_fast': 12,
    'ppo_slow': 26
}

# Volatility features
VOLATILITY_PARAMS = {
    'atr_length': 14,
    'bb_length': 20,
    'bb_std': 2.0,
    'kc_length': 20,
    'kc_scalar': 2.0,
    'donchian_length': 20,
    'historical_vol_length': 30,
    'chaikin_volatility_length': 10
}

# Volume features
VOLUME_PARAMS = {
    'mfi_length': 14,
    'vwma_length': 20,
    'cmf_length': 20,
    'kvo_fast': 34,
    'kvo_slow': 55,
    'kvo_signal': 13,
    'vol_fast': 14,
    'vol_slow': 28,
    'vzo_length': 14
}

# Statistical features
STATISTICAL_PARAMS = {
    'window_size': 20,
    'z_score_length': 20,
    'hurst_window': 100,
    'hurst_max_lag': 20,
    'entropy_window': 20,
    'autocorr_lag': 1
}

# Pattern features
PATTERN_PARAMS = {
    'doji_threshold': 0.1,
    'hammer_body_threshold': 0.3,
    'hammer_shadow_threshold': 2.0,
    'support_resistance_length': 20
}

# Time features
TIME_PARAMS = {
    'asian_session_start': 0,
    'asian_session_end': 8,
    'european_session_start': 8,
    'european_session_end': 16,
    'american_session_start': 14,
    'american_session_end': 22
}

# Multi-timeframe features
MULTI_TIMEFRAME_PARAMS = {
    'resample_rules': {
        '4h': '4h',
        '1d': '1d'
    },
    'min_points': {
        '4h': 24,  # 4h window = 1 day
        '1d': 30   # 1d window = ~1 month
    }
}

# Label features
LABEL_PARAMS = {
    'horizons': {
        '1h': 1,
        '4h': 4,
        '12h': 12,
        '1d': 24,
        '3d': 72,
        '1w': 168,
        '2w': 336
    },
    'max_return_window': 25,  # 24h = 24 candles in 1h timeframe
    'max_drawdown_window': 13  # 12h = 12 candles in 1h timeframe
}

# Cross-pair features
CROSS_PAIR_PARAMS = {
    'correlation_window': 24
}

# ---------------------------
# Configuration Management
# ---------------------------
class ConfigManager:
    """Handles loading and managing configuration settings"""
    
    def __init__(self, config_path=None, credentials_path=None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        
        # Load environmental variables
        if os.path.exists(self.credentials_path):
            load_dotenv(dotenv_path=self.credentials_path)
            logging.info(f"Loaded credentials from {self.credentials_path}")
        else:
            logging.warning(f"Credentials file not found: {self.credentials_path}")
            
        # Load config
        self.config = self._load_config()

        # Check GPU availability at initialization
        self.gpu_available = self._check_gpu_availability()
        if self.gpu_available:
            logging.info("GPU acceleration is available")
    
    def _check_gpu_availability(self):
        """Check if CuPy is available for GPU acceleration"""
        try:
            import cupy
            # Test basic GPU operation
            x = cupy.array([1, 2, 3])
            y = x * 2
            cupy.cuda.Stream.null.synchronize()
            
            # Get GPU info
            device = cupy.cuda.Device()
            mem_info = device.mem_info
            total_memory = mem_info[1] / (1024**3)  # Convert to GB
            device_id = device.id
            logging.info(f"GPU detected: Device #{device_id}, Memory: {total_memory:.2f} GB")
            return True
        except (ImportError, Exception) as e:
            logging.warning(f"GPU acceleration not available: {str(e)}")
            return False
    
    def use_gpu(self):
        """Check if GPU acceleration should be used"""
        if not self.gpu_available:
            return False
            
        if 'GENERAL' in self.config and 'USE_GPU' in self.config['GENERAL']:
            return self.config['GENERAL'].getboolean('USE_GPU')
        return False
    
    def _load_config(self):
        """Load configuration from file"""
        config = configparser.ConfigParser()
        if os.path.exists(self.config_path):
            config.read(self.config_path)
            logging.info(f"Loaded config from {self.config_path}")
        else:
            logging.warning(f"Config file not found: {self.config_path}")
            # Create minimal defaults
            config['DATABASE'] = {
                'DB_HOST': 'localhost',
                'DB_PORT': '5432',
                'DB_NAME': 'okxsignal'
            }
            config['GENERAL'] = {
                'COMPUTE_MODE': 'rolling_update',
                'ROLLING_WINDOW': str(ROLLING_WINDOW),
                'USE_TALIB': 'False',  # We're not using TA-Lib
                'USE_NUMBA': 'True',
                'USE_GPU': 'False'
            }
            config['FEATURES'] = {
                'PRICE_ACTION': 'True',
                'MOMENTUM': 'True',
                'VOLATILITY': 'True',
                'VOLUME': 'True',
                'STATISTICAL': 'True',
                'PATTERN': 'True',
                'TIME': 'True',
                'MULTI_TIMEFRAME': 'True',
                'CROSS_PAIR': 'True',
                'LABELS': 'True'
            }
        return config
    
    def get_db_params(self):
        """Get database connection parameters"""
        if 'DATABASE' not in self.config:
            raise ValueError("DATABASE section not found in config file")
            
        db_config = self.config['DATABASE']
        
        return {
            'host': db_config.get('DB_HOST', 'localhost'),
            'port': db_config.get('DB_PORT', '5432'),
            'dbname': db_config.get('DB_NAME', 'okxsignal'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
    def get_connection_string(self):
        """Create a SQLAlchemy connection string from config"""
        db_params = self.get_db_params()
        return (
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@"
            f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
    def get_rolling_window(self):
        """Get rolling window size from config"""
        if 'GENERAL' in self.config and 'ROLLING_WINDOW' in self.config['GENERAL']:
            return int(self.config['GENERAL']['ROLLING_WINDOW'])
        return ROLLING_WINDOW
        
    def get_compute_mode(self):
        """Get compute mode from config"""
        if 'GENERAL' in self.config and 'COMPUTE_MODE' in self.config['GENERAL']:
            return self.config['GENERAL']['COMPUTE_MODE']
        return 'rolling_update'
        
    def use_numba(self):
        """Check if Numba should be used for optimization"""
        if 'GENERAL' in self.config and 'USE_NUMBA' in self.config['GENERAL']:
            return self.config['GENERAL'].getboolean('USE_NUMBA')
        return True
        
    def get_batch_size(self):
        """Get batch size for parallel processing"""
        if 'GENERAL' in self.config and 'BATCH_SIZE' in self.config['GENERAL']:
            return int(self.config['GENERAL']['BATCH_SIZE'])
        
        # Auto-determine batch size based on available memory and CPU cores
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            # Estimate 250MB per pair
            memory_based_limit = int(available_memory_gb / 0.25)
            # Use the smaller of memory-based limit and CPU-based limit
            return min(memory_based_limit, BATCH_SIZE)
        except:
            return BATCH_SIZE
        
        
    def is_feature_enabled(self, feature_name):
        """Check if a specific feature group is enabled"""
        if 'FEATURES' in self.config and feature_name.upper() in self.config['FEATURES']:
            return self.config['FEATURES'].getboolean(feature_name.upper())
        return True  # By default, all features are enabled
        
    def get_feature_params(self, feature_name):
        """Get parameters for a specific feature group"""
        feature_name = feature_name.upper()
        
        # Get the default parameters map
        default_params = {
            'PRICE_ACTION': PRICE_ACTION_PARAMS,
            'MOMENTUM': MOMENTUM_PARAMS,
            'VOLATILITY': VOLATILITY_PARAMS,
            'VOLUME': VOLUME_PARAMS,
            'STATISTICAL': STATISTICAL_PARAMS,
            'PATTERN': PATTERN_PARAMS,
            'TIME': TIME_PARAMS,
            'MULTI_TIMEFRAME': MULTI_TIMEFRAME_PARAMS,
            'LABEL': LABEL_PARAMS,
            'CROSS_PAIR': CROSS_PAIR_PARAMS
        }
        
        # Return default params if no custom values in config
        if feature_name not in default_params or f"{feature_name}_PARAMS" not in self.config:
            return default_params.get(feature_name, {})
            
        # Get custom params from config
        custom_params = {}
        param_section = self.config[f"{feature_name}_PARAMS"]
        
        for key, value in param_section.items():
            # Try to convert to appropriate type
            if key in default_params.get(feature_name, {}):
                default_value = default_params[feature_name][key]
                
                if isinstance(default_value, bool):
                    custom_params[key] = param_section.getboolean(key)
                elif isinstance(default_value, int):
                    custom_params[key] = int(value)
                elif isinstance(default_value, float):
                    custom_params[key] = float(value)
                elif isinstance(default_value, dict):
                    # Dictionary values need specific handling
                    # For simplicity, we'll skip them for now and use defaults
                    custom_params[key] = default_value
                else:
                    custom_params[key] = value
            else:
                custom_params[key] = value
                
        # Merge with defaults, prioritizing custom values
        merged_params = default_params.get(feature_name, {}).copy()
        merged_params.update(custom_params)
        
        return merged_params
        
    def get_smallint_columns(self):
        """Get the set of column names that should be smallint"""
        return SMALLINT_COLUMNS