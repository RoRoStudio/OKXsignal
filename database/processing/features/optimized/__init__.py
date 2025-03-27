#!/usr/bin/env python3
"""
Optimized functions for feature computation
- Numba-accelerated functions
- GPU-accelerated functions using CuPy
"""

from database.processing.features.optimized.numba_functions import (
    moving_average_numba,
    moving_std_numba,
    ewma_numba,
    compute_candle_body_features_numba,
    compute_price_features_numba,
    compute_rsi_numba,
    compute_macd_numba,
    compute_z_score_numba,
    hurst_exponent_numba,
    shannon_entropy_numba,
    compute_future_return_numba,
    compute_max_future_return_numba,
    compute_max_future_drawdown_numba,
    resample_ohlcv_numba
)

# Check if CuPy is available before importing GPU functions
try:
    import cupy
    from database.processing.features.optimized.gpu_functions import (
        initialize_gpu,
        is_gpu_available,
        compute_candle_body_features_gpu,
        compute_z_score_gpu,
        hurst_exponent_gpu,
        shannon_entropy_gpu,
        compute_future_return_gpu,
        compute_max_future_return_gpu,
        compute_max_future_drawdown_gpu,
        compute_batch_features_gpu
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    # Numba functions
    'moving_average_numba',
    'moving_std_numba',
    'ewma_numba',
    'compute_candle_body_features_numba',
    'compute_price_features_numba',
    'compute_rsi_numba',
    'compute_macd_numba',
    'compute_z_score_numba',
    'hurst_exponent_numba',
    'shannon_entropy_numba',
    'compute_future_return_numba',
    'compute_max_future_return_numba',
    'compute_max_future_drawdown_numba',
    'resample_ohlcv_numba',
    'GPU_AVAILABLE'
]

# Add GPU functions if available
if GPU_AVAILABLE:
    __all__.extend([
        'initialize_gpu',
        'is_gpu_available',
        'compute_candle_body_features_gpu',
        'compute_z_score_gpu',
        'hurst_exponent_gpu',
        'shannon_entropy_gpu',
        'compute_future_return_gpu',
        'compute_max_future_return_gpu',
        'compute_max_future_drawdown_gpu',
        'compute_batch_features_gpu'
    ])

    