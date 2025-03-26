#!/usr/bin/env python3
"""
OKXsignal Feature Computation Package
- High-performance implementation using NumPy, Numba, and CuPy
"""

__version__ = "2.0.0"

# Import config and utilities
from features.config import ConfigManager, SMALLINT_COLUMNS
from features.utils import PerformanceMonitor

# Import optimized feature processor
from features.optimized.feature_processor import OptimizedFeatureProcessor

# Check for GPU availability
try:
    import cupy
    from features.optimized import is_gpu_available
    GPU_AVAILABLE = is_gpu_available()
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    'ConfigManager',
    'PerformanceMonitor',
    'OptimizedFeatureProcessor',
    'GPU_AVAILABLE',
    'SMALLINT_COLUMNS'
]