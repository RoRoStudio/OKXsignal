#!/usr/bin/env python3
"""
OKXsignal Feature Computation Package
- High-performance implementation using NumPy, Numba, and CuPy
"""

__version__ = "2.0.0"

# Import config and utilities
from database.processing.features.config import ConfigManager, SMALLINT_COLUMNS
from database.processing.features.utils import PerformanceMonitor

# Import optimized feature processor
from database.processing.features.optimized.feature_processor import OptimizedFeatureProcessor

# Check for GPU availability
try:
    import cupy
    from database.processing.features.optimized import is_gpu_available
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