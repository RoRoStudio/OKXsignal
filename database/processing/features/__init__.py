#!/usr/bin/env python3
"""
OKXsignal Feature Computation Package
- High-performance implementation using NumPy, Numba, and CuPy
"""

__version__ = "2.0.0"

# Import config and utilities
from database.processing.features.config import ConfigManager, SMALLINT_COLUMNS

# Check for GPU availability
try:
    import cupy
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    'ConfigManager',
    'GPU_AVAILABLE',
    'SMALLINT_COLUMNS'
]