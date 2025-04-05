# OKXsignal Feature Computation System

A modular system for computing technical indicators and features for cryptocurrency OHLCV data.

## Overview

This modular feature computation system replaces the monolithic `compute_candles.py` script with a well-organized, maintainable, and extensible architecture. Each feature group is implemented in its own module, making it easier to modify, debug, and extend functionality.

## Features

The system computes the following feature groups:

- **Price Action**: Candle body characteristics, gaps, velocity, etc.
- **Momentum**: RSI, MACD, Stochastics, Williams %R, etc.
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, etc.
- **Volume**: MFI, OBV, Chaikin Money Flow, etc.
- **Statistical**: Z-score, Hurst exponent, entropy, etc.
- **Pattern**: Candlestick patterns (doji, engulfing, hammer, etc.)
- **Time**: Hour of day, day of week, trading sessions, etc.
- **Multi-Timeframe**: Features computed from higher timeframes (4h, 1d)
- **Cross-Pair**: Relative performance and correlations between pairs
- **Labels**: Future returns and profit targets for supervised learning

## Performance Optimization

The system offers two levels of performance optimization:

1. **Numba**: JIT compilation for CPU-intensive calculations
2. **GPU Acceleration**: CuPy-based parallelization for compatible operations

## Configuration

All feature parameters are configurable through:

1. The `config.py` file (default parameters)
2. A configuration file (`config.ini`)
3. Command-line arguments

## Usage

Run the main script with:

```bash
python compute_features.py
```

Common command-line options:

```bash
# Process only specific pairs
python compute_features.py --pairs BTC-USDT,ETH-USDT,SOL-USDT

# Use GPU acceleration
python compute_features.py --use-gpu

# Disable specific feature groups
python compute_features.py --disable-features momentum,pattern

# Set custom rolling window
python compute_features.py --rolling-window 200
```

## Directory Structure

```
/features/
├── __init__.py         # Package initialization
├── config.py           # Configuration management 
├── utils.py            # Utility functions
├── base.py             # Base feature computer class
├── price_action.py     # Price action features
├── momentum.py         # Momentum indicators
├── volatility.py       # Volatility indicators
├── volume.py           # Volume indicators
├── statistical.py      # Statistical features
├── pattern.py          # Candlestick patterns
├── time.py             # Time-based features
├── multi_timeframe.py  # Multi-timeframe features
├── cross_pair.py       # Cross-pair analyses
├── labels.py           # Target/label features
├── optimized/          # Performance-optimized functions
│   ├── __init__.py
│   ├── numba_functions.py
│   └── gpu_functions.py
└── README.md           # This file
```

## Extending the System

To add a new feature group:

1. Create a new module (e.g., `my_features.py`)
2. Implement a class derived from `BaseFeatureComputer`
3. Add parameters to `config.py`
4. Register the class in `__init__.py`
5. Add computation to `compute_features.py`