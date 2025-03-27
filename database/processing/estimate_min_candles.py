#!/usr/bin/env python3
"""
Estimate recommended values for:
- MIN_CANDLES_REQUIRED
- ROLLING_WINDOW
Used for live trading and safe feature computation.
"""

def estimate_requirements():
    # --- Label windows (lookahead targets) ---
    future_return_horizons = {
        '1h': 1,
        '4h': 4,
        '12h': 12,
        '1d': 24,
        '3d': 72,
        '1w': 168,
        '2w': 336
    }
    max_future_return = max(future_return_horizons.values())  # 336
    max_return_window = 24   # for future_max_return_24h_pct
    max_drawdown_window = 12 # for future_max_drawdown_12h_pct

    # --- Rolling indicators (backward window) ---
    ta_windows = {
        'rsi_14': 14,
        'macd_slope': 26,
        'bollinger': 20,
        'atr': 14,
        'mfi': 14,
        'obv_slope': 10,
        'stoch': 14,
        'cci': 14,
        'roc': 10,
        'vwma': 20,
        'zscore': 20,
        'hurst': 100,
        'entropy': 100,
        'autocorr': 10,
        'kurtosis': 20,
        'skewness': 20
    }
    max_indicator_window = max(ta_windows.values())  # 100

    # --- Multi-timeframe dependencies (1d resample + indicator window) ---
    multi_tf_max = 24 + max_indicator_window  # 1D resample + 100

    # --- Cross-pair window ---
    cross_pair_window = 24  # 24h for BTC correlation

    # --- Total required history ---
    min_candles_required = (
        max_future_return +
        max_return_window +
        max_drawdown_window +
        max_indicator_window +
        multi_tf_max +
        cross_pair_window +
        20  # safety buffer
    )

    # --- Rolling window recommendation ---
    # You need to recompute all rows affected by future return shift
    rolling_window = max_future_return + 100  # recompute shift zone + margin

    print("📊 Recommended Configuration:")
    print(f"➡️  MIN_CANDLES_REQUIRED = {min_candles_required}")
    print(f"➡️  ROLLING_WINDOW = {rolling_window} (to cover label shifts and smoothing)")

    return min_candles_required, rolling_window


if __name__ == "__main__":
    estimate_requirements()
