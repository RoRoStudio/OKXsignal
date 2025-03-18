"""
rsi.py
Computes the Relative Strength Index (RSI), a momentum oscillator.

Usage:
    df = compute_rsi(df, period=14, col_name="RSI")
    # Creates an RSI column in the DataFrame.

Notes:
    - RSI is extremely popular for "overbought" (above 70) and "oversold" (below 30) signals.
    - With cryptos, you might see frequent extremes; 
      adapt your thresholds or period accordingly for OKXsignal.
"""

import pandas as pd
import numpy as np

def compute_rsi(
    df: pd.DataFrame,
    period: int = 14,
    col_name: str = "RSI"
) -> pd.DataFrame:
    """
    Adds an RSI column based on the 'close' column.

    The typical formula for RSI uses the smoothed average of up moves vs. down moves.

    :param df: DataFrame with 'close' column.
    :param period: RSI lookback period (14 is typical).
    :param col_name: name of the new RSI column.
    :return: DataFrame with the RSI appended.
    """
    df = df.copy()

    delta = df["close"].diff()
    # Gains/losses
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # Use exponential weighting or simple weighting:
    # Here we use a "Wilders" smoothing recommended for RSI. 
    # Essentially an EMA with alpha = 1/period
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    df[col_name] = rsi

    return df
