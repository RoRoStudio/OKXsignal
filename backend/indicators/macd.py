"""
macd.py
Computes the Moving Average Convergence Divergence (MACD) indicator.

Usage:
    df = compute_macd(df, fast=12, slow=26, signal=9, col_prefix="MACD")
    # Produces columns: MACD_Line, MACD_Signal, MACD_Hist

Notes:
    - For OKXsignal, MACD is a powerful momentum/trend indicator, 
      especially relevant for volatile pairs with strong directional moves.
"""

import pandas as pd

def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col_prefix: str = "MACD"
) -> pd.DataFrame:
    """
    Adds columns: 
      - {prefix}_Line: The MACD line (fast EMA - slow EMA)
      - {prefix}_Signal: Signal line (EMA of MACD line)
      - {prefix}_Hist: MACD Histogram (MACD line - Signal line)

    :param df: DataFrame with a 'close' column.
    :param fast: period for the "fast" EMA.
    :param slow: period for the "slow" EMA.
    :param signal: period for the signal EMA of the MACD line.
    :param col_prefix: prefix for the columns. 
    :return: DataFrame with the 3 columns appended.
    """

    df = df.copy()

    # Exponential Moving Averages (EMAs)
    # By default, pandas .ewm() uses adjust=False for typical EMA if we want.
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist_line = macd_line - signal_line

    df[f"{col_prefix}_Line"] = macd_line
    df[f"{col_prefix}_Signal"] = signal_line
    df[f"{col_prefix}_Hist"] = hist_line

    return df
