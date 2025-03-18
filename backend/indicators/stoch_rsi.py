"""
stoch_rsi.py
Computes the Stochastic RSI, giving a more responsive oscillator for 
"wild swings" typical in crypto markets.

Usage:
    df = compute_stoch_rsi(df, rsi_period=14, stoch_period=14, smoothK=3, smoothD=3)
    # Produces columns "StochRSI_K" and "StochRSI_D"

Notes:
    - 0 to 1 range. Over 0.8 often considered overbought, under 0.2 oversold. 
    - More sensitive than standard RSI.
"""

import pandas as pd
import numpy as np

from backend.indicators.rsi import compute_rsi  # if you want to reuse your RSI function
# OR you could compute a standard RSI inline.

def compute_stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smoothK: int = 3,
    smoothD: int = 3,
    col_prefix: str = "StochRSI"
) -> pd.DataFrame:
    """
    Calculate Stochastic RSI:
      1) RSI (standard) over 'rsi_period'
      2) Stoch of that RSI over 'stoch_period'
      3) Then smooth the %K line with 'smoothK' 
         and create %D line by smoothing %K again with 'smoothD'.

    Returns DataFrame with:
      {prefix}_K  and {prefix}_D

    :param df: DataFrame with 'close' column.
    :param rsi_period: period for computing RSI
    :param stoch_period: lookback for stoch. Typically same as RSI period.
    :param smoothK: smoothing factor for %K line.
    :param smoothD: smoothing factor for %D line.
    :param col_prefix: e.g. "StochRSI" 
    :return: df with new columns appended.
    """
    df = df.copy()

    # 1) Compute RSI
    df = compute_rsi(df, period=rsi_period, col_name="temp_rsi")

    # 2) Stoch of RSI => (RSI - min(RSI)) / (max(RSI) - min(RSI))
    min_rsi = df["temp_rsi"].rolling(window=stoch_period, min_periods=1).min()
    max_rsi = df["temp_rsi"].rolling(window=stoch_period, min_periods=1).max()

    stoch_rsi = (df["temp_rsi"] - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi.fillna(0, inplace=True)  # in case of zeros in denominator

    # 3) Smooth %K via an EMA or SMA
    # let's do SMA for simplicity here:
    k_sma = stoch_rsi.rolling(window=smoothK, min_periods=1).mean()
    d_sma = k_sma.rolling(window=smoothD, min_periods=1).mean()

    df[f"{col_prefix}_K"] = k_sma
    df[f"{col_prefix}_D"] = d_sma

    # Cleanup
    df.drop("temp_rsi", axis=1, inplace=True)

    return df
