"""
atr.py
Calculates the Average True Range (ATR), a measure of volatility.

Usage:
    df = compute_atr(df, period=14, fillna=True)
    # df now has a new column "ATR"

Notes:
    - For OKXsignal, ATR helps size positions in a volatile market or to set dynamic stop-loss distances.
"""

import pandas as pd
import numpy as np

def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    fillna: bool = False,
    column_name: str = "ATR"
) -> pd.DataFrame:
    """
    Adds a new column with ATR values.

    ATR is computed by:
      TR = max( (high - low), abs(high - previous_close), abs(low - previous_close) )
      Then we take an EMA or SMA of TR over 'period' bars. Here we default to an SMA.

    :param df: DataFrame with columns ['high', 'low', 'close'].
               Must have multiple rows for a valid ATR.
    :param period: the lookback period for ATR, typically 14.
    :param fillna: if True, fill NaN with last valid value.
    :param column_name: the name of the new column for ATR.
    :return: the original DataFrame with an 'ATR' column appended.
    """

    df = df.copy()

    # 1) True Range calculation
    df["prev_close"] = df["close"].shift(1)
    df["high_low"] = df["high"] - df["low"]
    df["high_pc"] = (df["high"] - df["prev_close"]).abs()
    df["low_pc"] = (df["low"] - df["prev_close"]).abs()

    df["TR"] = df[["high_low", "high_pc", "low_pc"]].max(axis=1)

    # 2) Average True Range: simple moving average of TR
    df[column_name] = df["TR"].rolling(window=period, min_periods=period).mean()

    # Optional: fill NaN
    if fillna:
        df[column_name].fillna(method="ffill", inplace=True)

    # Cleanup intermediate columns
    df.drop(["prev_close", "high_low", "high_pc", "low_pc", "TR"], axis=1, inplace=True)

    return df
