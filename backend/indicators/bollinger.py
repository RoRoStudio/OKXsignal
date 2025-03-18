"""
bollinger.py
Computes Bollinger Bands (Middle, Upper, Lower) based on a moving average 
and a standard deviation multiplier.

Usage:
    df = compute_bollinger_bands(df, period=20, std_multiplier=2.0, fillna=False)
    # Produces columns: BB_Middle, BB_Upper, BB_Lower

Notes:
    - For OKXsignal, Bollinger Bands help identify volatility expansions/contractions 
      and potential breakouts in highly liquid pairs.
"""

import pandas as pd

def compute_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_multiplier: float = 2.0,
    fillna: bool = False,
    col_prefix: str = "BB"
) -> pd.DataFrame:
    """
    Adds Bollinger Band columns to the DataFrame:
      - <prefix>_Middle
      - <prefix>_Upper
      - <prefix>_Lower

    :param df: DataFrame with 'close' column.
    :param period: lookback period for the moving average (often 20).
    :param std_multiplier: standard deviation factor (often 2.0).
    :param fillna: if True, forward fill NaN values.
    :param col_prefix: prefix for the new columns.
    :return: the original DataFrame with Bollinger columns appended.
    """

    df = df.copy()

    # Middle Band = SMA of the close
    middle_col = f"{col_prefix}_Middle"
    upper_col = f"{col_prefix}_Upper"
    lower_col = f"{col_prefix}_Lower"

    df[middle_col] = df["close"].rolling(period, min_periods=period).mean()

    # rolling std
    rolling_std = df["close"].rolling(period, min_periods=period).std()

    df[upper_col] = df[middle_col] + (std_multiplier * rolling_std)
    df[lower_col] = df[middle_col] - (std_multiplier * rolling_std)

    if fillna:
        df[middle_col].fillna(method="ffill", inplace=True)
        df[upper_col].fillna(method="ffill", inplace=True)
        df[lower_col].fillna(method="ffill", inplace=True)

    return df
