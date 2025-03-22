"""
compute_indicators.py

Removes old manual calculations. Now uses TA-Lib for all indicators (EMA, RSI, MACD, ADX, etc.)
and pandas for multi-timeframe aggregation. Also computes advanced features like
parabolic SAR, Ichimoku, Keltner, Donchian, Supertrend, etc.

Steps:
1) Fetch all 1H rows for a given pair (from DB).
2) Compute 1H indicators (TA-Lib-based) and certain custom logic (perf vs BTC, fib, future returns, etc.).
3) Resample to 4H and 1D (and optionally 1W) to compute same indicators. Merge them back into the 1H DataFrame.
4) Batch update the DB with everything in columns.
"""

import os
import time
import psycopg2
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from database.db import get_connection

# Database access
from database.db import fetch_data

RESAMPLE_MAPPING = {"H": "h", "4H": "4h", "1D": "1d", "1W": "1W"}

def safe_resample(df, timeframe):
    """Fix deprecated resampling values (e.g. H->h)."""
    return df.resample(RESAMPLE_MAPPING.get(timeframe, timeframe))

# ------------------------------------------------------------
# 1) Fetch All 1H Candles from DB
# ------------------------------------------------------------
def fetch_candles_for_pair(pair: str):
    """
    Fetch all 1H candles from DB for the given pair, ordered ascending.
    Expects table public.candles_1h with columns:
      [id, pair, timestamp_ms, open_1h, high_1h, low_1h, close_1h,
       volume_1h, quote_volume_1h, taker_buy_base_1h].
    """
    query = """
    SELECT 
        id,
        pair,
        timestamp_ms,
        open_1h,
        high_1h,
        low_1h,
        close_1h,
        volume_1h,
        quote_volume_1h,
        taker_buy_base_1h
    FROM public.candles_1h
    WHERE pair = %s
    ORDER BY timestamp_ms ASC
    """
    return fetch_data(query, (pair,))


# ------------------------------------------------------------
# 2) Compute Indicators for a Single Timeframe
# ------------------------------------------------------------
def compute_timeframe_indicators(df: pd.DataFrame, prefix: str = "1h") -> pd.DataFrame:
    ocol = f"open_{prefix}"
    hcol = f"high_{prefix}"
    lcol = f"low_{prefix}"
    ccol = f"close_{prefix}"
    vcol = f"volume_{prefix}"

    if not all(col in df.columns for col in [ocol, hcol, lcol, ccol, vcol]):
        return df

    df = df.sort_index()

    # Convert to numeric
    df[ocol] = pd.to_numeric(df[ocol], errors="coerce")
    df[hcol] = pd.to_numeric(df[hcol], errors="coerce")
    df[lcol] = pd.to_numeric(df[lcol], errors="coerce")
    df[ccol] = pd.to_numeric(df[ccol], errors="coerce")
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")

    # For TA-Lib
    c = df[ccol].values
    h = df[hcol].values
    l = df[lcol].values
    v = df[vcol].values

    # 1) EMAs
    df[f"ema_12_{prefix}"]  = talib.EMA(c, timeperiod=12)
    df[f"ema_26_{prefix}"]  = talib.EMA(c, timeperiod=26)
    df[f"ema_50_{prefix}"]  = talib.EMA(c, timeperiod=50)
    df[f"ema_100_{prefix}"] = talib.EMA(c, timeperiod=100)
    df[f"ema_200_{prefix}"] = talib.EMA(c, timeperiod=200)

    # 2) RSI
    rsi_vals = talib.RSI(c, timeperiod=14)
    df[f"rsi_{prefix}"] = rsi_vals
    df[f"rsi_slope_{prefix}"] = pd.Series(rsi_vals, index=df.index).diff()

    # 3) Stoch RSI
    fastk, fastd = talib.STOCHRSI(c, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
    df[f"stoch_rsi_k_{prefix}"] = fastk
    df[f"stoch_rsi_d_{prefix}"] = fastd

    # 4) MACD
    macd_line, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df[f"macd_line_{prefix}"]     = macd_line
    df[f"macd_signal_{prefix}"]   = macd_signal
    df[f"macd_hist_{prefix}"]     = macd_hist
    df[f"macd_slope_{prefix}"] = pd.Series(macd_line, index=df.index).diff()
    df[f"macd_hist_slope_{prefix}"] = pd.Series(macd_hist, index=df.index).diff()

    # 5) ADX & CCI
    df[f"adx_{prefix}"] = talib.ADX(h, l, c, timeperiod=14)
    df[f"cci_{prefix}"] = talib.CCI(h, l, c, timeperiod=20)

    # 6) ATR
    df[f"atr_{prefix}"] = talib.ATR(h, l, c, timeperiod=14)

    # 7) Bollinger
    upb, mid, lob = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df[f"bollinger_upper_{prefix}"] = upb
    df[f"bollinger_middle_{prefix}"] = mid
    df[f"bollinger_lower_{prefix}"] = lob
    df[f"bollinger_width_{prefix}"] = (upb - lob) / pd.Series(mid, index=df.index).replace(0, np.nan)

    # 8) Keltner
    ema_20 = talib.EMA(c, timeperiod=20)
    df[f"keltner_channel_upper_{prefix}"] = ema_20 + 2.0 * df[f"atr_{prefix}"]
    df[f"keltner_channel_lower_{prefix}"] = ema_20 - 2.0 * df[f"atr_{prefix}"]

    # 9) OBV & MFI
    obv_vals = talib.OBV(c, v)
    df[f"obv_{prefix}"]       = obv_vals
    df[f"obv_slope_{prefix}"] = pd.Series(obv_vals, index=df.index).diff()
    df[f"money_flow_index_{prefix}"] = talib.MFI(h, l, c, v, timeperiod=14)

    # Only do buy_volume_ratio for 1H
    if prefix == "1h":
        df["buy_volume_ratio_1h"] = np.where(
            df["volume_1h"] == 0,
            np.nan,
            df["taker_buy_base_1h"] / df["volume_1h"]
        )

    # 10) Donchian + highest/lowest 20
    dc_period = 20
    df[f"donchian_channel_upper_{prefix}"] = df[hcol].rolling(dc_period).max()
    df[f"donchian_channel_lower_{prefix}"] = df[lcol].rolling(dc_period).min()
    df[f"highest_high_20_{prefix}"]        = df[hcol].rolling(dc_period).max()
    df[f"lowest_low_20_{prefix}"]          = df[lcol].rolling(dc_period).min()

    # 11) Supertrend
    df = _compute_supertrend_proper(df, period=10, multiplier=3.0, prefix=prefix)

    # 12) Ichimoku
    df = _compute_ichimoku(df, prefix=prefix)

    # 13) Parabolic SAR
    psar_vals = talib.SAR(h, l, acceleration=0.02, maximum=0.2)
    df[f"parabolic_sar_{prefix}"] = psar_vals

    # 14) Performance vs. BTC/ETH
    btc_close_col = f"btc_close_{prefix}"
    eth_close_col = f"eth_close_{prefix}"
    if btc_close_col in df.columns:
        df[f"perf_vs_btc_pct_{prefix}"] = (df[ccol] / df[btc_close_col] - 1.0) * 100.0
    if eth_close_col in df.columns:
        df[f"perf_vs_eth_pct_{prefix}"] = (df[ccol] / df[eth_close_col] - 1.0) * 100.0

    # 15) Fibonacci levels
    roll_max20 = df[hcol].rolling(20).max()
    roll_min20 = df[lcol].rolling(20).min()
    rng20 = roll_max20 - roll_min20
    fibs = [("0_236", 0.236), ("0_382", 0.382), ("0_5", 0.5), ("0_618", 0.618), ("0_786", 0.786)]
    for lvl_str, lvl_val in fibs:
        df[f"fib_level_{lvl_str}_{prefix}"] = roll_max20 - rng20 * lvl_val

    # 16) VWAP
    df[f"vwap_{prefix}"] = (
        (df[ccol] * df[vcol]).groupby(df.index.date).cumsum()
        /
        df[vcol].groupby(df.index.date).cumsum()
    )

    # 17) Future Returns
    horizon_shifts = {
        "1h": {"1h": 1, "4h": 4, "12h": 12, "1d": 24, "3d": 72, "1w": 168, "2w": 336},
        "4h": {"4h": 1, "12h": 3, "1d": 6, "3d": 18, "1w": 42,  "2w": 84},
        "1d": {"1d": 1, "3d": 3, "1w": 7,  "2w": 14},
    }
    if prefix in horizon_shifts:
        for horizon_label, shift_count in horizon_shifts[prefix].items():
            df[f"future_return_{horizon_label}_pct"] = (
                df[ccol].shift(-shift_count) / df[ccol] - 1.0
            ) * 100.0

    # 18) Market Regime
    conds = [
        ((df[ccol] > df[f"ema_200_{prefix}"]) & (df[f"adx_{prefix}"] > 25)),
        (df[ccol] > df[f"ema_50_{prefix}"]),
    ]
    df[f"market_regime_{prefix}"] = np.select(conds, [2, 1], default=0)

    # 19) Seasonality
    df[f"hour_of_day"] = df.index.hour
    df[f"day_of_week"] = df.index.dayofweek

    # 20) Extra
    df[f"volume_change_pct_{prefix}"] = df[vcol].pct_change()

    return df

def _compute_supertrend_proper(df: pd.DataFrame, period=10, multiplier=3.0, prefix="1h") -> pd.DataFrame:
    hcol = f"high_{prefix}"
    lcol = f"low_{prefix}"
    ccol = f"close_{prefix}"

    high = df[hcol].values
    low = df[lcol].values
    close = df[ccol].values

    tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values

    hl2 = (high + low) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    supertrend = np.zeros(len(df))
    direction = np.ones(len(df))

    for i in range(1, len(df)):
        curr_close = close[i]
        prev_close = close[i - 1]

        if (upperband[i] < final_upper[i - 1]) or (prev_close > final_upper[i - 1]):
            final_upper[i] = upperband[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if (lowerband[i] > final_lower[i - 1]) or (prev_close < final_lower[i - 1]):
            final_lower[i] = lowerband[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if supertrend[i - 1] == final_upper[i - 1] and curr_close <= final_upper[i]:
            direction[i] = -1
        elif supertrend[i - 1] == final_upper[i - 1] and curr_close > final_upper[i]:
            direction[i] = 1
        elif supertrend[i - 1] == final_lower[i - 1] and curr_close >= final_lower[i]:
            direction[i] = 1
        elif supertrend[i - 1] == final_lower[i - 1] and curr_close < final_lower[i]:
            direction[i] = -1

        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    df[f"supertrend_{prefix}"] = supertrend
    df[f"supertrend_direction_{prefix}"] = direction
    df[f"supertrend_upper_{prefix}"] = final_upper
    df[f"supertrend_lower_{prefix}"] = final_lower
    return df

def _compute_ichimoku(df: pd.DataFrame, prefix="1h") -> pd.DataFrame:
    hcol = f"high_{prefix}"
    lcol = f"low_{prefix}"

    nine_high = df[hcol].rolling(9).max()
    nine_low  = df[lcol].rolling(9).min()
    conv_line = (nine_high + nine_low) / 2.0

    twenty_six_high = df[hcol].rolling(26).max()
    twenty_six_low  = df[lcol].rolling(26).min()
    base_line       = (twenty_six_high + twenty_six_low) / 2.0

    df[f"ichimoku_cloud_upper_{prefix}"] = ((conv_line + base_line) / 2.0).shift(26)
    fifty_two_high = df[hcol].rolling(52).max()
    fifty_two_low  = df[lcol].rolling(52).min()
    df[f"ichimoku_cloud_lower_{prefix}"] = ((fifty_two_high + fifty_two_low) / 2.0).shift(26)

    return df

# ------------------------------------------------------------
# 3) Build 4H / 1D / (optionally 1W) Aggregations
# ------------------------------------------------------------
def aggregate_timeframe(df_1h: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe not in ["4H", "1D", "1W"]:
        raise ValueError("Timeframe must be one of: 4H, 1D, 1W.")

    if df_1h.index.name != "datetime_1h":
        df_1h = df_1h.set_index("datetime_1h", drop=False)

    df_agg = safe_resample(df_1h, timeframe).agg({
        "open_1h": "first",
        "high_1h": "max",
        "low_1h": "min",
        "close_1h": "last",
        "volume_1h": "sum",
    })
    df_agg.rename(columns={
        "open_1h": f"open_{timeframe.lower()}",
        "high_1h": f"high_{timeframe.lower()}",
        "low_1h": f"low_{timeframe.lower()}",
        "close_1h": f"close_{timeframe.lower()}",
        "volume_1h": f"volume_{timeframe.lower()}",
    }, inplace=True)

    return df_agg

def merge_timeframe_into_1h(df_1h: pd.DataFrame, df_other: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df_1h.index.name != "datetime_1h":
        df_1h = df_1h.set_index("datetime_1h", drop=False)
    if df_other.index.name is None:
        df_other.index.name = f"datetime_{timeframe.lower()}"

    df_1h["idx_1h"] = df_1h.index.astype(np.int64)
    df_other["idx_other"] = df_other.index.astype(np.int64)

    merged = pd.merge_asof(
        df_1h.sort_values("idx_1h"),
        df_other.sort_values("idx_other"),
        left_on="idx_1h",
        right_on="idx_other",
        direction="backward",
        suffixes=("", f"_{timeframe.lower()}"),
    )
    merged.drop(["idx_1h", "idx_other"], axis=1, inplace=True)

    merged = merged.sort_index()
    return merged


# ------------------------------------------------------------
# 4) Final Update to Database
# ------------------------------------------------------------
def update_candles_table(df_final: pd.DataFrame):
    """
    Ultra-fast update method using temp table + COPY + UPDATE JOIN.
    Replaces the old batch-update with execute_values.
    """

    import io
    import time

    # Step 1: Clean NaNs/Infs
    s_preclean = time.time()
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final = df_final.where(pd.notnull(df_final), None)
    e_preclean = time.time() - s_preclean
    print(f"🚀 Cleaned NaNs/Infs in {e_preclean:.3f}s")

    # Step 2: Convert all object columns to floats
    for col in df_final.columns:
        if col != "id" and df_final[col].dtype == "object":
            try:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
            except Exception:
                pass

    row_count = len(df_final)
    if row_count == 0:
        print("⚠️ No rows to update.")
        return

    print(f"👷 Preparing {row_count} rows for fast update...")

    # Step 3: Copy DataFrame to CSV buffer
    s_buffer = time.time()
    buffer = io.StringIO()
        # Define the exact order of columns that match your actual DB schema (exclude non-DB columns)
    columns_to_write = [
        "id", "ema_12_1h", "ema_26_1h", ..., "volume_1w", "market_regime_1w"  # all 1H/4H/1D/1W DB columns only
    ]

    df_final[columns_to_write].to_csv(
        buffer, sep="|", header=False, index=False, na_rep="\\N", float_format="%.10f"
    )

    buffer.seek(0)
    e_buffer = time.time() - s_buffer
    print(f"🧃 Prepared CSV buffer for COPY in {e_buffer:.3f}s")

    rows = []

    # Clean up object columns into floats
    for col in df_final.columns:
        if col != "id" and df_final[col].dtype == "object":
            try:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
            except Exception:
                pass

    row_count = len(df_final)
    if row_count == 0:
        print("⚠️ No rows to update.")
        return

    print(f"👷 Building row tuples for {row_count} rows...")
    start_time = time.time()

    # We'll do a quick pass to show partial progress
    for i, (idx, row) in enumerate(df_final.iterrows()):
        if i % 1000 == 0 or i == row_count - 1:
            fraction = (i+1) / row_count
            elapsed = time.time() - start_time
            remain_sec = elapsed / fraction - elapsed if fraction else 0
            remain_str = str(timedelta(seconds=int(remain_sec)))
            print(f"    Progress: {i+1}/{row_count} (~{fraction*100:.1f}%), Elapsed={elapsed:.2f}s, ETA={remain_str}")

    # 4) Actually build the tuples
    s_tuple_build = time.time()
    rows = []
    for idx, row in df_final.iterrows():
        def g(col):
            val = row.get(col, None)
            if pd.isna(val):
                return None
            return float(val)

        def g_int(col):
            val = row.get(col, None)
            if pd.isna(val):
                return None
            return int(float(val))

        val_tuple = (
            # We have to ensure the order of these fields EXACTLY matches
            # the order in the sub-select above.

            row["id"],

            # 1H columns (floats)
            g("ema_12_1h"), g("ema_26_1h"), g("ema_50_1h"), g("ema_100_1h"), g("ema_200_1h"),
            g("rsi_1h"), g("slope_rsi_1h"), g("stoch_rsi_k_1h"), g("stoch_rsi_d_1h"),
            g("macd_line_1h"), g("macd_signal_1h"), g("macd_hist_1h"), g("macd_slope_1h"), g("macd_hist_slope_1h"),
            g("adx_1h"), g("cci_1h"), g("atr_1h"),
            g("bollinger_upper_1h"), g("bollinger_middle_1h"), g("bollinger_lower_1h"), g("bollinger_width_1h"),
            g("keltner_channel_upper_1h"), g("keltner_channel_lower_1h"),
            g("obv_1h"), g("obv_slope_1h"), g("money_flow_index_1h"),
            g("donchian_channel_upper_1h"), g("donchian_channel_lower_1h"),
            g("supertrend_upper_1h"), g("supertrend_lower_1h"),
            g("ichimoku_cloud_upper_1h"), g("ichimoku_cloud_lower_1h"), g("parabolic_sar_1h"),
            g("perf_vs_btc_pct_1h"), g("perf_vs_eth_pct_1h"),
            g("fib_level_0_236_1h"), g("fib_level_0_382_1h"), g("fib_level_0_5_1h"), g("fib_level_0_618_1h"), g("fib_level_0_786_1h"),
            g("vwap_1h"),
            g("future_return_1h_pct"), g("future_return_4h_pct"), g("future_return_12h_pct"), g("future_return_1d_pct"), g("future_return_3d_pct"), g("future_return_1w_pct"), g("future_return_2w_pct"),
            g_int("market_regime_1h"),
            g_int("hour_of_day"),
            g_int("day_of_week"),
            g("volume_change_pct_1h"),

            # 4H
            g("open_4h"), g("high_4h"), g("low_4h"), g("close_4h"), g("volume_4h"),
            g("ema_12_4h"), g("ema_26_4h"), g("ema_50_4h"), g("ema_100_4h"), g("ema_200_4h"),
            g("rsi_4h"), g("rsi_slope_4h"), g("stoch_rsi_k_4h"), g("stoch_rsi_d_4h"),
            g("macd_line_4h"), g("macd_signal_4h"), g("macd_hist_4h"), g("macd_slope_4h"), g("macd_hist_slope_4h"),
            g("adx_4h"), g("cci_4h"), g("atr_4h"),
            g("bollinger_upper_4h"), g("bollinger_middle_4h"), g("bollinger_lower_4h"), g("bollinger_width_4h"),
            g("keltner_channel_upper_4h"), g("keltner_channel_lower_4h"),
            g("obv_4h"), g("obv_slope_4h"), g("money_flow_index_4h"),
            g("donchian_channel_upper_4h"), g("donchian_channel_lower_4h"),
            g("supertrend_upper_4h"), g("supertrend_lower_4h"),
            g("ichimoku_cloud_upper_4h"), g("ichimoku_cloud_lower_4h"), g("parabolic_sar_4h"),
            g("perf_vs_btc_pct_4h"), g("perf_vs_eth_pct_4h"),
            g("fib_level_0_236_4h"), g("fib_level_0_382_4h"), g("fib_level_0_5_4h"), g("fib_level_0_618_4h"), g("fib_level_0_786_4h"),
            g("vwap_4h"),
            g_int("market_regime_4h"),
            g("volume_change_pct_4h"),

            # 1D
            g("open_1d"), g("high_1d"), g("low_1d"), g("close_1d"), g("volume_1d"),
            g("ema_12_1d"), g("ema_26_1d"), g("ema_50_1d"), g("ema_100_1d"), g("ema_200_1d"),
            g("rsi_1d"), g("rsi_slope_1d"), g("stoch_rsi_k_1d"), g("stoch_rsi_d_1d"),
            g("macd_line_1d"), g("macd_signal_1d"), g("macd_hist_1d"), g("macd_slope_1d"), g("macd_hist_slope_1d"),
            g("adx_1d"), g("cci_1d"), g("atr_1d"),
            g("bollinger_upper_1d"), g("bollinger_middle_1d"), g("bollinger_lower_1d"), g("bollinger_width_1d"),
            g("keltner_channel_upper_1d"), g("keltner_channel_lower_1d"),
            g("obv_1d"), g("obv_slope_1d"), g("money_flow_index_1d"),
            g("donchian_channel_upper_1d"), g("donchian_channel_lower_1d"),
            g("supertrend_upper_1d"), g("supertrend_lower_1d"),
            g("ichimoku_cloud_upper_1d"), g("ichimoku_cloud_lower_1d"), g("parabolic_sar_1d"),
            g("perf_vs_btc_pct_1d"), g("perf_vs_eth_pct_1d"),
            g("fib_level_0_236_1d"), g("fib_level_0_382_1d"), g("fib_level_0_5_1d"), g("fib_level_0_618_1d"), g("fib_level_0_786_1d"),
            g("vwap_1d"),
            g_int("market_regime_1d"),
            g("volume_change_pct_1d"),

            # 1W
            g("open_1w"), g("high_1w"), g("low_1w"), g("close_1w"), g("volume_1w"),
            g_int("market_regime_1w"),
        )
        rows.append(val_tuple)
    e_tuple_build = time.time() - s_tuple_build
    print(f"✅ Done building {len(rows)} row tuples in {e_tuple_build:.3f}s")

    if not rows:
        print("⚠️ No rows to update after all.")
        return

    # Step 4: COPY INTO temp table + UPDATE JOIN
    with get_connection() as conn:
        with conn.cursor() as cur:
            s_sql = time.time()
            # Create temp table
            cur.execute("""
                DROP TABLE IF EXISTS temp_candles_1h;
                CREATE TEMP TABLE temp_candles_1h (LIKE public.candles_1h INCLUDING DEFAULTS) ON COMMIT DROP;
            """)
            # Copy data into temp table
            cur.copy_from(buffer, 'temp_candles_1h', sep='|', null='\\N')
            conn.commit()
            print("📥 COPY into temp_candles_1h completed.")

            # Perform in-place update using join on ID
            cur.execute("""
                UPDATE public.candles_1h AS t
                    -- 1H columns
                    ema_12_1h = CAST(d.ema_12_1h AS double precision),
                    ema_26_1h = CAST(d.ema_26_1h AS double precision),
                    ema_50_1h = CAST(d.ema_50_1h AS double precision),
                    ema_100_1h = CAST(d.ema_100_1h AS double precision),
                    ema_200_1h = CAST(d.ema_200_1h AS double precision),
                    rsi_1h = CAST(d.rsi_1h AS double precision),
                    slope_rsi_1h = CAST(d.slope_rsi_1h AS double precision),
                    stoch_rsi_k_1h = CAST(d.stoch_rsi_k_1h AS double precision),
                    stoch_rsi_d_1h = CAST(d.stoch_rsi_d_1h AS double precision),
                    macd_line_1h = CAST(d.macd_line_1h AS double precision),
                    macd_signal_1h = CAST(d.macd_signal_1h AS double precision),
                    macd_hist_1h = CAST(d.macd_hist_1h AS double precision),
                    macd_slope_1h = CAST(d.macd_slope_1h AS double precision),
                    macd_hist_slope_1h = CAST(d.macd_hist_slope_1h AS double precision),
                    adx_1h = CAST(d.adx_1h AS double precision),
                    cci_1h = CAST(d.cci_1h AS double precision),
                    atr_1h = CAST(d.atr_1h AS double precision),
                    bollinger_upper_1h = CAST(d.bollinger_upper_1h AS double precision),
                    bollinger_middle_1h = CAST(d.bollinger_middle_1h AS double precision),
                    bollinger_lower_1h = CAST(d.bollinger_lower_1h AS double precision),
                    bollinger_width_1h = CAST(d.bollinger_width_1h AS double precision),
                    keltner_channel_upper_1h = CAST(d.keltner_channel_upper_1h AS double precision),
                    keltner_channel_lower_1h = CAST(d.keltner_channel_lower_1h AS double precision),
                    obv_1h = CAST(d.obv_1h AS double precision),
                    obv_slope_1h = CAST(d.obv_slope_1h AS double precision),
                    money_flow_index_1h = CAST(d.money_flow_index_1h AS double precision),
                    donchian_channel_upper_1h = CAST(d.donchian_channel_upper_1h AS double precision),
                    donchian_channel_lower_1h = CAST(d.donchian_channel_lower_1h AS double precision),
                    supertrend_upper_1h = CAST(d.supertrend_upper_1h AS double precision),
                    supertrend_lower_1h = CAST(d.supertrend_lower_1h AS double precision),
                    ichimoku_cloud_upper_1h = CAST(d.ichimoku_cloud_upper_1h AS double precision),
                    ichimoku_cloud_lower_1h = CAST(d.ichimoku_cloud_lower_1h AS double precision),
                    parabolic_sar_1h = CAST(d.parabolic_sar_1h AS double precision),
                    perf_vs_btc_pct_1h = CAST(d.perf_vs_btc_pct_1h AS double precision),
                    perf_vs_eth_pct_1h = CAST(d.perf_vs_eth_pct_1h AS double precision),
                    fib_level_0_236_1h = CAST(d.fib_level_0_236_1h AS double precision),
                    fib_level_0_382_1h = CAST(d.fib_level_0_382_1h AS double precision),
                    fib_level_0_5_1h = CAST(d.fib_level_0_5_1h AS double precision),
                    fib_level_0_618_1h = CAST(d.fib_level_0_618_1h AS double precision),
                    fib_level_0_786_1h = CAST(d.fib_level_0_786_1h AS double precision),
                    vwap_1h = CAST(d.vwap_1h AS double precision),
                    future_return_1h_pct = CAST(d.future_return_1h_pct AS double precision),
                    future_return_4h_pct = CAST(d.future_return_4h_pct AS double precision),
                    future_return_12h_pct = CAST(d.future_return_12h_pct AS double precision),
                    future_return_1d_pct = CAST(d.future_return_1d_pct AS double precision),
                    future_return_3d_pct = CAST(d.future_return_3d_pct AS double precision),
                    future_return_1w_pct = CAST(d.future_return_1w_pct AS double precision),
                    future_return_2w_pct = CAST(d.future_return_2w_pct AS double precision),
                    market_regime_1h = CAST(d.market_regime_1h AS int),
                    hour_of_day = CAST(d.hour_of_day AS int),
                    day_of_week = CAST(d.day_of_week AS int),
                    volume_change_pct_1h = CAST(d.volume_change_pct_1h AS double precision),

                    -- 4H columns
                    open_4h = CAST(d.open_4h AS double precision),
                    high_4h = CAST(d.high_4h AS double precision),
                    low_4h = CAST(d.low_4h AS double precision),
                    close_4h = CAST(d.close_4h AS double precision),
                    volume_4h = CAST(d.volume_4h AS double precision),
                    ema_12_4h = CAST(d.ema_12_4h AS double precision),
                    ema_26_4h = CAST(d.ema_26_4h AS double precision),
                    ema_50_4h = CAST(d.ema_50_4h AS double precision),
                    ema_100_4h = CAST(d.ema_100_4h AS double precision),
                    ema_200_4h = CAST(d.ema_200_4h AS double precision),
                    rsi_4h = CAST(d.rsi_4h AS double precision),
                    rsi_slope_4h = CAST(d.rsi_slope_4h AS double precision),
                    stoch_rsi_k_4h = CAST(d.stoch_rsi_k_4h AS double precision),
                    stoch_rsi_d_4h = CAST(d.stoch_rsi_d_4h AS double precision),
                    macd_line_4h = CAST(d.macd_line_4h AS double precision),
                    macd_signal_4h = CAST(d.macd_signal_4h AS double precision),
                    macd_hist_4h = CAST(d.macd_hist_4h AS double precision),
                    macd_slope_4h = CAST(d.macd_slope_4h AS double precision),
                    macd_hist_slope_4h = CAST(d.macd_hist_slope_4h AS double precision),
                    adx_4h = CAST(d.adx_4h AS double precision),
                    cci_4h = CAST(d.cci_4h AS double precision),
                    atr_4h = CAST(d.atr_4h AS double precision),
                    bollinger_upper_4h = CAST(d.bollinger_upper_4h AS double precision),
                    bollinger_middle_4h = CAST(d.bollinger_middle_4h AS double precision),
                    bollinger_lower_4h = CAST(d.bollinger_lower_4h AS double precision),
                    bollinger_width_4h = CAST(d.bollinger_width_4h AS double precision),
                    keltner_channel_upper_4h = CAST(d.keltner_channel_upper_4h AS double precision),
                    keltner_channel_lower_4h = CAST(d.keltner_channel_lower_4h AS double precision),
                    obv_4h = CAST(d.obv_4h AS double precision),
                    obv_slope_4h = CAST(d.obv_slope_4h AS double precision),
                    money_flow_index_4h = CAST(d.money_flow_index_4h AS double precision),
                    donchian_channel_upper_4h = CAST(d.donchian_channel_upper_4h AS double precision),
                    donchian_channel_lower_4h = CAST(d.donchian_channel_lower_4h AS double precision),
                    supertrend_upper_4h = CAST(d.supertrend_upper_4h AS double precision),
                    supertrend_lower_4h = CAST(d.supertrend_lower_4h AS double precision),
                    ichimoku_cloud_upper_4h = CAST(d.ichimoku_cloud_upper_4h AS double precision),
                    ichimoku_cloud_lower_4h = CAST(d.ichimoku_cloud_lower_4h AS double precision),
                    parabolic_sar_4h = CAST(d.parabolic_sar_4h AS double precision),
                    perf_vs_btc_pct_4h = CAST(d.perf_vs_btc_pct_4h AS double precision),
                    perf_vs_eth_pct_4h = CAST(d.perf_vs_eth_pct_4h AS double precision),
                    fib_level_0_236_4h = CAST(d.fib_level_0_236_4h AS double precision),
                    fib_level_0_382_4h = CAST(d.fib_level_0_382_4h AS double precision),
                    fib_level_0_5_4h = CAST(d.fib_level_0_5_4h AS double precision),
                    fib_level_0_618_4h = CAST(d.fib_level_0_618_4h AS double precision),
                    fib_level_0_786_4h = CAST(d.fib_level_0_786_4h AS double precision),
                    vwap_4h = CAST(d.vwap_4h AS double precision),
                    market_regime_4h = CAST(d.market_regime_4h AS int),
                    volume_change_pct_4h = CAST(d.volume_change_pct_4h AS double precision),

                    -- 1D columns
                    open_1d = CAST(d.open_1d AS double precision),
                    high_1d = CAST(d.high_1d AS double precision),
                    low_1d = CAST(d.low_1d AS double precision),
                    close_1d = CAST(d.close_1d AS double precision),
                    volume_1d = CAST(d.volume_1d AS double precision),
                    ema_12_1d = CAST(d.ema_12_1d AS double precision),
                    ema_26_1d = CAST(d.ema_26_1d AS double precision),
                    ema_50_1d = CAST(d.ema_50_1d AS double precision),
                    ema_100_1d = CAST(d.ema_100_1d AS double precision),
                    ema_200_1d = CAST(d.ema_200_1d AS double precision),
                    rsi_1d = CAST(d.rsi_1d AS double precision),
                    rsi_slope_1d = CAST(d.rsi_slope_1d AS double precision),
                    stoch_rsi_k_1d = CAST(d.stoch_rsi_k_1d AS double precision),
                    stoch_rsi_d_1d = CAST(d.stoch_rsi_d_1d AS double precision),
                    macd_line_1d = CAST(d.macd_line_1d AS double precision),
                    macd_signal_1d = CAST(d.macd_signal_1d AS double precision),
                    macd_hist_1d = CAST(d.macd_hist_1d AS double precision),
                    macd_slope_1d = CAST(d.macd_slope_1d AS double precision),
                    macd_hist_slope_1d = CAST(d.macd_hist_slope_1d AS double precision),
                    adx_1d = CAST(d.adx_1d AS double precision),
                    cci_1d = CAST(d.cci_1d AS double precision),
                    atr_1d = CAST(d.atr_1d AS double precision),
                    bollinger_upper_1d = CAST(d.bollinger_upper_1d AS double precision),
                    bollinger_middle_1d = CAST(d.bollinger_middle_1d AS double precision),
                    bollinger_lower_1d = CAST(d.bollinger_lower_1d AS double precision),
                    bollinger_width_1d = CAST(d.bollinger_width_1d AS double precision),
                    keltner_channel_upper_1d = CAST(d.keltner_channel_upper_1d AS double precision),
                    keltner_channel_lower_1d = CAST(d.keltner_channel_lower_1d AS double precision),
                    obv_1d = CAST(d.obv_1d AS double precision),
                    obv_slope_1d = CAST(d.obv_slope_1d AS double precision),
                    money_flow_index_1d = CAST(d.money_flow_index_1d AS double precision),
                    donchian_channel_upper_1d = CAST(d.donchian_channel_upper_1d AS double precision),
                    donchian_channel_lower_1d = CAST(d.donchian_channel_lower_1d AS double precision),
                    supertrend_upper_1d = CAST(d.supertrend_upper_1d AS double precision),
                    supertrend_lower_1d = CAST(d.supertrend_lower_1d AS double precision),
                    ichimoku_cloud_upper_1d = CAST(d.ichimoku_cloud_upper_1d AS double precision),
                    ichimoku_cloud_lower_1d = CAST(d.ichimoku_cloud_lower_1d AS double precision),
                    parabolic_sar_1d = CAST(d.parabolic_sar_1d AS double precision),
                    perf_vs_btc_pct_1d = CAST(d.perf_vs_btc_pct_1d AS double precision),
                    perf_vs_eth_pct_1d = CAST(d.perf_vs_eth_pct_1d AS double precision),
                    fib_level_0_236_1d = CAST(d.fib_level_0_236_1d AS double precision),
                    fib_level_0_382_1d = CAST(d.fib_level_0_382_1d AS double precision),
                    fib_level_0_5_1d = CAST(d.fib_level_0_5_1d AS double precision),
                    fib_level_0_618_1d = CAST(d.fib_level_0_618_1d AS double precision),
                    fib_level_0_786_1d = CAST(d.fib_level_0_786_1d AS double precision),
                    vwap_1d = CAST(d.vwap_1d AS double precision),
                    market_regime_1d = CAST(d.market_regime_1d AS int),
                    volume_change_pct_1d = CAST(d.volume_change_pct_1d AS double precision),

                    -- Optionally if you did 1W
                    open_1w = CAST(d.open_1w AS double precision),
                    high_1w = CAST(d.high_1w AS double precision),
                    low_1w = CAST(d.low_1w AS double precision),
                    close_1w = CAST(d.close_1w AS double precision),
                    volume_1w = CAST(d.volume_1w AS double precision),
                    market_regime_1w = CAST(d.market_regime_1w AS int)
            """)
            affected = cur.rowcount
            conn.commit()
            e_sql = time.time() - s_sql
            print(f"✅ Updated {affected} rows using COPY+JOIN in {e_sql:.3f}s.")


# ------------------------------------------------------------
# 5) Main Orchestrator
# ------------------------------------------------------------
def compute_all_indicators_for_pair(pair: str):
    print(f"🔍 Fetching all 1H candles for pair={pair} ...")
    data = fetch_candles_for_pair(pair)
    if not data:
        print(f"⚠️ No 1H candles found for pair={pair}. Nothing to do.")
        return

    df_1h = pd.DataFrame(data, columns=[
        "id", "pair", "timestamp_ms",
        "open_1h", "high_1h", "low_1h", "close_1h",
        "volume_1h", "quote_volume_1h", "taker_buy_base_1h"
    ])
    df_1h["datetime_1h"] = pd.to_datetime(df_1h["timestamp_ms"], unit="ms", utc=True)
    df_1h.set_index("datetime_1h", inplace=True, drop=False)

    # Merge BTC & ETH
    btc_data = fetch_candles_for_pair("BTC-USDT")
    eth_data = fetch_candles_for_pair("ETH-USDT")
    df_btc = pd.DataFrame(btc_data, columns=[
        "id", "pair", "timestamp_ms",
        "open_1h", "high_1h", "low_1h", "close_1h",
        "volume_1h", "quote_volume_1h", "taker_buy_base_1h"
    ])
    df_btc["datetime_1h"] = pd.to_datetime(df_btc["timestamp_ms"], unit="ms", utc=True)
    df_btc.set_index("datetime_1h", inplace=True, drop=False)
    df_btc.rename(columns={"close_1h": "btc_close_1h"}, inplace=True)

    df_eth = pd.DataFrame(eth_data, columns=[
        "id", "pair", "timestamp_ms",
        "open_1h", "high_1h", "low_1h", "close_1h",
        "volume_1h", "quote_volume_1h", "taker_buy_base_1h"
    ])
    df_eth["datetime_1h"] = pd.to_datetime(df_eth["timestamp_ms"], unit="ms", utc=True)
    df_eth.set_index("datetime_1h", inplace=True, drop=False)
    df_eth.rename(columns={"close_1h": "eth_close_1h"}, inplace=True)

    df_1h = df_1h.merge(df_btc[["btc_close_1h"]], left_index=True, right_index=True, how="left")
    df_1h = df_1h.merge(df_eth[["eth_close_1h"]], left_index=True, right_index=True, how="left")
    print("✅ Merged BTC & ETH 1H closes into main DF.")

    # Compute 1H
    df_1h = compute_timeframe_indicators(df_1h, prefix="1h")

    # Build 4H
    df_4h = aggregate_timeframe(df_1h, "4H")
    df_btc_4h = safe_resample(df_btc, "4H").agg({"btc_close_1h": "last"}).rename(columns={"btc_close_1h": "btc_close_4h"})
    df_eth_4h = safe_resample(df_eth, "4H").agg({"eth_close_1h": "last"}).rename(columns={"eth_close_1h": "eth_close_4h"})
    df_4h = df_4h.merge(df_btc_4h, left_index=True, right_index=True, how="left")
    df_4h = df_4h.merge(df_eth_4h, left_index=True, right_index=True, how="left")
    df_4h = compute_timeframe_indicators(df_4h, prefix="4h")
    df_merged_4h = merge_timeframe_into_1h(df_1h, df_4h, "4H")

    # Build 1D
    df_1d = aggregate_timeframe(df_1h, "1D")
    df_btc_1d = safe_resample(df_btc, "1D").agg({"btc_close_1h": "last"}).rename(columns={"btc_close_1h": "btc_close_1d"})
    df_eth_1d = safe_resample(df_eth, "1D").agg({"eth_close_1h": "last"}).rename(columns={"eth_close_1h": "eth_close_1d"})
    df_1d = df_1d.merge(df_btc_1d, left_index=True, right_index=True, how="left")
    df_1d = df_1d.merge(df_eth_1d, left_index=True, right_index=True, how="left")
    df_1d = compute_timeframe_indicators(df_1d, prefix="1d")
    df_merged_1d = merge_timeframe_into_1h(df_merged_4h, df_1d, "1D")

    # Build 1W
    df_1w = aggregate_timeframe(df_1h, "1W")
    df_btc_1w = safe_resample(df_btc, "1W").agg({"btc_close_1h": "last"}).rename(columns={"btc_close_1h": "btc_close_1w"})
    df_eth_1w = safe_resample(df_eth, "1W").agg({"eth_close_1h": "last"}).rename(columns={"eth_close_1h": "eth_close_1w"})
    df_1w = df_1w.merge(df_btc_1w, left_index=True, right_index=True, how="left")
    df_1w = df_1w.merge(df_eth_1w, left_index=True, right_index=True, how="left")
    df_1w = compute_timeframe_indicators(df_1w, prefix="1w")
    df_final = merge_timeframe_into_1h(df_merged_1d, df_1w, "1W")

    if "id" not in df_final.columns:
        df_final = df_final.merge(df_1h[["id"]], left_index=True, right_index=True, how="left")

    # Update DB
    update_candles_table(df_final)

    print(f"✅ All computations done for pair={pair}!")

if __name__ == "__main__":
    test_pair = "ETH-USDT"
    compute_all_indicators_for_pair(test_pair)
