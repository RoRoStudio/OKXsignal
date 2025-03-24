#This file should be a python script that #extremely carefully analyses all candles #from the candles_1h table that are marked as #computed (bool). It should thoroughly #validate whether the the data is complete #and fully correct for both training the #signals model and for feeding the data to #the model once it has been fetched. It #should raise very clear logging for both a #positive and negative outcome.


import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from ta.momentum import RSIIndicator
from ta.trend import MACD, PSARIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from datetime import datetime, timedelta

# Load environment variables
load_dotenv(dotenv_path='P:/OKXsignal/config/credentials.env')

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Setup logging
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
log_file = os.path.join("logs", f"validation_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("validator")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# === Core indicator validations ===
def validate_indicator(name, original, computed):
    diff = np.abs(original - computed)
    return name, diff.max(), diff.mean()

def validate_indicators(df):
    results = []

    # RSI and Slope
    rsi = RSIIndicator(df['close_1h'], window=14).rsi()
    rsi_slope = rsi.rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    results.append(validate_indicator('RSI', df['rsi_1h'], rsi))
    results.append(validate_indicator('RSI_SLOPE', df['rsi_slope_1h'], rsi_slope))

    # MACD and Hist Slope
    macd = MACD(df['close_1h'])
    results.append(validate_indicator('MACD_SLOPE', df['macd_slope_1h'], macd.macd().diff()))
    results.append(validate_indicator('MACD_HIST_SLOPE', df['macd_hist_slope_1h'], macd.macd_diff().diff()))

    # ATR
    atr = AverageTrueRange(df['high_1h'], df['low_1h'], df['close_1h']).average_true_range()
    results.append(validate_indicator('ATR', df['atr_1h'], atr))

    # Bollinger
    bb = BollingerBands(df['close_1h'])
    boll_width = bb.bollinger_hband() - bb.bollinger_lband()
    results.append(validate_indicator('BOLLINGER_WIDTH', df['bollinger_width_1h'], boll_width))

    # Parabolic SAR
    psar = PSARIndicator(df['high_1h'], df['low_1h'], df['close_1h']).psar()
    results.append(validate_indicator('PARABOLIC_SAR', df['parabolic_sar_1h'], psar))

    # MFI
    mfi = MFIIndicator(df['high_1h'], df['low_1h'], df['close_1h'], df['volume_1h']).money_flow_index()
    results.append(validate_indicator('MFI', df['money_flow_index_1h'], mfi))

    # OBV slope
    obv = OnBalanceVolumeIndicator(df['close_1h'], df['volume_1h']).on_balance_volume()
    obv_slope = obv.rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    results.append(validate_indicator('OBV_SLOPE', df['obv_slope_1h'], obv_slope))

    # Volume change
    volume_change = df['volume_1h'].pct_change()
    results.append(validate_indicator('VOLUME_CHANGE', df['volume_change_pct_1h'], volume_change))

    # Spread
    results.append(validate_indicator('SPREAD', df['bid_ask_spread_1h'], df['close_1h'] - df['open_1h']))
    results.append(validate_indicator('SLIPPAGE_ESTIMATE', df['estimated_slippage_1h'], df['high_1h'] - df['low_1h']))

    return results

# === Gap check ===
def validate_gaps(df):
    df = df.sort_values('timestamp_utc')
    df['expected'] = df['timestamp_utc'].shift(1) + timedelta(hours=1)
    gaps = df[df['timestamp_utc'] != df['expected']]
    return gaps

# === Null check ===
def validate_nulls(df):
    critical_columns = [col for col in df.columns if '_1h' in col or col.startswith('future_') or col.startswith('performance_') or col.endswith('_rank')]
    null_report = df[critical_columns].isnull().sum()
    return null_report[null_report > 0]

# === Range check ===
def validate_cross_pair_ranges(df):
    violations = {}
    for col in ['volume_rank_1h', 'volatility_rank_1h', 'performance_rank_btc_1h', 'performance_rank_eth_1h']:
        if col in df.columns:
            min_val = df[col].min()
max_val = df[col].max()
if min_val < -1e-3 or max_val > 100 + 1e-3:
    violations[col] = (min_val, max_val)
                violations[col] = (df[col].min(), df[col].max())
    return violations

# === Entry point ===
def validate_computed_candles():
    conn = get_connection()
    logger.info("Connected to DB... loading computed candles")
    df = pd.read_sql("SELECT * FROM candles_1h WHERE features_computed = TRUE AND targets_computed = TRUE", conn)
    conn.close()

    if df.empty:
        logger.warning("No computed candles found.")
        return

    logger.info(f"Validating {len(df)} computed rows...")

    # Check indicators
    indicator_results = validate_indicators(df)
    for name, max_diff, mean_diff in indicator_results:
        if max_diff > 1e-6:
            logger.warning(f"{name}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")
        else:
            logger.info(f"{name}: Validation PASSED (diffs negligible)")

    # Check for gaps
    gaps = validate_gaps(df)
    if not gaps.empty:
        logger.warning(f"❗ Found {len(gaps)} timestamp gaps in 1h data.")
    else:
        logger.info("✅ No timestamp gaps detected.")

    # Check nulls
    nulls = validate_nulls(df)
    if not nulls.empty:
        logger.warning("❗ Null values found in critical columns:")
        for col, count in nulls.items():
            logger.warning(f" - {col}: {count} nulls")
    else:
        logger.info("✅ No null values in critical fields.")

    # Cross-pair range check
    violations = validate_cross_pair_ranges(df)
    if violations:
        logger.warning("❗ Cross-pair ranks out of expected [0–100] range:")
        for col, (min_val, max_val) in violations.items():
            logger.warning(f" - {col}: min={min_val:.2f}, max={max_val:.2f}")
    else:
        logger.info("✅ Cross-pair rankings within expected range.")

    logger.info("✅ Validation complete.")

if __name__ == "__main__":
    validate_computed_candles()