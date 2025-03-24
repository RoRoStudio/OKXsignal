#This file should be a python script that #extremely carefully analyses all candles #from the candles_1h table that are marked as #computed (bool). It should thoroughly #validate whether the the data is complete #and fully correct for both training the #signals model and for feeding the data to #the model once it has been fetched. It #should raise very clear logging for both a #positive and negative outcome.


import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands
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
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("validator")

# Database connection
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# Validation functions
def validate_rsi(df):
    original_rsi = df['rsi_1h']
    computed_rsi = RSIIndicator(df['close_1h'], window=14).rsi()
    diff = np.abs(original_rsi - computed_rsi)
    return diff.max(), diff.mean()

def validate_macd(df):
    original_macd = df['macd_slope_1h']
    macd = MACD(df['close_1h'])
    computed_macd = macd.macd().diff()
    diff = np.abs(original_macd - computed_macd)
    return diff.max(), diff.mean()

def validate_atr(df):
    original_atr = df['atr_1h']
    computed_atr = AverageTrueRange(df['high_1h'], df['low_1h'], df['close_1h']).average_true_range()
    diff = np.abs(original_atr - computed_atr)
    return diff.max(), diff.mean()

def validate_bollinger_width(df):
    original_bw = df['bollinger_width_1h']
    bb = BollingerBands(df['close_1h'])
    computed_bw = bb.bollinger_hband() - bb.bollinger_lband()
    diff = np.abs(original_bw - computed_bw)
    return diff.max(), diff.mean()

def validate_gaps(df):
    df = df.sort_values('timestamp_utc')
    df['next_timestamp'] = df['timestamp_utc'].shift(-1)
    df['expected_next_timestamp'] = df['timestamp_utc'] + timedelta(hours=1)
    gaps = df[df['next_timestamp'] != df['expected_next_timestamp']]
    return gaps

def validate_all(df):
    validations = {
        'RSI': validate_rsi(df),
        'MACD': validate_macd(df),
        'ATR': validate_atr(df),
        'Bollinger Width': validate_bollinger_width(df),
    }
    return validations

# Main validation process
def validate_computed_candles():
    conn = get_connection()
    logger.info("Connected to DB... loading computed candles")

    query = """
    SELECT * FROM candles_1h WHERE features_computed = TRUE
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        logger.warning("No computed candles found.")
        return

    logger.info(f"Validating {len(df)} computed candles...")

    # Perform validations
    validations = validate_all(df)

    for indicator, (max_diff, mean_diff) in validations.items():
        logger.info(f"Validation results for {indicator}: Max Diff = {max_diff}, Mean Diff = {mean_diff}")

    # Check for gaps or missing data
    gaps = validate_gaps(df)
    if not gaps.empty:
        logger.warning(f"Found {len(gaps)} gaps or missing data points:")
        for _, row in gaps.iterrows():
            logger.warning(f"Missing data between {row['timestamp_utc']} and {row['next_timestamp']}")

    logger.info("Validation completed.")

if __name__ == "__main__":
    validate_computed_candles()