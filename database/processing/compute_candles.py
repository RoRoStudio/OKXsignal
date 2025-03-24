"""
OKXsignal - compute_candles.py
Efficient, production-grade feature and label computation for candles_1h.
Supports incremental and full backfill modes.
Includes multi-timeframe (4h, 1d) indicators and cross-pair intelligence.
Logs to file and console based on LOG_LEVEL.
"""

import os
import configparser
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, PSARIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from concurrent.futures import ProcessPoolExecutor
from psycopg2.extras import execute_batch
from datetime import datetime
import logging
#from sqlalchemy import create_engine

#engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
#df = pd.read_sql("SELECT ...", engine)

# ---------------------------
# Load Configuration
# ---------------------------
CONFIG_PATH = os.path.join('P:/OKXsignal/config/config.ini')
CREDENTIALS_PATH = os.path.join('P:/OKXsignal/config/credentials.env')

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
load_dotenv(dotenv_path=CREDENTIALS_PATH)

DB = config['DATABASE']
MODE = config['GENERAL'].get('COMPUTE_MODE', 'rolling_update').lower()
LOG_LEVEL = config['GENERAL'].get('LOG_LEVEL', 'INFO').upper()
ROLLING_WINDOW = config['GENERAL'].getint('ROLLING_WINDOW', fallback=128)

# ---------------------------
# Setup Logging
# ---------------------------
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
log_file = os.path.join("logs", f"compute_{timestamp}.log")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='[%(levelname)s] %(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("compute")
RUNTIME_LOG_PATH = os.path.join("logs", "runtime_compute.log")
start_time_global = datetime.now()

# ---------------------------
# Database Connection
# ---------------------------
def get_connection():
    try:
        conn = psycopg2.connect(
            host=DB['DB_HOST'],
            port=DB['DB_PORT'],
            dbname=DB['DB_NAME'],
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        logger.info("Database connection established.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

# ---------------------------
# Feature Computation Logic
# ---------------------------
def compute_1h_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.sort_values('timestamp_utc')

        df['rsi_1h'] = RSIIndicator(df['close_1h'], window=14).rsi()
        df['rsi_slope_1h'] = df['rsi_1h'].rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

        macd = MACD(df['close_1h'])
        df['macd_slope_1h'] = macd.macd().diff()
        df['macd_hist_slope_1h'] = macd.macd_diff().diff()

        df['atr_1h'] = AverageTrueRange(df['high_1h'], df['low_1h'], df['close_1h']).average_true_range()

        bb = BollingerBands(df['close_1h'])
        df['bollinger_width_1h'] = bb.bollinger_hband() - bb.bollinger_lband()

        df['donchian_channel_width_1h'] = df['high_1h'].rolling(20).max() - df['low_1h'].rolling(20).min()
        df['supertrend_direction_1h'] = np.nan  # placeholder
        df['parabolic_sar_1h'] = PSARIndicator(df['high_1h'], df['low_1h'], df['close_1h']).psar()

        df['money_flow_index_1h'] = MFIIndicator(df['high_1h'], df['low_1h'], df['close_1h'], df['volume_1h']).money_flow_index()

        obv = OnBalanceVolumeIndicator(df['close_1h'], df['volume_1h']).on_balance_volume()
        df['obv_slope_1h'] = obv.rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

        df['volume_change_pct_1h'] = df['volume_1h'].pct_change()
        df['estimated_slippage_1h'] = df['high_1h'] - df['low_1h']
        df['bid_ask_spread_1h'] = df['close_1h'] - df['open_1h']
        df['hour_of_day'] = df['timestamp_utc'].dt.hour
        df['day_of_week'] = df['timestamp_utc'].dt.weekday
        df['was_profitable_12h'] = (df['close_1h'].shift(-12) > df['close_1h']).astype(int)
        df['prev_close_change_pct'] = df['close_1h'].pct_change()
        df['prev_volume_rank'] = df['volume_1h'].rank(pct=True).shift(1) * 100

        return df
    except Exception as e:
        logger.error(f"Error in compute_1h_features: {e}")
        raise

def compute_multi_tf_features(df: pd.DataFrame, tf_label: str, rule: str) -> pd.DataFrame:
    try:
        df = df.set_index('timestamp_utc')
        ohlcv = df[['open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h']]
        resampled = ohlcv.resample(rule).agg({
            'open_1h': 'first',
            'high_1h': 'max',
            'low_1h': 'min',
            'close_1h': 'last',
            'volume_1h': 'sum'
        }).dropna()
        resampled.columns = [col.replace('1h', tf_label) for col in resampled.columns]

        resampled[f'rsi_{tf_label}'] = RSIIndicator(resampled[f'close_{tf_label}'], window=14).rsi()
        resampled[f'rsi_slope_{tf_label}'] = resampled[f'rsi_{tf_label}'].rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        macd = MACD(resampled[f'close_{tf_label}'])
        resampled[f'macd_slope_{tf_label}'] = macd.macd().diff()
        resampled[f'macd_hist_slope_{tf_label}'] = macd.macd_diff().diff()
        resampled[f'atr_{tf_label}'] = AverageTrueRange(
            resampled[f'high_{tf_label}'],
            resampled[f'low_{tf_label}'],
            resampled[f'close_{tf_label}']
        ).average_true_range()
        bb = BollingerBands(resampled[f'close_{tf_label}'])
        resampled[f'bollinger_width_{tf_label}'] = bb.bollinger_hband() - bb.bollinger_lband()
        resampled[f'donchian_channel_width_{tf_label}'] = resampled[f'high_{tf_label}'].rolling(20).max() - resampled[f'low_{tf_label}'].rolling(20).min()
        resampled[f'supertrend_direction_{tf_label}'] = np.nan
        resampled[f'money_flow_index_{tf_label}'] = MFIIndicator(
            resampled[f'high_{tf_label}'], resampled[f'low_{tf_label}'], resampled[f'close_{tf_label}'], resampled[f'volume_{tf_label}']
        ).money_flow_index()
        obv = OnBalanceVolumeIndicator(resampled[f'close_{tf_label}'], resampled[f'volume_{tf_label}']).on_balance_volume()
        resampled[f'obv_slope_{tf_label}'] = obv.rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        resampled[f'volume_change_pct_{tf_label}'] = resampled[f'volume_{tf_label}'].pct_change()

        df = df.merge(resampled, how='left', left_index=True, right_index=True)
        df = df.reset_index()
        return df
    except Exception as e:
        logger.error(f"Error in compute_multi_tf_features: {e}")
        raise

# ---------------------------
# Label Computation Logic
# ---------------------------
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.sort_values('timestamp_utc')
        for horizon, shift in [('1h', 1), ('4h', 4), ('12h', 12), ('1d', 24), ('3d', 72), ('1w', 168), ('2w', 336)]:
            df[f'future_return_{horizon}_pct'] = (df['close_1h'].shift(-shift) - df['close_1h']) / df['close_1h']

        df['future_max_return_24h_pct'] = (df['high_1h'].rolling(window=24).max() - df['close_1h']) / df['close_1h']

        rolling_low = df['low_1h'].shift(-1).rolling(12).min()
        df['future_max_drawdown_12h_pct'] = (rolling_low - df['close_1h']) / df['close_1h']

        df['targets_computed'] = True

        return df
    except Exception as e:
        logger.error(f"Error in compute_labels: {e}")
        raise

# ---------------------------
# Cross-Pair Intelligence
# ---------------------------
def compute_cross_pair_features(latest_df: pd.DataFrame) -> pd.DataFrame:
    try:
        latest_df['volume_rank_1h'] = latest_df['volume_1h'].rank(pct=True) * 100
        latest_df['volatility_rank_1h'] = latest_df['atr_1h'].rank(pct=True) * 100
        btc_row = latest_df[latest_df['pair'] == 'BTC-USDT']
        eth_row = latest_df[latest_df['pair'] == 'ETH-USDT']
        if not btc_row.empty:
            btc_return = btc_row['future_return_1h_pct'].values[0]
            latest_df['performance_rank_btc_1h'] = ((latest_df['future_return_1h_pct'] - btc_return) / abs(btc_return + 1e-9)).rank(pct=True) * 100
        if not eth_row.empty:
            eth_return = eth_row['future_return_1h_pct'].values[0]
            latest_df['performance_rank_eth_1h'] = ((latest_df['future_return_1h_pct'] - eth_return) / abs(eth_return + 1e-9)).rank(pct=True) * 100
        return latest_df
    except Exception as e:
        logger.error(f"Error in compute_cross_pair_features: {e}")
        raise

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == '__main__':
    logger.info(f"Starting compute_candles.py in {MODE.upper()} mode")
    try:
        conn = get_connection()
    except Exception as e:
        logger.critical(f"Unable to start due to database connection issue: {e}")
        exit(1)

    logger.info("Connected to DB... loading pair list")

    try:
        all_pairs = pd.read_sql("SELECT DISTINCT pair FROM candles_1h", conn)['pair'].tolist()
        logger.info(f"Found {len(all_pairs)} pairs")
    except Exception as e:
        logger.critical(f"Error loading pair list: {e}")
        exit(1)

    if not all_pairs:
        logger.warning("No pairs found in candles_1h. Exiting early.")
        exit()

    if MODE == "rolling_update":
        conn = get_connection()
        logger.info("Connected to DB for rolling update")
        try:
            all_pairs = pd.read_sql("SELECT DISTINCT pair FROM candles_1h", conn)['pair'].tolist()
            logger.info(f"Found {len(all_pairs)} pairs for rolling update")
        except Exception as e:
            logger.critical(f"Error loading pair list for rolling update: {e}")
            exit(1)
        finally:
            conn.close()
        logger.info(f"Rolling update mode: computing last {ROLLING_WINDOW} rows per pair")

        def process_pair_rolling(pair: str):
            logger.info(f"🔁 Computing features for {pair}")
            try:
                conn = get_connection()
                query = f"""
                    SELECT * FROM candles_1h
                    WHERE pair = %s
                    ORDER BY timestamp_utc DESC
                    LIMIT {ROLLING_WINDOW}
                """
                df = pd.read_sql(query, conn, params=(pair,))
                if df.empty or len(df) < ROLLING_WINDOW // 2:
                    logger.warning(f"Not enough data for {pair}")
                    return

                df = compute_1h_features(df)
                df = compute_labels(df)
                df = compute_multi_tf_features(df, '4h', '4H')
                df = compute_multi_tf_features(df, '1d', '1D')
                df['features_computed'] = True
                df['targets_computed'] = True

                with conn.cursor() as cur:
                    for _, row in df.iterrows():
                        update_query = """
                    UPDATE candles_1h SET
                        rsi_1h = %s,
                        rsi_slope_1h = %s,
                        macd_slope_1h = %s,
                        macd_hist_slope_1h = %s,
                        atr_1h = %s,
                        bollinger_width_1h = %s,
                        donchian_channel_width_1h = %s,
                        supertrend_direction_1h = %s,
                        parabolic_sar_1h = %s,
                        money_flow_index_1h = %s,
                        obv_slope_1h = %s,
                        volume_change_pct_1h = %s,
                        estimated_slippage_1h = %s,
                        bid_ask_spread_1h = %s,
                        hour_of_day = %s,
                        day_of_week = %s,
                        rsi_4h = %s,
                        rsi_slope_4h = %s,
                        macd_slope_4h = %s,
                        macd_hist_slope_4h = %s,
                        atr_4h = %s,
                        bollinger_width_4h = %s,
                        donchian_channel_width_4h = %s,
                        supertrend_direction_4h = %s,
                        money_flow_index_4h = %s,
                        obv_slope_4h = %s,
                        volume_change_pct_4h = %s,
                        rsi_1d = %s,
                        rsi_slope_1d = %s,
                        macd_slope_1d = %s,
                        macd_hist_slope_1d = %s,
                        atr_1d = %s,
                        bollinger_width_1d = %s,
                        donchian_channel_width_1d = %s,
                        supertrend_direction_1d = %s,
                        money_flow_index_1d = %s,
                        obv_slope_1d = %s,
                        volume_change_pct_1d = %s,
                        performance_rank_btc_1h = %s,
                        performance_rank_eth_1h = %s,
                        volume_rank_1h = %s,
                        volatility_rank_1h = %s,
                        was_profitable_12h = %s,
                        prev_close_change_pct = %s,
                        prev_volume_rank = %s,
                        future_max_return_24h_pct = %s,
                        future_max_drawdown_12h_pct = %s,
                        features_computed = TRUE,
                        targets_computed = TRUE
                    WHERE pair = %s AND timestamp_utc = %s;
                """

                        values = [row.get(col) for col in [
                            'rsi_1h', 'rsi_slope_1h', 'macd_slope_1h', 'macd_hist_slope_1h', 'atr_1h',
                            'bollinger_width_1h', 'donchian_channel_width_1h', 'supertrend_direction_1h',
                            'parabolic_sar_1h', 'money_flow_index_1h', 'obv_slope_1h', 'volume_change_pct_1h',
                            'estimated_slippage_1h', 'bid_ask_spread_1h', 'hour_of_day', 'day_of_week',
                            'rsi_4h', 'rsi_slope_4h', 'macd_slope_4h', 'macd_hist_slope_4h', 'atr_4h',
                            'bollinger_width_4h', 'donchian_channel_width_4h', 'supertrend_direction_4h',
                            'money_flow_index_4h', 'obv_slope_4h', 'volume_change_pct_4h',
                            'rsi_1d', 'rsi_slope_1d', 'macd_slope_1d', 'macd_hist_slope_1d', 'atr_1d',
                            'bollinger_width_1d', 'donchian_channel_width_1d', 'supertrend_direction_1d',
                            'money_flow_index_1d', 'obv_slope_1d', 'volume_change_pct_1d', 'was_profitable_12h',
                            'prev_close_change_pct', 'prev_volume_rank', 'future_max_return_24h_pct',
                            'future_max_drawdown_12h_pct', 'pair', 'timestamp_utc'
                        ]]
                        cur.execute(update_query, values)
                        conn.commit()

            except Exception as e:
                logger.error(f"Error processing {pair}: {e}")
            finally:
                conn.close()

        with ProcessPoolExecutor() as executor:
            executor.map(process_pair_rolling, all_pairs)

        logger.info("✅ Rolling update mode completed.")

    elif MODE == "full_backfill":
        logger.info("Full backfill mode: fetching all candles per pair and computing everything...")

        from database.db import execute_copy_update

        all_pairs = pd.read_sql("SELECT DISTINCT pair FROM candles_1h", conn)['pair'].tolist()
        conn.close()

        all_rows = []

        columns_for_update = [
            'pair', 'timestamp_utc',
            'rsi_1h', 'rsi_slope_1h', 'macd_slope_1h', 'macd_hist_slope_1h', 'atr_1h',
            'bollinger_width_1h', 'donchian_channel_width_1h', 'supertrend_direction_1h',
            'parabolic_sar_1h', 'money_flow_index_1h', 'obv_slope_1h', 'volume_change_pct_1h',
            'estimated_slippage_1h', 'bid_ask_spread_1h', 'hour_of_day', 'day_of_week',
            'rsi_4h', 'rsi_slope_4h', 'macd_slope_4h', 'macd_hist_slope_4h', 'atr_4h',
            'bollinger_width_4h', 'donchian_channel_width_4h', 'supertrend_direction_4h',
            'money_flow_index_4h', 'obv_slope_4h', 'volume_change_pct_4h',
            'rsi_1d', 'rsi_slope_1d', 'macd_slope_1d', 'macd_hist_slope_1d', 'atr_1d',
            'bollinger_width_1d', 'donchian_channel_width_1d', 'supertrend_direction_1d',
            'money_flow_index_1d', 'obv_slope_1d', 'volume_change_pct_1d',
            'performance_rank_btc_1h', 'performance_rank_eth_1h',
            'volume_rank_1h', 'volatility_rank_1h',
            'was_profitable_12h', 'prev_close_change_pct', 'prev_volume_rank',
            'future_max_return_24h_pct', 'future_max_drawdown_12h_pct'
        ]

        for pair in all_pairs:
            conn = get_connection()
            df = pd.read_sql("SELECT * FROM candles_1h WHERE pair = %s ORDER BY timestamp_utc ASC;", conn, params=(pair,))
            conn.close()
            if df.empty or len(df) < 100:
                logger.warning(f"Skipping {pair} due to insufficient candles.")
                continue

            df = compute_1h_features(df)
            df = compute_labels(df)
            df = compute_multi_tf_features(df, '4h', '4H')
            df = compute_multi_tf_features(df, '1d', '1D')
            df['features_computed'] = True
            df['targets_computed'] = True

            latest = compute_cross_pair_features(df.tail(ROLLING_WINDOW))
            df = df.merge(latest[['pair', 'timestamp_utc',
                                  'performance_rank_btc_1h', 'performance_rank_eth_1h',
                                  'volume_rank_1h', 'volatility_rank_1h']],
                          on=['pair', 'timestamp_utc'], how='left')

            for _, row in df.iterrows():
                row_values = [row.get(col) for col in columns_for_update]
                all_rows.append(row_values)

        update_query = """
        UPDATE candles_1h AS c SET
        """ + ",\n".join([
            f"{col} = t.{col}" for col in columns_for_update[2:]
        ]) + """
        , features_computed = TRUE,
          targets_computed = TRUE
        FROM {temp_table} t
        WHERE c.pair = t.pair AND c.timestamp_utc = t.timestamp_utc;
        """

        execute_copy_update(
            temp_table_name="temp_full_backfill",
            column_names=columns_for_update,
            values=all_rows,
            update_query=update_query
        )

        logger.info("✅ Full backfill completed successfully.")


    # Runtime logging
    end_time = datetime.now()
    duration = (end_time - start_time_global).total_seconds()
    with open(RUNTIME_LOG_PATH, "a") as f:
        f.write(f"[{end_time}] compute_candles.py (rolling_update) completed in {duration:.2f} seconds\n")
