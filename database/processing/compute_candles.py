import os
import configparser
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import numpy as np
import sys
import math
from ta.momentum import RSIIndicator
from ta.trend import MACD, PSARIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import signal
from sqlalchemy import create_engine

# ---------------------------
# Load Configuration
# ---------------------------
BATCH_SIZE = os.cpu_count() or 8  # parallel processes

CONFIG_PATH = os.path.join('P:/OKXsignal/config/config.ini')
CREDENTIALS_PATH = os.path.join('P:/OKXsignal/config/credentials.env')

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
load_dotenv(dotenv_path=CREDENTIALS_PATH)

DB = config['DATABASE']
MODE = config['GENERAL'].get('COMPUTE_MODE', 'rolling_update').lower()
LOG_LEVEL = config['GENERAL'].get('LOG_LEVEL', 'INFO').upper()
ROLLING_WINDOW = config['GENERAL'].getint('ROLLING_WINDOW', fallback=128)
# Minimum candles needed to compute all features and labels (2w horizon + safety margin)
MIN_CANDLES_REQUIRED = 388

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


def force_exit_on_ctrl_c():
    """
    Allows Ctrl-C to forcibly exit all threads.
    """
    import threading
    import ctypes
    def handler(signum, frame):
        print("\nInterrupted. Forcing thread exit...")
        for t in threading.enumerate():
            if t is not threading.main_thread():
                try:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(t.ident), ctypes.py_object(SystemExit))
                except Exception:
                    pass
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

# ---------------------------
# Database Connection
# ---------------------------
def get_connection():
    """
    Create and return a SQLAlchemy Engine.
    """
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
            f"{DB['DB_HOST']}:{DB['DB_PORT']}/{DB['DB_NAME']}"
        )
        logger.info("Database connection established.")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

# Smallint columns in candles_1h that must be cast to int
SMALLINT_COLUMNS = {
    'supertrend_direction_1h',
    'supertrend_direction_4h',
    'supertrend_direction_1d',
    'performance_rank_btc_1h',
    'performance_rank_eth_1h',
    'volatility_rank_1h',
    'volume_rank_1h',
    'hour_of_day',
    'day_of_week',
    'was_profitable_12h'
}

# ---------------------------
# Casting Helper
# ---------------------------
def cast_for_sqlalchemy(col_name, val):
    """
    Convert `val` into a Python scalar suitable for SQLAlchemy param binding.
    - Convert numpy types to standard Python (float, int).
    - Convert datetime64 or pd.Timestamp to Python datetime.
    - Convert NaN to None.
    - Force columns in SMALLINT_COLUMNS to int if not None.
    """
    # If it's null or NaN, convert to None
    if pd.isna(val):
        return None

    # If it's a numpy type
    if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
        val = int(val)
    elif isinstance(val, (np.float64, np.float32)):
        val = float(val)
    elif isinstance(val, np.datetime64):
        # convert to Python datetime
        val = pd.to_datetime(val).to_pydatetime()

    # If it's a pandas Timestamp
    if isinstance(val, pd.Timestamp):
        val = val.to_pydatetime()

    # If the column is smallint in DB, cast to int
    if col_name in SMALLINT_COLUMNS:
        # we only do this if the value is not None
        if val is not None:
            val = int(round(float(val)))  # round then int
    # Next, if it's not a standard python numeric/datetime/bool/str, fallback to string
    if not isinstance(val, (int, float, bool, datetime, str, type(None))):
        val = str(val)

    return val

# ---------------------------
# Feature Computation Logic
# ---------------------------
def compute_1h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1h-based indicators on the DataFrame.
    """
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, PSARIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import MFIIndicator, OnBalanceVolumeIndicator

    try:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        df = df.sort_values('timestamp_utc')

        # RSI & slopes
        df['rsi_1h'] = RSIIndicator(df['close_1h'], window=14).rsi()
        df['rsi_slope_1h'] = df['rsi_1h'].rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

        # MACD
        macd = MACD(df['close_1h'])
        df['macd_slope_1h'] = macd.macd().diff()
        df['macd_hist_slope_1h'] = macd.macd_diff().diff()

        # ATR
        df['atr_1h'] = AverageTrueRange(df['high_1h'], df['low_1h'], df['close_1h']).average_true_range()

        # Bollinger
        bb = BollingerBands(df['close_1h'])
        df['bollinger_width_1h'] = bb.bollinger_hband() - bb.bollinger_lband()

        # Donchian
        df['donchian_channel_width_1h'] = (
            df['high_1h'].rolling(20).max() - df['low_1h'].rolling(20).min()
        )
        df['supertrend_direction_1h'] = np.nan  # placeholder
        df['parabolic_sar_1h'] = PSARIndicator(
            df['high_1h'], df['low_1h'], df['close_1h']
        ).psar()

        # MFI
        df['money_flow_index_1h'] = MFIIndicator(
            df['high_1h'], df['low_1h'], df['close_1h'], df['volume_1h']
        ).money_flow_index()

        # OBV
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
        logger.error(f"Error in compute_1h_features: {e}", exc_info=True)
        raise

def compute_multi_tf_features(df: pd.DataFrame, tf_label: str, rule: str) -> pd.DataFrame:
    """
    Resample from 1h to 4h or 1d, then compute relevant multi-timeframe indicators.
    """
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
    try:
        required_points = {
            '4h': 24,   # 4h window = 1 day
            '1d': 30    # 1d window = ~1 month
        }
        min_points = required_points.get(rule, 20)

        if len(df) < min_points:
            if 'pair' in df.columns and not df.empty:
                logger.warning(
                    f"Skipping {df['pair'].iloc[0]} {tf_label} features: only {len(df)} rows (need >= {min_points})"
                )
            return df

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

        # RSI
        resampled[f'rsi_{tf_label}'] = RSIIndicator(resampled[f'close_{tf_label}'], window=14).rsi()
        # RSI slope
        resampled[f'rsi_slope_{tf_label}'] = resampled[f'rsi_{tf_label}'].rolling(3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        # MACD
        macd = MACD(resampled[f'close_{tf_label}'])
        resampled[f'macd_slope_{tf_label}'] = macd.macd().diff()
        resampled[f'macd_hist_slope_{tf_label}'] = macd.macd_diff().diff()
        # ATR
        resampled[f'atr_{tf_label}'] = AverageTrueRange(
            resampled[f'high_{tf_label}'],
            resampled[f'low_{tf_label}'],
            resampled[f'close_{tf_label}']
        ).average_true_range()
        # Bollinger
        bb = BollingerBands(resampled[f'close_{tf_label}'])
        resampled[f'bollinger_width_{tf_label}'] = (
            bb.bollinger_hband() - bb.bollinger_lband()
        )
        # Donchian
        resampled[f'donchian_channel_width_{tf_label}'] = (
            resampled[f'high_{tf_label}'].rolling(20).max() -
            resampled[f'low_{tf_label}'].rolling(20).min()
        )
        # placeholders
        resampled[f'supertrend_direction_{tf_label}'] = np.nan
        # MFI
        resampled[f'money_flow_index_{tf_label}'] = MFIIndicator(
            resampled[f'high_{tf_label}'],
            resampled[f'low_{tf_label}'],
            resampled[f'close_{tf_label}'],
            resampled[f'volume_{tf_label}']
        ).money_flow_index()
        # OBV
        obv = OnBalanceVolumeIndicator(
            resampled[f'close_{tf_label}'], resampled[f'volume_{tf_label}']
        ).on_balance_volume()
        resampled[f'obv_slope_{tf_label}'] = obv.rolling(3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        # Volume change
        resampled[f'volume_change_pct_{tf_label}'] = resampled[f'volume_{tf_label}'].pct_change()

        # Merge
        df = df.merge(resampled, how='left', left_index=True, right_index=True)
        df = df.reset_index()
        return df
    except Exception as e:
        logger.error(f"Error in compute_multi_tf_features: {e}", exc_info=True)
        raise

# ---------------------------
# Label Computation Logic
# ---------------------------
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward returns and related label columns.
    """
    try:
        df = df.sort_values('timestamp_utc')
        for horizon, shift_ in [('1h', 1), ('4h', 4), ('12h', 12),
                                ('1d', 24), ('3d', 72), ('1w', 168), ('2w', 336)]:
            df[f'future_return_{horizon}_pct'] = (
                df['close_1h'].shift(-shift_) - df['close_1h']
            ) / df['close_1h']

        df['future_max_return_24h_pct'] = (
            df['high_1h'].rolling(window=24).max() - df['close_1h']
        ) / df['close_1h']

        rolling_low = df['low_1h'].shift(-1).rolling(12).min()
        df['future_max_drawdown_12h_pct'] = (
            rolling_low - df['close_1h']
        ) / df['close_1h']

        df['targets_computed'] = True
        return df
    except Exception as e:
        logger.error(f"Error in compute_labels: {e}", exc_info=True)
        raise

# ---------------------------
# Cross-Pair Intelligence
# ---------------------------
def compute_cross_pair_features(latest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-pair metrics relative to BTC-USDT and ETH-USDT.
    """
    try:
        latest_df['volume_rank_1h'] = latest_df['volume_1h'].rank(pct=True) * 100
        latest_df['volatility_rank_1h'] = latest_df['atr_1h'].rank(pct=True) * 100

        btc_row = latest_df[latest_df['pair'] == 'BTC-USDT']
        eth_row = latest_df[latest_df['pair'] == 'ETH-USDT']

        if not btc_row.empty:
            btc_return = btc_row['future_return_1h_pct'].values[0]
            latest_df['performance_rank_btc_1h'] = (
                (latest_df['future_return_1h_pct'] - btc_return) / abs(btc_return + 1e-9)
            ).rank(pct=True) * 100

        if not eth_row.empty:
            eth_return = eth_row['future_return_1h_pct'].values[0]
            latest_df['performance_rank_eth_1h'] = (
                (latest_df['future_return_1h_pct'] - eth_return) / abs(eth_return + 1e-9)
            ).rank(pct=True) * 100

        return latest_df
    except Exception as e:
        logger.error(f"Error in compute_cross_pair_features: {e}", exc_info=True)
        raise

# ---------------------------
# Single-Pair Processing
# ---------------------------
def fetch_data(pair: str, engine) -> pd.DataFrame:
    """
    Fetch all candles_1h for the given pair, sorted by timestamp_utc ASC.
    """
    query = """
        SELECT *
        FROM candles_1h
        WHERE pair = %s
        ORDER BY timestamp_utc ASC
    """
    return pd.read_sql(query, engine, params=(pair,))

def process_pair(pair: str, engine, rolling_window: int) -> None:
    """
    For the given pair, compute 1h, multi-timeframe features, labels, etc.
    Then update the database row by row.
    """
    logger.info(f"Computing features for {pair}")
    updated_rows = 0

    try:
        df = fetch_data(pair, engine)
        row_count = len(df)
        logger.debug(f"{pair}: initial row_count = {row_count}")

        if df.empty or row_count < MIN_CANDLES_REQUIRED:
            logger.warning(
                f"Skipping {pair}: only {row_count} candles, need >= {MIN_CANDLES_REQUIRED}"
            )
            return

        # Compute features
        df = compute_1h_features(df)
        df = compute_labels(df)
        df = compute_multi_tf_features(df, '4h', '4h')
        df = compute_multi_tf_features(df, '1d', '1d')
        df['features_computed'] = True
        df['targets_computed'] = True

        # The columns we want to update
        columns_for_update = [
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
            'performance_rank_btc_1h', 'performance_rank_eth_1h', 'volume_rank_1h',
            'volatility_rank_1h', 'was_profitable_12h', 'prev_close_change_pct', 'prev_volume_rank',
            'future_max_return_24h_pct', 'future_max_drawdown_12h_pct',
            'pair', 'timestamp_utc'
        ]

        update_query = """
        UPDATE candles_1h
        SET
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

        with engine.connect() as conn:
            for i, row_data in df.iterrows():
                # Build param list
                param_list = []
                for col in columns_for_update:
                    raw_val = row_data.get(col, None)
                    safe_val = cast_for_sqlalchemy(col, raw_val)
                    param_list.append(safe_val)

                logger.debug(
                    f"{pair}: Row {i} param types = {[type(x) for x in param_list]}"
                )

                try:
                    conn.execute(update_query, tuple(param_list))
                    updated_rows += 1
                except Exception as inner_e:
                    logger.error(
                        f"Exception updating row idx={i} for {pair} with param_list={param_list}",
                        exc_info=True
                    )
                    continue

    except Exception as e:
        logger.error(f"Error processing {pair}: {e}", exc_info=True)
    finally:
        logger.info(f"{pair}: Updated {updated_rows} rows")

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    force_exit_on_ctrl_c()
    logger.info(f"Starting compute_candles.py in {MODE.upper()} mode")

    try:
        engine = get_connection()
    except Exception as e:
        logger.critical(f"Unable to start due to database connection issue: {e}")
        sys.exit(1)

    logger.info("Connected to DB... loading pair list")

    try:
        all_pairs = pd.read_sql("SELECT DISTINCT pair FROM candles_1h", engine)['pair'].tolist()
        logger.info(f"Found {len(all_pairs)} pairs")
    except Exception as e:
        logger.critical(f"Error loading pair list: {e}")
        sys.exit(1)

    if not all_pairs:
        logger.warning("No pairs found in candles_1h. Exiting early.")
        sys.exit()

    if MODE == "rolling_update":
        logger.info(f"Rolling update mode: computing last {ROLLING_WINDOW} rows per pair")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_pair, pair, engine, ROLLING_WINDOW) for pair in all_pairs]
            for i, future in enumerate(futures):
                try:
                    future.result()
                    if (i + 1) % 25 == 0:
                        logger.info(f"Progress: {i + 1}/{len(all_pairs)} pairs processed")
                except Exception as e:
                    logger.error(f"Error in thread for pair {all_pairs[i]}: {e}", exc_info=True)

        logger.info("Rolling update mode completed.")

    elif MODE == "full_backfill":
        logger.info("Full backfill mode: fetching all candles per pair and computing everything...")

        from database.db import execute_copy_update

        all_pairs = pd.read_sql("SELECT DISTINCT pair FROM candles_1h", engine)['pair'].tolist()

        all_rows = []
        skipped = 0
        columns_for_update = [
            'pair', 'timestamp_utc',
            'rsi_1h', 'rsi_slope_1h', 'macd_slope_1h', 'macd_hist_slope_1h', 'atr_1h',
            'bollinger_width_1h', 'donchian_channel_width_1h', 'supertrend_direction_1h',
            'parabolic_sar_1h', 'money_flow_index_1h', 'obv_slope_1h', 'volume_change_pct_1h',
            'estimated_slippage_1h', 'bid_ask_spread_1h', 'hour_of_day', 'day_of_week',
            'rsi_4h', 'rsi_slope_4h', 'macd_hist_slope_4h', 'macd_slope_4h', 'atr_4h',
            'bollinger_width_4h', 'donchian_channel_width_4h', 'supertrend_direction_4h',
            'money_flow_index_4h', 'obv_slope_4h', 'volume_change_pct_4h',
            'rsi_1d', 'rsi_slope_1d', 'macd_hist_slope_1d', 'macd_slope_1d', 'atr_1d',
            'bollinger_width_1d', 'donchian_channel_width_1d', 'supertrend_direction_1d',
            'money_flow_index_1d', 'obv_slope_1d', 'volume_change_pct_1d',
            'performance_rank_btc_1h', 'performance_rank_eth_1h',
            'volume_rank_1h', 'volatility_rank_1h',
            'was_profitable_12h', 'prev_close_change_pct', 'prev_volume_rank',
            'future_max_return_24h_pct', 'future_max_drawdown_12h_pct'
        ]

        def compute_and_collect(pair: str) -> list:
            """
            Processes a single pair, returns a list of rows for bulk update in the full_backfill mode.
            """
            import os
            from dotenv import load_dotenv
            import pandas as pd
            from sqlalchemy import create_engine

            load_dotenv(dotenv_path=CREDENTIALS_PATH)
            config = configparser.ConfigParser()
            config.read(CONFIG_PATH)
            DB = config['DATABASE']

            local_engine = create_engine(
                f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
                f"{DB['DB_HOST']}:{DB['DB_PORT']}/{DB['DB_NAME']}"
            )
            batch_rows = []

            try:
                df = pd.read_sql(
                    "SELECT * FROM candles_1h WHERE pair = %s ORDER BY timestamp_utc ASC;",
                    local_engine, params=(pair,)
                )
                if df.empty or len(df) < MIN_CANDLES_REQUIRED:
                    logger.warning(f"Skipping {pair} (full_backfill): only {len(df)} candles; need >= {MIN_CANDLES_REQUIRED}")
                    return batch_rows

                df = compute_1h_features(df)
                df = compute_labels(df)
                df = compute_multi_tf_features(df, '4h', '4h')
                df = compute_multi_tf_features(df, '1d', '1d')
                df['features_computed'] = True
                df['targets_computed'] = True

                # cross-pair for last ROLLING_WINDOW
                latest = compute_cross_pair_features(df.tail(ROLLING_WINDOW).copy())
                df = df.merge(
                    latest[[
                        'pair', 'timestamp_utc',
                        'performance_rank_btc_1h', 'performance_rank_eth_1h',
                        'volume_rank_1h', 'volatility_rank_1h'
                    ]],
                    on=['pair', 'timestamp_utc'], how='left'
                )

                # Prepare rows for bulk update
                for _, row_data in df.iterrows():
                    row_to_insert = []
                    for col in columns_for_update:
                        raw_val = row_data.get(col, None)
                        safe_val = cast_for_sqlalchemy(col, raw_val)
                        row_to_insert.append(safe_val)
                    batch_rows.append(row_to_insert)

            except Exception as pair_error:
                logger.error(f"Error in {pair} during full_backfill: {pair_error}", exc_info=True)
                return batch_rows

            return batch_rows

        logger.info(f"Running full backfill with {BATCH_SIZE} parallel workers...")

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures_map = {executor.submit(compute_and_collect, pair): pair for pair in all_pairs}
            for i, future in enumerate(futures_map):
                pair_str = futures_map[future]
                try:
                    result = future.result()
                    all_rows.extend(result)
                    if (i + 1) % 25 == 0:
                        logger.info(f"Progress: {i + 1}/{len(all_pairs)} pairs processed")
                except Exception as e:
                    logger.error(f"Future error for pair {pair_str}: {e}", exc_info=True)

        rows_written = len(all_rows)
        pairs_processed = len(set(row[0] for row in all_rows))  # row[0] is 'pair'

        if rows_written == 0:
            logger.critical("No rows were collected for update. Aborting write.")
        else:
            logger.info(f"Writing {rows_written} rows across {pairs_processed} pairs to DB...")

            update_query = """
            UPDATE candles_1h AS c SET
            """ + ",\n".join([
                f"{col} = t.{col}" for col in columns_for_update[2:]  # skip pair, timestamp_utc
            ]) + """
            , features_computed = TRUE,
              targets_computed = TRUE
            FROM {temp_table} t
            WHERE c.pair = t.pair AND c.timestamp_utc = t.timestamp_utc;
            """

            # We'll rely on your custom function execute_copy_update
            execute_copy_update(
                temp_table_name="temp_full_backfill",
                column_names=columns_for_update,
                values=all_rows,
                update_query=update_query
            )

            logger.info(
                f"Full backfill complete: {rows_written} rows updated for {pairs_processed} pairs (skipped {skipped})."
            )

    # Runtime logging
    end_time = datetime.now()
    duration = (end_time - start_time_global).total_seconds()
    with open(RUNTIME_LOG_PATH, "a") as f:
        f.write(f"[{end_time}] compute_candles.py ({MODE}) completed in {duration:.2f} seconds\n")
