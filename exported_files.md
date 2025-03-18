# Exported Code Files

## `(personal) commands.md`

```python
cd E:\Programming\OKXsignal
conda activate okxsignal
python backend/main.py

```

## `exported_files.md`

```python

```

## `export_markdown.py`

```python
import os

# Define root directory
ROOT_DIR = r"E:\Programming\OKXsignal"
OUTPUT_FILE = os.path.join(ROOT_DIR, "exported_files.md")

# Allowed file extensions (only source code files are included)
ALLOWED_EXTENSIONS = {".py", ".json", ".md"}

# Any file or directory you want to exclude completely
EXCLUDED_PATHS = {
    os.path.join(ROOT_DIR, "frontend", "grafana"),
    os.path.join(ROOT_DIR, "package-lock.json"),
    os.path.join(ROOT_DIR, "supabase"),
    os.path.join(ROOT_DIR, "credentials.env"),
}

def is_excluded(path):
    """
    Returns True if 'path' is explicitly excluded,
    either as an exact file or because it's within an excluded directory.
    """
    norm_path = os.path.normcase(path)
    for excluded in EXCLUDED_PATHS:
        norm_excluded = os.path.normcase(excluded)

        # Exact match (file or directory)
        if norm_path == norm_excluded:
            return True

        # Path inside an excluded directory
        if norm_path.startswith(norm_excluded + os.sep):
            return True

    return False

def should_include(file_path):
    """Check if a file should be included in the export."""
    if is_excluded(file_path):
        return False

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False

    return True

def export_markdown():
    """Recursively export all relevant code files, respecting excluded paths."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as md_file:
        md_file.write("# Exported Code Files\n\n")

        for root, dirs, files in os.walk(ROOT_DIR):
            # Exclude directories that match EXCLUDED_PATHS
            dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d))]

            for filename in files:
                file_path = os.path.join(root, filename)

                if should_include(file_path):
                    relative_path = os.path.relpath(file_path, ROOT_DIR)
                    md_file.write(f"## `{relative_path}`\n\n```python\n")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            md_file.write(f.read())
                    except Exception as e:
                        md_file.write(f"# Skipped {filename}: {e}")
                    
                    md_file.write("\n```\n\n")

    print(f"✅ Exported code to: {OUTPUT_FILE}")

if __name__ == "__main__":
    export_markdown()

```

## `main.py`

```python
"""
main.py
🚀 Entry point for OKXsignal's Streamlit Dashboard
"""

import streamlit as st
import os
import sys

# 1) Adjust Python path to ensure we can import backend & dashboard modules
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "backend"))
sys.path.append(os.path.join(BASE_DIR, "dashboard"))

# 2) Optional: import custom CSS
def load_custom_css():
    css_file = os.path.join(BASE_DIR, "dashboard", "assets", "custom.css")
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3) Import your layout / navigation if desired
from dashboard.layout.navigation import build_sidebar  # example
# from dashboard.layout.theme import get_theme_settings  # example theming

# 4) Import pages (show them in a multi-page style or custom routing)
from dashboard.pages.home import show_home
from dashboard.pages.market_analysis.overview import show_market_overview
from dashboard.pages.portfolio.holdings import show_holdings
from dashboard.pages.trade_execution import show_trade_execution
# etc.

# 5) If you need to fetch market data (like from fetch_market.py):
# from backend.execution.fetch_market import main as fetch_market_data

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="OKXsignal", layout="wide")

def main():
    # Load CSS
    load_custom_css()

    # Optional: build a sidebar or top bar
    # E.g. "build_sidebar()" could return the selected page
    st.sidebar.title("OKXsignal Navigation")
    pages = {
        "🏠 Home": "home",
        "📈 Market Analysis": "market_overview",
        "💰 Portfolio": "holdings",
        "⚡ Trade Execution": "trade_execution",
    }
    chosen_page = st.sidebar.radio("Go to:", list(pages.keys()))

    # Show whichever page user picked
    if pages[chosen_page] == "home":
        show_home()
    elif pages[chosen_page] == "market_overview":
        show_market_overview()
    elif pages[chosen_page] == "holdings":
        show_holdings()
    elif pages[chosen_page] == "trade_execution":
        show_trade_execution()
    else:
        show_home()  # default fallback

if __name__ == "__main__":
    main()

```

## `package.json`

```python
{
  "name": "okxsignal",
  "version": "1.0.0",
  "description": "```\r OKXsignal\r ├─ (personal) commands.md\r ├─ backend\r │  ├─ api\r │  │  ├─ routes.py\r │  │  └─ server.py\r │  ├─ config\r │  │  ├─ credentials.env\r │  │  └─ settings.py\r │  ├─ execution\r │  │  ├─ api.py\r │  │  ├─ async_requests.py\r │  │  ├─ fetch_market.py\r │  │  ├─ fetch_portfolio.py\r │  │  ├─ grafana.ini\r │  │  ├─ run_grafana.ps1\r │  │  └─ __pycache__\r │  │     ├─ api.cpython-311.pyc\r │  │     ├─ api.cpython-312.pyc\r │  │     └─ fetch_market.cpython-311.pyc\r │  ├─ indicators\r │  │  ├─ bollinger.py\r │  │  ├─ macd.py\r │  │  └─ rsi.py\r │  ├─ requirements.txt\r │  ├─ signal_engine\r │  │  └─ strategy.py\r │  ├─ test_rest_client.py\r │  ├─ utils\r │  │  ├─ helpers.py\r │  │  └─ logger.py\r │  ├─ __init__.py\r │  └─ __pycache__\r │     ├─ test_rest_client.cpython-311.pyc\r │     ├─ __init__.cpython-311.pyc\r │     └─ __init__.cpython-312.pyc\r ├─ dashboard\r │  ├─ charts.py\r │  ├─ layout.py\r │  ├─ pages\r │  │  ├─ market_analysis.py\r │  │  ├─ portfolio.py\r │  │  ├─ settings.py\r │  │  └─ __init__.py\r │  ├─ requirements.txt\r │  ├─ sidebar.py\r │  ├─ __init__.py\r │  └─ __pycache__\r │     ├─ charts.cpython-311.pyc\r │     ├─ sidebar.cpython-311.pyc\r │     └─ __init__.cpython-311.pyc\r ├─ exported_files.md\r ├─ export_markdown.py\r ├─ grafana_debug.log\r ├─ main.py\r ├─ okx_api\r │  ├─ auth.py\r │  ├─ rest_client.py\r │  ├─ __init__.py\r │  └─ __pycache__\r │     ├─ auth.cpython-311.pyc\r │     ├─ rest_client.cpython-311.pyc\r │     └─ __init__.cpython-311.pyc\r ├─ README.md\r └─ __pycache__\r    ├─ export_markdown.cpython-311.pyc\r    └─ export_pdf.cpython-311.pyc",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "supabase": "^2.19.7"
  },
  "dependencies": {
    "dotenv": "^16.4.7"
  }
}

```

## `README.md`

```python

```

## `.vscode\extensions.json`

```python
{
  "recommendations": ["denoland.vscode-deno"]
}

```

## `.vscode\settings.json`

```python
{
  "deno.enablePaths": [
    "supabase/functions"
  ],
  "deno.lint": true,
  "deno.unstable": [
    "bare-node-builtins",
    "byonm",
    "sloppy-imports",
    "unsafe-proto",
    "webgpu",
    "broadcast-channel",
    "worker-options",
    "cron",
    "kv",
    "ffi",
    "fs",
    "http",
    "net"
  ],
  "[typescript]": {
    "editor.defaultFormatter": "denoland.vscode-deno"
  }
}

```

## `backend\test_rest_client.py`

```python
import json
from okx_api.rest_client import OKXRestClient

client = OKXRestClient()

print('\n===== Testing OKX REST API Connection =====\n')

try:
    balance = client.get_balance()
    print('? Account Balance:', json.dumps(balance, indent=2))
except Exception as e:
    print('? Failed to fetch balance:', e)

try:
    ticker = client.get_ticker('BTC-USDT')
    print('? BTC-USDT Ticker:', json.dumps(ticker, indent=2))
except Exception as e:
    print('? Failed to fetch ticker:', e)

print('\n===== Test Completed =====')

```

## `backend\__init__.py`

```python

```

## `backend\config\config_loader.py`

```python
# General configuration settings
"""
config_loader.py
Loads configuration settings from config.ini.
"""

import os
import configparser

# Locate config.ini in project root
CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.ini"))

def load_config():
    """
    Reads config.ini and returns a dictionary of relevant info.
    """
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"❌ Config file not found: {CONFIG_FILE}")

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    settings = {
        "SIMULATED_TRADING": config["OKX"].getboolean("SIMULATED_TRADING", fallback=False),

        "SUPABASE_URL": config["SUPABASE"]["SUPABASE_URL"],

        "DEFAULT_PAIR": config["GENERAL"]["DEFAULT_PAIR"],
        "DEFAULT_TIMEFRAME": config["GENERAL"]["DEFAULT_TIMEFRAME"],
        "ORDER_SIZE_LIMIT": config["GENERAL"].getint("ORDER_SIZE_LIMIT", fallback=5),
        "LOG_LEVEL": config["GENERAL"]["LOG_LEVEL"]
    }
    return settings

```

## `backend\controllers\data_retrieval.py`

```python
"""
data_retrieval.py
Handles supabase table retrieval for the 1H and 1D candles.

Requires:
  pip install supabase-py

Assumes you have the following environment variables or direct config:
    SUPABASE_URL
    SUPABASE_ANON_KEY
"""

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from backend.config.config_loader import load_config

SUPABASE_URL = os.getenv("SUPABASE_URL", "<your-supabase-url>")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "<your-anon-key>")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("Supabase credentials missing. Set SUPABASE_URL and SUPABASE_ANON_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def get_candles_1H(pair: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch candle data from the 'candles_1H' table for a given pair.
    :param pair: e.g. "BTC-USDT"
    :param limit: how many recent rows to fetch
    :return: pandas DataFrame with columns:
        [pair, timestamp_ms, open, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote]
    """
    response = supabase.table("candles_1H") \
        .select("*") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    data = response.data  # list of dict
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values("timestamp_ms", ascending=True).reset_index(drop=True)
    return df

def get_candles_1D(pair: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch candle data from the 'candles_1D' table for a given pair.
    :param pair: e.g. "BTC-USDT"
    :param limit: how many recent rows to fetch
    :return: pandas DataFrame with columns:
        [pair, timestamp_ms, open, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote]
    """
    response = supabase.table("candles_1D") \
        .select("*") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    data = response.data
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values("timestamp_ms", ascending=True).reset_index(drop=True)
    return df


def get_recent_candles(pair: str, timeframe: str = "1H", limit: int = 1000) -> pd.DataFrame:
    """
    Simple wrapper to fetch either 1H or 1D from a single function.
    """
    if timeframe == "1H":
        return get_candles_1H(pair, limit)
    elif timeframe == "1D":
        return get_candles_1D(pair, limit)
    else:
        raise ValueError("Unsupported timeframe. Use '1H' or '1D'.")

# Optional: You can add more specialized queries for date ranges, e.g. timestamp_ms >= ...
# For example:

def get_candles_by_range(pair: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetch candles within a specific timestamp range (inclusive).
    :param start_ts: earliest timestamp in ms
    :param end_ts: latest timestamp in ms
    """
    table_name = "candles_1H" if timeframe == "1H" else "candles_1D"
    response = supabase.table(table_name) \
        .select("*") \
        .eq("pair", pair) \
        .gte("timestamp_ms", start_ts) \
        .lte("timestamp_ms", end_ts) \
        .order("timestamp_ms", desc=False).execute()

    data = response.data
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Simple test
    df_1h = get_candles_1H("BTC-USDT", limit=5)
    print("=== Last 5 of 1H ===")
    print(df_1h)

    df_1d = get_candles_1D("BTC-USDT", limit=5)
    print("=== Last 5 of 1D ===")
    print(df_1d)

```

## `backend\controllers\order_execution.py`

```python
"""
order_execution.py
Handles placing, amending, and canceling orders via the OKX V5 API endpoints.
"""

from okx_api.rest_client import OKXRestClient

def place_spot_order(inst_id: str, side: str, ord_type: str, qty: float, price: float = None):
    """
    Places a SPOT order on OKX.
    :param inst_id: e.g. 'BTC-USDT'
    :param side: 'buy' or 'sell'
    :param ord_type: 'limit', 'market', 'post_only', 'fok', 'ioc', ...
    :param qty: the size of the trade, in base or quote currency depending on the ord_type
    :param price: required if it's a 'limit' or 'post_only' or 'fok' order
    """
    client = OKXRestClient()

    data = {
        "instId": inst_id,
        "tdMode": "cash",    # spot trading
        "side": side,
        "ordType": ord_type,
        "sz": str(qty),      # must be string
    }
    if price and ord_type in ["limit", "post_only", "fok", "ioc"]:
        data["px"] = str(price)

    endpoint = "/api/v5/trade/order"
    response = client._request("POST", endpoint, data=data)
    return response

def cancel_spot_order(inst_id: str, ord_id: str = None, cl_ord_id: str = None):
    """
    Cancel an open SPOT order by either 'ordId' or 'clOrdId' (client ID).
    instId is required. If both ord_id & cl_ord_id are passed, ord_id is used.
    """
    client = OKXRestClient()
    endpoint = "/api/v5/trade/cancel-order"

    data = {
        "instId": inst_id
    }
    if ord_id:
        data["ordId"] = ord_id
    elif cl_ord_id:
        data["clOrdId"] = cl_ord_id

    response = client._request("POST", endpoint, data=data)
    return response

def amend_spot_order(inst_id: str, new_qty: float = None, new_px: float = None, ord_id: str = None, cl_ord_id: str = None):
    """
    Amend a pending SPOT order. E.g., modify the price or size (unfilled portion).
    The order ID or client order ID must be specified.
    """
    client = OKXRestClient()
    endpoint = "/api/v5/trade/amend-order"

    data = {
        "instId": inst_id
    }
    if ord_id:
        data["ordId"] = ord_id
    elif cl_ord_id:
        data["clOrdId"] = cl_ord_id

    if new_qty:
        data["newSz"] = str(new_qty)
    if new_px:
        data["newPx"] = str(new_px)

    response = client._request("POST", endpoint, data=data)
    return response

```

## `backend\controllers\trading_account.py`

```python
"""
trading_account.py
Retrieves account-related information like balances, trade fee, positions, etc.
"""

from okx_api.rest_client import OKXRestClient
from backend.config.config_loader import load_config

config = load_config()
simulated_trading = config["SIMULATED_TRADING"]

client = OKXRestClient(simulated_trading=simulated_trading)

def get_balance_info():
    """
    Returns account balances from /api/v5/account/balance
    """
    client = OKXRestClient()
    resp = client.get_balance()
    return resp

def get_positions_info():
    """
    Returns positions info from /api/v5/account/positions
    (Applicable if you're using margin/Futures/Swap.)
    """
    client = OKXRestClient()
    resp = client.get_positions()
    return resp

def get_account_config():
    """
    Returns the account configuration from /api/v5/account/config
    e.g. position mode, risk settings.
    """
    client = OKXRestClient()
    resp = client.get_account_config()
    return resp

def get_trade_fee(inst_type: str = "SPOT", inst_id: str = None):
    """
    Returns fee rate (maker & taker) from /api/v5/account/trade-fee
    :param inst_type: 'SPOT', 'FUTURES', 'SWAP', etc.
    :param inst_id: specific pair e.g. 'BTC-USDT' if you want a more precise fee
    """
    client = OKXRestClient()
    endpoint = "/api/v5/account/trade-fee"
    params = {
        "instType": inst_type
    }
    if inst_id:
        params["instId"] = inst_id

    resp = client._request("GET", endpoint, params=params)
    return resp

```

## `backend\indicators\atr.py`

```python
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

```

## `backend\indicators\bollinger.py`

```python
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

```

## `backend\indicators\macd.py`

```python
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

```

## `backend\indicators\rsi.py`

```python
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

```

## `backend\indicators\stoch_rsi.py`

```python
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

```

## `backend\ml\model_trainer.py`

```python
# Where we’d eventually load data & train
```

## `backend\ml\predictor.py`

```python
# Possibly used at runtime for inference
```

## `backend\signal_engine\strategy.py`

```python
"""
strategy.py
A simple example strategy for OKXsignal that:
  1) Fetches candle data from Supabase (1H or 1D)
  2) Computes RSI, Bollinger Bands, and MACD
  3) Generates a naive "buy/sell/hold" recommendation

Notes:
    - This is purely illustrative. 
    - Adjust thresholds or add more logic for real production usage.
"""

import pandas as pd
from backend.controllers.data_retrieval import get_recent_candles
from backend.indicators.rsi import compute_rsi
from backend.indicators.macd import compute_macd
from backend.indicators.bollinger import compute_bollinger_bands
from backend.config.config_loader import load_config

config = load_config()
DEFAULT_PAIR = config["DEFAULT_PAIR"]
DEFAULT_TIMEFRAME = config["DEFAULT_TIMEFRAME"]

def generate_signal(
    pair: str = "BTC-USDT", 
    timeframe: str = "1H", 
    limit: int = 100
) -> dict:
    """
    Fetches candles, applies indicators, and decides on a naive action.

    :param pair: e.g. "BTC-USDT"
    :param timeframe: "1H" or "1D"
    :param limit: how many rows of data to fetch
    :return: dict with keys { "pair", "timeframe", "action", "reason" }
             action can be "BUY", "SELL", "HOLD"
             reason is a short string explaining the logic.
    """

    # 1) Retrieve Data from Supabase
    df = get_recent_candles(pair, timeframe, limit)
    if df.empty or len(df) < 20:
        return {
            "pair": pair,
            "timeframe": timeframe,
            "action": "HOLD",
            "reason": "Insufficient candle data."
        }

    # 2) Compute Indicators
    df_ind = df.copy()

    # RSI
    df_ind = compute_rsi(df_ind, period=14, col_name="RSI")
    # MACD
    df_ind = compute_macd(df_ind, fast=12, slow=26, signal=9, col_prefix="MACD")
    # Bollinger
    df_ind = compute_bollinger_bands(df_ind, period=20, std_multiplier=2.0, col_prefix="BB")

    # 3) Check latest row
    last_row = df_ind.iloc[-1]
    rsi_val = last_row["RSI"]
    macd_line = last_row["MACD_Line"]
    macd_signal = last_row["MACD_Signal"]
    close_price = last_row["close"]

    # 4) Some naive logic
    # If RSI < 30 and MACD_Line > MACD_Signal => "BUY"
    if rsi_val < 30 and macd_line > macd_signal:
        action = "BUY"
        reason = f"RSI {rsi_val:.2f} < 30 and MACD crossing up."
    # If RSI > 70 and MACD_Line < MACD_Signal => "SELL"
    elif rsi_val > 70 and macd_line < macd_signal:
        action = "SELL"
        reason = f"RSI {rsi_val:.2f} > 70 and MACD crossing down."
    else:
        action = "HOLD"
        reason = "No strong signal from RSI/MACD."

    return {
        "pair": pair,
        "timeframe": timeframe,
        "action": action,
        "reason": reason,
        "latest_close": close_price
    }

if __name__ == "__main__":
    # Example usage
    signal_info = generate_signal("BTC-USDT", "1H", 100)
    print(signal_info)

```

## `dashboard\components\alerts.py`

```python
# alerts.py\n# For toast-style notifications, warnings, or info messages in the UI.
```

## `dashboard\components\metrics.py`

```python
# metrics.py\n# Functions to display KPI cards (e.g., total portfolio value, 24h change).
```

## `dashboard\components\charts\candle_chart.py`

```python
# candle_chart.py\n# Functions to render candlestick charts (e.g., Plotly, Matplotlib).
```

## `dashboard\components\charts\macd_plot.py`

```python
# macd_plot.py\n# Renders MACD lines, signal line, and histogram subchart.
```

## `dashboard\components\charts\rsi_plot.py`

```python
# rsi_plot.py\n# Functions to generate RSI subcharts (or overlays).
```

## `dashboard\components\forms\filter_form.py`

```python
# filter_form.py\n# Common filtering widget for date ranges, pairs, timeframes.
```

## `dashboard\components\forms\order_form.py`

```python
# order_form.py\n# Streamlit form to place or cancel orders (buy/sell inputs).
```

## `dashboard\components\tables\portfolio_table.py`

```python
# portfolio_table.py\n# Shows holdings in a table with columns for PnL, cost basis, etc.
```

## `dashboard\components\tables\trades_table.py`

```python
# trades_table.py\n# Displays past trades in a table with consistent formatting.
```

## `dashboard\layout\base_page.py`

```python
# base_page.py\n# A reusable page template or layout function for consistent look.
```

## `dashboard\layout\navigation.py`

```python
# navigation.py\n# Builds the sidebar or topbar navigation for the dashboard.
```

## `dashboard\layout\theme.py`

```python
# theme.py\n# Contains theming constants, color palettes, and overall UI style.
```

## `dashboard\pages\home.py`

```python
# home.py\n# Landing page with quick stats, welcome text, and shortcuts.
```

## `dashboard\pages\trade_execution.py`

```python
# trade_execution.py\n# Page that allows single-click buy/sell with advanced order forms.
```

## `dashboard\pages\__init__.py`

```python
# __init__.py\n# Marks this as a Python package.
```

## `dashboard\pages\market_analysis\advanced_charts.py`

```python
# advanced_charts.py\n# Bollinger/MACD/RSI overlays and multi-subplot analysis.
```

## `dashboard\pages\market_analysis\overview.py`

```python
# overview.py\n# Basic candlestick chart, volume, and quick indicators.
```

## `dashboard\pages\market_analysis\signals.py`

```python
# signals.py\n# List of current signal triggers and recommended actions.
```

## `dashboard\pages\portfolio\fees.py`

```python

```

## `dashboard\pages\portfolio\holdings.py`

```python

```

## `dashboard\pages\portfolio\order_history.py`

```python

```

## `dashboard\pages\settings\risk_config.py`

```python
# risk_config.py\n# Global risk tolerance settings (max position size, etc.).
```

## `dashboard\pages\settings\user_prefs.py`

```python
# user_prefs.py\n# Personal preferences: default pair, default timeframe, etc.
```

## `dashboard\utils\data_loader.py`

```python
# data_loader.py\n# Helper to fetch data from Supabase or caching layer, used by multiple pages.
```

## `dashboard\utils\session_manager.py`

```python
# session_manager.py\n# Manages Streamlit session state (user data, caching, etc.).
```

## `okx_api\auth.py`

```python
# Handles authentication for OKX API
"""
auth.py
Handles authentication for OKX API using HMAC SHA256 signatures.
"""

import hmac
import hashlib
import base64
import datetime
import os
from dotenv import load_dotenv

# Force-load .env file from the exact path
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "config", "credentials.env"))
if not os.path.exists(env_path):
    raise FileNotFoundError(f"❌ credentials.env file not found at {env_path}")

loaded = load_dotenv(env_path)
if not loaded:
    raise ValueError(f"❌ Failed to load environment variables from {env_path}")



API_KEY = os.getenv("OKX_API_KEY", "").strip()
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "").strip()
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "").strip()

if not API_KEY or not SECRET_KEY or not PASSPHRASE:
    raise ValueError(f"❌ Missing API credentials.\nLoaded Values:\n"
                     f"OKX_API_KEY='{API_KEY}'\nOKX_SECRET_KEY='{SECRET_KEY}'\nOKX_PASSPHRASE='{PASSPHRASE}'")

def get_timestamp() -> str:
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

def get_signature(timestamp: str, method: str, path: str, body: str = "") -> str:
    message = f"{timestamp}{method}{path}{body}"
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

def get_headers(timestamp: str, method: str, path: str, body: str = "", simulated: bool = False) -> dict:
    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": get_signature(timestamp, method, path, body),
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }
    if simulated:
        headers["x-simulated-trading"] = "1"
    return headers


```

## `okx_api\rest_client.py`

```python
# OKX REST API client
"""
rest_client.py
Handles REST API requests for OKX V5 endpoints.
"""

import requests
import json
from okx_api.auth import get_headers, get_timestamp

BASE_URL = "https://my.okx.com"

class OKXRestClient:
    """
    REST client to interact with OKX V5 endpoints.
    """

    def __init__(self, simulated_trading: bool = False):
        self.simulated_trading = simulated_trading

    def _request(self, method: str, endpoint: str, params=None, data=None) -> dict:
        """
        Internal method to handle signed requests.
        """
        timestamp = get_timestamp()
        body = json.dumps(data) if data else ""

        headers = get_headers(
            timestamp=timestamp,
            method=method,
            path=endpoint,
            body=body,
            simulated=self.simulated_trading
        )

        url = f"{BASE_URL}{endpoint}"
        response = requests.request(method, url=url, headers=headers, params=params, json=data)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
        
        return response.json()

    # ✅ Market Data
    def get_ticker(self, instId: str) -> dict:
        return self._request("GET", "/api/v5/market/ticker", params={"instId": instId})

    def get_orderbook(self, instId: str, depth: int = 5) -> dict:
        return self._request("GET", "/api/v5/market/books", params={"instId": instId, "sz": depth})

    def get_candlesticks(self, instId: str, bar: str = "1D", limit: int = 100) -> dict:
        return self._request("GET", "/api/v5/market/candles", params={"instId": instId, "bar": bar, "limit": limit})

    # ✅ Account Information
    def get_balance(self) -> dict:
        return self._request("GET", "/api/v5/account/balance")

    def get_positions(self) -> dict:
        return self._request("GET", "/api/v5/account/positions")

    def get_account_config(self) -> dict:
        return self._request("GET", "/api/v5/account/config")

    def get_instruments(self, instType: str = "SPOT") -> dict:
        """
        Fetches a list of trading instruments (e.g., all SPOT pairs).
        """
        return self._request("GET", "/api/v5/public/instruments", params={"instType": instType})

```

## `okx_api\__init__.py`

```python
# Initializes OKX API module

```

