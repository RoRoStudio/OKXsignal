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
    os.path.join(ROOT_DIR, "supabase"),
    os.path.join(ROOT_DIR, "config\\credentials.env"),
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

## `README.md`

```python

```
OKXsignal
├─ (personal) commands.md
├─ backend
│  ├─ controllers
│  │  ├─ data_retrieval.py
│  │  ├─ order_execution.py
│  │  ├─ strategy.py
│  │  └─ trading_account.py
│  ├─ ml
│  │  ├─ model_trainer.py
│  │  └─ predictor.py
│  ├─ requirements.txt
│  ├─ test_rest_client.py
│  └─ __init__.py
├─ config
│  ├─ config.ini
│  ├─ config_loader.py
│  └─ credentials.env
├─ dashboard
│  ├─ assets
│  │  ├─ custom.css
│  │  └─ images
│  ├─ components
│  │  ├─ alerts.py
│  │  ├─ charts
│  │  │  ├─ candle_chart.py
│  │  │  ├─ macd_plot.py
│  │  │  └─ rsi_plot.py
│  │  ├─ forms
│  │  │  ├─ filter_form.py
│  │  │  └─ order_form.py
│  │  ├─ metrics.py
│  │  └─ tables
│  │     ├─ portfolio_table.py
│  │     └─ trades_table.py
│  ├─ layout
│  │  ├─ base_page.py
│  │  ├─ navigation.py
│  │  └─ theme.py
│  ├─ pages
│  │  ├─ home.py
│  │  ├─ market_analysis
│  │  │  ├─ advanced_charts.py
│  │  │  ├─ overview.py
│  │  │  └─ signals.py
│  │  ├─ portfolio
│  │  │  ├─ fees.py
│  │  │  ├─ holdings.py
│  │  │  └─ order_history.py
│  │  ├─ settings
│  │  │  ├─ risk_config.py
│  │  │  └─ user_prefs.py
│  │  ├─ trade_execution.py
│  │  └─ __init__.py
│  └─ utils
│     ├─ data_loader.py
│     └─ session_manager.py
├─ export_markdown.py
├─ main.py
├─ okx_api
│  ├─ auth.py
│  ├─ rest_client.py
│  └─ __init__.py
├─ README.md
├─ requirements.txt
└─ supabase
   ├─ .branches
   │  └─ _current_branch
   ├─ .temp
   │  └─ cli-latest
   └─ functions
      ├─ backfill_missing_1h_candles.py
      ├─ fetch_new_1d_candles.py
      ├─ fetch_new_1h_candles.py
      ├─ fetch_old_1d_candles.py
      ├─ fetch_old_1h_candles.py
      └─ __init__.py

```
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

## `backend\controllers\data_retrieval.py`

```python
"""
data_retrieval.py
Handles fetching price and indicator data from Supabase for the 1H and 1D candles.
"""

import os
import pandas as pd
from supabase import create_client, Client

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("❌ Supabase credentials are missing.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def fetch_market_data(pair: str, timeframe: str = "1H", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch recent market data (candlesticks + indicators) from Supabase.

    :param pair: Trading pair, e.g., "BTC-USDT".
    :param timeframe: "1H" or "1D".
    :param limit: Number of most recent rows to fetch.
    :return: Pandas DataFrame containing market data with indicators.
    """
    table_name = f"candles_{timeframe}"

    response = supabase.table(table_name) \
        .select("pair", "timestamp_ms", "open", "high", "low", "close", "volume",
                "rsi", "macd_line", "macd_signal", "macd_hist",
                "bollinger_middle", "bollinger_upper", "bollinger_lower",
                "atr", "stoch_rsi_k", "stoch_rsi_d") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    data = response.data
    return pd.DataFrame(data) if data else pd.DataFrame()

if __name__ == "__main__":
    df = fetch_market_data("BTC-USDT", "1H", 5)
    print(df)

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

## `backend\controllers\strategy.py`

```python
"""
strategy.py
Fetches precomputed indicators from Supabase and generates a trading signal.
"""

from backend.controllers.data_retrieval import fetch_market_data

def generate_signal(pair: str = "BTC-USDT", timeframe: str = "1H", limit: int = 100) -> dict:
    """
    Fetches precomputed indicators and generates a trading signal.
    """
    df = fetch_market_data(pair, timeframe, limit)

    if df.empty or len(df) < 20:
        return {
            "pair": pair,
            "timeframe": timeframe,
            "action": "HOLD",
            "reason": "Insufficient candle data."
        }

    # Retrieve indicators from the table
    last_row = df.iloc[-1]
    rsi = last_row["rsi"]
    macd_line = last_row["macd_line"]
    macd_signal = last_row["macd_signal"]
    macd_hist = last_row["macd_hist"]
    close_price = last_row["close"]
    atr = last_row["atr"]
    stoch_rsi_k = last_row["stoch_rsi_k"]
    stoch_rsi_d = last_row["stoch_rsi_d"]
    bollinger_upper = last_row["bollinger_upper"]
    bollinger_middle = last_row["bollinger_middle"]
    bollinger_lower = last_row["bollinger_lower"]

    # Trading Logic (Updated)
    if (
        rsi < 30 and 
        macd_line > macd_signal and 
        stoch_rsi_k > 0.8 and 
        close_price < bollinger_lower
    ):
        action = "BUY"
        reason = f"RSI {rsi:.2f} < 30, MACD crossover, Stoch RSI K > 0.8, Price near lower Bollinger Band."
    elif (
        rsi > 70 and 
        macd_line < macd_signal and 
        stoch_rsi_k < 0.2 and 
        close_price > bollinger_upper
    ):
        action = "SELL"
        reason = f"RSI {rsi:.2f} > 70, MACD crossover, Stoch RSI K < 0.2, Price near upper Bollinger Band."
    else:
        action = "HOLD"
        reason = "No strong signal."

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

## `backend\controllers\trading_account.py`

```python
"""
trading_account.py
Retrieves account-related information like balances, trade fee, positions, etc.
"""

from okx_api.rest_client import OKXRestClient
from config.config_loader import load_config

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

## `backend\ml\model_trainer.py`

```python
# Where we’d eventually load data & train
```

## `backend\ml\predictor.py`

```python
# Possibly used at runtime for inference
```

## `config\config_loader.py`

```python
"""
config_loader.py
Loads configuration settings from config/config.ini.
"""

import os
import configparser

# Locate config.ini in the root config/ directory
CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.ini"))

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

# Load environment variables from credentials.env in config/
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "credentials.env"))
load_dotenv(env_path)

# Retrieve credentials from environment variables
API_KEY = os.getenv("OKX_API_KEY", "").strip()
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "").strip()
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "").strip()

if not API_KEY or not SECRET_KEY or not PASSPHRASE:
    raise ValueError("❌ Missing OKX API credentials. Check config/credentials.env.")

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

