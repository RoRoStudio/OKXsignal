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

## `backend\config\settings.py`

```python
# General configuration settings

```

## `backend\controllers\data_retrieval.py`

```python

```

## `backend\controllers\order_execution.py`

```python

```

## `backend\controllers\trading_account.py`

```python

```

## `backend\execution\api.py`

```python
﻿# api.py
# FastAPI backend for triggering Python scripts from Grafana
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend API is working!"}

```

## `backend\execution\async_requests.py`

```python
# Handles asynchronous batch API requests

```

## `backend\execution\fetch_market.py`

```python
"""
fetch_market.py
Fetches historical market data (candlesticks) for all USDT spot pairs on OKX.
Designed to be triggered by Streamlit.
"""

import asyncio
import time
import random
import pandas as pd
import streamlit as st
from tqdm.asyncio import tqdm  # Async progress bar
from okx_api.rest_client import OKXRestClient

CANDLE_LIMIT = 100  # OKX API max per request
TOTAL_CANDLES = 1000  # Number of daily candles we want
TIMEFRAME = "1D"  # 1-day timeframe
MAX_CONCURRENT_REQUESTS = 20  # ✅ Rate-limiting

async def fetch_usdt_pairs():
    """Fetches active USDT spot trading pairs from OKX."""
    client = OKXRestClient()
    response = client.get_instruments(instType="SPOT")

    raw_pairs = [x["instId"] for x in response["data"] if x["instId"].endswith("-USDT") and x["state"] == "live"]

    valid_pairs = []
    batch_size = 10
    for i in range(0, len(raw_pairs), batch_size):
        batch = raw_pairs[i : i + batch_size]

        for pair in batch:
            test_response = client.get_ticker(pair)
            if "code" in test_response and test_response["code"] == "51001":
                st.warning(f"🛑 Skipping non-existent pair: {pair}")
            else:
                valid_pairs.append(pair)

        time.sleep(1)  # ✅ Add delay to avoid rate-limiting

    st.success(f"✅ Found {len(valid_pairs)} valid USDT trading pairs.")
    return valid_pairs

async def fetch_candles(pair, semaphore, progress_bar):
    """Fetches historical candlestick data for a trading pair."""
    client = OKXRestClient()
    all_data = []

    async with semaphore:
        while len(all_data) < TOTAL_CANDLES:
            response = client.get_candlesticks(instId=pair, bar=TIMEFRAME, limit=CANDLE_LIMIT)

            if "code" in response and response["code"] == "50011":
                st.warning(f"⚠️ Rate limited. Retrying {pair}...")
                await asyncio.sleep(random.uniform(3, 5))  
                continue  

            if "data" not in response or len(response["data"]) == 0:
                return pair, pd.DataFrame()

            all_data.extend(response["data"])
            await asyncio.sleep(random.uniform(0.2, 0.5))  # ✅ Avoid hammering API

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")

    progress_bar.update(1)
    return pair, df[:TOTAL_CANDLES]

async def fetch_all_pairs():
    """Fetches market data for all USDT trading pairs."""
    pairs = await fetch_usdt_pairs()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    progress_bar = st.progress(0)
    total_pairs = len(pairs)

    async def limited_fetch(pair):
        return await fetch_candles(pair, semaphore, progress_bar)

    tasks = [limited_fetch(pair) for pair in pairs]
    results = await asyncio.gather(*tasks)

    progress_bar.empty()
    return {pair: df for pair, df in results}

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(fetch_all_pairs())


```

## `backend\execution\fetch_portfolio.py`

```python
# Fetches user portfolio balances from OKX API

```

## `backend\indicators\atr.py`

```python
# For volatility-based sizing
```

## `backend\indicators\bollinger.py`

```python
# Computes Bollinger Bands indicator

```

## `backend\indicators\macd.py`

```python
# Computes MACD indicator

```

## `backend\indicators\rsi.py`

```python
# Computes RSI indicator

```

## `backend\indicators\stoch_rsi.py`

```python
# recommended for “wild swings” 
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
# Defines strategy for Buy/Sell/Hold recommendations

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

