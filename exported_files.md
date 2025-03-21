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
🚀 Entry point for OKXsignal Dashboard (Streamlit).
"""

import streamlit as st

# ✅ First Streamlit command (MUST be first)
st.set_page_config(
    page_title="OKXsignal",
    page_icon="⚡",
    layout="wide",
)

import os
import sys

# Ensure correct imports from dashboard/
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "dashboard"))

# Import layout & theming
from dashboard.layout.theme import inject_global_css
from dashboard.layout.base_page import apply_base_layout
from dashboard.layout.navigation import build_sidebar
from dashboard.components.forms.filter_form import filter_form

# Inject global CSS (ensure styling is applied correctly)
inject_global_css()

# Load navigation sidebar
selected_page_name = build_sidebar()

# ✅ **Fix: Ensure module names match exactly**
valid_pages = {
    "home": "dashboard.pages.home",
    "market_analysis.overview": "dashboard.pages.market_analysis.overview",  
    "portfolio.holdings": "dashboard.pages.portfolio.holdings",
    "trade_execution": "dashboard.pages.trade_execution",
    "settings.user_prefs": "dashboard.pages.settings.user_prefs",
}

if selected_page_name in valid_pages:
    try:
        page_module = __import__(valid_pages[selected_page_name], fromlist=["show_page"])
        
        # Ensure the module has `show_page` function
        if hasattr(page_module, "show_page"):
            page_module.show_page()
        else:
            st.error(f"❌ Page '{selected_page_name}' does not have a `show_page()` function.")
    except ImportError as e:
        st.error(f"❌ Page '{selected_page_name}' not found. Error: {e}")
else:
    st.error(f"❌ Invalid page selection: {selected_page_name}")

```

## `README.md`

```python

```
OKXsignal
├─ (personal) commands.md
├─ .streamlit
│  └─ config.toml
├─ backend
│  ├─ controllers
│  │  ├─ data_retrieval.py
│  │  ├─ order_execution.py
│  │  ├─ strategy.py
│  │  └─ trading_account.py
│  ├─ indicators
│  ├─ ml
│  │  ├─ model_trainer.py
│  │  └─ predictor.py
│  ├─ requirements.txt
│  ├─ signal_engine
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
│  │  │  ├─ macd_plot.py
│  │  │  ├─ master_chart.py
│  │  │  └─ rsi_plot.py
│  │  ├─ forms
│  │  │  ├─ filter_form.py
│  │  │  └─ order_form.py
│  │  ├─ metrics.py
│  │  └─ tables
│  │     ├─ candle_table.py
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

## `dashboard\components\charts\macd_plot.py`

```python
# macd_plot.py\n# Renders MACD lines, signal line, and histogram subchart.
```

## `dashboard\components\charts\master_chart.py`

```python
"""
master_chart.py
📊 The all-in-one Master Chart for OKXsignal, dynamically adjusting zoom & range.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pytz
from dashboard.utils.data_loader import fetch_market_data

# 🎨 Colors
PRIMARY_COLOR = "#03fcb6"  # ✅ Green for bullish candles
BEARISH_COLOR = "#e74c3c"  # 🔻 Red for bearish candles
VOLUME_COLOR = "#3498db"   # 🔵 Blue for volume bars

# ✅ Default indicator settings
DEFAULT_INDICATORS = {
    "Bollinger Bands": True,
    "MACD": True,
    "RSI": True,
    "ATR": True,
}

def show_master_chart(data):
    """Displays the Master Chart with price, volume, and key indicators."""

    # ✅ Fetch updated market data when filters change
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")
    date_range = st.session_state.get("selected_date_range", [])

    # ✅ Fetch new market data based on the selected filters
    data = fetch_market_data(pair=pair, timeframe=timeframe, limit=100)

    if not data:
        st.warning("No data available for this pair and timeframe.")
        return

    # ✅ Convert list of dicts to DataFrame
    df = pd.DataFrame(data)

    # ✅ Convert timestamps to local time
    local_tz = pytz.timezone("Europe/Amsterdam")
    df["timestamp_ms"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(local_tz)

    # ✅ Apply Date Filtering (Handles partial selections)
    if date_range and isinstance(date_range, list):
        if len(date_range) == 1:  # ✅ Only start date selected
            start_timestamp = int(pd.Timestamp(date_range[0]).timestamp() * 1000)
            df = df[df["timestamp_ms"] >= start_timestamp]
        elif len(date_range) == 2:  # ✅ Both start & end date selected
            start_timestamp = int(pd.Timestamp(date_range[0]).timestamp() * 1000)
            end_timestamp = int(pd.Timestamp(date_range[1]).timestamp() * 1000)
            df = df[(df["timestamp_ms"] >= start_timestamp) & (df["timestamp_ms"] <= end_timestamp)]

    # ✅ Auto-Adjust Zoom: Get min & max price for Y-axis
    min_price = df["low"].min() * 0.99
    max_price = df["high"].max() * 1.01

    # ✅ User selection for indicators
    with st.expander("⚙️ Chart Settings", expanded=True):
        st.markdown("**Toggle Indicators**")
        indicators = {key: st.checkbox(key, DEFAULT_INDICATORS[key]) for key in DEFAULT_INDICATORS}

    # ✅ Initialize Plotly figure (Double Height)
    fig = go.Figure()

    # 📈 **Candlestick Chart**
    fig.add_trace(go.Candlestick(
        x=df["timestamp_ms"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing=dict(line=dict(color=PRIMARY_COLOR), fillcolor=PRIMARY_COLOR),
        decreasing=dict(line=dict(color=BEARISH_COLOR), fillcolor=BEARISH_COLOR),
        name="Candles"
    ))

    # 📊 **Volume Bars**
    fig.add_trace(go.Bar(
        x=df["timestamp_ms"],
        y=df["volume"],
        marker_color=VOLUME_COLOR,
        opacity=0.5,
        name="Volume"
    ))

    # 🎚️ **Bollinger Bands**
    if indicators["Bollinger Bands"]:
        fig.add_trace(go.Scatter(x=df["timestamp_ms"], y=df["bollinger_upper"], line=dict(color="gray", width=1, dash="dot"), name="Bollinger Upper"))
        fig.add_trace(go.Scatter(x=df["timestamp_ms"], y=df["bollinger_middle"], line=dict(color="gray", width=1, dash="dash"), name="Bollinger Middle"))
        fig.add_trace(go.Scatter(x=df["timestamp_ms"], y=df["bollinger_lower"], line=dict(color="gray", width=1, dash="dot"), name="Bollinger Lower"))

    # 📉 **MACD**
    if indicators["MACD"]:
        fig.add_trace(go.Scatter(x=df["timestamp_ms"], y=df["macd_line"], line=dict(color="blue", width=1), name="MACD Line"))
        fig.add_trace(go.Scatter(x=df["timestamp_ms"], y=df["macd_signal"], line=dict(color="orange", width=1, dash="dot"), name="MACD Signal"))

    # 📊 **RSI**
    if indicators["RSI"]:
        fig.add_trace(go.Scatter(x=df["timestamp_ms"], y=df["rsi"], line=dict(color="purple", width=1), name="RSI"))

    # 📊 **ATR (Volatility Indicator)**
    if indicators["ATR"]:
        fig.add_trace(go.Scatter(x=df["timestamp_ms"], y=df["atr"], line=dict(color="red", width=1), name="ATR (Volatility)"))

    # ✅ Chart Settings (Auto-Adjust Zoom & Secondary Y-Axis for Volume)
    fig.update_layout(
        height=900,
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        yaxis=dict(range=[min_price, max_price]),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1),
        
        # ✅ Volume now has a separate Y-axis
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)

```

## `dashboard\components\charts\rsi_plot.py`

```python
# rsi_plot.py\n# Functions to generate RSI subcharts (or overlays).
```

## `dashboard\components\forms\filter_form.py`

```python
"""
filter_form.py
🔍 Provides filtering options (pair, timeframe, date range) for market data.
"""

"""
filter_form.py
🔍 Provides filtering options (pair, timeframe, date range) for market data.
"""

import streamlit as st
import datetime
from dashboard.utils.data_loader import fetch_trading_pairs

def filter_form():
    """Displays the filtering UI inside a styled container."""
    
    with st.container():
        st.markdown("### 🔍 Filter Data")

        # Fetch available trading pairs
        trading_pairs = fetch_trading_pairs()  

        # Select Trading Pair
        st.selectbox("Trading Pair", trading_pairs, key="selected_pair")

        # Select Timeframe
        st.radio("Timeframe", ["1H", "1D"], key="selected_timeframe", horizontal=True)

        # Select Date Range
        date_range = st.date_input("Select Date Range", [], key="selected_date_range")

        # ✅ Apply Filters Button (Now required to trigger updates)
        if st.button("Apply Filters"):
            st.session_state["filters_applied"] = True  # ✅ Force refresh table & chart

# ✅ Ensure session state variables are initialized
if "selected_pair" not in st.session_state:
    st.session_state["selected_pair"] = "BTC-USDT"
if "selected_timeframe" not in st.session_state:
    st.session_state["selected_timeframe"] = "1H"
if "selected_date_range" not in st.session_state:
    st.session_state["selected_date_range"] = []

```

## `dashboard\components\forms\order_form.py`

```python
# order_form.py\n# Streamlit form to place or cancel orders (buy/sell inputs).
```

## `dashboard\components\tables\candle_table.py`

```python
"""
candle_table.py
📊 Displays fetched market data in a Streamlit table.
"""

import streamlit as st
from dashboard.utils.data_loader import fetch_market_data

def show_candle_table(pair="BTC-USDT", timeframe="1H"):
    """
    Displays a table of candlestick data with indicators.
    """
    st.subheader(f"{pair} - {timeframe} Candles")

    # Fetch Data
    data = fetch_market_data(pair, timeframe, 100)

    if not data:
        st.warning("No data available.")
        return

    # Display Table
    st.dataframe(data, use_container_width=True)


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
"""
base_page.py
📌 Provides a base layout for all pages.
"""

import streamlit as st

def apply_base_layout(page_function):
    """
    Applies the base layout to the page.
    :param page_function: The function that renders the actual page content.
    """

    st.container()
    page_function()  # Render the selected page

```

## `dashboard\layout\navigation.py`

```python
"""
navigation.py
🧭 Handles sidebar navigation for OKXsignal.
"""

import streamlit as st

def build_sidebar():
    """
    Creates the sidebar for navigation and returns the selected page function.
    """
    with st.sidebar:
        st.title("⚡ OKXsignal")
        st.markdown("---")

        pages = {
            "🏠 Home": "home",
            "📈 Market Analysis": "market_analysis.overview",
            "💰 Portfolio": "portfolio.holdings",
            "⚡ Trade Execution": "trade_execution",
            "⚙️ Settings": "settings.user_prefs",
        }

        selected_page = st.radio("Navigate to:", list(pages.keys()))

        return pages[selected_page]

```

## `dashboard\layout\theme.py`

```python
"""
theme.py
🎨 Defines global theming & styling for OKXsignal's Streamlit Dashboard.
"""

import streamlit as st

def inject_global_css():
    """Injects global CSS for consistent UI styling inspired by InsightBig."""
    custom_css = """
    <style>
        /* 📌 Global Styling */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        /* 🌍 Dark Theme Colors */
        :root {
            --primary-color: #03fcb6;
            --background-dark: #131722;
            --text-dark: #f6f6f6;
            --secondary-background: #0c0e15;
            --box-shadow: -6px 8px 20px 1px rgba(0, 0, 0, 0.52);
        }

        /* 🌙 Dark Mode Only */
        body {
            background-color: var(--background-dark);
            color: var(--text-dark);
        }

        /* 🎨 Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background);
            padding: 20px;
            box-shadow: var(--box-shadow);
            border-radius: 10px;
        }

        /* 📦 Containers */
        div[data-testid="stVerticalBlock"] {
            background-color: var(--secondary-background);
            border-radius: 12px;
            padding: 20px;
            box-shadow: var(--box-shadow);
        }

        /* 🔘 Custom Radio Buttons */
        div[data-testid="stRadio"] label {
            color: var(--text-dark) !important;
            font-weight: bold !important;
        }

        /* 🚀 Custom Buttons */
        .stButton > button {
            background-color: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: bold !important;
            transition: all 0.2s ease-in-out;
            box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.15);
        }
        .stButton > button:hover {
            background-color: #02e3a5 !important;
        }

        /* 🛠️ Forms & Dropdowns */
        .stSelectbox, .stMultiSelect {
            border: 0px solid var(--primary-color) !important;
            background-color: var(--secondary-background);
            color: var(--text-dark);
            border-radius: 5px;
            padding: 8px;
        }

        /* 📢 Custom Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-dark) !important;
            font-weight: bold !important;
        }

        /* 📝 Tables */
        table {
            border: 1px solid var(--primary-color) !important;
            border-radius: 10px !important;
        }

        /* 📦 Custom Containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            box-shadow: var(--box-shadow);
            border-radius: 10px;
        }

        /* 📏 Dashboard Padding */
        div.block-container {
            padding-left: 5rem !important;
            padding-right: 5rem !important;
            padding-top: 15px !important;
            padding-bottom: 40px !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

```

## `dashboard\pages\home.py`

```python
"""
home.py
🏠 Home Page for OKXsignal Dashboard.
"""

import streamlit as st

def show_page():
    st.title("🏠 Home")
    st.markdown("Welcome to **OKXsignal**!")

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
"""
overview.py
📊 Market Overview Page: Displays the latest 1H & 1D candles with filtering options.
"""

import streamlit as st
import datetime
import pandas as pd
import pytz
from dashboard.components.forms.filter_form import filter_form
from dashboard.utils.data_loader import fetch_market_data, fetch_trading_pairs
from dashboard.components.charts.master_chart import show_master_chart

def show_page():
    st.title("📈 Market Overview")

    # ✅ Step 1: Show Filter Form
    filter_form()

    # ✅ Step 2: Fetch Data (Only when "Apply Filters" is clicked)
    if st.session_state.get("filters_applied", False):
        trading_pairs = fetch_trading_pairs()
        selected_data = fetch_selected_market_data()

        # ✅ Step 3: Show Market Summary (Gainers, Losers, Volume Movers)
        show_market_summary(trading_pairs)

        # ✅ Step 4: Show Selected Pair Data (Instant Update)
        show_filtered_data(selected_data)

        # ✅ Step 5: Show Master Chart (Connected to Filters)
        show_master_chart(selected_data)

        # ✅ Reset applied filter flag to avoid unnecessary refreshes
        st.session_state["filters_applied"] = False
    else:
        st.warning("👆 Adjust filters and click 'Apply Filters' to refresh the table & chart.")

@st.cache_data(ttl=3600)
def fetch_selected_market_data():
    """Fetch data for the currently selected pair & timeframe (cached for 1 hour)."""
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")
    return fetch_market_data(pair=pair, timeframe=timeframe)

def show_market_summary(trading_pairs):
    """Displays key market summary statistics for rapid insight."""
    st.markdown("### 📊 Market Summary")

    summary_data = []
    for pair in trading_pairs[:50]:  # Limit for performance
        market_data = fetch_market_data(pair, "1D", 2)
        if len(market_data) >= 2:
            prev_close, latest_close = market_data[1]["close"], market_data[0]["close"]
            percent_change = ((latest_close - prev_close) / prev_close) * 100
            summary_data.append({
                "pair": pair,
                "close": latest_close,
                "percent_change": percent_change,
                "rsi": market_data[0]["rsi"],
                "volume": market_data[0]["volume"]
            })

    if not summary_data:
        st.warning("No market summary data available.")
        return

    df = pd.DataFrame(summary_data)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🚀 Top 5 Gainers")
        st.dataframe(df.nlargest(5, "percent_change")[["pair", "percent_change"]], use_container_width=True)

    with col2:
        st.markdown("#### 🔻 Top 5 Losers")
        st.dataframe(df.nsmallest(5, "percent_change")[["pair", "percent_change"]], use_container_width=True)

    with col3:
        st.markdown("#### 🔥 Biggest Volume Movers")
        st.dataframe(df.nlargest(5, "volume")[["pair", "volume"]], use_container_width=True)

def show_filtered_data(data):
    """Displays the filtered market data in a table and auto-updates."""
    pair = st.session_state.get("selected_pair", "BTC-USDT")
    timeframe = st.session_state.get("selected_timeframe", "1H")
    date_range = st.session_state.get("selected_date_range", [])

    # ✅ Gracefully Handle Partial Date Selection
    start_timestamp, end_timestamp = None, None
    if date_range:
        if isinstance(date_range, list) and len(date_range) > 0:
            if len(date_range) == 1:  # ✅ Only one date selected
                start_timestamp = int(datetime.datetime.combine(date_range[0], datetime.time.min).timestamp() * 1000)
            else:  # ✅ Both start and end dates are selected
                start_timestamp = int(datetime.datetime.combine(date_range[0], datetime.time.min).timestamp() * 1000)
                end_timestamp = int(datetime.datetime.combine(date_range[1], datetime.time.max).timestamp() * 1000)

    # ✅ Apply Date Filtering (Only if Both Dates Exist)
    if start_timestamp and end_timestamp:
        data = [d for d in data if start_timestamp <= d['timestamp_ms'] <= end_timestamp]
    elif start_timestamp:  # ✅ If only start date is selected, show from that date onward
        data = [d for d in data if start_timestamp <= d['timestamp_ms']]
    elif end_timestamp:  # ✅ If only end date is selected, show up to that date
        data = [d for d in data if d['timestamp_ms'] <= end_timestamp]

    # ✅ Convert to DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        df["RSI Alert"] = df["rsi"].apply(lambda x: "🔥 Overbought" if x > 70 else ("💎 Oversold" if x < 30 else ""))
        df["MACD Signal"] = df.apply(lambda row: "📈 Bullish Crossover" if row["macd_line"] > row["macd_signal"] else "📉 Bearish Crossover", axis=1)

    st.dataframe(df, use_container_width=True)
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
"""
data_loader.py
🔹 Fetches trading pairs & candle data from Supabase with caching & fixes.
"""

import streamlit as st
from supabase import create_client, Client
import os
import datetime
from dotenv import load_dotenv
from config.config_loader import load_config

# ✅ Load environment variables from credentials.env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "credentials.env"))
load_dotenv(env_path)

# ✅ Load config (ensures SUPABASE_URL comes from config.ini)
config = load_config()

SUPABASE_URL = config.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")  # Loaded from environment

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("❌ Supabase credentials missing. Check config.ini and credentials.env.")

# ✅ Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

@st.cache_data(ttl=3600)
def fetch_trading_pairs():
    """
    Fetches all unique trading pairs from the `distinct_trading_pairs` materialized view.
    Ensures all pairs are retrieved and sorted alphabetically.
    """
    response = supabase.table("distinct_trading_pairs").select("pair").execute()

    if not response.data:
        return ["BTC-USDT"]  # Default fallback if query fails

    pairs = sorted([item["pair"] for item in response.data])
    return pairs


@st.cache_data(ttl=3600)  # Cache market data for 1 hour
def fetch_market_data(pair: str, timeframe: str = "1H", limit: int = 1000):
    """
    Fetches recent market data (candles + indicators) from Supabase.
    
    - ⏳ Caches results for 1 hour (to avoid excessive queries).
    - 📅 Fetches 1H data every hour, 1D data once per day.

    :param pair: Trading pair, e.g., "BTC-USDT".
    :param timeframe: "1H" or "1D".
    :param limit: Number of most recent rows to fetch.
    :return: List of market data rows.
    """
    table_name = f"candles_{timeframe}"

    # Query Supabase
    response = supabase.table(table_name) \
        .select("pair", "timestamp_ms", "open", "high", "low", "close", "volume",
                "rsi", "macd_line", "macd_signal", "macd_hist",
                "bollinger_middle", "bollinger_upper", "bollinger_lower",
                "atr", "stoch_rsi_k", "stoch_rsi_d") \
        .eq("pair", pair) \
        .order("timestamp_ms", desc=True) \
        .limit(limit).execute()

    return response.data if response.data else []

# ✅ Auto-fetch on dashboard start
def auto_fetch_data():
    """Fetches 1H data every hour & 1D data once per day."""
    now = datetime.datetime.utcnow()
    last_fetch_time = st.session_state.get("last_fetch_time", None)

    if not last_fetch_time or (now - last_fetch_time).seconds >= 3600:
        st.session_state["last_fetch_time"] = now
        st.session_state["1H_data"] = fetch_market_data("BTC-USDT", "1H", 1000)
        st.session_state["1D_data"] = fetch_market_data("BTC-USDT", "1D", 365)  # Fetch last 365 days
        print("✅ Market data updated")

# ✅ Auto-run on Streamlit start
auto_fetch_data()

```

## `dashboard\utils\session_manager.py`

```python
"""
session_manager.py
🎯 Manages user session state: selected pair, timeframe, & caching.
"""

import streamlit as st

def initialize_session_state():
    """Ensures session variables exist to track user settings."""
    if "selected_pair" not in st.session_state:
        st.session_state["selected_pair"] = "BTC-USDT"  # Default trading pair
    if "timeframe" not in st.session_state:
        st.session_state["timeframe"] = "1H"  # Default timeframe

def get_selected_pair():
    """Returns the currently selected trading pair."""
    return st.session_state["selected_pair"]

def get_timeframe():
    """Returns the currently selected timeframe."""
    return st.session_state["timeframe"]

def update_selected_pair(pair: str):
    """Updates the selected trading pair."""
    st.session_state["selected_pair"] = pair

def update_timeframe(timeframe: str):
    """Updates the selected timeframe."""
    st.session_state["timeframe"] = timeframe

# Ensure session state is initialized
initialize_session_state()

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

## `supabase\functions\backfill_missing_1h_candles.py`

```python
import requests
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# ✅ Load .env if running locally
if not os.getenv("GITHUB_ACTIONS"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

# ✅ Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# ✅ OKX API
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit (20 requests per 2s)
CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

# ✅ Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ✅ Fetch Missing Pairs from Supabase using the correct function
def fetch_missing_pairs():
    """Fetch pairs that exist in 1D but are missing from 1H using Supabase function."""
    response = supabase.rpc("find_missing_1h_pairs").execute()
    
    if response.data:
        missing_pairs = [row["pair"] for row in response.data]
        print(f"✅ Found {len(missing_pairs)} missing 1H pairs.")
        return missing_pairs
    else:
        print("✅ No missing pairs found.")
        return []

# ✅ Fetch Latest Timestamp in Supabase
def fetch_latest_supabase_timestamp(pair):
    """Fetch the most recent timestamp stored for a given pair."""
    response = (
        supabase.table("candles_1H")
        .select("timestamp_ms")
        .eq("pair", pair)
        .order("timestamp_ms", desc=True)
        .limit(1)
        .execute()
    )
    return int(response.data[0]["timestamp_ms"]) if response.data else None

# ✅ Fetch Candles from OKX
def fetch_candles(pair, after_timestamp=None):
    """Fetch 1H historical candles from OKX using `after` to paginate properly."""
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
    }
    if after_timestamp:
        params["after"] = str(after_timestamp)  # ✅ Use `after` for correct pagination

    print(f"📡 Sending request to OKX: {params}")  # Debugging output

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON for {pair}: {e}")
        return None


# ✅ Insert Candles into Supabase
def insert_candles(pair, candles):
    """Insert fetched candles into Supabase."""
    rows = [
        {
            "timestamp_ms": int(c[0]),
            "pair": pair,
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "quote_volume": float(c[6]),
            "taker_buy_base": float(c[7]),
            "taker_buy_quote": float(c[8]),
        }
        for c in candles
    ]

    if not rows:
        return 0

    response = supabase.table("candles_1H").upsert(rows, on_conflict="pair,timestamp_ms").execute()
    return len(response.data) if response.data else 0

# ✅ Enforce Rate Limit
def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

# ✅ Main Function
def main():
    print("🚀 Script started: Backfilling missing 1H candles...")

    missing_pairs = fetch_missing_pairs()
    if not missing_pairs:
        print("✅ No missing pairs found. Exiting.")
        return

    print(f"🚀 Backfilling {len(missing_pairs)} missing 1H candles...")

    total_fixed = 0
    failed_pairs = []
    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(missing_pairs, start=1):
        try:
            print(f"🔍 Fetching {pair} missing candles...")

            # ✅ Start from the latest known candle in Supabase
            latest_supabase_timestamp = fetch_latest_supabase_timestamp(pair)

            # ✅ If no data in Supabase, start from latest OKX candle
            if latest_supabase_timestamp is None:
                first_candle = fetch_candles(pair, after_timestamp=None)  # Get the latest candle
                if not first_candle:
                    print(f"⚠️ {pair}: No candles found in OKX.")
                    continue
                latest_supabase_timestamp = int(first_candle[0][0])

            # ✅ Backfill candles using `before=<timestamp>`
            while True:
                candles = fetch_candles(pair, after_timestamp=latest_supabase_timestamp)
                
                if not candles:
                    print(f"⏳ No more missing candles found for {pair}, stopping.")
                    break

                inserted = insert_candles(pair, candles)
                total_fixed += inserted

                # ✅ Fix: Use the earliest candle timestamp instead of the latest
                latest_supabase_timestamp = int(candles[-1][0])  # ✅ Use oldest timestamp in batch for proper backfilling

                print(f"📌 {pair} → Inserted {inserted} missing candles. Now fetching before {latest_supabase_timestamp}...")

                request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_CANDLES_URL], start_time)

            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(missing_pairs)} | Fixed: {total_fixed}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(missing_pairs)}, Fixed={total_fixed}, Failed={len(failed_pairs)}")

if __name__ == "__main__":
    main()

```

## `supabase\functions\fetch_new_1d_candles.py`

```python
import requests
import os
import time
import smtplib
import ssl
from dateutil import parser
from email.message import EmailMessage
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load .env only if running locally (not in GitHub Actions)
if not os.getenv("GITHUB_ACTIONS"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# OKX API URLs
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"  # ✅ New, faster endpoint

# Rate Limit Settings (40 requests per 2s)
CANDLES_RATE_LIMIT = 40
BATCH_INTERVAL = 2

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_active_pairs():
    """Fetch active trading pairs with USDT or USDC."""
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [
        inst["instId"]
        for inst in data.get("data", [])
        if inst["quoteCcy"] in {"USDT"} and inst["state"] == "live"
    ]


def fetch_latest_timestamp(pair):
    """Fetch the latest timestamp for a given pair."""
    response = supabase.table("candles_1D").select("timestamp_ms").eq("pair", pair).order("timestamp_ms", desc=True).limit(1).execute()
    return response.data[0]["timestamp_ms"] if response.data else None


def enforce_rate_limit(request_count, start_time):
    """Ensure API rate limits are respected."""
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def fetch_candles(pair, after_timestamp_ms=None):
    """Fetch new 1D candles using the OKX market API."""
    params = {"instId": pair, "bar": "1D", "limit": 100}
    if after_timestamp_ms:
        params["before"] = str(after_timestamp_ms)  # ✅ Use milliseconds directly

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []


def insert_candles(pair, candles):
    """Insert new candle data into Supabase and return inserted count."""
    rows = [{
        "timestamp_ms": int(c[0]),
        "pair": pair,
        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
        "close": float(c[4]), "volume": float(c[5]),
        "quote_volume": float(c[6]), "taker_buy_base": float(c[7]),
        "taker_buy_quote": float(c[8])
    } for c in candles]

    if not rows:
        return 0

    response = supabase.table("candles_1D").upsert(rows, on_conflict="pair,timestamp_ms").execute()
    return len(response.data) if response.data else 0


def send_email(subject, body):
    """Send an email notification with a report."""
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USERNAME
    msg["To"] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    failed_pairs = []

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    print(f"✅ Found {len(pairs)} active USDT pairs.")
    print(f"🚀 Fetching new 1D candles...")

    for index, pair in enumerate(pairs, start=1):
        try:
            latest_timestamp_ms = fetch_latest_timestamp(pair)
            pair_inserted = 0

            if latest_timestamp_ms:
                candles = fetch_candles(pair, after_timestamp_ms=latest_timestamp_ms)
                inserted = insert_candles(pair, candles)
                total_inserted += inserted
                pair_inserted += inserted

            # ✅ Log progress every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Inserted: {total_inserted}")

            request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_CANDLES_URL], start_time)

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Inserted={total_inserted}, Failed={len(failed_pairs)}")

    if total_inserted > 0:
        send_email("New 1D OKX Candle Sync Report", f"Processed: {len(pairs)}\nInserted: {total_inserted}\nFailed: {len(failed_pairs)}")


if __name__ == "__main__":
    main()

```

## `supabase\functions\fetch_new_1h_candles.py`

```python
import requests
import os
import time
import smtplib
import ssl
from dateutil import parser
from email.message import EmailMessage
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load .env only if running locally (not in GitHub Actions)
if not os.getenv("GITHUB_ACTIONS"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# OKX API URLs
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"  # ✅ New, faster endpoint

# Rate Limit Settings (40 requests per 2s)
CANDLES_RATE_LIMIT = 40
BATCH_INTERVAL = 2

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_active_pairs():
    """Fetch active trading pairs with USDT or USDC."""
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [
        inst["instId"]
        for inst in data.get("data", [])
        if inst["quoteCcy"] in {"USDT"} and inst["state"] == "live"
    ]


def fetch_latest_timestamp(pair):
    """Fetch the latest timestamp for a given pair."""
    response = supabase.table("candles_1H").select("timestamp_ms").eq("pair", pair).order("timestamp_ms", desc=True).limit(1).execute()
    return response.data[0]["timestamp_ms"] if response.data else None


def enforce_rate_limit(request_count, start_time):
    """Ensure API rate limits are respected."""
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def fetch_candles(pair, after_timestamp_ms=None):
    """Fetch new 1H candles using the OKX market API."""
    params = {"instId": pair, "bar": "1H", "limit": 100}
    if after_timestamp_ms:
        params["before"] = str(after_timestamp_ms)  # ✅ Use milliseconds directly

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []


def insert_candles(pair, candles):
    """Insert new candle data into Supabase and return inserted count."""
    rows = [{
        "timestamp_ms": int(c[0]),
        "pair": pair,
        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
        "close": float(c[4]), "volume": float(c[5]),
        "quote_volume": float(c[6]), "taker_buy_base": float(c[7]),
        "taker_buy_quote": float(c[8])
    } for c in candles]

    if not rows:
        return 0

    response = supabase.table("candles_1H").upsert(rows, on_conflict="pair,timestamp_ms").execute()
    return len(response.data) if response.data else 0


def send_email(subject, body):
    """Send an email notification with a report."""
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USERNAME
    msg["To"] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    failed_pairs = []

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    print(f"✅ Found {len(pairs)} active USDT pairs.")
    print(f"🚀 Fetching new 1H candles...")

    for index, pair in enumerate(pairs, start=1):
        try:
            latest_timestamp_ms = fetch_latest_timestamp(pair)
            pair_inserted = 0

            if latest_timestamp_ms:
                candles = fetch_candles(pair, after_timestamp_ms=latest_timestamp_ms)
                inserted = insert_candles(pair, candles)
                total_inserted += inserted
                pair_inserted += inserted

            # ✅ Log progress every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Inserted: {total_inserted}")

            request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_CANDLES_URL], start_time)

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Inserted={total_inserted}, Failed={len(failed_pairs)}")

    if total_inserted > 0:
        send_email("New 1H OKX Candle Sync Report", f"Processed: {len(pairs)}\nInserted: {total_inserted}\nFailed: {len(failed_pairs)}")


if __name__ == "__main__":
    main()

```

## `supabase\functions\fetch_old_1d_candles.py`

```python
import requests
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load .env only if running locally (not in GitHub Actions)
if not os.getenv("GITHUB_ACTIONS"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

# ✅ Load Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# ✅ OKX API
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit (20 requests per 2s)
HISTORY_CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

# ✅ Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ✅ Fetch Active Trading Pairs
def fetch_active_pairs():
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [inst["instId"] for inst in data.get("data", []) if inst["quoteCcy"] in {"USDT"} and inst["state"] == "live"]

# ✅ Fetch Oldest Timestamp from Supabase
def fetch_oldest_timestamp(pair):
    response = supabase.table("candles_1D").select("timestamp_ms").eq("pair", pair).order("timestamp_ms").limit(1).execute()
    return response.data[0]["timestamp_ms"] if response.data else None

# ✅ Enforce Rate Limit
def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

# ✅ Fetch Historical Candles from OKX API
def fetch_candles(pair, after_timestamp_ms):
    params = {"instId": pair, "bar": "1D", "limit": 100, "after": str(after_timestamp_ms)}
    
    print(f"🔍 Fetching {pair} older candles from {after_timestamp_ms}...")
    
    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)
    try:
        data = response.json().get("data", [])
        if not data:
            print(f"⚠️ No older candles found for {pair}")
        return data
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []

# ✅ Insert Candles into Supabase
def insert_candles(pair, candles):
    rows = [{
        "timestamp_ms": int(c[0]),
        "pair": pair,
        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
        "close": float(c[4]), "volume": float(c[5]),
        "quote_volume": float(c[6]), "taker_buy_base": float(c[7]),
        "taker_buy_quote": float(c[8])
    } for c in candles]

    if not rows:
        return 0

    response = supabase.table("candles_1D").upsert(rows, on_conflict="pair,timestamp_ms").execute()
    return len(response.data) if response.data else 0

# ✅ Send Email Notification
def send_email(subject, body):
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USERNAME
    msg["To"] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")

# ✅ Main Function
def main():
    pairs = fetch_active_pairs()
    total_fixed = 0
    failed_pairs = []

    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    print(f"✅ Found {len(pairs)} active USDT pairs.")
    print(f"🚀 Fetching historical 1D candles...")

    for index, pair in enumerate(pairs, start=1):
        try:
            oldest_timestamp_ms = fetch_oldest_timestamp(pair)
            if oldest_timestamp_ms is None:
                print(f"⚠️ No timestamp found for {pair}, skipping...")
                continue

            print(f"⏳ {pair} → Fetching candles older than {oldest_timestamp_ms}")

            # ✅ Start from the oldest available timestamp and move forward
            while True:
                candles = fetch_candles(pair, after_timestamp_ms=oldest_timestamp_ms)
                
                if not candles:
                    print(f"⏳ No more older candles found for {pair}, stopping.")
                    break

                inserted = insert_candles(pair, candles)
                total_fixed += inserted
                oldest_timestamp_ms = int(candles[-1][0])  # ✅ Move forward in time

                print(f"📌 {pair} → Inserted {inserted} older candles.")

                request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_HISTORY_CANDLES_URL], start_time)

            # ✅ Log progress every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Fixed: {total_fixed}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Fixed={total_fixed}, Failed={len(failed_pairs)}")

    if total_fixed > 0:
        send_email("Historical 1D OKX Candle Sync Report", f"Processed: {len(pairs)}\nFixed: {total_fixed}\nFailed: {len(failed_pairs)}")

if __name__ == "__main__":
    main()

```

## `supabase\functions\fetch_old_1h_candles.py`

```python
import requests
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# ✅ Load .env if running locally
if not os.getenv("GITHUB_ACTIONS"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

# ✅ Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# ✅ OKX API
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit (20 requests per 2s)
HISTORY_CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

# ✅ Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ✅ Fetch Active Trading Pairs
def fetch_active_pairs():
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [inst["instId"] for inst in data.get("data", []) if inst["quoteCcy"] in {"USDT"} and inst["state"] == "live"]

# ✅ Fetch Oldest Timestamp from Supabase
def fetch_oldest_timestamp(pair):
    """Fetch the oldest available timestamp in the database."""
    response = supabase.table("candles_1D") \
        .select("timestamp_ms") \
        .eq("pair", pair) \
        .order("timestamp_ms") \
        .limit(1) \
        .execute()
    
    if response.data:
        oldest_timestamp = response.data[0]["timestamp_ms"]
        print(f"✅ Oldest timestamp for {pair}: {oldest_timestamp}")  # 🔍 Debug logging
        return oldest_timestamp
    
    print(f"⚠️ No oldest timestamp found for {pair}, returning None")
    return None


# ✅ Enforce Rate Limit
def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

# ✅ Fetch Historical Candles from OKX API
def fetch_candles(pair, after_timestamp_ms):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
        "after": str(int(after_timestamp_ms))
    }

    print(f"🔍 Fetching {pair} older candles from {after_timestamp_ms}...")
    print(f"🕒 Checking {pair} - Fetching candles with `after`: {after_timestamp_ms}")

    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)

    try:
        data = response.json()
        print(f"📩 Full API Response for {pair}: {data}")  # Log full response
        return data.get("data", [])

    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []

# ✅ Insert Candles into Supabase
def insert_candles(pair, candles):
    rows = [{
        "timestamp_ms": int(c[0]),
        "pair": pair,
        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
        "close": float(c[4]), "volume": float(c[5]),
        "quote_volume": float(c[6]), "taker_buy_base": float(c[7]),
        "taker_buy_quote": float(c[8])
    } for c in candles]

    if not rows:
        return 0

    response = supabase.table("candles_1H").upsert(rows, on_conflict="pair,timestamp_ms").execute()
    return len(response.data) if response.data else 0

# ✅ Send Email Notification
def send_email(subject, body):
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USERNAME
    msg["To"] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")

# ✅ Main Function
def main():
    pairs = fetch_active_pairs()
    total_fixed = 0
    failed_pairs = []

    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    print(f"✅ Found {len(pairs)} active USDT pairs.")
    print(f"🚀 Fetching historical 1H candles...")

    for index, pair in enumerate(pairs, start=1):
        try:
            oldest_timestamp_ms = fetch_oldest_timestamp(pair)
            if oldest_timestamp_ms is None:
                print(f"⚠️ No timestamp found for {pair}, skipping...")
                continue

            print(f"⏳ {pair} → Fetching candles older than {oldest_timestamp_ms}")

            # ✅ Start from the oldest available timestamp and move forward
            while True:
                candles = fetch_candles(pair, after_timestamp_ms=oldest_timestamp_ms)

                if not candles:
                    print(f"⏳ No more older candles found for {pair}, stopping.")
                    break

                inserted = insert_candles(pair, candles)
                total_fixed += inserted
                oldest_timestamp_ms = int(candles[-1][0])  # ✅ Move forward in time


                print(f"📌 {pair} → Inserted {inserted} older candles.")

                request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_HISTORY_CANDLES_URL], start_time)

            # ✅ Log progress every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Fixed: {total_fixed}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Fixed={total_fixed}, Failed={len(failed_pairs)}")

    if total_fixed > 0:
        send_email("Historical 1H OKX Candle Sync Report", f"Processed: {len(pairs)}\nFixed: {total_fixed}\nFailed: {len(failed_pairs)}")

if __name__ == "__main__":
    main()

```

## `supabase\functions\__init__.py`

```python

```

