# Exported Code Files

## `(personal) commands.md`

```python
cd E:\Programming\OKXsignal
conda activate okxsignal
python backend/main.py


Restart-Service postgresql-x64-17

```

## `exported_files.md`

```python

```

## `export_markdown.py`

```python
import os

# Define root directory
ROOT_DIR = r"P:\OKXsignal"
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

## `IDEAS.md`

```python
1. Use Unnest to improve insert speeds for postgresql (with timescaledb)
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

## `__init__.py`

```python

```

## `backend\__init__.py`

```python

```

## `backend\backtesting\metrics.py`

```python

```

## `backend\backtesting\run_backtest.py`

```python
#Example Values:
#Column	Example Value
#source	"backtest"
#model_version	"v2.1.3" or "commit:fa23a7c"
#backtest_config	'{"slippage_model": "v1.2", "filter": "confidence>70%", "risk_factor": 0.95}'
#✅ When to Set These
#In run_backtest.py:

#Set source = 'backtest'

#Include model_version from git tag or file

#Include backtest_config from loaded .yaml or .json config

#In executor.py (live trading):

#Set source = 'live'

#Optionally tag model_version if you’re deploying updated models regularly
```

## `backend\backtesting\strategy_wrapper.py`

```python

```

## `backend\backtesting\__init__.py`

```python

```

## `backend\live-feed\websocket_subscriptions.py`

```python
#WebSocket Channels
#trades-all: every public trade, real-time (used in slippage detection)

#books / books-l2-tbt: full order book updates every 100ms / 10ms

#tickers: live price/volume changes

#Use:

#Ideal for live deployment, e.g. real-time model scoring + execution

#Not used for training

#➡️ Use in live_feed/ws_subscriptions.py for real-time strategy + alerts.


```

## `backend\models\signal_model.py`

```python

```

## `backend\models\slippage_model.py`

```python

```

## `backend\models\__init__.py`

```python

```

## `backend\post_model\market_filter.py`

```python
#GET /api/v5/market/tickers
#Purpose: 24h stats snapshot for all pairs (volume, 24h high/low, last price, etc.)
#Backfill range: ❌ None

#Use:

#Live market health filtering (e.g. skip low-volume pairs)

#Use in post_model/market_filter.py to reduce noise during sideways/ranging markets

#➡️ Only for live pre-trade checks.
```

## `backend\post_model\signal_filtering.py`

```python

```

## `backend\post_model\slippage_adjustment.py`

```python

```

## `backend\post_model\slippage_guard.py`

```python
#GET /api/v5/market/books-full
#Purpose: Retrieve full order book snapshot (max 5,000 bids/asks)
#Backfill range: ❌ None — only real-time usage

#Use:

#Live execution-aware logic: check liquidity before placing order

#Estimate real slippage cost in post_model/slippage_guard.py

#➡️ Use live. Don't use for training.
```

## `backend\post_model\throttle_logic.py`

```python

```

## `backend\post_model\trade_sizing.py`

```python

```

## `backend\post_model\__init__.py`

```python

```

## `backend\trading\account.py`

```python

```

## `backend\trading\executor.py`

```python

```

## `backend\trading\portfolio.py`

```python

```

## `backend\trading\recorder.py`

```python

```

## `backend\trading\__init__.py`

```python

```

## `backend\training\dataloader.py`

```python

```

## `backend\training\features.py`

```python

```

## `backend\training\signal\train_signal_model.py`

```python

```

## `backend\training\slippage\fetch_trade_history.py`

```python
# GET /api/v5/market/history-trades

#Purpose: Returns individual trade-level data (price, size, direction) for last 3 months
#Backfill range: ❗Only last 3 months
#Pagination: via tradeId (type=1) or ts (type=2)

#Use:

#Train the slippage model

#Label high-slippage conditions per candle (e.g. price impact)

#Detect microstructure patterns pre-trade

#➡️ Add to training/slippage/fetch_trade_history.py
```

## `backend\training\slippage\train_slippage_model.py`

```python

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
        # ✅ Trading configuration
        "SIMULATED_TRADING": config["OKX"].getboolean("SIMULATED_TRADING", fallback=False),

        # ✅ Default trading settings
        "DEFAULT_PAIR": config["GENERAL"]["DEFAULT_PAIR"],
        "DEFAULT_TIMEFRAME": config["GENERAL"]["DEFAULT_TIMEFRAME"],
        "ORDER_SIZE_LIMIT": config["GENERAL"].getint("ORDER_SIZE_LIMIT", fallback=5),
        "LOG_LEVEL": config["GENERAL"]["LOG_LEVEL"],

        # ✅ Database connection details
        "DB_HOST": config["DATABASE"]["DB_HOST"],
        "DB_PORT": config["DATABASE"]["DB_PORT"],
        "DB_NAME": config["DATABASE"]["DB_NAME"],
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
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(local_tz)

    # ✅ Apply Date Filtering (Handles partial selections)
    if date_range and isinstance(date_range, list):
        if len(date_range) == 1:  # ✅ Only start date selected
            start_timestamp = int(pd.Timestamp(date_range[0]).timestamp() * 1000)
            df = df[df["timestamp_utc"] >= start_timestamp]
        elif len(date_range) == 2:  # ✅ Both start & end date selected
            start_timestamp = int(pd.Timestamp(date_range[0]).timestamp() * 1000)
            end_timestamp = int(pd.Timestamp(date_range[1]).timestamp() * 1000)
            df = df[(df["timestamp_utc"] >= start_timestamp) & (df["timestamp_utc"] <= end_timestamp)]

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
        x=df["timestamp_utc"],
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
        x=df["timestamp_utc"],
        y=df["volume"],
        marker_color=VOLUME_COLOR,
        opacity=0.5,
        name="Volume"
    ))

    # 🎚️ **Bollinger Bands**
    if indicators["Bollinger Bands"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["bollinger_upper"], line=dict(color="gray", width=1, dash="dot"), name="Bollinger Upper"))
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["bollinger_middle"], line=dict(color="gray", width=1, dash="dash"), name="Bollinger Middle"))
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["bollinger_lower"], line=dict(color="gray", width=1, dash="dot"), name="Bollinger Lower"))

    # 📉 **MACD**
    if indicators["MACD"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["macd_line"], line=dict(color="blue", width=1), name="MACD Line"))
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["macd_signal"], line=dict(color="orange", width=1, dash="dot"), name="MACD Signal"))

    # 📊 **RSI**
    if indicators["RSI"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["rsi"], line=dict(color="purple", width=1), name="RSI"))

    # 📊 **ATR (Volatility Indicator)**
    if indicators["ATR"]:
        fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["atr"], line=dict(color="red", width=1), name="ATR (Volatility)"))

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
        data = [d for d in data if start_timestamp <= d['timestamp_utc'] <= end_timestamp]
    elif start_timestamp:  # ✅ If only start date is selected, show from that date onward
        data = [d for d in data if start_timestamp <= d['timestamp_utc']]
    elif end_timestamp:  # ✅ If only end date is selected, show up to that date
        data = [d for d in data if d['timestamp_utc'] <= end_timestamp]

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
        .select("pair", "timestamp_utc", "open", "high", "low", "close", "volume",
                "rsi", "macd_line", "macd_signal", "macd_hist",
                "bollinger_middle", "bollinger_upper", "bollinger_lower",
                "atr", "stoch_rsi_k", "stoch_rsi_d") \
        .eq("pair", pair) \
        .order("timestamp_utc", desc=True) \
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

## `database\db.py`

```python
import psycopg2
import os
import time
from dotenv import load_dotenv
from config.config_loader import load_config
from psycopg2.extras import execute_values
from psycopg2.extensions import adapt

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", "credentials.env"))

# Load configuration settings
config = load_config()

DB_HOST = config["DB_HOST"]
DB_PORT = config["DB_PORT"]
DB_NAME = config["DB_NAME"]
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_connection():
    """Establishes and returns a PostgreSQL database connection."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def fetch_data(query, params=None):
    """Fetches data from PostgreSQL and returns results as a list of dictionaries."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    results = []
    try:
        cursor.execute(query, params or ())
        results = cursor.fetchall()
    except Exception as e:
        print(f"❌ Database fetch error: {e}")
    finally:
        cursor.close()
        conn.close()
    return results

def execute_query(query, params=None):
    """Executes a query (INSERT, UPDATE, DELETE) on PostgreSQL."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, params or ())
        conn.commit()
    except Exception as e:
        print(f"❌ Database error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def execute_copy_update(temp_table_name, column_names, values, update_query):
    """
    Performs a high-speed COPY into a temp table and executes an UPDATE ... FROM ... join.
    """
    import io
    import time

    conn = get_connection()
    cursor = conn.cursor()

    try:
        start_all = time.time()

        print(f"🧪 Creating temporary table: {temp_table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
        create_stmt = f"""
        CREATE TEMP TABLE {temp_table_name} (
            {', '.join(f"{col} double precision" if col != 'id' else 'id bigint' for col in column_names)}
        ) ON COMMIT DROP;
        """
        cursor.execute(create_stmt)

        print("📥 Starting COPY INTO temp table...")
        output = io.StringIO()
        for row in values:
            output.write("\t".join("" if v is None else str(v) for v in row) + "\n")
        output.seek(0)

        copy_start = time.time()
        cursor.copy_from(output, temp_table_name, sep="\t", null="")
        print(f"✅ COPY completed in {time.time() - copy_start:.2f}s")

        print("🔁 Running UPDATE FROM temp table...")
        update_start = time.time()
        cursor.execute(update_query.format(temp_table=temp_table_name))
        print(f"✅ UPDATE completed in {time.time() - update_start:.2f}s")

        conn.commit()
        print(f"🎉 Total COPY + UPDATE time: {time.time() - start_all:.2f}s")

    except Exception as e:
        print(f"❌ Error during COPY+UPDATE: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
```

## `database\__init__.py`

```python

```

## `database\fetching\backfill_missing_1h_candles.py`

```python
"""
backfill_missing_1h_candles.py
Finds and fills missing 1-hour candles in PostgreSQL using OKX API.
"""

import requests
import time
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2


def fetch_all_pairs():
    query = "SELECT DISTINCT pair FROM public.candles_1h;"
    return [row["pair"] for row in fetch_data(query)]


def fetch_timestamps(pair):
    query = """
    SELECT timestamp_utc FROM public.candles_1h 
    WHERE pair = %s ORDER BY timestamp_utc ASC;
    """
    return [row["timestamp_utc"] for row in fetch_data(query, (pair,))]


def find_gaps(timestamps):
    gaps = []
    expected_delta = timedelta(hours=1)
    for i in range(1, len(timestamps)):
        current = timestamps[i]
        prev = timestamps[i - 1]
        delta = current - prev
        if delta > expected_delta:
            missing_start = prev + expected_delta
            while missing_start < current:
                gaps.append(missing_start)
                missing_start += expected_delta
    return gaps


def fetch_candles(pair, after_utc):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
        "after": str(int(after_utc.timestamp() * 1000))
    }

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error fetching {pair} after {after_utc}: {e}")
        return []


def insert_candles(pair, candles):
    query = """
    INSERT INTO public.candles_1h 
    (pair, timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h, quote_volume_1h, taker_buy_base_1h)
    VALUES %s
    ON CONFLICT (pair, timestamp_utc) DO NOTHING;
    """
    rows = []
    for c in candles:
        try:
            utc_ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc) - timedelta(hours=8)
            row = (
                pair,
                utc_ts,
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
                float(c[6]),
                float(c[7]),
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Malformed candle for {pair}: {e} | Raw: {c}")

    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} gap candles for {pair} | {rows[0][1]} → {rows[-1][1]}")
            return len(rows)
        except Exception as e:
            print(f"❌ Insert failed for {pair}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    return 0


def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def main():
    print("🚀 Scanning for gaps in 1H candle history...")

    pairs = fetch_all_pairs()
    print(f"✅ Found {len(pairs)} pairs with existing data")

    total_inserted = 0
    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        try:
            print(f"\n🔁 Checking {pair}")
            timestamps = fetch_timestamps(pair)
            if len(timestamps) < 2:
                print(f"⚠️ Not enough data to find gaps for {pair}")
                continue

            gaps = find_gaps(timestamps)
            print(f"🧩 Found {len(gaps)} missing 1H timestamps for {pair}")

            for gap_start in gaps:
                candles = fetch_candles(pair, gap_start)
                inserted = insert_candles(pair, candles)
                total_inserted += inserted

                request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(
                    request_count[OKX_CANDLES_URL], start_time
                )

            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Total inserted: {total_inserted}")

        except Exception as e:
            print(f"❌ Failed to process {pair}: {e}")

    print(f"\n✅ Backfill complete: Inserted {total_inserted} missing candles across {len(pairs)} pairs")


if __name__ == "__main__":
    main()

```

## `database\fetching\fetch_new_1h_candles.py`

```python
"""
fetch_new_1h_candles.py
Fetches all historical 1-hour candles from OKX API and inserts them into PostgreSQL.
"""

import requests
import time
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

config = load_config()

OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
CANDLES_RATE_LIMIT = 40
BATCH_INTERVAL = 2

def fetch_active_pairs():
    OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    if "data" in data:
        return [
            inst["instId"]
            for inst in data["data"]
            if inst["quoteCcy"] == "USDT" and inst["state"] == "live"
        ]
    return []

def fetch_candles(pair, before_timestamp_utc=None):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100
    }
    if before_timestamp_utc:
        params["after"] = str(int(before_timestamp_utc.timestamp() * 1000))

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error fetching {pair}: {e}")
        return []

def insert_candles(pair, candles):
    query = """
    INSERT INTO public.candles_1h 
    (pair, timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h, quote_volume_1h, taker_buy_base_1h)
    VALUES %s
    ON CONFLICT (pair, timestamp_utc) DO NOTHING;
    """
    rows = []
    for c in candles:
        try:
            # OKX timestamps are in ms, HK-time-aligned → subtract 8h to convert to UTC
            utc_ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc) - timedelta(hours=8)
            row = (
                pair,
                utc_ts,
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
                float(c[6]),
                float(c[7]),
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Malformed row for {pair}: {e} | Raw: {c}")

    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} candles for {pair} | {rows[-1][1]} → {rows[0][1]}")
            return rows[-1][1]  # Return earliest timestamp in this batch
        except Exception as e:
            print(f"❌ Insert failed for {pair}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    return None

def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            print(f"⏳ Sleeping for {BATCH_INTERVAL - elapsed:.2f}s (rate limit)")
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

def main():
    print("🚀 Backfilling full 1H candle history from OKX...")

    pairs = fetch_active_pairs()
    print(f"✅ {len(pairs)} active USDT spot pairs found")

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        print(f"\n🔁 Processing {pair}")
        before = None
        total_rows = 0
        earliest = None

        while True:
            candles = fetch_candles(pair, before)
            if not candles:
                print(f"⛔ No more candles for {pair}. Total inserted: {total_rows}")
                break

            before = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
            inserted_timestamp = insert_candles(pair, candles)
            if not inserted_timestamp:
                break

            earliest = inserted_timestamp
            total_rows += len(candles)

            request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(
                request_count[OKX_CANDLES_URL], start_time
            )

        print(f"📦 Finished {pair}: {total_rows} candles (oldest = {earliest})")

if __name__ == "__main__":
    main()

```

## `database\fetching\fetch_old_1h_candles.py`

```python
"""
fetch_old_1h_candles.py
Finds and fetches older 1-hour candles from OKX API and stores them in PostgreSQL.
"""

import requests
import time
from datetime import datetime, timezone, timedelta
from config.config_loader import load_config
from database.db import fetch_data, get_connection
from psycopg2.extras import execute_values

# ✅ Load configuration settings
config = load_config()

# ✅ OKX API Endpoints
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit Settings
HISTORY_CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

def fetch_active_pairs():
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    if "data" in data:
        return [
            inst["instId"]
            for inst in data["data"]
            if inst["quoteCcy"] == "USDT" and inst["state"] == "live"
        ]
    return []

def fetch_oldest_timestamp(pair):
    query = "SELECT MIN(timestamp_utc) FROM public.candles_1h WHERE pair = %s;"
    result = fetch_data(query, (pair,))
    return result[0]["min"] if result and result[0]["min"] else None

def fetch_candles(pair, after_timestamp_utc):
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
        "after": str(int(after_timestamp_utc.timestamp() * 1000))
    }

    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []

def insert_candles(pair, candles):
    query = """
    INSERT INTO public.candles_1h 
    (pair, timestamp_utc, open_1h, high_1h, low_1h, close_1h, volume_1h, quote_volume_1h, taker_buy_base_1h)
    VALUES %s
    ON CONFLICT (pair, timestamp_utc) DO NOTHING;
    """

    rows = []
    for c in candles:
        try:
            utc_ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc) - timedelta(hours=8)  # HK → UTC
            row = (
                pair,
                utc_ts,
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
                float(c[6]),
                float(c[7])
            )
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Malformed candle for {pair}: {e} | Raw: {c}")

    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, rows)
            conn.commit()
            print(f"✅ Inserted {len(rows)} historical candles for {pair} | {rows[0][1]} → {rows[-1][1]}")
            return rows[-1][1]  # Return latest timestamp in batch
        except Exception as e:
            print(f"❌ Insert failed for {pair}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    return None

def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

def main():
    print("🚀 Fetching older 1H candles from OKX...")

    pairs = fetch_active_pairs()
    print(f"✅ {len(pairs)} active USDT spot pairs found")

    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(pairs, start=1):
        print(f"\n🔁 Processing {pair}")
        after = fetch_oldest_timestamp(pair)
        if after is None:
            print(f"⚠️ No local data for {pair}, skipping.")
            continue

        total = 0

        while True:
            candles = fetch_candles(pair, after_timestamp_utc=after)
            if not candles:
                print(f"⛔ No more older candles for {pair}")
                break

            after = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=timezone.utc)
            inserted_timestamp = insert_candles(pair, candles)
            if not inserted_timestamp:
                break

            total += len(candles)

            request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(
                request_count[OKX_HISTORY_CANDLES_URL], start_time
            )

        print(f"📦 Finished {pair}: Inserted {total} older candles")

if __name__ == "__main__":
    main()

```

## `database\fetching\__init__.py`

```python

```

## `database\processing\compute_candles.py`

```python
"""
OKXsignal - compute.py
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

# ---------------------------
# Load Configuration
# ---------------------------
CONFIG_PATH = os.path.join('P:/OKXsignal/config/config.ini')
CREDENTIALS_PATH = os.path.join('P:/OKXsignal/config/credentials.env')

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
load_dotenv(dotenv_path=CREDENTIALS_PATH)

DB = config['DATABASE']
MODE = config['GENERAL'].get('COMPUTE_MODE', 'incremental').lower()
LOG_LEVEL = config['GENERAL'].get('LOG_LEVEL', 'INFO').upper()

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

# ---------------------------
# Database Connection
# ---------------------------
def get_connection():
    return psycopg2.connect(
        host=DB['DB_HOST'],
        port=DB['DB_PORT'],
        dbname=DB['DB_NAME'],
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

# ---------------------------
# Feature Computation Logic
# ---------------------------
def compute_1h_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df['quote_volume_1h'] = df['close_1h'] * df['volume_1h']

    return df

def compute_multi_tf_features(df: pd.DataFrame, tf_label: str, rule: str) -> pd.DataFrame:
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

# ---------------------------
# Label Computation Logic
# ---------------------------
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('timestamp_utc')
    for horizon, shift in [('1h', 1), ('4h', 4), ('12h', 12), ('1d', 24), ('3d', 72), ('1w', 168), ('2w', 336)]:
        df[f'future_return_{horizon}_pct'] = (df['close_1h'].shift(-shift) - df['close_1h']) / df['close_1h']
    df['targets_computed'] = True
    return df

# ---------------------------
# Cross-Pair Intelligence
# ---------------------------
def compute_cross_pair_features(latest_df: pd.DataFrame) -> pd.DataFrame:
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

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == '__main__':
    logger.info(f"Starting compute.py in {MODE.upper()} mode")
    conn = get_connection()
    all_pairs = pd.read_sql("SELECT DISTINCT pair FROM candles_1h", conn)['pair'].tolist()
    latest_rows = []
    for pair in all_pairs:
        df = pd.read_sql("SELECT * FROM candles_1h WHERE pair = %s ORDER BY timestamp_utc DESC LIMIT 1;", conn, params=(pair,))
        if not df.empty:
            latest_rows.append(df.iloc[0])
    latest_df = pd.DataFrame(latest_rows)
    if not latest_df.empty:
        latest_df = compute_cross_pair_features(latest_df)
    conn.close()
    logger.info("Cross-pair features computed. Starting per-pair computation...")

    def process_pair(pair: str):
        try:
            conn = get_connection()
            query = "SELECT * FROM candles_1h WHERE pair = %s ORDER BY timestamp_utc DESC LIMIT 1050;"
            df = pd.read_sql(query, conn, params=(pair,))
            if df.empty:
                return
            df = compute_1h_features(df)
            df = compute_labels(df)
            df = compute_multi_tf_features(df, '4h', '4H')
            df = compute_multi_tf_features(df, '1d', '1D')
            df['features_computed'] = True
            latest_row = df.iloc[-1]
            with conn.cursor() as cur:
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
                        volatility_rank_1h = %s
                        row.get('performance_rank_btc_1h'),
                        row.get('performance_rank_eth_1h'),
                        row.get('volume_rank_1h'),
                        row.get('volatility_rank_1h'),
                        features_computed = TRUE,
                        targets_computed = TRUE
                    WHERE pair = %s AND timestamp_utc = %s;
                """

                # Ensure no critical NaNs before writing
                required_columns = [
                    'rsi_1h', 'rsi_slope_1h', 'macd_slope_1h', 'macd_hist_slope_1h', 'atr_1h',
                    'bollinger_width_1h', 'donchian_channel_width_1h', 'parabolic_sar_1h',
                    'money_flow_index_1h', 'obv_slope_1h', 'volume_change_pct_1h',
                    'estimated_slippage_1h', 'bid_ask_spread_1h',
                    'rsi_4h', 'rsi_slope_4h', 'macd_slope_4h', 'macd_hist_slope_4h', 'atr_4h',
                    'bollinger_width_4h', 'donchian_channel_width_4h', 'money_flow_index_4h',
                    'obv_slope_4h', 'volume_change_pct_4h',
                    'rsi_1d', 'rsi_slope_1d', 'macd_slope_1d', 'macd_hist_slope_1d', 'atr_1d',
                    'bollinger_width_1d', 'donchian_channel_width_1d', 'money_flow_index_1d',
                    'obv_slope_1d', 'volume_change_pct_1d'
                ]
                if latest_row[required_columns].isnull().any():
                    logger.warning(f"Skipping update for {pair} at {latest_row['timestamp_utc']} due to NaNs.")
                    return

                values = [latest_row.get(col) for col in [
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
                    'pair', 'timestamp_utc'
                ]]

                execute_batch(cur, update_query, [tuple(values)])
                conn.commit()

        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
        finally:
            conn.close()

    with ProcessPoolExecutor() as executor:
        executor.map(process_pair, all_pairs)

    logger.info("Feature + target computation completed!")

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

