# Exported Code Files

## 📂 frontend

## 📂 backend\utils

### `helpers.py`

```python
# Miscellaneous helper functions

```

### `logger.py`

```python
# Logging functionality

```

## 📂 backend

### `test_rest_client.py`

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

### `__init__.py`

```python

```

## 📂 backend\signal_engine

### `strategy.py`

```python
# Defines strategy for Buy/Sell/Hold recommendations

```

## 📂 backend\indicators

### `bollinger.py`

```python
# Computes Bollinger Bands indicator

```

### `macd.py`

```python
# Computes MACD indicator

```

### `rsi.py`

```python
# Computes RSI indicator

```

## 📂 backend\api

### `routes.py`

```python
# Defines API routes (e.g., GET /signals, GET /portfolio)

```

### `server.py`

```python
# Runs FastAPI/Flask backend server

```

## 📂 frontend\dashboards

### `okxsignal.json`

```python
﻿{
    "dashboard": {
      "id": null,
      "title": "OKXsignal Trading Dashboard",
      "timezone": "browser",
      "version": 1,
      "refresh": "10s",
      "panels": [
        {
          "title": "Fetch Market Data",
          "type": "text",
          "content": "[![Fetch Market Data](https://img.shields.io/badge/Fetch%20Market%20Data-blue)](http://localhost:8000/fetch_market)",
          "gridPos": { "x": 0, "y": 0, "w": 4, "h": 2 }
        },
        {
          "title": "Process Candles",
          "type": "text",
          "content": "[![Process Candles](https://img.shields.io/badge/Process%20Candles-orange)](http://localhost:8000/process_candles)",
          "gridPos": { "x": 4, "y": 0, "w": 4, "h": 2 }
        },
        {
          "title": "Analyze Signals",
          "type": "text",
          "content": "[![Analyze Signals](https://img.shields.io/badge/Analyze%20Signals-red)](http://localhost:8000/analyze_signals)",
          "gridPos": { "x": 8, "y": 0, "w": 4, "h": 2 }
        },
        {
          "title": "Market Data Fetch Progress",
          "type": "logs",
          "targets": [{ "expr": "{job=\"okxsignal_logs\"}", "refId": "A" }],
          "gridPos": { "x": 0, "y": 2, "w": 12, "h": 6 }
        },
        {
          "title": "Candlestick Data",
          "type": "timeseries",
          "targets": [{ "expr": "{job=\"okxsignal_candles\"}", "refId": "B" }],
          "gridPos": { "x": 0, "y": 8, "w": 12, "h": 8 }
        }
      ]
    }
  }
  
```

## 📂 backend\config

### `settings.py`

```python
# General configuration settings

```

## 📂 okx_api

### `auth.py`

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

### `rest_client.py`

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

### `__init__.py`

```python
# Initializes OKX API module

```

## 📂 frontend\loki

## 📂 backend\execution

### `api.py`

```python
﻿# api.py
# FastAPI backend for triggering Python scripts from Grafana

```

### `async_requests.py`

```python
# Handles asynchronous batch API requests

```

### `fetch_market.py`

```python
"""
fetch_market.py
Fetches historical market data (candlesticks) for all USDT spot pairs on OKX.
Designed to be triggered by Grafana.
"""

import asyncio
import time
import random
import json
import pandas as pd
import aiohttp
from tqdm.asyncio import tqdm  # Async progress bar
from okx_api.rest_client import OKXRestClient
import logging

# ✅ Loki Logging Configuration
LOG_FILE = "fetch_market.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

CANDLE_LIMIT = 100  # OKX API max per request
TOTAL_CANDLES = 1000  # Number of daily candles we want
TIMEFRAME = "1D"  # 1-day timeframe
MAX_CONCURRENT_REQUESTS = 20  # ✅ Rate-limiting

async def fetch_usdt_pairs():
    """Fetches active USDT spot trading pairs from OKX."""
    client = OKXRestClient()
    response = client.get_instruments(instType="SPOT")

    raw_pairs = [x["instId"] for x in response["data"] if x["instId"].endswith("-USDT") and x["state"] == "live"]

    # ✅ Validate pairs in batches
    valid_pairs = []
    batch_size = 10
    for i in range(0, len(raw_pairs), batch_size):
        batch = raw_pairs[i : i + batch_size]

        for pair in batch:
            test_response = client.get_ticker(pair)
            if "code" in test_response and test_response["code"] == "51001":
                logging.info(f"🛑 Skipping non-existent pair: {pair}")
            else:
                valid_pairs.append(pair)

        time.sleep(1)  # ✅ Add delay to avoid rate-limiting

    logging.info(f"✅ Found {len(valid_pairs)} valid USDT trading pairs.")
    return valid_pairs

async def fetch_candles(pair, semaphore):
    """Fetches historical candlestick data for a trading pair."""
    client = OKXRestClient()
    all_data = []

    async with semaphore:
        while len(all_data) < TOTAL_CANDLES:
            response = client.get_candlesticks(instId=pair, bar=TIMEFRAME, limit=CANDLE_LIMIT)

            if "code" in response and response["code"] == "50011":
                logging.warning(f"⚠️ Rate limited. Retrying {pair}...")
                await asyncio.sleep(random.uniform(3, 5))  
                continue  

            if "data" not in response or len(response["data"]) == 0:
                return pair, pd.DataFrame()

            all_data.extend(response["data"])
            await asyncio.sleep(random.uniform(0.2, 0.5))  # ✅ Avoid hammering API

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")

    logging.info(f"✅ {pair}: {len(df)} candles fetched.")
    return pair, df[:TOTAL_CANDLES]

async def fetch_all_pairs():
    """Fetches market data for all USDT trading pairs."""
    pairs = await fetch_usdt_pairs()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = [fetch_candles(pair, semaphore) for pair in pairs]
    results = await asyncio.gather(*tasks)

    return {pair: df for pair, df in results}

def main():
    loop = asyncio.get_event_loop()
    historical_data = loop.run_until_complete(fetch_all_pairs())

    logging.info("\n=== Data Fetch Completed ===")
    return historical_data

if __name__ == "__main__":
    main()

```

### `fetch_portfolio.py`

```python
# Fetches user portfolio balances from OKX API

```

