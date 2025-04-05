### OKXsignal Documentation

**Project Overview:**
OKXsignal is an AI-powered cryptocurrency spot trading platform designed for personal use. It aims to provide real-time trading signals, backtesting capabilities, and a user-friendly dashboard to facilitate efficient trading on the OKX exchange.

**Repository Structure:**

```
OKXsignal
├── .streamlit
│   └── config.toml
├── backend
│   ├── backtesting
│   │   ├── metrics.py
│   │   ├── portfolio_simulator.py
│   │   ├── run_backtest.py
│   │   ├── strategy_wrapper.py
│   │   ├── trade_logger.py
│   │   └── __init__.py
│   ├── live-feed
│   │   └── websocket_subscriptions.py
│   ├── models
│   │   ├── signal_model.py
│   │   ├── slippage_model.py
│   │   └── __init__.py
│   ├── post_model
│   │   ├── market_filter.py
│   │   ├── signal_filtering.py
│   │   ├── slippage_adjustment.py
│   │   ├── slippage_guard.py
│   │   ├── throttle_logic.py
│   │   ├── trade_sizing.py
│   │   └── __init__.py
│   ├── trading
│   │   ├── account.py
│   │   ├── executor.py
│   │   ├── portfolio.py
│   │   ├── recorder.py
│   │   └── __init__.py
│   ├── training
│   │   ├── dataloader.py
│   │   ├── features.py
│   │   ├── signal
│   │   │   └── train_signal_model.py
│   │   └── slippage
│   │       └── train_slippage_model.py
│   └── __init__.py
├── config
│   ├── config.ini
│   ├── config_loader.py
│   └── __init__.py
├── dashboard
│   ├── assets
│   │   └── custom.css
│   ├── components
│   │   ├── alerts.py
│   │   ├── charts
│   │   │   ├── macd_plot.py
│   │   │   ├── master_chart.py
│   │   │   └── rsi_plot.py
│   │   ├── forms
│   │   │   ├── filter_form.py
│   │   │   └── order_form.py
│   │   ├── metrics.py
│   │   └── tables
│   │       ├── candle_table.py
│   │       ├── portfolio_table.py
│   │       └── trades_table.py
│   ├── layout
│   │   ├── base_page.py
│   │   ├── navigation.py
│   │   └── theme.py
│   ├── pages
│   │   ├── home.py
│   │   ├── market_analysis
│   │   │   ├── advanced_charts.py
│   │   │   ├── overview.py
│   │   │   └── signals.py
│   │   ├── portfolio
│   │   │   ├── fees.py
│   │   │   ├── holdings.py
│   │   │   └── order_history.py
│   │   ├── settings
│   │   │   ├── risk_config.py
│   │   │   └── user_prefs.py
│   │   ├── trade_execution.py
│   │   └── __init__.py
│   └── utils
│       ├── data_loader.py
│       └── session_manager.py
├── database
│   ├── db.py
│   ├── fetching
│   │   ├── .env
│   │   ├── backfill_missing_1h_candles.py
│   │   ├── fetch_new_1h_candles.py
│   │   ├── fetch_old_1h_candles.py
│   │   ├── fetch_trade_history.py
│   │   └── __init__.py
│   ├── processing
│   │   └── compute_candles.py
│   └── __init__.py
├── documentation
│   ├── (personal) commands.md
│   ├── IDEAS.md
│   └── trainable_features.md
├── export_markdown.py
├── main.py
├── okx_api
│   ├── auth.py
│   ├── rest_client.py
│   └── __init__.py
├── README.md
├── requirements.txt
├── run_hourly_pipeline.py
└── __init__.py
```


OKXsignal – Full System Overview

Purpose

OKXsignal is an intelligent, production-grade trading research and signal execution platform built to:

Compute deep multi-timeframe, multi-asset features

Train powerful models to generate trading signals

Backtest those signals accurately with realistic constraints

Run live with execution-aware, slippage-aware decision logic

Automatically fetch, process, and store fresh data on a rolling basis



---

1. Data Ingestion Layer

Scripts

fetch_old_1h_candles.py – Full historical backfill (uses before)

backfill_missing_1h_candles.py – Detects + fills gaps per pair

fetch_new_1h_candles.py – Efficient, async hourly fetch with concurrency, rate limiting, deduplication, and precise pagination using:

Initial before fetch (newest 100)

Paginate backward using after with deduplication against known timestamp_utc

ON CONFLICT DO NOTHING on insert



PostgreSQL Table: candles_1h

Stores raw and processed OHLCV candle data for all pairs

Timestamp: timestamp_utc (TIMESTAMPTZ) for chunking & hypertables (TimescaleDB)

Primary key: (pair, timestamp_utc)

Raw columns:

open_1h, high_1h, low_1h, close_1h, volume_1h, quote_volume_1h, taker_buy_base_1h


Feature columns: 60+ technical, statistical, and rank-based features

Label columns: 10+ future return targets, max drawdowns



---

2. Feature Computation Pipeline

Script

compute_candles.py


Modes

rolling_update – every hour, updates last N rows (e.g. 400) per pair

full_backfill – re-computes every row for all pairs

incremental – append-only for new rows (optional)

Mode read from config.ini: COMPUTE_MODE = rolling_update, etc.


Features Computed

1H timeframe:

RSI, MACD slopes, ATR, Bollinger Width, Donchian, OBV slope, PSAR, MFI, volume change %, etc.


Multi-timeframe (4H, 1D):

Resampled & recomputed same features


Seasonality:

hour_of_day, day_of_week


Cross-pair intelligence:

volume_rank_1h, volatility_rank_1h, performance_rank_btc_1h, performance_rank_eth_1h



Labels Computed

Future return % over: 1h, 4h, 12h, 1d, 3d, 1w, 2w

Future max return (24h), max drawdown (12h)

Binary outcome: was_profitable_12h

Prev-change: prev_close_change_pct, prev_volume_rank


Database Write Logic

execute_copy_update() for full backfills (uses fast COPY + UPDATE)

execute_batch() for incremental updates (can be replaced later)



---

3. Orchestration

Script

run_hourly_pipeline.py

Runs:

1. fetch_new_1h_candles.py


2. compute_candles.py




Logging

Logs per-script runtime in logs/process_durations.log

Logs real-time stdout for live visibility



---

4. Modeling & Training

Folder: backend/training

Signal Model (signal/train_signal_model.py)

Uses processed feature set

Target: future_return_{1h, 4h, 1d}, configurable

Data loading via dataloader.py

Model saved and versioned


Slippage Model (slippage/train_slippage_model.py)

Uses historical trade-level data from slippage_training_data table

Predicts slippage % for live filtering/adjustment




---

5. Post-Model Logic

Folder: backend/post_model

Applies intelligent constraints after model prediction to decide:

Whether to skip or execute the trade

How much to size

Whether to sell, hold, throttle, etc.


Modules

market_filter.py – skip assets if liquidity/spread is poor

signal_filtering.py – fee-aware thresholding, slippage filtering

slippage_adjustment.py – adjusts predictions using slippage model

slippage_guard.py – warns/skips when recent slippage exceeds threshold

throttle_logic.py – drawdown-aware throttling, trend conditions

trade_sizing.py – allocates usdt_amount = capital × confidence × risk_factor



---

6. Backtesting Framework

Folder: backend/backtesting

Supports setting starting capital (e.g. 2000 USDT)

Realistic constraints:

Slippage

Spread

Position holding limits

No double buying/selling


Logs all trades, final metrics

Key files:

run_backtest.py

strategy_wrapper.py

metrics.py




---

7. Trade Execution Engine

Folder: backend/trading

account.py – mock or real balance tracking

portfolio.py – current positions, PnL

executor.py – sends trades to OKX or simulates them

recorder.py – logs all trades for backtesting + analysis



---

8. Dashboard Interface

Folder: dashboard/

Built using Streamlit

Custom charts: RSI, MACD, candle overlays

Pages:

Market overview

Strategy signals

Portfolio analytics

Settings (risk, user prefs)

Execution page (manual override)


Real-time websocket support for ticker data via websocket_subscriptions.py



---

9. Configuration Management

All configs centralized in config/config.ini

Modes, limits, defaults


Credentials in .env file

Loaded via config_loader.py



---

10. Logging

Per-script logs stored in logs/

Runtime durations: runtime_fetch.log, runtime_compute.log, process_durations.log

Async fetch log progress every 50 pairs

All major steps wrapped in try/except with clean tracebacks



---

11. Performance Engineering

Hypertables via TimescaleDB on candles_1h

Parallel feature computation (ProcessPoolExecutor)

execute_copy_update() bulk write optimization

Async + rate-limited fetching of candles

Uses Pandas, TA-Lib, NumPy, PyTorch



---

12. Future Enhancements (Planned)

[ ] Model explainability module

[ ] Continual online learning with fresh labels

[ ] Dynamic strategy switching

[ ] Confidence-weighted ensemble predictions

[ ] Full end-to-end signal trace logging
