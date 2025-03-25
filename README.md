
```
OKXsignal
├─ .deepsource.toml
├─ .streamlit
│  └─ config.toml
├─ backend
│  ├─ backtesting
│  │  ├─ metrics.py
│  │  ├─ portfolio_simulator.py
│  │  ├─ run_backtest.py
│  │  ├─ strategy_wrapper.py
│  │  ├─ trade_logger.py
│  │  └─ __init__.py
│  ├─ live-feed
│  │  └─ websocket_subscriptions.py
│  ├─ models
│  │  ├─ signal_model.py
│  │  ├─ slippage_model.py
│  │  └─ __init__.py
│  ├─ post_model
│  │  ├─ market_filter.py
│  │  ├─ signal_filtering.py
│  │  ├─ slippage_adjustment.py
│  │  ├─ slippage_guard.py
│  │  ├─ throttle_logic.py
│  │  ├─ trade_sizing.py
│  │  └─ __init__.py
│  ├─ trading
│  │  ├─ account.py
│  │  ├─ executor.py
│  │  ├─ portfolio.py
│  │  ├─ recorder.py
│  │  └─ __init__.py
│  ├─ training
│  │  ├─ dataloader.py
│  │  ├─ features.py
│  │  ├─ signal
│  │  │  └─ train_signal_model.py
│  │  └─ slippage
│  │     └─ train_slippage_model.py
│  └─ __init__.py
├─ config
│  ├─ config.ini
│  ├─ config_loader.py
│  └─ __init__.py
├─ dashboard
│  ├─ assets
│  │  └─ custom.css
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
├─ database
│  ├─ db.py
│  ├─ fetching
│  │  ├─ .env
│  │  ├─ backfill_missing_1h_candles.py
│  │  ├─ fetch_new_1h_candles.py
│  │  ├─ fetch_old_1h_candles.py
│  │  ├─ fetch_trade_history.py
│  │  └─ __init__.py
│  ├─ processing
│  │  ├─ compute_candles.py
│  │  └─ validate_computed_candles.py
│  └─ __init__.py
├─ documentation
│  ├─ (personal) commands.md
│  ├─ documentation.md
│  ├─ IDEAS.md
│  └─ trainable_features.md
├─ exported_files.md
├─ export_markdown.py
├─ main.py
├─ okx_api
│  ├─ auth.py
│  ├─ rest_client.py
│  └─ __init__.py
├─ README.md
├─ requirements.txt
├─ run_hourly_pipeline.py
└─ __init__.py

```
```
OKXsignal
├─ .deepsource.toml
├─ .streamlit
│  └─ config.toml
├─ backend
│  ├─ backtesting
│  │  ├─ metrics.py
│  │  ├─ portfolio_simulator.py
│  │  ├─ run_backtest.py
│  │  ├─ strategy_wrapper.py
│  │  ├─ trade_logger.py
│  │  └─ __init__.py
│  ├─ live-feed
│  │  └─ websocket_subscriptions.py
│  ├─ models
│  │  ├─ signal_model.py
│  │  ├─ slippage_model.py
│  │  └─ __init__.py
│  ├─ post_model
│  │  ├─ market_filter.py
│  │  ├─ signal_filtering.py
│  │  ├─ slippage_adjustment.py
│  │  ├─ slippage_guard.py
│  │  ├─ throttle_logic.py
│  │  ├─ trade_sizing.py
│  │  └─ __init__.py
│  ├─ trading
│  │  ├─ account.py
│  │  ├─ executor.py
│  │  ├─ portfolio.py
│  │  ├─ recorder.py
│  │  └─ __init__.py
│  ├─ training
│  │  ├─ dataloader.py
│  │  ├─ features.py
│  │  ├─ signal
│  │  │  └─ train_signal_model.py
│  │  └─ slippage
│  │     └─ train_slippage_model.py
│  └─ __init__.py
├─ config
│  ├─ config.ini
│  ├─ config_loader.py
│  └─ __init__.py
├─ dashboard
│  ├─ assets
│  │  └─ custom.css
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
├─ database
│  ├─ db.py
│  ├─ fetching
│  │  ├─ .env
│  │  ├─ backfill_missing_1h_candles.py
│  │  ├─ fetch_new_1h_candles.py
│  │  ├─ fetch_old_1h_candles.py
│  │  ├─ fetch_trade_history.py
│  │  └─ __init__.py
│  ├─ processing
│  │  ├─ compute_candles.py
│  │  └─ validate_computed_candles.py
│  └─ __init__.py
├─ documentation
│  ├─ (personal) commands.md
│  ├─ documentation.md
│  ├─ IDEAS.md
│  └─ trainable_features.md
├─ exported_files.md
├─ export_markdown.py
├─ main.py
├─ okx_api
│  ├─ auth.py
│  ├─ rest_client.py
│  └─ __init__.py
├─ README.md
├─ requirements.txt
├─ run_hourly_pipeline.py
└─ __init__.py

```