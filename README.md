
```
OKXsignal
├─ OKXsignal.egg-info
│  ├─ PKG-INFO
│  ├─ SOURCES.txt
│  ├─ dependency_links.txt
│  └─ top_level.txt
├─ README.md
├─ __init__.py
├─ __pycache__
│  ├─ check_db_env.cpython-39.pyc
│  ├─ database_test.cpython-39.pyc
│  ├─ export_markdown.cpython-311.pyc
│  ├─ run_hourly_pipeline.cpython-311.pyc
│  ├─ test.cpython-311.pyc
│  ├─ test.cpython-39.pyc
│  └─ test_feature.cpython-39.pyc
├─ backend
│  ├─ __init__.py
│  ├─ backtesting
│  │  ├─ __init__.py
│  │  ├─ metrics.py
│  │  ├─ portfolio_simulator.py
│  │  ├─ run_backtest.py
│  │  ├─ strategy_wrapper.py
│  │  └─ trade_logger.py
│  ├─ live-feed
│  │  └─ websocket_subscriptions.py
│  ├─ models
│  │  ├─ __init__.py
│  │  ├─ signal_model.py
│  │  └─ slippage_model.py
│  ├─ post_model
│  │  ├─ __init__.py
│  │  ├─ market_filter.py
│  │  ├─ signal_filtering.py
│  │  ├─ slippage_adjustment.py
│  │  ├─ slippage_guard.py
│  │  ├─ throttle_logic.py
│  │  └─ trade_sizing.py
│  ├─ trading
│  │  ├─ __init__.py
│  │  ├─ account.py
│  │  ├─ executor.py
│  │  ├─ portfolio.py
│  │  └─ recorder.py
│  └─ training
│     ├─ dataloader.py
│     ├─ features.py
│     ├─ signal
│     │  └─ train_signal_model.py
│     └─ slippage
│        └─ train_slippage_model.py
├─ config
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-311.pyc
│  │  ├─ __init__.cpython-39.pyc
│  │  ├─ config_loader.cpython-311.pyc
│  │  └─ config_loader.cpython-39.pyc
│  ├─ config.ini
│  ├─ config_loader.py
│  └─ credentials.env
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
│  │  ├─ __init__.py
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
│  │  └─ trade_execution.py
│  └─ utils
│     ├─ data_loader.py
│     └─ session_manager.py
├─ database
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-311.pyc
│  │  ├─ __init__.cpython-39.pyc
│  │  ├─ db.cpython-311.pyc
│  │  └─ db.cpython-39.pyc
│  ├─ db.py
│  ├─ fetching
│  │  ├─ .env
│  │  ├─ __init__.py
│  │  ├─ __pycache__
│  │  │  ├─ __init__.cpython-311.pyc
│  │  │  ├─ fetch_new_1h_candles.cpython-311.pyc
│  │  │  ├─ fetch_old_1h_candles.cpython-311.pyc
│  │  │  └─ fetch_trade_history.cpython-311.pyc
│  │  ├─ backfill_missing_1h_candles.py
│  │  ├─ fetch_new_1h_candles.py
│  │  ├─ fetch_old_1h_candles.py
│  │  └─ fetch_trade_history.py
│  └─ processing
│     ├─ __init__.py
│     ├─ __pycache__
│     │  ├─ __init__.cpython-39.pyc
│     │  ├─ compute_candles.cpython-311.pyc
│     │  ├─ compute_candles.cpython-39.pyc
│     │  └─ compute_features.cpython-39.pyc
│     ├─ compute_features.py
│     └─ features
│        ├─ README.md
│        ├─ __init__.py
│        ├─ __pycache__
│        │  ├─ __init__.cpython-39.pyc
│        │  ├─ base.cpython-39.pyc
│        │  ├─ config.cpython-39.pyc
│        │  ├─ cross_pair.cpython-39.pyc
│        │  ├─ db_operations.cpython-39.pyc
│        │  ├─ db_pool.cpython-39.pyc
│        │  ├─ labels.cpython-39.pyc
│        │  ├─ momentum.cpython-39.pyc
│        │  ├─ multi_timeframe.cpython-39.pyc
│        │  ├─ pattern.cpython-39.pyc
│        │  ├─ performance_monitor.cpython-39.pyc
│        │  ├─ price_action.cpython-39.pyc
│        │  ├─ statistical.cpython-39.pyc
│        │  ├─ test_features.cpython-39.pyc
│        │  ├─ time.cpython-39.pyc
│        │  ├─ utils.cpython-39.pyc
│        │  ├─ volatility.cpython-39.pyc
│        │  └─ volume.cpython-39.pyc
│        ├─ base.py
│        ├─ config.py
│        ├─ cross_pair.py
│        ├─ db_operations.py
│        ├─ db_pool.py
│        ├─ labels.py
│        ├─ momentum.py
│        ├─ multi_timeframe.py
│        ├─ optimized
│        │  ├─ __init__.py
│        │  ├─ __pycache__
│        │  │  ├─ __init__.cpython-39.pyc
│        │  │  ├─ feature_processor.cpython-39.pyc
│        │  │  ├─ gpu_functions.cpython-39.pyc
│        │  │  └─ numba_functions.cpython-39.pyc
│        │  ├─ feature_processor.py
│        │  ├─ gpu_functions.py
│        │  └─ numba_functions.py
│        ├─ pattern.py
│        ├─ performance_monitor.py
│        ├─ price_action.py
│        ├─ statistical.py
│        ├─ time.py
│        ├─ utils.py
│        ├─ volatility.py
│        └─ volume.py
├─ database_test.py
├─ documentation
│  ├─ (personal) commands.md
│  ├─ IDEAS.md
│  ├─ documentation.md
│  └─ trainable_features.md
├─ export_markdown.py
├─ logs
│  ├─ compute_2025-03-27_134339.log
│  ├─ compute_2025-03-27_134953.log
│  ├─ compute_2025-03-27_135018.log
│  ├─ computing_durations_2025-03-27_134339.log
│  ├─ computing_durations_2025-03-27_134953.log
│  ├─ computing_durations_2025-03-27_135018.log
│  ├─ performance_report_2025-03-27_134505.txt
│  ├─ performance_report_2025-03-27_135149.txt
│  ├─ performance_summary_2025-03-27_134505.json
│  ├─ performance_summary_2025-03-27_135149.json
│  └─ runtime_stats.log
├─ main.py
├─ okx_api
│  ├─ __init__.py
│  ├─ auth.py
│  └─ rest_client.py
├─ okxsignal_backup.yml
├─ okxsignal_packages_backup.txt
├─ pyproject.toml
├─ requirements.txt
├─ run_hourly_pipeline.py
└─ setup.cfg

```