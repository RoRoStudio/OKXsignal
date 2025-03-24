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

**1. Purpose:**
OKXsignal is a comprehensive solution designed to automate and optimize cryptocurrency spot trading. It leverages AI models to generate trading signals, execute trades, and provide extensive backtesting capabilities.

**2. Architecture:**
- **Backend:**
  - **Backtesting:** Contains modules for simulating trading strategies, evaluating performance metrics, and logging trade activities.
  - **Live-feed:** Manages real-time data feed from the OKX exchange using WebSockets.
  - **Models:** Implements various models for signal generation and slippage adjustments.
  - **Post Model:** Includes filtering and adjustment logic for market signals and slippage.
  - **Trading:** Handles actual trading operations, including account management and trade execution.
  - **Training:** Contains scripts for training signal and slippage models using historical data.

- **Config:**
  - Centralizes configuration settings, allowing easy adjustments and environment-specific configurations.

- **Dashboard:**
  - Provides a user-friendly interface for monitoring trading activities, analyzing market trends, and configuring trading preferences.

- **Database:**
  - Manages data storage and retrieval, ensuring efficient handling of historical market data and trade logs.

- **Documentation:**
  - Includes various markdown files detailing commands, ideas for future development, and features to be trained.

**3. Flow:**
1. **Data Collection:**
   - Historical and real-time market data is fetched using scripts in the `database/fetching` directory.
2. **Signal Generation:**
   - AI models process the collected data to generate trading signals.
3. **Backtesting:**
   - Strategies are backtested using historical data to evaluate their performance.
4. **Trading Execution:**
   - Validated signals are used to execute trades in real-time through modules in the `trading` directory.
5. **User Interaction:**
   - The dashboard provides real-time updates and allows users to configure settings and monitor their portfolio.

**4. Next Steps:**
- **Complete TODOs:**
  - Address any incomplete or placeholder code throughout the repository.
- **Enhance AI Models:**
  - Improve the training and accuracy of AI models for better signal generation.
- **Documentation:**
  - Expand and refine documentation to provide clear guidance on setup, usage, and contributions.
- **Error Handling:**
  - Implement robust error handling across all modules to ensure reliability.

By following these steps, you can enhance the functionality and user experience of the OKXsignal platform, making it a leading tool for AI-powered crypto trading.
