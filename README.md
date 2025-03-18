
```
OKXsignal
├─ (personal) commands.md
├─ backend
│  ├─ config
│  ├─ controllers
│  │  ├─ data_retrieval.py
│  │  ├─ order_execution.py
│  │  └─ trading_account.py
│  ├─ execution
│  │  ├─ api.py
│  │  ├─ async_requests.py
│  │  ├─ fetch_market.py
│  │  ├─ fetch_portfolio.py
│  │  ├─ grafana.ini
│  │  └─ run_grafana.ps1
│  ├─ indicators
│  │  ├─ atr.py
│  │  ├─ bollinger.py
│  │  ├─ macd.py
│  │  ├─ rsi.py
│  │  └─ stoch_rsi.py
│  ├─ ml
│  │  ├─ model_trainer.py
│  │  └─ predictor.py
│  ├─ requirements.txt
│  ├─ signal_engine
│  │  └─ strategy.py
│  ├─ test_rest_client.py
│  └─ __init__.py
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
├─ exported_files.md
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
   ├─ config.toml
   └─ functions
      ├─ backfill_missing_1h_candles.py
      ├─ fetch_new_1d_candles.py
      ├─ fetch_new_1h_candles.py
      ├─ fetch_old_1d_candles.py
      ├─ fetch_old_1h_candles.py
      └─ __init__.py

```