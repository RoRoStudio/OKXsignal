
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
│  ├─ components
│  │  ├─ charts.py
│  │  ├─ filters.py
│  │  └─ tables.py
│  ├─ pages
│  │  ├─ market_analysis.py
│  │  ├─ portfolio.py
│  │  ├─ settings.py
│  │  └─ __init__.py
│  ├─ requirements.txt
│  ├─ sidebar.py
│  └─ __init__.py
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