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

# ----------------------

#NEW
#run_backtest.py (entry point)
#Loads historical candles (filtered by dates, pairs)

#Loads model predictions (future returns or classes)

#Passes it to the strategy

#Passes signals to the simulator

#Logs trades and returns metrics

#➡️ You'll be able to run it like:

#bash
#Kopiëren
#Bewerken
#python backend/backtesting/run_backtest.py --start 2023-01-01 --capital 2000 --pair BTC-USDT
