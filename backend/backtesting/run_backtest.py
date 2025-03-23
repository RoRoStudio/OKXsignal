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