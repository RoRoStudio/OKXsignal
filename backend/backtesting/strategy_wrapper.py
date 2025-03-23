#strategy_wrapper.py
#Supports plug-and-play strategies:

#Rule-based (e.g. momentum, MA cross)

#Model-driven (load predictions from DB or .csv)

#Enhanced logic (e.g. no trade if slippage risk is high, throttle trades)

#Each strategy returns:

{
  "timestamp": ...,
  "pair": "BTC-USDT",
  "signal": "BUY",  # or "SELL", "HOLD"
  "confidence": 0.92,
  "expected_return_1h": 0.045,
  "risk_score": 0.3,
}