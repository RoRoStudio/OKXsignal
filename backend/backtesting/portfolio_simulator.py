#portfolio_simulator.py
#Handles capital, slippage, position tracking:

#Starting capital (STARTING_CAPITAL = 2000)

#Tracks position size and USDT balance

#Applies slippage, fees, and max position size

#Checks constraints (e.g. cooldown, daily trade limit)

#It emits structured logs like:

{
  "timestamp": "2023-01-01 12:00",
  "pair": "ETH-USDT",
  "action": "BUY",
  "price": 1324.55,
  "size_usdt": 300,
  "slippage_pct": 0.06,
  "fees": 0.1,
  "new_balance": 1700.0,
}