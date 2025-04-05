#GET /api/v5/market/books-full
#Purpose: Retrieve full order book snapshot (max 5,000 bids/asks)
#Backfill range: ❌ None — only real-time usage

#Use:

#Live execution-aware logic: check liquidity before placing order

#Estimate real slippage cost in post_model/slippage_guard.py

#➡️ Use live. Don't use for training.