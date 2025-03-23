#GET /api/v5/market/tickers
#Purpose: 24h stats snapshot for all pairs (volume, 24h high/low, last price, etc.)
#Backfill range: ❌ None

#Use:

#Live market health filtering (e.g. skip low-volume pairs)

#Use in post_model/market_filter.py to reduce noise during sideways/ranging markets

#➡️ Only for live pre-trade checks.