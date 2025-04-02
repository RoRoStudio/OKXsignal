# Basic usage
python -m database.fetching.backfill_missing_1h_candles 

# Look back specific number of days (default is 30)
python -m database.fetching.backfill_missing_1h_candles --days=7

# Process specific pairs
python -m database.fetching.backfill_missing_1h_candles BTC-USDT,ETH-USDT

# Debug modes
python -m database.fetching.backfill_missing_1h_candles --debug          # General debug
