import json
from okx_api.rest_client import OKXRestClient

client = OKXRestClient()

print('\n===== Testing OKX REST API Connection =====\n')

try:
    balance = client.get_balance()
    print('? Account Balance:', json.dumps(balance, indent=2))
except Exception as e:
    print('? Failed to fetch balance:', e)

try:
    ticker = client.get_ticker('BTC-USDT')
    print('? BTC-USDT Ticker:', json.dumps(ticker, indent=2))
except Exception as e:
    print('? Failed to fetch ticker:', e)

print('\n===== Test Completed =====')
