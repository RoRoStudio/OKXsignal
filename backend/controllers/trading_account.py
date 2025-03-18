"""
trading_account.py
Retrieves account-related information like balances, trade fee, positions, etc.
"""

from okx_api.rest_client import OKXRestClient

def get_balance_info():
    """
    Returns account balances from /api/v5/account/balance
    """
    client = OKXRestClient()
    resp = client.get_balance()
    return resp

def get_positions_info():
    """
    Returns positions info from /api/v5/account/positions
    (Applicable if you're using margin/Futures/Swap.)
    """
    client = OKXRestClient()
    resp = client.get_positions()
    return resp

def get_account_config():
    """
    Returns the account configuration from /api/v5/account/config
    e.g. position mode, risk settings.
    """
    client = OKXRestClient()
    resp = client.get_account_config()
    return resp

def get_trade_fee(inst_type: str = "SPOT", inst_id: str = None):
    """
    Returns fee rate (maker & taker) from /api/v5/account/trade-fee
    :param inst_type: 'SPOT', 'FUTURES', 'SWAP', etc.
    :param inst_id: specific pair e.g. 'BTC-USDT' if you want a more precise fee
    """
    client = OKXRestClient()
    endpoint = "/api/v5/account/trade-fee"
    params = {
        "instType": inst_type
    }
    if inst_id:
        params["instId"] = inst_id

    resp = client._request("GET", endpoint, params=params)
    return resp
