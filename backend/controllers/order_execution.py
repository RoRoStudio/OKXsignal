"""
order_execution.py
Handles placing, amending, and canceling orders via the OKX V5 API endpoints.
"""

from okx_api.rest_client import OKXRestClient

def place_spot_order(inst_id: str, side: str, ord_type: str, qty: float, price: float = None):
    """
    Places a SPOT order on OKX.
    :param inst_id: e.g. 'BTC-USDT'
    :param side: 'buy' or 'sell'
    :param ord_type: 'limit', 'market', 'post_only', 'fok', 'ioc', ...
    :param qty: the size of the trade, in base or quote currency depending on the ord_type
    :param price: required if it's a 'limit' or 'post_only' or 'fok' order
    """
    client = OKXRestClient()

    data = {
        "instId": inst_id,
        "tdMode": "cash",    # spot trading
        "side": side,
        "ordType": ord_type,
        "sz": str(qty),      # must be string
    }
    if price and ord_type in ["limit", "post_only", "fok", "ioc"]:
        data["px"] = str(price)

    endpoint = "/api/v5/trade/order"
    response = client._request("POST", endpoint, data=data)
    return response

def cancel_spot_order(inst_id: str, ord_id: str = None, cl_ord_id: str = None):
    """
    Cancel an open SPOT order by either 'ordId' or 'clOrdId' (client ID).
    instId is required. If both ord_id & cl_ord_id are passed, ord_id is used.
    """
    client = OKXRestClient()
    endpoint = "/api/v5/trade/cancel-order"

    data = {
        "instId": inst_id
    }
    if ord_id:
        data["ordId"] = ord_id
    elif cl_ord_id:
        data["clOrdId"] = cl_ord_id

    response = client._request("POST", endpoint, data=data)
    return response

def amend_spot_order(inst_id: str, new_qty: float = None, new_px: float = None, ord_id: str = None, cl_ord_id: str = None):
    """
    Amend a pending SPOT order. E.g., modify the price or size (unfilled portion).
    The order ID or client order ID must be specified.
    """
    client = OKXRestClient()
    endpoint = "/api/v5/trade/amend-order"

    data = {
        "instId": inst_id
    }
    if ord_id:
        data["ordId"] = ord_id
    elif cl_ord_id:
        data["clOrdId"] = cl_ord_id

    if new_qty:
        data["newSz"] = str(new_qty)
    if new_px:
        data["newPx"] = str(new_px)

    response = client._request("POST", endpoint, data=data)
    return response
