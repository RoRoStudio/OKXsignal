"""
strategy.py
A simple example strategy for OKXsignal that:
  1) Fetches candle data from Supabase (1H or 1D)
  2) Computes RSI, Bollinger Bands, and MACD
  3) Generates a naive "buy/sell/hold" recommendation

Notes:
    - This is purely illustrative. 
    - Adjust thresholds or add more logic for real production usage.
"""

from backend.controllers.data_retrieval import get_recent_candles
from backend.indicators.rsi import compute_rsi
from backend.indicators.macd import compute_macd
from backend.indicators.bollinger import compute_bollinger_bands
from config.config_loader import load_config

config = load_config()
DEFAULT_PAIR = config["DEFAULT_PAIR"]
DEFAULT_TIMEFRAME = config["DEFAULT_TIMEFRAME"]

def generate_signal(
    pair: str = "BTC-USDT", 
    timeframe: str = "1H", 
    limit: int = 100
) -> dict:
    """
    Fetches candles, applies indicators, and decides on a naive action.

    :param pair: e.g. "BTC-USDT"
    :param timeframe: "1H" or "1D"
    :param limit: how many rows of data to fetch
    :return: dict with keys { "pair", "timeframe", "action", "reason" }
             action can be "BUY", "SELL", "HOLD"
             reason is a short string explaining the logic.
    """

    # 1) Retrieve Data from Supabase
    df = get_recent_candles(pair, timeframe, limit)
    if df.empty or len(df) < 20:
        return {
            "pair": pair,
            "timeframe": timeframe,
            "action": "HOLD",
            "reason": "Insufficient candle data."
        }

    # 2) Compute Indicators
    df_ind = df.copy()

    # RSI
    df_ind = compute_rsi(df_ind, period=14, col_name="RSI")
    # MACD
    df_ind = compute_macd(df_ind, fast=12, slow=26, signal=9, col_prefix="MACD")
    # Bollinger
    df_ind = compute_bollinger_bands(df_ind, period=20, std_multiplier=2.0, col_prefix="BB")

    # 3) Check latest row
    last_row = df_ind.iloc[-1]
    rsi_val = last_row["RSI"]
    macd_line = last_row["MACD_Line"]
    macd_signal = last_row["MACD_Signal"]
    close_price = last_row["close"]

    # 4) Some naive logic
    # If RSI < 30 and MACD_Line > MACD_Signal => "BUY"
    if rsi_val < 30 and macd_line > macd_signal:
        action = "BUY"
        reason = f"RSI {rsi_val:.2f} < 30 and MACD crossing up."
    # If RSI > 70 and MACD_Line < MACD_Signal => "SELL"
    elif rsi_val > 70 and macd_line < macd_signal:
        action = "SELL"
        reason = f"RSI {rsi_val:.2f} > 70 and MACD crossing down."
    else:
        action = "HOLD"
        reason = "No strong signal from RSI/MACD."

    return {
        "pair": pair,
        "timeframe": timeframe,
        "action": action,
        "reason": reason,
        "latest_close": close_price
    }

if __name__ == "__main__":
    # Example usage
    signal_info = generate_signal("BTC-USDT", "1H", 100)
    print(signal_info)
