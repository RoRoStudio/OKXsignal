"""
strategy.py
Fetches precomputed indicators from Supabase and generates a trading signal.
"""

from backend.controllers.data_retrieval import fetch_market_data

def generate_signal(pair: str = "BTC-USDT", timeframe: str = "1H", limit: int = 100) -> dict:
    """
    Fetches precomputed indicators and generates a trading signal.
    """
    df = fetch_market_data(pair, timeframe, limit)

    if df.empty or len(df) < 20:
        return {
            "pair": pair,
            "timeframe": timeframe,
            "action": "HOLD",
            "reason": "Insufficient candle data."
        }

    # Retrieve indicators from the table
    last_row = df.iloc[-1]
    rsi = last_row["rsi"]
    macd_line = last_row["macd_line"]
    macd_signal = last_row["macd_signal"]
    macd_hist = last_row["macd_hist"]
    close_price = last_row["close"]
    atr = last_row["atr"]
    stoch_rsi_k = last_row["stoch_rsi_k"]
    stoch_rsi_d = last_row["stoch_rsi_d"]
    bollinger_upper = last_row["bollinger_upper"]
    bollinger_middle = last_row["bollinger_middle"]
    bollinger_lower = last_row["bollinger_lower"]

    # Trading Logic (Updated)
    if (
        rsi < 30 and 
        macd_line > macd_signal and 
        stoch_rsi_k > 0.8 and 
        close_price < bollinger_lower
    ):
        action = "BUY"
        reason = f"RSI {rsi:.2f} < 30, MACD crossover, Stoch RSI K > 0.8, Price near lower Bollinger Band."
    elif (
        rsi > 70 and 
        macd_line < macd_signal and 
        stoch_rsi_k < 0.2 and 
        close_price > bollinger_upper
    ):
        action = "SELL"
        reason = f"RSI {rsi:.2f} > 70, MACD crossover, Stoch RSI K < 0.2, Price near upper Bollinger Band."
    else:
        action = "HOLD"
        reason = "No strong signal."

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
