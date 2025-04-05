# 🧠 OKXsignal — Trainable Features Overview

This file documents the features and labels used for training OKXsignal's deep learning models. Features are grouped into three main categories: **primary features**, **auxiliary features**, and **labels**.

---

## ✅ Primary Features

These are the **core input features** used by the model to learn market behavior. They represent technical indicators, statistical patterns, and market structure extracted from raw candle data.

| Feature | Description |
|--------|-------------|
| `rsi_1h` | Relative Strength Index (14) on 1h close prices |
| `rsi_slope_1h` | Short-term slope of RSI over past 3 hours |
| `macd_slope_1h` | Rate of change of MACD line (trend momentum) |
| `macd_hist_slope_1h` | Rate of change of MACD histogram (trend strength) |
| `atr_1h` | Average True Range (volatility indicator) |
| `bollinger_width_1h` | Width of Bollinger Bands (volatility + squeeze detection) |
| `donchian_channel_width_1h` | Range of high/low over 20 periods (trend range) |
| `parabolic_sar_1h` | Parabolic SAR (trend reversal detection) |
| `money_flow_index_1h` | Volume-weighted RSI (detects overbought/oversold) |
| `obv_slope_1h` | OBV slope: volume–price agreement |
| `volume_change_pct_1h` | % change in volume from previous candle |
| `estimated_slippage_1h` | High–low spread (used as proxy for slippage risk) |
| `bid_ask_spread_1h` | Close – Open difference (used to estimate spread/skew) |
| `hour_of_day` | Hour of the day (0–23, helps detect hourly seasonality) |
| `day_of_week` | Weekday (0 = Monday, 6 = Sunday) |
| `rsi_4h` | RSI on 4h timeframe |
| `rsi_slope_4h` | Slope of RSI over past 3 x 4h candles |
| `macd_slope_4h` | 4h MACD line delta |
| `macd_hist_slope_4h` | 4h MACD histogram delta |
| `atr_4h` | 4h Average True Range |
| `bollinger_width_4h` | 4h Bollinger Band width |
| `donchian_channel_width_4h` | 4h Donchian Channel range |
| `money_flow_index_4h` | 4h MFI |
| `obv_slope_4h` | 4h OBV slope |
| `volume_change_pct_4h` | 4h volume % change |
| `rsi_1d` | Daily RSI |
| `rsi_slope_1d` | Daily RSI slope |
| `macd_slope_1d` | Daily MACD slope |
| `macd_hist_slope_1d` | Daily MACD histogram slope |
| `atr_1d` | Daily ATR |
| `bollinger_width_1d` | Daily Bollinger width |
| `donchian_channel_width_1d` | Daily Donchian width |
| `money_flow_index_1d` | Daily MFI |
| `obv_slope_1d` | Daily OBV slope |
| `volume_change_pct_1d` | Daily volume change percentage |
| `volume_rank_1h` | Relative volume rank compared to other USDT pairs |
| `volatility_rank_1h` | Relative ATR rank vs other pairs (cross-pair intelligence) |
| `performance_rank_btc_1h` | Return vs BTC percentile |
| `performance_rank_eth_1h` | Return vs ETH percentile |

---

## 🛠️ Auxiliary Features

These are **contextual signals** used for analysis, feature engineering, or optional model branches. They may or may not be used for training depending on experiments.

| Feature | Description |
|--------|-------------|
| `was_profitable_12h` | Binary label (1 if price after 12h > now, else 0) |
| `prev_close_change_pct` | % price change compared to previous candle |
| `prev_volume_rank` | Previous candle’s relative volume rank |
| `future_max_return_24h_pct` | Best return (high vs close) within next 24 hours |

---

## 🎯 Labels (Targets)

These are **what the model is trying to predict**. All are calculated based on future price changes. They are not used as inputs.

| Label | Description |
|-------|-------------|
| `future_return_1h_pct` | Return after 1 hour (close-to-close) |
| `future_return_4h_pct` | Return after 4 hours |
| `future_return_12h_pct` | Return after 12 hours |
| `future_return_1d_pct` | Return after 1 day |
| `future_return_3d_pct` | Return after 3 days |
| `future_return_1w_pct` | Return after 1 week |
| `future_return_2w_pct` | Return after 2 weeks |
| `future_max_drawdown_12h_pct` | Max drawdown from close over next 12h (risk label) |

---

📘 **Notes**:

- All percentage returns are normalized: `(future_close - now_close) / now_close`
- Timezone is UTC across all candles and calculations
- Daily and 4h features are backfilled from 1h candles using proper aggregation
- All cross-pair rankings are calculated at each hourly timestamp