import requests
import os
import time
import sys  # Needed for stdout flushing
import smtplib
import ssl
from dateutil import parser
from email.message import EmailMessage
from datetime import datetime, timedelta
from supabase import create_client, Client

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# OKX API URLs
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# Rate Limit Settings
HISTORY_CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

# Define how far back to fetch if table is empty
INITIAL_HISTORICAL_LOOKBACK_DAYS = 30

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_active_pairs():
    """Fetch active trading pairs with USDT or USDC."""
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [inst["instId"] for inst in data.get("data", []) if inst["quoteCcy"] in {"USDT", "USDC"} and inst["state"] == "live"]


def fetch_timestamps(pair):
    """Fetch both the latest and oldest timestamps in a single query."""
    latest_response = supabase.table("candles_1H").select("timestamp").eq("pair", pair).order("timestamp", desc=True).limit(1).execute()
    oldest_response = supabase.table("candles_1H").select("timestamp").eq("pair", pair).order("timestamp").limit(1).execute()

    latest_timestamp = parser.isoparse(latest_response.data[0]['timestamp']) if latest_response.data else None
    oldest_timestamp = parser.isoparse(oldest_response.data[0]['timestamp']) if oldest_response.data else None

    return latest_timestamp, oldest_timestamp


def enforce_rate_limit(request_count, start_time):
    """Ensure API rate limits are respected using batch processing."""
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def fetch_candles(pair, after_timestamp=None, before_timestamp=None):
    """Fetch historical candles efficiently for the 1H timeframe."""
    params = {"instId": pair, "bar": "1H", "limit": 100}
    if after_timestamp:
        params["after"] = str(int(after_timestamp.timestamp() * 1000))
    if before_timestamp:
        params["before"] = str(int(before_timestamp.timestamp() * 1000))

    print(f"🔍 Fetching {pair} candles with: {params}", flush=True)

    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)
    try:
        data = response.json()
        print(f"✅ API Response for {pair}: {len(data.get('data', []))} candles", flush=True)  # ✅ Debugging output
        return data.get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}", flush=True)
        return []


def insert_candles(pair, candles):
    """Insert new candle data into Supabase and return the actual inserted count."""
    rows = []
    for c in candles:
        try:
            ts = datetime.utcfromtimestamp(int(c[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S')
            rows.append({
                "timestamp": ts,
                "pair": pair,
                "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
                "close": float(c[4]), "volume": float(c[5]),
                "quote_volume": float(c[6]), "taker_buy_base": float(c[7]),
                "taker_buy_quote": float(c[8])
            })
        except Exception as e:
            print(f"⚠️ Error processing candle data for {pair}: {e}", flush=True)

    if not rows:
        print(f"⚠️ No valid rows to insert for {pair}, skipping...", flush=True)
        return 0

    print(f"📌 Attempting to insert {len(rows)} rows for {pair}", flush=True)
    response = supabase.table("candles_1H").upsert(rows, on_conflict="pair,timestamp").execute()
    print(f"🔍 Supabase Insert Response for {pair}: {response}", flush=True)

    return len(response.data) if response.data else 0


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    total_fixed = 0
    failed_pairs = []
    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()
    last_log_time = time.time()

    print(f"✅ Found {len(pairs)} active pairs.", flush=True)
    print(f"🚀 Syncing {len(pairs)} trading pairs for 1H candles...", flush=True)

    for index, pair in enumerate(pairs, start=1):
        try:
            latest_timestamp, oldest_timestamp = fetch_timestamps(pair)
            print(f"⏳ {pair} → Latest: {latest_timestamp}, Oldest: {oldest_timestamp}", flush=True)

            pair_inserted, pair_fixed = 0, 0

            # 🟢 **Handle completely empty table**
            if latest_timestamp is None and oldest_timestamp is None:
                historical_start = datetime.utcnow() - timedelta(days=INITIAL_HISTORICAL_LOOKBACK_DAYS)
                print(f"🟡 {pair} → No data found. Fetching from {historical_start}...", flush=True)
                candles = fetch_candles(pair, before_timestamp=historical_start)
                inserted = insert_candles(pair, candles)
                total_inserted += inserted
                pair_inserted += inserted

            # 🔍 Fetch new candles (latest first)
            if latest_timestamp:
                candles = fetch_candles(pair, after_timestamp=latest_timestamp)
                inserted = insert_candles(pair, candles)
                total_inserted += inserted
                pair_inserted += inserted

            # 🔍 Quick backfill check (fetch 1 batch before the oldest timestamp)
            if oldest_timestamp:
                backfill_candles = fetch_candles(pair, before_timestamp=oldest_timestamp)
                fixed = insert_candles(pair, backfill_candles)
                total_fixed += fixed
                pair_fixed += fixed

            # ✅ Log every 50 pairs or 2 minutes
            if index % 50 == 0 or time.time() - last_log_time > 120:
                print(f"📊 Progress: {index}/{len(pairs)} | Inserted: {total_inserted} | Fixed: {total_fixed}", flush=True)
                last_log_time = time.time()

        except Exception as e:
            print(f"⚠️ Error with {pair}: {e}", flush=True)
            failed_pairs.append(pair)

    # ✅ Final summary
    print(f"\n✅ Sync complete: Processed={len(pairs)}, Inserted={total_inserted}, Fixed={total_fixed}, Failed={len(failed_pairs)}", flush=True)


if __name__ == "__main__":
    main()
