import requests
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# ✅ Load .env if running locally
if not os.getenv("GITHUB_ACTIONS"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

# ✅ Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# ✅ OKX API
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit (20 requests per 2s)
CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

# ✅ Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ✅ Fetch Missing Pairs from Supabase using the correct function
def fetch_missing_pairs():
    """Fetch pairs that exist in 1D but are missing from 1H using Supabase function."""
    response = supabase.rpc("find_missing_1h_pairs").execute()
    
    if response.data:
        missing_pairs = [row["pair"] for row in response.data]
        print(f"✅ Found {len(missing_pairs)} missing 1H pairs.")
        return missing_pairs
    else:
        print("✅ No missing pairs found.")
        return []

# ✅ Fetch Latest Timestamp in Supabase
def fetch_latest_supabase_timestamp(pair):
    """Fetch the most recent timestamp stored for a given pair."""
    response = (
        supabase.table("candles_1H")
        .select("timestamp_ms")
        .eq("pair", pair)
        .order("timestamp_ms", desc=True)
        .limit(1)
        .execute()
    )
    return int(response.data[0]["timestamp_ms"]) if response.data else None

# ✅ Fetch Candles from OKX
def fetch_candles(pair, after_timestamp=None):
    """Fetch 1H historical candles from OKX using `after` to paginate properly."""
    params = {
        "instId": pair,
        "bar": "1H",
        "limit": 100,
    }
    if after_timestamp:
        params["after"] = str(after_timestamp)  # ✅ Use `after` for correct pagination

    print(f"📡 Sending request to OKX: {params}")  # Debugging output

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON for {pair}: {e}")
        return None


# ✅ Insert Candles into Supabase
def insert_candles(pair, candles):
    """Insert fetched candles into Supabase."""
    rows = [
        {
            "timestamp_ms": int(c[0]),
            "pair": pair,
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "quote_volume": float(c[6]),
            "taker_buy_base": float(c[7]),
            "taker_buy_quote": float(c[8]),
        }
        for c in candles
    ]

    if not rows:
        return 0

    response = supabase.table("candles_1H").upsert(rows, on_conflict="pair,timestamp_ms").execute()
    return len(response.data) if response.data else 0

# ✅ Enforce Rate Limit
def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

# ✅ Main Function
def main():
    print("🚀 Script started: Backfilling missing 1H candles...")

    missing_pairs = fetch_missing_pairs()
    if not missing_pairs:
        print("✅ No missing pairs found. Exiting.")
        return

    print(f"🚀 Backfilling {len(missing_pairs)} missing 1H candles...")

    total_fixed = 0
    failed_pairs = []
    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    for index, pair in enumerate(missing_pairs, start=1):
        try:
            print(f"🔍 Fetching {pair} missing candles...")

            # ✅ Start from the latest known candle in Supabase
            latest_supabase_timestamp = fetch_latest_supabase_timestamp(pair)

            # ✅ If no data in Supabase, start from latest OKX candle
            if latest_supabase_timestamp is None:
                first_candle = fetch_candles(pair, after_timestamp=None)  # Get the latest candle
                if not first_candle:
                    print(f"⚠️ {pair}: No candles found in OKX.")
                    continue
                latest_supabase_timestamp = int(first_candle[0][0])

            # ✅ Backfill candles using `before=<timestamp>`
            while True:
                candles = fetch_candles(pair, after_timestamp=latest_supabase_timestamp)
                
                if not candles:
                    print(f"⏳ No more missing candles found for {pair}, stopping.")
                    break

                inserted = insert_candles(pair, candles)
                total_fixed += inserted

                # ✅ Fix: Use the earliest candle timestamp instead of the latest
                latest_supabase_timestamp = int(candles[-1][0])  # ✅ Use oldest timestamp in batch for proper backfilling

                print(f"📌 {pair} → Inserted {inserted} missing candles. Now fetching before {latest_supabase_timestamp}...")

                request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_CANDLES_URL], start_time)

            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(missing_pairs)} | Fixed: {total_fixed}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(missing_pairs)}, Fixed={total_fixed}, Failed={len(failed_pairs)}")

if __name__ == "__main__":
    main()
