import requests
import os
import time
import smtplib
import ssl
from dateutil import parser
from email.message import EmailMessage
from datetime import datetime
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

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_active_pairs():
    """Fetch active trading pairs with USDT or USDC."""
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [
        inst["instId"]
        for inst in data.get("data", [])
        if inst["quoteCcy"] in {"USDT", "USDC"} and inst["state"] == "live"
    ]


def fetch_timestamps(pair):
    """Fetch both the latest and oldest timestamps in a single query."""
    latest_response = supabase.table("candles_1H") \
        .select("timestamp") \
        .eq("pair", pair) \
        .order("timestamp", desc=True) \
        .limit(1) \
        .execute()

    latest_timestamp = parser.isoparse(latest_response.data[0]['timestamp']) if latest_response.data else None

    oldest_response = supabase.table("candles_1H") \
        .select("timestamp") \
        .eq("pair", pair) \
        .order("timestamp") \
        .limit(1) \
        .execute()

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


def fetch_candles(pair, after_timestamp=None):
    """Fetch historical candles efficiently for the 1H timeframe."""
    params = {"instId": pair, "bar": "1H", "limit": 100}
    if after_timestamp:
        params["after"] = str(int(after_timestamp.timestamp() * 1000))

    print(f"🔍 Fetching {pair} candles with: {params}")  # ✅ Debugging

    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)
    print(f"✅ Raw API Response for {pair}: {response.text}")  # ✅ Debugging

    try:
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []

def insert_candles(pair, candles):
    """Insert new candle data into Supabase and return the actual inserted count."""
    rows = [{
        "timestamp": datetime.utcfromtimestamp(int(c[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S'),
        "pair": pair,
        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
        "close": float(c[4]), "volume": float(c[5]),
        "quote_volume": float(c[6]), "taker_buy_base": float(c[7]),
        "taker_buy_quote": float(c[8])
    } for c in candles]

    print(f"📌 Attempting to insert {len(rows)} rows for {pair}")  # ✅ Debugging

    if not rows:
        print(f"⚠️ No valid rows to insert for {pair}, skipping...")
        return 0

    response = supabase.table("candles_1H").upsert(rows, on_conflict="pair,timestamp").execute()
    print(f"🔍 Supabase Insert Response: {response}")  # ✅ Debugging

    return len(response.data) if response.data else 0
    
def send_email(subject, body):
    """Send an email notification with a report."""
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        print("⚠️ Error: Missing email credentials. Skipping email notification.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USERNAME
    msg["To"] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())
        print(f"📧 Email sent successfully to {EMAIL_RECIPIENT}")
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    total_missing_fixed = 0
    failed_pairs = []

    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()
    last_log_time = time.time()  # ✅ Track last log time for updates every 2 minutes

    print(f"🚀 Syncing {len(pairs)} trading pairs for 1H candles...")

    for index, pair in enumerate(pairs, start=1):
        try:
            latest_timestamp, oldest_timestamp = fetch_timestamps(pair)
            pair_inserted, pair_missing_fixed = 0, 0

            # 🔍 Fetch new candles (latest first)
            if latest_timestamp:
                print(f"🔄 Fetching new candles for {pair} after {latest_timestamp}...")
                candles = fetch_candles(pair, after_timestamp=latest_timestamp)
                
                if candles:
                    print(f"✅ API returned {len(candles)} candles for {pair}")  # 🔍 Immediate logging
                else:
                    print(f"⚠️ No new candles found for {pair}")

                inserted = insert_candles(pair, candles)
                total_inserted += inserted
                pair_inserted += inserted

                print(f"📌 Inserted {inserted} new candles for {pair}")  # 🔍 Immediate logging

            # 🔍 Fetch historical candles (earliest first)
            while oldest_timestamp:
                print(f"🔄 Fetching older candles for {pair} before {oldest_timestamp}...")
                candles = fetch_candles(pair, after_timestamp=oldest_timestamp)

                if not candles or len(candles) < 100:
                    print(f"⏳ No more historical candles found for {pair}, stopping fetch.")
                    break

                inserted = insert_candles(pair, candles)
                total_missing_fixed += inserted
                pair_missing_fixed += inserted
                oldest_timestamp = datetime.utcfromtimestamp(int(candles[-1][0]) / 1000)

                print(f"📌 Inserted {inserted} missing candles for {pair}")  # 🔍 Immediate logging

                request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(
                    request_count[OKX_HISTORY_CANDLES_URL], start_time
                )

            # ✅ Log every 10 pairs, so you can stop early if needed
            if index % 10 == 0 or time.time() - last_log_time > 120:
                print(f"📊 Progress: {index}/{len(pairs)} pairs processed... Inserted: {total_inserted}, Fixed: {total_missing_fixed}, Failed: {len(failed_pairs)}")
                last_log_time = time.time()

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Inserted={total_inserted}, Fixed={total_missing_fixed}, Failed={len(failed_pairs)}")

    if total_inserted > 0 or total_missing_fixed > 0:
        send_email("Hourly OKX Candle Sync Report", f"Processed: {len(pairs)}\nInserted: {total_inserted}\nFixed: {total_missing_fixed}\nFailed: {len(failed_pairs)}")

if __name__ == "__main__":
    main()
