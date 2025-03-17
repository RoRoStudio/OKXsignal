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
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"  # ✅ New, faster endpoint

# Rate Limit Settings (40 requests per 2s)
CANDLES_RATE_LIMIT = 40
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


def fetch_latest_timestamp(pair):
    """Fetch the latest timestamp for a given pair."""
    response = supabase.table("candles_1D").select("timestamp_ms").eq("pair", pair).order("timestamp_ms", desc=True).limit(1).execute()
    return response.data[0]["timestamp_ms"] if response.data else None


def enforce_rate_limit(request_count, start_time):
    """Ensure API rate limits are respected."""
    request_count += 1
    if request_count >= CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def fetch_candles(pair, after_timestamp_ms=None):
    """Fetch new 1D candles using the OKX market API."""
    params = {"instId": pair, "bar": "1D", "limit": 100}
    if after_timestamp_ms:
        params["after"] = str(after_timestamp_ms)  # ✅ Use milliseconds directly

    response = requests.get(OKX_CANDLES_URL, params=params)
    try:
        return response.json().get("data", [])
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []


def insert_candles(pair, candles):
    """Insert new candle data into Supabase and return inserted count."""
    rows = [{
        "timestamp_ms": int(c[0]),
        "pair": pair,
        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
        "close": float(c[4]), "volume": float(c[5]),
        "quote_volume": float(c[6]), "taker_buy_base": float(c[7]),
        "taker_buy_quote": float(c[8])
    } for c in candles]

    if not rows:
        return 0

    response = supabase.table("candles_1D").upsert(rows, on_conflict="pair,timestamp_ms").execute()
    return len(response.data) if response.data else 0


def send_email(subject, body):
    """Send an email notification with a report."""
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
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
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    failed_pairs = []

    request_count = {OKX_CANDLES_URL: 0}
    start_time = time.time()

    print(f"✅ Found {len(pairs)} active pairs.")
    print(f"🚀 Fetching new 1D candles...")

    for index, pair in enumerate(pairs, start=1):
        try:
            latest_timestamp_ms = fetch_latest_timestamp(pair)
            pair_inserted = 0

            if latest_timestamp_ms:
                candles = fetch_candles(pair, after_timestamp_ms=latest_timestamp_ms)
                inserted = insert_candles(pair, candles)
                total_inserted += inserted
                pair_inserted += inserted

            # ✅ Log progress every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Inserted: {total_inserted}")

            request_count[OKX_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_CANDLES_URL], start_time)

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Inserted={total_inserted}, Failed={len(failed_pairs)}")

    if total_inserted > 0:
        send_email("New 1D OKX Candle Sync Report", f"Processed: {len(pairs)}\nInserted: {total_inserted}\nFailed: {len(failed_pairs)}")


if __name__ == "__main__":
    main()
