import requests
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timedelta
from supabase import create_client, Client

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")  # SMTP Email
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # SMTP Password
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# OKX API Endpoints
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"

# Rate Limit Handling (OKX allows 40 requests per 2 seconds)
REQUESTS_PER_BATCH = 40
BATCH_INTERVAL = 2  # seconds

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_active_pairs():
    """Fetch all live USDT/USDC spot trading pairs."""
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [
        inst["instId"]
        for inst in data.get("data", [])
        if inst["quoteCcy"] in {"USDT", "USDC"} and inst["state"] == "live"
    ]


def fetch_latest_timestamp(pair):
    """Get the most recent timestamp stored in Supabase for the given pair."""
    response = supabase.table("candles_1D") \
        .select("timestamp") \
        .eq("pair", pair) \
        .order("timestamp", desc=True) \
        .limit(1) \
        .execute()

    if response.data:
        return datetime.strptime(response.data[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
    return None  # No existing data


def fetch_missing_timestamps(pair):
    """Detect missing candles (gaps) in Supabase and return a list of timestamps to fill."""
    existing_timestamps = supabase.table("candles_1D") \
        .select("timestamp") \
        .eq("pair", pair) \
        .order("timestamp", asc=True) \
        .execute()

    existing_timestamps = [
        datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S") for row in existing_timestamps.data
    ]

    missing_timestamps = []
    if existing_timestamps:
        start_time = existing_timestamps[0]
        end_time = existing_timestamps[-1]
        current_time = start_time

        while current_time <= end_time:
            if current_time not in existing_timestamps:
                missing_timestamps.append(current_time)
            current_time += timedelta(days=1)

    return missing_timestamps


def fetch_candles(pair, after_timestamp=None):
    """Fetch daily candles from OKX for a given pair."""
    params = {"instId": pair, "bar": "1D", "limit": "1500"}
    if after_timestamp:
        params["after"] = str(int(after_timestamp.timestamp() * 1000))

    response = requests.get(OKX_CANDLES_URL, params=params)
    return response.json().get("data", [])


def insert_candles(pair, candles):
    """Insert candles into Supabase."""
    rows = []
    for candle in candles:
        ts, open_, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote = (
            candle + ["0"] * (9 - len(candle))  # Handle missing fields
        )
        rows.append(
            {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(int(ts) / 1000)),
                "pair": pair,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
                "quote_volume": float(quote_volume),
                "taker_buy_base": float(taker_buy_base),
                "taker_buy_quote": float(taker_buy_quote),
            }
        )

    response = supabase.table("candles_1D").upsert(rows).execute()
    return len(rows)


def send_email(subject, body):
    """Send an email summary after execution."""
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USERNAME
    msg["To"] = EMAIL_RECIPIENT

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    total_missing_fixed = 0
    failed_pairs = []

    request_count = 0
    start_time = time.time()

    for pair in pairs:
        try:
            last_timestamp = fetch_latest_timestamp(pair)
            missing_timestamps = fetch_missing_timestamps(pair)

            # Fetch and insert latest candles
            if last_timestamp:
                candles = fetch_candles(pair, after_timestamp=last_timestamp)
            else:
                candles = fetch_candles(pair)

            if candles:
                inserted_count = insert_candles(pair, candles)
                total_inserted += inserted_count

            # Fetch and insert missing candles
            for missing_ts in missing_timestamps:
                candles = fetch_candles(pair, after_timestamp=missing_ts)
                if candles:
                    fixed_count = insert_candles(pair, candles)
                    total_missing_fixed += fixed_count

            request_count += 2  # Two requests per pair (latest + missing candles)

            # Ensure we respect OKX rate limits
            if request_count >= REQUESTS_PER_BATCH:
                elapsed = time.time() - start_time
                if elapsed < BATCH_INTERVAL:
                    time.sleep(BATCH_INTERVAL - elapsed)
                request_count = 0
                start_time = time.time()

        except Exception as e:
            print(f"Failed for {pair}: {str(e)}")
            failed_pairs.append(pair)

    # Send summary email
    email_subject = "Daily OKX Candle Sync Report"
    email_body = (
        f"Pairs Processed: {len(pairs)}\n"
        f"New Candles Inserted: {total_inserted}\n"
        f"Missing Candles Fixed: {total_missing_fixed}\n"
        f"Failed Pairs: {', '.join(failed_pairs) if failed_pairs else 'None'}\n"
    )

    send_email(email_subject, email_body)
    print("Sync complete. Summary email sent.")


if __name__ == "__main__":
    main()
