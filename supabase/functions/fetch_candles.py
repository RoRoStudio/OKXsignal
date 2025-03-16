import requests
import os
import time
import smtplib
import ssl
from dateutil import parser
from email.message import EmailMessage
from datetime import datetime, timedelta
from supabase import create_client, Client

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# OKX API Endpoints
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# Rate Limit Handling (Separate limits for each endpoint)
CANDLES_RATE_LIMIT = 40  # /market/candles → 40 requests per 2 seconds
HISTORY_CANDLES_RATE_LIMIT = 20  # /market/history-candles → 20 requests per 2 seconds
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
        return parser.isoparse(response.data[0]['timestamp'])  # Auto-detects ISO format
    return None


def fetch_oldest_timestamp(pair):
    """Get the oldest timestamp stored in Supabase for the given pair."""
    response = supabase.table("candles_1D") \
        .select("timestamp") \
        .eq("pair", pair) \
        .order("timestamp", desc=False) \
        .limit(1) \
        .execute()

    if response.data:
        return parser.isoparse(response.data[0]['timestamp'])  # Auto-detects ISO format
    return None

def enforce_rate_limit(endpoint, request_count, start_time):
    """Handles rate limit enforcement for different OKX endpoints."""
    rate_limit = CANDLES_RATE_LIMIT if endpoint == OKX_CANDLES_URL else HISTORY_CANDLES_RATE_LIMIT

    request_count += 1
    if request_count >= rate_limit:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    
    return request_count, start_time


def fetch_candles(pair, endpoint, before_timestamp=None, after_timestamp=None):
    """Fetch daily candles from OKX, filtering out existing ones."""
    params = {"instId": pair, "bar": "1D", "limit": 100}

    if before_timestamp:
        params["before"] = str(int(before_timestamp.timestamp() * 1000))
    if after_timestamp:
        params["after"] = str(int(after_timestamp.timestamp() * 1000))

    response = requests.get(endpoint, params=params)
    raw_candles = response.json().get("data", [])

    existing_timestamps = set(
        row["timestamp"]
        for row in supabase.table("candles_1D")
        .select("timestamp")
        .eq("pair", pair)
        .execute()
        .data
    )

    filtered_candles = []
    for candle in raw_candles:
        # Convert Unix timestamp (milliseconds) to ISO format
        timestamp_iso = datetime.utcfromtimestamp(int(candle[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S')

        if timestamp_iso not in existing_timestamps:
            filtered_candles.append(candle)

def insert_candles(pair, candles):
    """Insert candles into Supabase, handling duplicates properly."""
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

    response = supabase.table("candles_1D") \
        .upsert(rows, on_conflict="pair,timestamp") \
        .execute()

    return len(rows)


def send_email(subject, body):
    """Send an email summary after execution."""
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        print("Error: Missing email credentials. Skipping email notification.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USERNAME
    msg["To"] = EMAIL_RECIPIENT

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())
    except Exception as e:
        print(f"Error sending email: {e}")


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    total_missing_fixed = 0
    failed_pairs = []

    request_count = {OKX_CANDLES_URL: 0, OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    for pair in pairs:
        try:
            last_timestamp = fetch_latest_timestamp(pair)
            oldest_timestamp = fetch_oldest_timestamp(pair)

            # Fetch latest candles using /market/candles
            candles = fetch_candles(pair, OKX_CANDLES_URL, after_timestamp=last_timestamp)
            if candles:
                total_inserted += insert_candles(pair, candles)

            # Fetch older candles using /market/history-candles
            while oldest_timestamp:
                candles = fetch_candles(pair, OKX_HISTORY_CANDLES_URL, before_timestamp=oldest_timestamp)
                if not candles:
                    break

                total_missing_fixed += insert_candles(pair, candles)
                oldest_timestamp = datetime.strptime(candles[-1][0], "%Y-%m-%d %H:%M:%S")

                # Enforce separate rate limits
                request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(
                    OKX_HISTORY_CANDLES_URL, request_count[OKX_HISTORY_CANDLES_URL], start_time
                )

        except Exception as e:
            print(f"Failed for {pair}: {str(e)}")
            failed_pairs.append(pair)

    # Send summary email
    email_subject = "Daily OKX Candle Sync Report"
    email_body = f"Pairs Processed: {len(pairs)}\nNew Candles Inserted: {total_inserted}\nMissing Candles Fixed: {total_missing_fixed}\nFailed Pairs: {', '.join(failed_pairs) if failed_pairs else 'None'}\n"
    send_email(email_subject, email_body)

    print("Sync complete. Summary email sent.")


if __name__ == "__main__":
    main()
