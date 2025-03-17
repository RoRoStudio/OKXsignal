import requests
import os
import time
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
    """Fetch the latest available timestamp from Supabase."""
    response = supabase.table("candles_1D") \
        .select("timestamp") \
        .eq("pair", pair) \
        .order("timestamp", desc=True) \
        .limit(1) \
        .execute()

    if response.data:
        return parser.isoparse(response.data[0]['timestamp'])
    return None


def fetch_oldest_timestamp(pair):
    """Fetch the oldest available timestamp from Supabase."""
    response = supabase.table("candles_1D") \
        .select("timestamp") \
        .eq("pair", pair) \
        .order("timestamp", desc=False) \
        .limit(1) \
        .execute()

    if response.data:
        return parser.isoparse(response.data[0]['timestamp'])
    return None


def enforce_rate_limit(request_count, start_time):
    """Ensure API rate limits are respected."""
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time


def fetch_candles(pair, after_timestamp=None):
    """
    Fetch historical candles using 'after' for correct pagination.
    """
    params = {"instId": pair, "bar": "1D", "limit": 100}
    if after_timestamp:
        params["after"] = str(int(after_timestamp.timestamp() * 1000))

    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)
    raw_candles = response.json().get("data", [])
    if not raw_candles:
        print(f"❌ No candles received from OKX for {pair}, skipping...")
        return []

    existing_timestamps = set(
        parser.isoparse(row["timestamp"]).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')
        for row in supabase.table("candles_1D")
        .select("timestamp")
        .eq("pair", pair)
        .execute()
        .data
    )

    filtered_candles = []
    for candle in raw_candles:
        timestamp_iso = datetime.utcfromtimestamp(int(candle[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        if timestamp_iso not in existing_timestamps:
            filtered_candles.append(candle)

    print(f"📌 Fetched {len(raw_candles)} candles for {pair}, after filtering: {len(filtered_candles)}")
    return filtered_candles


def insert_candles(pair, candles):
    """Insert new candle data into Supabase."""
    rows = []
    for candle in candles:
        ts, open_, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote = (
            candle + ["0"] * (9 - len(candle))
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

    if not rows:
        print(f"❌ No data to insert for {pair}, skipping...")
        return 0

    print(f"📌 Attempting to insert {len(rows)} rows for {pair}...")
    response = supabase.table("candles_1D").upsert(rows, on_conflict="pair,timestamp").execute()
    if response.data:
        print(f"✅ Successfully inserted {len(response.data)} rows for {pair}")
        return len(response.data)
    else:
        print(f"❌ Insert failed for {pair}: {response}")
        return 0


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
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_RECIPIENT, msg.as_string())
        print(f"✅ Email sent successfully to {EMAIL_RECIPIENT}")
    except smtplib.SMTPAuthenticationError:
        print("❌ SMTP Authentication Error: Check your email credentials.")
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")
    except Exception as e:
        print(f"❌ Unknown Error sending email: {e}")


def main():
    pairs = fetch_active_pairs()
    total_inserted = 0
    total_missing_fixed = 0
    failed_pairs = []

    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    print(f"🚀 Starting sync for {len(pairs)} trading pairs...")

    for index, pair in enumerate(pairs, start=1):
        try:
            last_timestamp = fetch_latest_timestamp(pair)
            oldest_timestamp = fetch_oldest_timestamp(pair)
            pair_inserted = 0
            pair_missing_fixed = 0

            # Fetch new candles (latest first)
            if last_timestamp:
                candles = fetch_candles(pair, after_timestamp=last_timestamp)
                if candles:
                    inserted = insert_candles(pair, candles)
                    total_inserted += inserted
                    pair_inserted += inserted

            # Fetch historical candles (earliest first)
            while oldest_timestamp:
                candles = fetch_candles(pair, after_timestamp=oldest_timestamp)
                if not candles or len(candles) < 100:
                    break
                inserted = insert_candles(pair, candles)
                total_missing_fixed += inserted
                pair_missing_fixed += inserted
                oldest_timestamp = datetime.utcfromtimestamp(int(candles[-1][0]) / 1000)

                request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(
                    request_count[OKX_HISTORY_CANDLES_URL], start_time
                )

            # ✅ Print progress only every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} pairs processed...")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    # 📌 Final summary logging
    print("\n✅ Sync complete.")
    print(f"🔄 Pairs Processed: {len(pairs)}")
    print(f"📈 New Candles Inserted: {total_inserted}")
    print(f"📉 Missing Candles Fixed: {total_missing_fixed}")
    print(f"❌ Failed Pairs: {len(failed_pairs)} ({', '.join(failed_pairs) if failed_pairs else 'None'})")
    
    # 📌 NEW FINAL SUMMARY LINE
    print(f"📝 Full Summary → Processed: {len(pairs)}, Inserted: {total_inserted}, Fixed: {total_missing_fixed}, Failed: {len(failed_pairs)}")

    # 📧 Email handling with explicit logging
    if total_inserted > 0 or total_missing_fixed > 0:
        email_subject = "Daily OKX Candle Sync Report"
        email_body = (
            f"Pairs Processed: {len(pairs)}\n"
            f"New Candles Inserted: {total_inserted}\n"
            f"Missing Candles Fixed: {total_missing_fixed}\n"
            f"Failed Pairs: {', '.join(failed_pairs) if failed_pairs else 'None'}\n"
        )
        send_email(email_subject, email_body)
        print(f"📧 Email sent successfully to {EMAIL_RECIPIENT}")
    else:
        print("📌 No new data inserted. Skipping email notification.")


if __name__ == "__main__":
    main()
