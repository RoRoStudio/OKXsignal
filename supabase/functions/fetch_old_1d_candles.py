import requests
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load .env only if running locally (not in GitHub Actions)
if not os.getenv("GITHUB_ACTIONS"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

# ✅ Load Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = "robert@rorostudio.com"

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587

# ✅ OKX API
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# ✅ Rate Limit (20 requests per 2s)
HISTORY_CANDLES_RATE_LIMIT = 20
BATCH_INTERVAL = 2

# ✅ Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ✅ Fetch Active Trading Pairs
def fetch_active_pairs():
    response = requests.get(OKX_INSTRUMENTS_URL)
    data = response.json()
    return [inst["instId"] for inst in data.get("data", []) if inst["quoteCcy"] in {"USDT"} and inst["state"] == "live"]

# ✅ Fetch Oldest Timestamp from Supabase
def fetch_oldest_timestamp(pair):
    response = supabase.table("candles_1D").select("timestamp_ms").eq("pair", pair).order("timestamp_ms").limit(1).execute()
    return response.data[0]["timestamp_ms"] if response.data else None

# ✅ Enforce Rate Limit
def enforce_rate_limit(request_count, start_time):
    request_count += 1
    if request_count >= HISTORY_CANDLES_RATE_LIMIT:
        elapsed = time.time() - start_time
        if elapsed < BATCH_INTERVAL:
            time.sleep(BATCH_INTERVAL - elapsed)
        return 0, time.time()
    return request_count, start_time

# ✅ Fetch Historical Candles from OKX API
def fetch_candles(pair, after_timestamp_ms):
    params = {"instId": pair, "bar": "1D", "limit": 100, "after": str(after_timestamp_ms)}
    
    print(f"🔍 Fetching {pair} older candles from {after_timestamp_ms}...")
    
    response = requests.get(OKX_HISTORY_CANDLES_URL, params=params)
    try:
        data = response.json().get("data", [])
        if not data:
            print(f"⚠️ No older candles found for {pair}")
        return data
    except Exception as e:
        print(f"❌ Error parsing JSON response for {pair}: {e}")
        return []

# ✅ Insert Candles into Supabase
def insert_candles(pair, candles):
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

# ✅ Send Email Notification
def send_email(subject, body):
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

# ✅ Main Function
def main():
    pairs = fetch_active_pairs()
    total_fixed = 0
    failed_pairs = []

    request_count = {OKX_HISTORY_CANDLES_URL: 0}
    start_time = time.time()

    print(f"✅ Found {len(pairs)} active USDT pairs.")
    print(f"🚀 Fetching historical 1D candles...")

    for index, pair in enumerate(pairs, start=1):
        try:
            oldest_timestamp_ms = fetch_oldest_timestamp(pair)
            if oldest_timestamp_ms is None:
                print(f"⚠️ No timestamp found for {pair}, skipping...")
                continue

            print(f"⏳ {pair} → Fetching candles older than {oldest_timestamp_ms}")

            # ✅ Start from the oldest available timestamp and move forward
            while True:
                candles = fetch_candles(pair, after_timestamp_ms=oldest_timestamp_ms)
                
                if not candles:
                    print(f"⏳ No more older candles found for {pair}, stopping.")
                    break

                inserted = insert_candles(pair, candles)
                total_fixed += inserted
                oldest_timestamp_ms = int(candles[-1][0])  # ✅ Move forward in time

                print(f"📌 {pair} → Inserted {inserted} older candles.")

                request_count[OKX_HISTORY_CANDLES_URL], start_time = enforce_rate_limit(request_count[OKX_HISTORY_CANDLES_URL], start_time)

            # ✅ Log progress every 50 pairs
            if index % 50 == 0:
                print(f"📊 Progress: {index}/{len(pairs)} | Fixed: {total_fixed}")

        except Exception as e:
            print(f"⚠️ Error with {pair}: {str(e)}")
            failed_pairs.append(pair)

    print(f"\n✅ Sync complete: Processed={len(pairs)}, Fixed={total_fixed}, Failed={len(failed_pairs)}")

    if total_fixed > 0:
        send_email("Historical 1D OKX Candle Sync Report", f"Processed: {len(pairs)}\nFixed: {total_fixed}\nFailed: {len(failed_pairs)}")

if __name__ == "__main__":
    main()
