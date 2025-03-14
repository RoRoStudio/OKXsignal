# Handles authentication for OKX API
"""
auth.py
Handles authentication for OKX API using HMAC SHA256 signatures.
"""

import hmac
import hashlib
import base64
import datetime
import os
from dotenv import load_dotenv

# Force-load .env file from the exact path
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "config", "credentials.env"))
if not os.path.exists(env_path):
    raise FileNotFoundError(f"❌ credentials.env file not found at {env_path}")

loaded = load_dotenv(env_path)
if not loaded:
    raise ValueError(f"❌ Failed to load environment variables from {env_path}")



API_KEY = os.getenv("OKX_API_KEY", "").strip()
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "").strip()
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "").strip()

if not API_KEY or not SECRET_KEY or not PASSPHRASE:
    raise ValueError(f"❌ Missing API credentials.\nLoaded Values:\n"
                     f"OKX_API_KEY='{API_KEY}'\nOKX_SECRET_KEY='{SECRET_KEY}'\nOKX_PASSPHRASE='{PASSPHRASE}'")

def get_timestamp() -> str:
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

def get_signature(timestamp: str, method: str, path: str, body: str = "") -> str:
    message = f"{timestamp}{method}{path}{body}"
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

def get_headers(timestamp: str, method: str, path: str, body: str = "", simulated: bool = False) -> dict:
    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": get_signature(timestamp, method, path, body),
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }
    if simulated:
        headers["x-simulated-trading"] = "1"
    return headers

