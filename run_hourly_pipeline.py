"""
run_hourly_pipeline.py
Runs fetch + compute pipeline and logs individual + total durations.
This script ensures the AI doesn't trade until data is fresh.
"""

import subprocess
import time
from datetime import datetime
import os

LOG_DIR = "P:/OKXsignal/logs"
os.makedirs(LOG_DIR, exist_ok=True)
DURATION_LOG_PATH = os.path.join(LOG_DIR, "process_durations.log")

def log_duration(process_name, duration):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DURATION_LOG_PATH, "a") as f:
        f.write(f"[{now}] {process_name:<30} {duration:.2f} seconds\n")

def run_and_time(script_path):
    start = time.time()
    subprocess.run(["python", script_path], check=True)
    return time.time() - start

def main():
    print("🚀 Starting hourly pipeline...")
    
    total_start = time.time()

    try:
        fetch_duration = run_and_time("database/fetching/fetch_new_1h_candles.py")
        log_duration("fetch_new_1h_candles.py", fetch_duration)

        compute_duration = run_and_time("database/processing/compute_candles.py")
        log_duration("compute_candles.py", compute_duration)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during script execution: {e}")
        return

    total_duration = time.time() - total_start
    log_duration("total_hourly_pipeline", total_duration)
    print(f"✅ Hourly pipeline completed in {total_duration:.2f} seconds")

if __name__ == "__main__":
    main()
