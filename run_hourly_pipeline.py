"""
run_hourly_pipeline.py
Runs fetch + compute pipeline and logs individual + total durations.
This script ensures the AI doesn't trade until data is fresh.
Features:
- Retries fetching candles every 15 seconds until successful
- Captures detailed metrics about fetched candles including first candle timestamp
- Runs compute_features.py after successful fetching
- Logs detailed statistics about both processes
- Runs continuously, executing once per hour
"""
import subprocess
import time
import re
import json
from datetime import datetime, timedelta
import os
import sys

# Configuration
# Calculate root directory based on script location
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DURATION_LOG_PATH = os.path.join(LOG_DIR, "process_durations.log")
PIPELINE_LOG_PATH = os.path.join(LOG_DIR, "hourly_pipeline.log")
FETCH_RETRY_DELAY = 15  # seconds
MAX_FETCH_RETRIES = 20  # 5 minutes worth of retries

def log_message(message, log_file=None):
    """Log a message to both console and log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    print(formatted_message)
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(formatted_message + "\n")

def log_duration(process_name, duration, candles_fetched=None, errors=None, first_candle_timestamp=None):
    """Log detailed information about a process run"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Base duration log
    duration_log = f"[{now}] {process_name:<30} {duration:.2f} seconds"
    
    # Add candles information if available
    if candles_fetched is not None:
        duration_log += f" | {candles_fetched} candles fetched"
    
    # Add first candle timestamp if available
    if first_candle_timestamp:
        duration_log += f" | First candle at {first_candle_timestamp}"
    
    # Add error information if available
    if errors and len(errors) > 0:
        duration_log += f" | {len(errors)} errors"
    
    with open(DURATION_LOG_PATH, "a") as f:
        f.write(duration_log + "\n")
    
    # Log detailed error information if available
    if errors and len(errors) > 0:
        with open(PIPELINE_LOG_PATH, "a") as f:
            f.write(f"[{now}] Errors during {process_name}:\n")
            for error in errors:
                f.write(f"  - {error}\n")

def run_and_capture_output(script_path):
    """Run a Python script and capture its output"""
    start = time.time()
    
    # Build absolute path to script
    full_script_path = os.path.join(ROOT_DIR, script_path)
    log_message(f"Running: {full_script_path}", PIPELINE_LOG_PATH)
    
    # Run the script and capture output
    process = subprocess.Popen(
        ["python", full_script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT_DIR,
        env={**os.environ, "PYTHONPATH": ROOT_DIR}
    )
    
    stdout, stderr = process.communicate()
    duration = time.time() - start
    
    # Return all information
    return {
        "success": process.returncode == 0,
        "duration": duration,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": process.returncode
    }

def parse_fetch_output(output):
    """Parse the output from fetch_new_1h_candles.py to extract metrics"""
    candles_fetched = 0
    errors = []
    first_candle_timestamp = None
    
    # Extract the number of candles fetched using regex
    inserted_matches = re.findall(r"Inserted (\d+) new candles for", output)
    for match in inserted_matches:
        candles_fetched += int(match)
    
    # Try to extract timestamp of the first fetched candle
    # This regex looks for a timestamp pattern that follows "Inserted X new candles for"
    timestamp_match = re.search(r"Inserted \d+ new candles for .* \| ([\d-]+ [\d:]+)", output)
    if timestamp_match:
        first_candle_timestamp = timestamp_match.group(1)
        log_message(f"First candle timestamp detected: {first_candle_timestamp}", PIPELINE_LOG_PATH)
    
    # Check for "Fetching latest 1H candles" message to detect successful run
    # even if no new candles were inserted
    if "Fetching latest 1H candles from OKX" in output:
        # If the script ran but didn't find any new candles, consider it a success
        if candles_fetched == 0 and "pairs found" in output:
            # Consider it successful but with 0 candles fetched
            log_message("Fetch completed successfully but no new candles were found", PIPELINE_LOG_PATH)
    
    # Extract errors
    error_lines = re.findall(r"ERROR.*|Error.*|Failed.*", output)
    for line in error_lines:
        errors.append(line.strip())
    
    return candles_fetched, errors, first_candle_timestamp

def parse_compute_output(output):
    """Parse the output from compute_features.py to extract metrics"""
    pairs_processed = 0
    errors = []
    
    # Count pairs processed
    processed_matches = re.findall(r"Updated \d+/\d+ rows", output)
    pairs_processed = len(processed_matches)
    
    # Extract errors
    error_lines = re.findall(r"ERROR.*|Error.*|Failed.*|❌.*", output)
    for line in error_lines:
        errors.append(line.strip())
    
    return pairs_processed, errors

def run_fetch_with_retry():
    """Run the fetch script with retries until ALL new candles are fetched"""
    log_message("Starting fetch_new_1h_candles.py with retry logic", PIPELINE_LOG_PATH)
    first_candle_timestamp = None
    
    # We expect to fetch a candle for almost every pair (around 306)
    EXPECTED_CANDLE_COUNT = 300  # Slightly less than total pairs to allow for some variation
    
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        # Run the fetch script
        result = run_and_capture_output("database/fetching/fetch_new_1h_candles.py")
        
        # Log the raw output for debugging (only on first few attempts to avoid log spam)
        if attempt <= 3:
            log_message(f"Fetch stdout: {result['stdout'][:500]}...", PIPELINE_LOG_PATH)
            if result['stderr']:
                log_message(f"Fetch stderr: {result['stderr'][:500]}...", PIPELINE_LOG_PATH)
        
        # Parse output to determine success
        candles_fetched, errors, timestamp = parse_fetch_output(result["stdout"] + result["stderr"])
        
        # Record first candle timestamp if found
        if timestamp and not first_candle_timestamp:
            first_candle_timestamp = timestamp
        
        # Log this attempt
        log_message(f"Fetch attempt {attempt}: {candles_fetched} candles fetched, {len(errors)} errors", PIPELINE_LOG_PATH)
        
        # Three success criteria:
        # 1. We fetched approximately the expected number of candles (success!)
        # 2. We're on the last retry and got at least some candles (partial success)
        # 3. Script ran successfully but confirmed there are no new candles to fetch
        
        got_all_candles = candles_fetched >= EXPECTED_CANDLE_COUNT
        last_attempt = attempt >= MAX_FETCH_RETRIES
        got_some_candles = candles_fetched > 0
        
        if got_all_candles:
            log_message(f"✅ Success: Fetched {candles_fetched} candles (expected ~306)", PIPELINE_LOG_PATH)
            log_duration("fetch_new_1h_candles.py", result["duration"], candles_fetched, errors, first_candle_timestamp)
            return {
                "success": True,
                "duration": result["duration"],
                "candles_fetched": candles_fetched,
                "errors": errors,
                "first_candle_timestamp": first_candle_timestamp
            }
        elif last_attempt and got_some_candles:
            log_message(f"⚠️ Partial success: Fetched only {candles_fetched} of ~306 candles after {MAX_FETCH_RETRIES} attempts", PIPELINE_LOG_PATH)
            log_duration("fetch_new_1h_candles.py", result["duration"], candles_fetched, errors, first_candle_timestamp)
            return {
                "success": True,
                "duration": result["duration"],
                "candles_fetched": candles_fetched,
                "errors": errors + [f"Incomplete fetch: Only {candles_fetched}/{EXPECTED_CANDLE_COUNT} candles fetched"],
                "first_candle_timestamp": first_candle_timestamp
            }
        elif result["returncode"] == 0 and "No new data for" in (result["stdout"] + result["stderr"]) and candles_fetched == 0:
            # Script ran successfully but confirmed there are no new candles yet
            log_message(f"⏳ No new candles available yet. Retrying in {FETCH_RETRY_DELAY} seconds...", PIPELINE_LOG_PATH)
        else:
            log_message(f"⚠️ Incomplete fetch: Only {candles_fetched} of ~306 candles. Retrying in {FETCH_RETRY_DELAY} seconds...", PIPELINE_LOG_PATH)
        
        # If we didn't get enough candles, wait and retry
        if attempt < MAX_FETCH_RETRIES:
            time.sleep(FETCH_RETRY_DELAY)
        else:
            log_message("Maximum retries reached. Giving up on fetching all candles.", PIPELINE_LOG_PATH)
    
    # If we get here, all retries failed
    return {
        "success": False,
        "duration": 0,
        "candles_fetched": 0,
        "errors": ["Maximum retries reached without fetching all candles"],
        "first_candle_timestamp": None
    }

def run_compute_features():
    """Run the compute features script"""
    log_message("Starting compute_features.py", PIPELINE_LOG_PATH)
    
    # Run the compute script
    result = run_and_capture_output("database/processing/compute_features.py")
    
    # Log the raw output for debugging
    log_message(f"Compute stdout: {result['stdout'][:500]}...", PIPELINE_LOG_PATH)
    if result['stderr']:
        log_message(f"Compute stderr: {result['stderr'][:500]}...", PIPELINE_LOG_PATH)
    
    # Parse output
    pairs_processed, errors = parse_compute_output(result["stdout"] + result["stderr"])
    
    # Log summary
    log_message(f"Compute complete: {pairs_processed} pairs processed, {len(errors)} errors", PIPELINE_LOG_PATH)
    log_duration("compute_features.py", result["duration"], pairs_processed, errors)
    
    return {
        "success": result["returncode"] == 0,
        "duration": result["duration"],
        "pairs_processed": pairs_processed,
        "errors": errors
    }

def is_fresh_hour(current_time=None):
    """
    Determine if we're in a fresh hour for pipeline execution
    Returns True if we're in the first 10 minutes of an hour
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Check if we're in the first 10 minutes of the hour
    return current_time.minute < 10


def wait_for_next_execution():
    """
    Wait until the appropriate time to execute the pipeline
    Returns the current hour that will be processed
    """
    now = datetime.now()
    current_hour = now.hour
    
    # If we're in a fresh hour already, no need to wait
    if is_fresh_hour(now):
        log_message(f"Current time is within execution window for hour {current_hour}", PIPELINE_LOG_PATH)
        return current_hour
    
    # Calculate time until the next hour
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_seconds = (next_hour - now).total_seconds()
    
    # Wait until just before the next hour (we'll start 1 minute early to prepare)
    wait_until = next_hour - timedelta(minutes=1)
    log_message(f"Waiting until {wait_until.strftime('%Y-%m-%d %H:%M:%S')} for next execution window...", PIPELINE_LOG_PATH)
    
    time.sleep(max(0, wait_seconds - 60))
    return next_hour.hour

def main():
    """Main pipeline execution function"""
    log_message("🚀 Starting hourly data pipeline...", PIPELINE_LOG_PATH)
    
    total_start = time.time()
    
    # Step 1: Fetch new candles with retry logic
    fetch_result = run_fetch_with_retry()
    
    # Step 2: Compute features ONLY if we actually fetched new candles
    compute_result = {"success": False, "duration": 0, "errors": []}
    if fetch_result["success"] and fetch_result["candles_fetched"] > 0:
        log_message("New candles fetched, proceeding to compute features", PIPELINE_LOG_PATH)
        compute_result = run_compute_features()
    else:
        compute_skipped_reason = "No new candles were fetched" if fetch_result["success"] else "Fetch failed"
        log_message(f"Skipping compute features: {compute_skipped_reason}", PIPELINE_LOG_PATH)
        compute_result["errors"] = [f"Compute skipped: {compute_skipped_reason}"]
    
    # Calculate total pipeline duration
    total_duration = time.time() - total_start
    
    # Log total duration
    log_duration(
        "total_hourly_pipeline", 
        total_duration, 
        fetch_result.get("candles_fetched", 0),
        None,  # No errors at this level
        fetch_result.get("first_candle_timestamp")
    )
    
    # Log pipeline summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hour_executed": datetime.now().hour,
        "total_duration_seconds": round(total_duration, 2),
        "fetch": {
            "success": fetch_result["success"],
            "duration_seconds": round(fetch_result["duration"], 2),
            "candles_fetched": fetch_result["candles_fetched"],
            "error_count": len(fetch_result["errors"]),
            "first_candle_timestamp": fetch_result.get("first_candle_timestamp")
        },
        "compute": {
            "success": compute_result["success"],
            "duration_seconds": round(compute_result["duration"], 2),
            "pairs_processed": compute_result.get("pairs_processed", 0),
            "error_count": len(compute_result["errors"]),
            "skipped": fetch_result["candles_fetched"] == 0
        }
    }
    
    # Write JSON summary to log
    with open(os.path.join(LOG_DIR, "pipeline_summary.json"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    
    # Final status message
    overall_success = fetch_result["success"] and (compute_result["success"] or fetch_result["candles_fetched"] == 0)
    status = "✅ completed successfully" if overall_success else "❌ completed with errors"
    log_message(f"Hourly pipeline {status} in {total_duration:.2f} seconds", PIPELINE_LOG_PATH)
    
    return fetch_result["candles_fetched"] > 0

def main_loop():
    """Run the pipeline continuously, at appropriate intervals"""
    log_message("Starting continuous hourly pipeline scheduler", PIPELINE_LOG_PATH)
    log_message("Pipeline will execute during the first 10 minutes of each hour", PIPELINE_LOG_PATH)
    
    # Track which hours we've already processed
    processed_hours = set()
    
    # First run: immediately try to fetch regardless of time
    log_message("Performing initial fetch attempt on startup...", PIPELINE_LOG_PATH)
    current_hour = datetime.now().hour
    candles_fetched = main()
    
    if candles_fetched:
        log_message(f"Initial run: Successfully fetched candles for hour {current_hour}", PIPELINE_LOG_PATH)
        processed_hours.add(current_hour)
    else:
        log_message("Initial run: No candles fetched, will wait for next execution window", PIPELINE_LOG_PATH)
    
    # Main continuous loop
    while True:
        try:
            # Check if we should execute now
            current_hour = wait_for_next_execution()
            
            # Skip if we've already processed this hour
            if current_hour in processed_hours:
                log_message(f"Hour {current_hour} already processed. Waiting for next execution window...", PIPELINE_LOG_PATH)
                time.sleep(60)  # Check again in a minute
                continue
            
            # Run the pipeline
            candles_fetched = main()
            
            if candles_fetched:
                # If we did fetch candles, mark this hour as processed
                processed_hours.add(current_hour)
                
                # Limit the size of processed_hours to avoid memory growth
                if len(processed_hours) > 24:
                    processed_hours = set(sorted(processed_hours)[-24:])
                
                # Wait for the next execution window
                time.sleep(600)  # Wait 10 minutes before checking again
            else:
                # If no candles were fetched yet, try again soon
                log_message(f"No new candles for hour {current_hour} yet. Retrying in {FETCH_RETRY_DELAY} seconds...", PIPELINE_LOG_PATH)
                time.sleep(FETCH_RETRY_DELAY)
            
        except KeyboardInterrupt:
            log_message("Pipeline interrupted by user. Exiting...", PIPELINE_LOG_PATH)
            break
        except Exception as e:
            log_message(f"Unexpected error in pipeline: {str(e)}", PIPELINE_LOG_PATH)
            # Log to file for troubleshooting
            with open(os.path.join(LOG_DIR, "pipeline_errors.log"), "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {str(e)}\n")
            
            # Wait before trying again
            time.sleep(60)

if __name__ == "__main__":
    main_loop()  # Use the continuous loop instead of a single run