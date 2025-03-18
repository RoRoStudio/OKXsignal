"""
config_loader.py
Loads configuration settings from config/config.ini.
"""

import os
import configparser

# Locate config.ini in the root config/ directory
CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.ini"))

def load_config():
    """
    Reads config.ini and returns a dictionary of relevant info.
    """
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"❌ Config file not found: {CONFIG_FILE}")

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    settings = {
        "SIMULATED_TRADING": config["OKX"].getboolean("SIMULATED_TRADING", fallback=False),

        "SUPABASE_URL": config["SUPABASE"]["SUPABASE_URL"],

        "DEFAULT_PAIR": config["GENERAL"]["DEFAULT_PAIR"],
        "DEFAULT_TIMEFRAME": config["GENERAL"]["DEFAULT_TIMEFRAME"],
        "ORDER_SIZE_LIMIT": config["GENERAL"].getint("ORDER_SIZE_LIMIT", fallback=5),
        "LOG_LEVEL": config["GENERAL"]["LOG_LEVEL"]
    }
    return settings
