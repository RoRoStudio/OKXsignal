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
        # ✅ Trading configuration
        "SIMULATED_TRADING": config["OKX"].getboolean("SIMULATED_TRADING", fallback=False),

        # ✅ Default trading settings
        "DEFAULT_PAIR": config["GENERAL"]["DEFAULT_PAIR"],
        "DEFAULT_TIMEFRAME": config["GENERAL"]["DEFAULT_TIMEFRAME"],
        "ORDER_SIZE_LIMIT": config["GENERAL"].getint("ORDER_SIZE_LIMIT", fallback=5),
        "LOG_LEVEL": config["GENERAL"]["LOG_LEVEL"],

        # ✅ Database connection details
        "DB_HOST": config["DATABASE"]["DB_HOST"],
        "DB_PORT": config["DATABASE"]["DB_PORT"],
        "DB_NAME": config["DATABASE"]["DB_NAME"],
    }
    return settings
