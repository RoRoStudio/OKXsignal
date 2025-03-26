#!/usr/bin/env python3
"""
Check if database environment variables are correctly set
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from credentials file
load_dotenv(os.path.join(os.path.dirname(__file__), "config", "credentials.env"))

def check_env_vars():
    required_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please make sure these variables are set in your environment or credentials.env file.")
        return False
    
    print("Environment variables check passed:")
    for var in required_vars:
        # Hide password value
        if var == 'DB_PASSWORD':
            print(f"  {var}: {'*' * min(len(os.getenv(var, '')), 8)}")
        else:
            print(f"  {var}: {os.getenv(var)}")
    
    return True

if __name__ == '__main__':
    if not check_env_vars():
        sys.exit(1)
    print("Database environment variables are properly set!")