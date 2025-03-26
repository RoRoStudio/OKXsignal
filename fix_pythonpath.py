#!/usr/bin/env python3
"""
Fix Python Path Issues
- Adds project root to Python path
- Diagnoses and fixes common import path problems
- Verifies critical imports are working
"""

import os
import sys
import inspect
import importlib
import traceback
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    print(f"Added root directory to path: {root_dir}")

# Add database directory to path if it exists
database_dir = os.path.join(root_dir, "database")
if os.path.exists(database_dir) and database_dir not in sys.path:
    sys.path.insert(0, database_dir)
    print(f"Added database directory to path: {database_dir}")

# Add database/processing directory to path if it exists
processing_dir = os.path.join(database_dir, "processing")
if os.path.exists(processing_dir) and processing_dir not in sys.path:
    sys.path.insert(0, processing_dir)
    print(f"Added processing directory to path: {processing_dir}")

def check_module_exists(module_name):
    """Check if a module exists without importing it"""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ModuleNotFoundError:
        return False

def safe_import(module_name):
    """Safely try to import a module and return success status"""
    try:
        module = importlib.import_module(module_name)
        return True, module
    except Exception as e:
        return False, str(e)

def diagnose_imports():
    """Diagnose import issues with critical modules"""
    print("\nDiagnosing import issues...")
    
    critical_modules = [
        # Standard libraries
        "os", "sys", "logging", "datetime", "pathlib",
        
        # Third-party libraries
        "numpy", "pandas", "psycopg2", 
        
        # Try to import project modules
        "database", "database.db", 
        "database.processing.features.config",
        "database.processing.features.utils",
        "database.processing.features.optimized.feature_processor"
    ]
    
    results = {}
    for module in critical_modules:
        exists = check_module_exists(module)
        status, details = safe_import(module) if exists else (False, "Module not found")
        
        results[module] = {
            "exists": exists,
            "importable": status,
            "details": details if not status else "Success"
        }
        
        status_str = "✅ OK" if status else "❌ FAILED"
        print(f"{status_str} - {module}")
        
        if not status and exists:
            print(f"  Error: {details}")
    
    return results

def fix_pythonpath():
    """Create a script to fix the Python path"""
    script_content = """
import sys
import os

# Get root directory (where this script is located)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories to add to path
paths_to_add = [
    ROOT_DIR,
    os.path.join(ROOT_DIR, "database"),
    os.path.join(ROOT_DIR, "database", "processing")
]

# Add directories to Python path
for path in paths_to_add:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
"""
    
    # Create script file
    path_fix_script = os.path.join(root_dir, "fix_path.py")
    with open(path_fix_script, "w") as f:
        f.write(script_content.strip())
    
    print(f"\nCreated Python path fix script: {path_fix_script}")
    print("Add the following line to the beginning of your compute_features.py script:")
    print("import fix_path  # Fix Python path issues\n")
    
    # Create .pth file in site-packages
    try:
        import site
        site_packages = site.getsitepackages()[0]
        pth_file = os.path.join(site_packages, "okxsignal.pth")
        
        with open(pth_file, "w") as f:
            f.write(root_dir)
        
        print(f"Created .pth file for permanent fix: {pth_file}")
        print("This will add the project root to the Python path for all scripts.")
    except Exception as e:
        print(f"Could not create .pth file: {e}")
        print("You'll need to use the fix_path.py script in each file.")

def print_debug_info():
    """Print debug information about the environment"""
    print("\nEnvironment Information:")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

if __name__ == "__main__":
    print("Python Path Diagnostics and Fix Utility")
    print("=======================================")
    
    print_debug_info()
    diagnose_imports()
    fix_pythonpath()
    
    print("\nDiagnostics complete. Please review the output above.")
    print("If import issues persist, try the following:")
    print("1. Add 'import fix_path' to the top of your main script")
    print("2. Run 'pip install -e .' in the project root directory")
    print("3. Make sure all required packages are installed")