#!/usr/bin/env python3
"""
IPFS Accelerate CLI Entry Point

This script provides the ipfs-accelerate command functionality.
"""
import sys
import os
from pathlib import Path

# Try to import storage wrapper
try:
    from .common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

# Add the package directory to the Python path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir))

# Import and run the CLI
try:
    from cli import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error importing CLI module: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -e .")
    sys.exit(1)
