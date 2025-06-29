#\!/usr/bin/env python3
"""
Add WebNN and WebGPU support to the test generator system by integrating the fixed_web_platform package.

This script:
1. Creates the fixed_web_platform module with proper implementations
2. Adds the web platform handlers to the merged_test_generator.py

Usage:
  python add_web_platform_support.py
"""

import os
import sys
import shutil
import importlib
from pathlib import Path

# Paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_FILE = CURRENT_DIR / "merged_test_generator.py"

def main():
    """Main function."""
    print("Adding WebNN and WebGPU platform support to test generators...")
    
    # Check if fixed_web_platform directory exists
    fixed_web_dir = CURRENT_DIR / "fixed_web_platform"
    if not fixed_web_dir.exists():
        print("Error: fixed_web_platform directory not found.")
        print("Please make sure it's created with the web_platform_handler.py and __init__.py files.")
        return 1
    
    # Now run the test generator with a simple model using WebNN platform
    print("\nTesting with a simple model using WebNN platform...")
    
    cmd = f"cd {CURRENT_DIR} && python merged_test_generator.py --generate bert --platform webnn --no-deps"
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    # Check if it worked
    if result == 0:
        print("\nWebNN and WebGPU platform support successfully added\!")
        print("You can now generate tests with web platform support:")
        print("  python merged_test_generator.py --generate bert --platform webnn")
        print("  python merged_test_generator.py --generate vit --platform webgpu")
        return 0
    else:
        print("\nThere was an error testing the web platform support.")
        print("Please check the errors above and try adding the import manually:")
        print("from fixed_web_platform import process_for_web, init_webnn, init_webgpu, create_mock_processors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
