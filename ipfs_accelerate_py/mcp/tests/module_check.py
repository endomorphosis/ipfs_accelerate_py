#!/usr/bin/env python
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print("Python path:", sys.path)
print("Looking for modules...")

try:
    print("Checking for fastmcp module...")
    import fastmcp
    print(f"Found fastmcp at {fastmcp.__file__}")
    
    print("Checking for ipfs_accelerate_py module...")
    import ipfs_accelerate_py
    print(f"Found ipfs_accelerate_py at {ipfs_accelerate_py.__file__}")
    
    print("Creating ipfs_accelerate_py instance...")
    instance = ipfs_accelerate_py.ipfs_accelerate_py()
    print("Successfully created instance")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()

input("Press Enter to exit...")
