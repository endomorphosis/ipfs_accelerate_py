#!/usr/bin/env python3
"""
Very basic test script to identify import issues.
"""
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

try:
    print("Trying to import ipfs_accelerate_py...")
    from ipfs_accelerate_py import ipfs_accelerate_py
    print("Successfully imported ipfs_accelerate_py")
    
    # Create instance
    print("Creating ipfs_accelerate_py instance...")
    instance = ipfs_accelerate_py()
    print("Successfully created ipfs_accelerate_py instance")
    
    print("Trying to import FastMCP...")
    from fastmcp import FastMCP
    print("Successfully imported FastMCP")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
