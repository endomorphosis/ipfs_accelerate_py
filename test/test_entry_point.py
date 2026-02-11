#!/usr/bin/env python3
"""
Test script to validate the ipfs-accelerate entry point is working correctly.
"""

import subprocess
import sys
import os

def test_entry_point():
    """Test the ipfs-accelerate entry point functionality."""
    print("🧪 Testing IPFS Accelerate Entry Point")
    print("=" * 50)
    
    # Test 1: Direct module execution
    print("\n✅ Test 1: Direct module execution")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'ipfs_accelerate_py.cli_entry', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and 'IPFS Accelerate CLI' in result.stdout:
            print("✅ SUCCESS: Module execution works")
        else:
            print("❌ FAILED: Module execution failed")
            print(f"Return code: {result.returncode}")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ FAILED: Exception during module execution: {e}")
    
    # Test 2: MCP commands
    print("\n✅ Test 2: MCP command structure")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'ipfs_accelerate_py.cli_entry', 'mcp', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and 'start' in result.stdout and 'dashboard' in result.stdout:
            print("✅ SUCCESS: MCP commands available")
        else:
            print("❌ FAILED: MCP commands not available")
            print(f"Return code: {result.returncode}")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ FAILED: Exception during MCP command test: {e}")
    
    # Test 3: MCP start command help
    print("\n✅ Test 3: MCP start command")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'ipfs_accelerate_py.cli_entry', 'mcp', 'start', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and '--dashboard' in result.stdout and '--port' in result.stdout:
            print("✅ SUCCESS: MCP start command works")
        else:
            print("❌ FAILED: MCP start command failed")
            print(f"Return code: {result.returncode}")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ FAILED: Exception during MCP start command test: {e}")
    
    # Show installation instructions
    print("\n🚀 Installation Instructions")
    print("=" * 50)
    print("To install the ipfs-accelerate command globally:")
    print("1. pip install -e .")
    print("2. Ensure ~/.local/bin is in your PATH")
    print("3. Run: ipfs-accelerate mcp start --dashboard")
    print("")
    print("Alternative direct usage:")
    print("python -m ipfs_accelerate_py.cli_entry mcp start --dashboard")
    print("")
    print("✅ Entry point validation complete!")

if __name__ == '__main__':
    test_entry_point()