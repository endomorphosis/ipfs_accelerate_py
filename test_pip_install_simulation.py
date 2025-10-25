#!/usr/bin/env python3
"""
Test script to simulate what happens after pip install -e .

This script tests the ipfs-accelerate command functionality without
requiring actual pip installation.
"""

import os
import sys
import subprocess

def test_cli_entry_point():
    """Test the CLI entry point functionality."""
    print("🧪 Testing IPFS Accelerate CLI Entry Point")
    print("=" * 50)
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test 1: Direct execution of cli_entry.py
    print("\n✅ Test 1: Direct CLI entry execution")
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join(current_dir, "ipfs_accelerate_py", "cli_entry.py"), 
            "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ SUCCESS: CLI entry point works")
            print(f"   📄 Output preview: {result.stdout.split('n')[0]}")
        else:
            print(f"   ❌ FAILED: Return code {result.returncode}")
            print(f"   📄 Error: {result.stderr[:200]}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    # Test 2: Module execution
    print("\n✅ Test 2: Python module execution")
    try:
        result = subprocess.run([
            sys.executable, 
            "-m", "ipfs_accelerate_py.cli_entry",
            "--help"
        ], capture_output=True, text=True, timeout=30, cwd=current_dir)
        
        if result.returncode == 0:
            print("   ✅ SUCCESS: Module execution works")
        else:
            print(f"   ❌ FAILED: Return code {result.returncode}")
            print(f"   📄 Error: {result.stderr[:200]}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    # Test 3: MCP start command
    print("\n✅ Test 3: MCP start command")
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join(current_dir, "ipfs_accelerate_py", "cli_entry.py"), 
            "mcp", "start", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ SUCCESS: MCP start command works")
            print("   📄 Help text available - command structure correct")
        else:
            print(f"   ❌ FAILED: Return code {result.returncode}")
            print(f"   📄 Error: {result.stderr[:200]}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    # Test 4: Import test
    print("\n✅ Test 4: Import functionality")
    try:
        # Add current directory to path
        sys.path.insert(0, current_dir)
        from ipfs_accelerate_py.cli_entry import main
        print("   ✅ SUCCESS: CLI entry import works")
    except Exception as e:
        print(f"   ❌ FAILED: Import error: {e}")
    
    print("\n🎯 **SOLUTION FOR VIRTUAL ENVIRONMENT:**")
    print("After activating your virtual environment, run:")
    print(f"   python {os.path.join(current_dir, 'ipfs_accelerate_py', 'cli_entry.py')} mcp start --dashboard")
    print("\nOr use module execution:")
    print(f"   cd {current_dir}")
    print("   python -m ipfs_accelerate_py.cli_entry mcp start --dashboard")
    
    print("\n🔧 **FOR PROPER INSTALLATION:**")
    print("The entry point should work after 'pip install -e .' with:")
    print("   ipfs-accelerate mcp start --dashboard")

if __name__ == "__main__":
    test_cli_entry_point()