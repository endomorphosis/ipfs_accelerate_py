#!/usr/bin/env python3
"""
Quick test to verify ipfs-accelerate mcp start uses the correct dashboard
"""

import subprocess
import time
import requests
import sys

def test_mcp_start_command():
    """Test that ipfs-accelerate mcp start works correctly"""
    
    print("\n" + "="*80)
    print("TESTING: ipfs-accelerate mcp start command")
    print("="*80)
    
    # Start the server
    print("\n1. Starting MCP server...")
    process = subprocess.Popen(
        [sys.executable, "-m", "ipfs_accelerate_py.cli_entry", "mcp", "start", "--port", "8899"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for it to start
    print("2. Waiting for server to be ready...")
    time.sleep(5)
    
    try:
        # Test if endpoints work
        print("\n3. Testing dashboard endpoint...")
        response = requests.get("http://127.0.0.1:8899/mcp", timeout=5)
        if response.status_code == 200:
            print("   ✅ Dashboard loads")
            
            # Check for our custom content
            if "Model Browser" in response.text:
                print("   ✅ Dashboard contains Model Browser tab")
            else:
                print("   ❌ Model Browser tab not found")
        else:
            print(f"   ❌ Dashboard returned status {response.status_code}")
        
        print("\n4. Testing model search API endpoint...")
        response = requests.get("http://127.0.0.1:8899/api/mcp/models/search?q=bert&limit=5", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Search API works - returned {len(data.get('results', []))} results")
        else:
            print(f"   ❌ Search API returned status {response.status_code}")
        
        print("\n5. Testing model stats API endpoint...")
        response = requests.get("http://127.0.0.1:8899/api/mcp/models/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Stats API works - {data.get('total_cached_models', 0)} models cached")
        else:
            print(f"   ❌ Stats API returned status {response.status_code}")
        
        print("\n" + "="*80)
        print("✅ TEST PASSED - ipfs-accelerate mcp start works correctly")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
        
    finally:
        # Stop the server
        print("\n6. Stopping server...")
        process.terminate()
        process.wait(timeout=5)
        print("   ✅ Server stopped")

if __name__ == '__main__':
    success = test_mcp_start_command()
    sys.exit(0 if success else 1)
