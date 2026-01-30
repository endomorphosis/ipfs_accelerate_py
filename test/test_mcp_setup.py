#!/usr/bin/env python3
"""
Test script to verify all MCP server components are working.
"""

import json
import requests
import subprocess
import time
import sys
import os

def test_json_rpc_server():
    """Test the JSON-RPC server."""
    print("🧪 Testing JSON-RPC server...")
    
    # Start the server
    process = None
    try:
        process = subprocess.Popen([
            sys.executable, "mcp_jsonrpc_server.py", "--port", "8007"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        # Test API
        response = requests.post(
            "http://localhost:8007/jsonrpc",
            json={"jsonrpc": "2.0", "method": "list_models", "id": 1},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                print("✅ JSON-RPC server working!")
                print(f"   Response: {data['result']}")
                return True
            else:
                print(f"❌ JSON-RPC server error: {data}")
                return False
        else:
            print(f"❌ JSON-RPC server HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ JSON-RPC server test failed: {e}")
        return False
    finally:
        if process:
            process.terminate()
            process.wait()

def test_cli_tools():
    """Test CLI tools."""
    print("🧪 Testing CLI tools...")
    
    try:
        # Test comprehensive server help
        result = subprocess.run([
            sys.executable, "tools/comprehensive_mcp_server.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✅ Comprehensive MCP server CLI working!")
        else:
            print(f"❌ Comprehensive MCP server CLI failed: {result.stderr}")
            return False
        
        # Test standalone server help
        result = subprocess.run([
            sys.executable, "-m", "ipfs_accelerate_py.mcp.standalone", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✅ Standalone MCP server CLI working!")
        else:
            print(f"❌ Standalone MCP server CLI failed: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ CLI tools test failed: {e}")
        return False

def test_vscode_wrapper():
    """Test VS Code MCP wrapper."""
    print("🧪 Testing VS Code MCP wrapper...")
    
    try:
        # Test that the wrapper starts without immediate errors
        process = subprocess.Popen([
            sys.executable, "vscode_mcp_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Let it run briefly
        time.sleep(2)
        
        # Check if it's still running (good sign)
        poll = process.poll()
        if poll is None:
            print("✅ VS Code MCP wrapper started successfully!")
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ VS Code MCP wrapper exited early: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ VS Code MCP wrapper test failed: {e}")
        return False

def test_dashboard_files():
    """Test that dashboard files exist."""
    print("🧪 Testing dashboard files...")
    
    required_files = [
        "templates/sdk_dashboard.html",
        "static/js/mcp-sdk.js",
        "static/js/kitchen-sink-sdk.js"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing dashboard files: {missing_files}")
        return False
    else:
        print("✅ All dashboard files present!")
        return True

def main():
    """Run all tests."""
    print("🚀 Starting IPFS Accelerate MCP verification tests...\n")
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    tests = [
        ("Dashboard Files", test_dashboard_files),
        ("CLI Tools", test_cli_tools),
        ("VS Code Wrapper", test_vscode_wrapper),
        ("JSON-RPC Server", test_json_rpc_server),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MCP servers are working correctly.")
        print("\nNext steps:")
        print("1. Start the JSON-RPC server: python mcp_jsonrpc_server.py --port 8003")
        print("2. Open http://localhost:8003 in your browser")
        print("3. Configure VS Code with the MCP server using mcp_config.json")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())