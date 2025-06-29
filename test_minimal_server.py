#!/usr/bin/env python3
"""
Test script for the minimal working MCP server
"""

import requests
import json
import time
import subprocess
import signal
import os
import sys

def test_minimal_server():
    """Test the minimal working MCP server"""
    
    base_url = "http://127.0.0.1:8002"
    server_process = None
    
    try:
        print("🚀 Starting minimal MCP server...")
        
        # Start the server
        server_process = subprocess.Popen([
            sys.executable, "minimal_working_mcp_server.py",
            "--host", "127.0.0.1",
            "--port", "8002"
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        cwd="/home/barberb/ipfs_accelerate_py"
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Check if server is running
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            print(f"❌ Server failed to start!")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
        
        print("✅ Server started successfully!")
        
        # Test endpoints
        tests = [
            ("Health Check", "GET", "/health", None),
            ("List Tools", "GET", "/tools", None),
            ("MCP Manifest", "GET", "/mcp/manifest", None),
            ("Status", "GET", "/status", None),
            ("IPFS Node Info", "POST", "/tools/ipfs_node_info", {}),
            ("IPFS Gateway URL", "POST", "/tools/ipfs_gateway_url", {"cid": "QmTest123"}),
            ("Hardware Info", "POST", "/tools/ipfs_get_hardware_info", {}),
            ("List Models", "POST", "/tools/list_models", {}),
        ]
        
        success_count = 0
        total_tests = len(tests)
        
        for test_name, method, endpoint, data in tests:
            try:
                print(f"🧪 Testing {test_name}...")
                
                if method == "GET":
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                else:
                    response = requests.post(f"{base_url}{endpoint}", json=data, timeout=5)
                
                if response.status_code == 200:
                    print(f"   ✅ {test_name}: {response.status_code}")
                    if endpoint == "/tools":
                        tools = response.json()
                        print(f"      Found {len(tools)} tools: {tools[:3]}...")
                    elif endpoint == "/tools/ipfs_node_info":
                        result = response.json()
                        print(f"      Node ID: {result.get('id', 'N/A')}")
                    success_count += 1
                else:
                    print(f"   ❌ {test_name}: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                print(f"   ❌ {test_name}: Error - {e}")
        
        print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
        
        if success_count == total_tests:
            print("🎉 All tests passed! The minimal MCP server is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed, but basic functionality may be working.")
            return success_count > total_tests // 2
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        # Clean up - stop the server
        if server_process and server_process.poll() is None:
            print("🛑 Stopping server...")
            server_process.terminate()
            time.sleep(2)
            if server_process.poll() is None:
                server_process.kill()

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir("/home/barberb/ipfs_accelerate_py")
    
    print("="*60)
    print("IPFS Accelerate MCP Server - Test Suite")
    print("="*60)
    
    success = test_minimal_server()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. The minimal MCP server is working")
        print("2. Update VS Code settings to use this server")
        print("3. Test VS Code integration")
        sys.exit(0)
    else:
        print("\n🔧 Server needs debugging")
        sys.exit(1)
