#!/usr/bin/env python3
"""
Test the working MCP server
"""

import requests
import json
import time

def test_working_server():
    base_url = "http://127.0.0.1:8003"
    
    # Wait a moment for server to start
    print("Testing Working MCP Server...")
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Health: {response.status_code} - {response.text}")
        
        # Test tools list
        print("2. Testing tools endpoint...")
        response = requests.get(f"{base_url}/tools", timeout=5)
        print(f"   Tools: {response.status_code}")
        if response.status_code == 200:
            tools = response.json()
            print(f"   Found {len(tools)} tools: {tools}")
        
        # Test a specific tool
        print("3. Testing ipfs_node_info tool...")
        response = requests.post(f"{base_url}/tools/ipfs_node_info", json={}, timeout=5)
        print(f"   ipfs_node_info: {response.status_code} - {response.text}")
        
        # Test gateway URL tool
        print("4. Testing ipfs_gateway_url tool...")
        response = requests.post(f"{base_url}/tools/ipfs_gateway_url", json={"cid": "QmTest123"}, timeout=5)
        print(f"   ipfs_gateway_url: {response.status_code} - {response.text}")
        
        print("✅ All tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Give server time to start
    time.sleep(2)
    test_working_server()
