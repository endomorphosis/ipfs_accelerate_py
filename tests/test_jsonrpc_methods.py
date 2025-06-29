#!/usr/bin/env python3
"""
Test script to verify all JSON-RPC methods are working correctly
"""

import json
import requests
import sys

def test_jsonrpc_method(url, method, params=None):
    """Test a JSON-RPC method"""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": 1
    }
    if params:
        payload["params"] = params
    
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print(f"❌ {method}: ERROR - {result['error']}")
                return False
            else:
                print(f"✅ {method}: SUCCESS")
                if method in ["tools/list", "get_tools", "list_tools"]:
                    tools = result.get("result", [])
                    print(f"   Found {len(tools)} tools")
                elif method == "ping":
                    print(f"   Response: {result.get('result')}")
                elif method == "get_server_info":
                    info = result.get("result", {})
                    print(f"   Version: {info.get('version')}, Tools: {info.get('registered_tools')}")
                return True
        else:
            print(f"❌ {method}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {method}: Exception - {e}")
        return False

def main():
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = "8004"
    
    url = f"http://127.0.0.1:{port}/jsonrpc"
    
    print(f"Testing JSON-RPC methods on {url}")
    print("=" * 50)
    
    # Test core MCP methods
    methods_to_test = [
        "ping",
        "get_server_info", 
        "tools/list",
        "get_tools",
        "list_tools"
    ]
    
    results = {}
    for method in methods_to_test:
        results[method] = test_jsonrpc_method(url, method)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All JSON-RPC methods are working!")
        return 0
    else:
        print("❌ Some methods failed")
        return 1

if __name__ == "__main__":
    exit(main())
