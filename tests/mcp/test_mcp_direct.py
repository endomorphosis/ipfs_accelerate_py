#!/usr/bin/env python3
"""
Direct MCP Server Test Script

This script tests the MCP server directly using HTTP requests without relying on the MCP client.
"""

import requests
import json
import sys
import time

def test_mcp_server(base_url="http://localhost:8002"):
    """Test the MCP server directly."""
    print(f"Testing MCP server at: {base_url}")
    
    # Test 1: Get the manifest
    try:
        response = requests.get(f"{base_url}/mcp/manifest")
        if response.status_code == 200:
            manifest = response.json()
            print("Server manifest:")
            print(json.dumps(manifest, indent=2))
            print("\nAvailable tools:")
            for tool_name in manifest.get("tools", {}):
                print(f"- {tool_name}")
            print()
        else:
            print(f"Failed to get manifest: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return False
    
    # Test 2: Call the get_hardware_info tool
    try:
        print("Testing get_hardware_info tool...")
        response = requests.post(
            f"{base_url}/mcp/tool/get_hardware_info",
            json={}
        )
        if response.status_code == 200:
            result = response.json()
            print("Hardware info:")
            print(json.dumps(result, indent=2))
            print("\nTest successful!")
        else:
            print(f"Failed to call get_hardware_info: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error calling get_hardware_info: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Get base URL from command line if provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
    
    # Run the test
    success = test_mcp_server(base_url)
    sys.exit(0 if success else 1)
