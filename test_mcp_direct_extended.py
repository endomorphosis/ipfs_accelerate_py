#!/usr/bin/env python3
"""
Extended MCP Server Test Script

This script tests more functionality of the MCP server directly using HTTP requests.
"""

import requests
import json
import sys
import os
import tempfile
import time

def test_mcp_server(base_url="http://localhost:8002"):
    """Test the MCP server directly with extended functionality."""
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
    
    # Test 2: Check server health
    try:
        print("Testing server health...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print("Health check result:")
            print(json.dumps(health, indent=2))
            print()
        else:
            print(f"Failed health check: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error checking health: {e}")
    
    # Test 3: Call the get_hardware_info tool
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
            print("\nget_hardware_info test successful!")
            print()
        else:
            print(f"Failed to call get_hardware_info: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error calling get_hardware_info: {e}")
    
    # Test 4: Test IPFS tools if available
    try:
        # Try ipfs_add_file if available
        if "ipfs_add_file" in manifest.get("tools", {}):
            print("Testing ipfs_add_file tool...")
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp:
                tmp.write("Test content for IPFS")
                tmp_path = tmp.name
            
            response = requests.post(
                f"{base_url}/mcp/tool/ipfs_add_file",
                json={"path": tmp_path}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("ipfs_add_file result:")
                print(json.dumps(result, indent=2))
                
                # If we got a CID, test ipfs_cat if available
                cid = result.get("result", {}).get("cid")
                if cid and "ipfs_cat" in manifest.get("tools", {}):
                    print("\nTesting ipfs_cat tool...")
                    response = requests.post(
                        f"{base_url}/mcp/tool/ipfs_cat",
                        json={"cid": cid}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print("ipfs_cat result:")
                        print(json.dumps(result, indent=2))
                        print("\nipfs_cat test successful!")
                    else:
                        print(f"Failed to call ipfs_cat: {response.status_code}")
                        print(response.text)
                
                print("\nipfs_add_file test successful!")
                print()
            else:
                print(f"Failed to call ipfs_add_file: {response.status_code}")
                print(response.text)
            
            # Clean up the temporary file
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error testing IPFS tools: {e}")
    
    # Check if SSE endpoint exists (for Claude MCP integration)
    try:
        print("Checking SSE endpoint (for Claude MCP integration)...")
        response = requests.get(f"{base_url}/mcp/sse")
        print(f"SSE endpoint status: {response.status_code}")
        
        # Also check the plain /sse endpoint
        response = requests.get(f"{base_url}/sse")
        print(f"Root SSE endpoint status: {response.status_code}")
        print()
    except Exception as e:
        print(f"Error checking SSE endpoint: {e}")
    
    print("All tests completed!")
    return True

if __name__ == "__main__":
    # Get base URL from command line if provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
    
    # Run the test
    success = test_mcp_server(base_url)
    sys.exit(0 if success else 1)
