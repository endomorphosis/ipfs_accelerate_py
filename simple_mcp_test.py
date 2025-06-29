#!/usr/bin/env python3
"""
Simple MCP Test Script

This script tests the MCP server using direct HTTP requests instead of SSE.
"""

import requests
import json
import time
import sys

# Terminal colors
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")

def print_success(msg):
    print(f"{GREEN}[SUCCESS]{RESET} {msg}")

def print_error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")

def print_header(msg):
    print(f"\n{CYAN}=== {msg} ==={RESET}\n")

def test_mcp_server(base_url="http://localhost:8002"):
    """Test the MCP server API endpoints"""
    
    print_header("MCP Server Test")
    
    # Check if server is running
    try:
        print_info(f"Testing connection to {base_url}/tools")
        response = requests.get(f"{base_url}/tools", timeout=5)
        response.raise_for_status()
        tools = response.json().get("tools", [])
        print_success(f"Server is running. Available tools: {', '.join(tools)}")
    except Exception as e:
        print_error(f"Server connection failed: {str(e)}")
        return False
    
    # Test tool call - health check
    print_header("Testing health_check Tool")
    try:
        response = requests.post(
            f"{base_url}/call_tool",
            json={"tool_name": "health_check", "arguments": {}},
            timeout=5
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        print_success("Health check successful:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False
    
    # Test MCP manifest
    print_header("Testing MCP Manifest")
    try:
        response = requests.get(f"{base_url}/mcp/manifest", timeout=5)
        response.raise_for_status()
        manifest = response.json()
        print_success("Got MCP manifest:")
        print(f"Name: {manifest.get('name')}")
        print(f"Version: {manifest.get('version')}")
        print(f"Description: {manifest.get('description')}")
        print(f"Tools: {len(manifest.get('tools', {}))}")
    except Exception as e:
        print_error(f"Manifest request failed: {str(e)}")
        return False
    
    # Test IPFS tools
    print_header("Testing IPFS Tools")
    
    # Test files_write
    try:
        test_path = "/mcp-test/test-file.txt"
        test_content = f"Test content at {time.time()}"
        
        response = requests.post(
            f"{base_url}/call_tool",
            json={
                "tool_name": "ipfs_files_write", 
                "arguments": {
                    "path": test_path,
                    "content": test_content
                }
            },
            timeout=5
        )
        response.raise_for_status()
        write_result = response.json().get("result", {})
        print_success("ipfs_files_write successful:")
        print(json.dumps(write_result, indent=2))
        
        # Test files_read
        response = requests.post(
            f"{base_url}/call_tool",
            json={
                "tool_name": "ipfs_files_read", 
                "arguments": {"path": test_path}
            },
            timeout=5
        )
        response.raise_for_status()
        read_result = response.json().get("result", {})
        print_success("ipfs_files_read successful:")
        print(json.dumps(read_result, indent=2))
        
        if read_result.get("content") == test_content:
            print_success("Read content matches written content")
        else:
            print_error("Content mismatch!")
            print(f"Expected: {test_content}")
            print(f"Actual: {read_result.get('content')}")
            
    except Exception as e:
        print_error(f"IPFS tests failed: {str(e)}")
        return False
    
    # Test model tools
    print_header("Testing Model Tools")
    
    try:
        # List models
        response = requests.post(
            f"{base_url}/call_tool",
            json={"tool_name": "list_models", "arguments": {}},
            timeout=5
        )
        response.raise_for_status()
        models_result = response.json().get("result", {})
        print_success("list_models successful:")
        print(json.dumps(models_result, indent=2))
        
        # Try to create an endpoint
        if models_result.get("models") and len(models_result.get("models")) > 0:
            model_name = next(iter(models_result.get("models").keys()))
            
            response = requests.post(
                f"{base_url}/call_tool",
                json={
                    "tool_name": "create_endpoint", 
                    "arguments": {"model_name": model_name}
                },
                timeout=5
            )
            response.raise_for_status()
            endpoint_result = response.json().get("result", {})
            print_success("create_endpoint successful:")
            print(json.dumps(endpoint_result, indent=2))
            
            # Get hardware info
            response = requests.post(
                f"{base_url}/call_tool",
                json={"tool_name": "get_hardware_info", "arguments": {}},
                timeout=5
            )
            response.raise_for_status()
            hardware_result = response.json().get("result", {})
            print_success("get_hardware_info successful:")
            print(json.dumps(hardware_result, indent=2))
            
    except Exception as e:
        print_error(f"Model tests failed: {str(e)}")
        return False
    
    print_header("Test Results")
    print_success("All tests passed!")
    print_success("The MCP server API is working correctly.")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCP server")
    parser.add_argument("--url", default="http://localhost:8002", help="Base URL of the MCP server")
    
    args = parser.parse_args()
    
    if test_mcp_server(args.url):
        sys.exit(0)
    else:
        sys.exit(1)
