#!/usr/bin/env python3
"""
Simple verification script for the IPFS Accelerate MCP Server

This script tests basic functionality of the MCP server endpoints
to verify they're working correctly.
"""

import sys
import json
import requests
import tempfile
import os
from typing import Dict, Any, Optional

def test_server_connection(server_url: str) -> bool:
    """Test connection to the MCP server."""
    try:
        response = requests.get(f"{server_url}/", timeout=5)
        response.raise_for_status()
        print(f"Server connection successful: {server_url}")
        print(f"Server response: {response.text[:100]}...")
        return True
    except Exception as e:
        print(f"Server connection failed: {e}")
        return False

def get_tools_list(server_url: str) -> Optional[list]:
    """Get the list of available tools."""
    try:
        response = requests.get(f"{server_url}/tools", timeout=5)
        response.raise_for_status()
        tools = response.json().get("tools", [])
        print(f"Available tools ({len(tools)}): {', '.join(sorted(tools))}")
        return tools
    except Exception as e:
        print(f"Failed to get tools list: {e}")
        return None

def call_tool(server_url: str, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call a tool with arguments."""
    if arguments is None:
        arguments = {}
    
    print(f"Calling tool {tool_name} with arguments: {arguments}")
    
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": tool_name, "arguments": arguments},
            timeout=10
        )
        
        # Print the raw response for debugging
        print(f"Raw response: {response.text[:200]}...")
        
        # If we got a non-200 status, print the details
        if response.status_code != 200:
            print(f"Error status: {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"HTTP Error: {response.status_code}", "success": False}
            
        # Parse the JSON response
        result = response.json().get("result", {})
        print(f"Tool result: {json.dumps(result, indent=2)[:200]}...")
        return result
    except Exception as e:
        print(f"Exception calling tool {tool_name}: {e}")
        return {"error": str(e), "success": False}

def test_health_check(server_url: str) -> bool:
    """Test the health check endpoint."""
    print("\n=== Testing health_check ===")
    result = call_tool(server_url, "health_check", {})
    return result.get("status") == "healthy"

def test_add_and_cat(server_url: str) -> bool:
    """Test adding a file to IPFS and retrieving it."""
    print("\n=== Testing ipfs_add_file and ipfs_cat ===")
    
    # Create a temporary file
    content = "Test content for IPFS"
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Add the file to IPFS
        add_result = call_tool(server_url, "ipfs_add_file", {"path": tmp_path})
        if not add_result.get("success", False):
            print(f"Failed to add file: {add_result.get('error', 'Unknown error')}")
            return False
        
        # Get the CID
        cid = add_result.get("cid")
        if not cid:
            print("No CID returned")
            return False
            
        print(f"File added with CID: {cid}")
        
        # Retrieve the file from IPFS
        cat_result = call_tool(server_url, "ipfs_cat", {"cid": cid})
        if cat_result == content:
            print("Successfully retrieved file content")
            return True
        else:
            print(f"Retrieved content doesn't match: {cat_result}")
            return False
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

def test_files_write_and_read(server_url: str) -> bool:
    """Test writing and reading from IPFS MFS."""
    print("\n=== Testing ipfs_files_write and ipfs_files_read ===")
    
    # Write to MFS
    path = "/test-mcp-verify.txt"
    content = "Test MFS content"
    
    write_result = call_tool(server_url, "ipfs_files_write", {
        "path": path,
        "content": content
    })
    
    if not write_result.get("success", False):
        print(f"Failed to write to MFS: {write_result.get('error', 'Unknown error')}")
        return False
    
    # Read from MFS
    read_result = call_tool(server_url, "ipfs_files_read", {"path": path})
    
    if read_result == content:
        print("Successfully read content from MFS")
        return True
    else:
        print(f"Read content doesn't match: {read_result}")
        return False

def test_hardware_info(server_url: str) -> bool:
    """Test getting hardware information."""
    print("\n=== Testing get_hardware_info ===")
    
    result = call_tool(server_url, "get_hardware_info", {})
    
    if "cpu" in result:
        print(f"Got hardware info with {result.get('cpu', {}).get('cores', 'unknown')} CPU cores")
        return True
    else:
        print("Failed to get hardware info")
        return False

def main():
    """Main function to run the verification."""
    server_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    print(f"Testing MCP server at: {server_url}")
    
    # Test server connection
    if not test_server_connection(server_url):
        print("Failed to connect to server, exiting...")
        return 1
    
    # Get tools list
    tools = get_tools_list(server_url)
    if not tools:
        print("Failed to get tools list, exiting...")
        return 1
    
    # Run basic tool tests
    results = {
        "health_check": test_health_check(server_url) if "health_check" in tools else None,
        "add_and_cat": test_add_and_cat(server_url) if "ipfs_add_file" in tools and "ipfs_cat" in tools else None,
        "files_write_and_read": test_files_write_and_read(server_url) if "ipfs_files_write" in tools and "ipfs_files_read" in tools else None,
        "hardware_info": test_hardware_info(server_url) if "get_hardware_info" in tools else None,
    }
    
    # Print summary
    print("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED (Tool not available)"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
            all_passed = False
        
        print(f"{test_name}: {status}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())