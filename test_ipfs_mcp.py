#!/usr/bin/env python3
"""
Test IPFS MCP Tools

This script tests the IPFS tools registered with the MCP server.
"""

import os
import sys
import json
import tempfile
import requests
from typing import Dict, Any, List, Optional

# Server configuration
SERVER_URL = "http://localhost:8002"
MCP_SERVER_NAME = "localhost:8002"

def make_request(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a request to the MCP server."""
    url = f"{SERVER_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response.json()
    except Exception as e:
        print(f"Error making request to {url}: {e}")
        sys.exit(1)

def get_manifest() -> Dict[str, Any]:
    """Get the MCP manifest."""
    print("\n🔍 Getting MCP manifest...")
    manifest = make_request("/mcp/manifest")
    
    # Print server info
    print(f"Server: {manifest.get('server_name')}")
    print(f"Description: {manifest.get('description')}")
    print(f"Version: {manifest.get('version')}")
    print(f"MCP Version: {manifest.get('mcp_version')}")
    
    # Print tool names
    tools = manifest.get("tools", {})
    print(f"\nAvailable Tools ({len(tools)}):")
    for name in tools:
        print(f"  - {name}")
    
    # Print resource names
    resources = manifest.get("resources", {})
    print(f"\nAvailable Resources ({len(resources)}):")
    for name in resources:
        print(f"  - {name}")
    
    return manifest

def call_tool(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call a tool on the MCP server."""
    if args is None:
        args = {}
    
    print(f"\n🛠️ Calling tool '{name}' with args: {args}")
    
    result = make_request(f"/mcp/tool/{name}", method="POST", data=args)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    return result

def access_resource(name: str) -> Dict[str, Any]:
    """Access a resource on the MCP server."""
    print(f"\n📚 Accessing resource '{name}'")
    
    result = make_request(f"/mcp/resources/{name}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    return result

def run_tests():
    """Run tests for IPFS MCP tools."""
    print("=== IPFS MCP TOOLS TEST ===")
    
    # Get manifest
    manifest = get_manifest()
    
    # Access system_info resource
    access_resource("system_info")
    
    # Test ipfs_node_info
    node_info = call_tool("ipfs_node_info", {})
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        test_content = "Hello, IPFS MCP test!"
        f.write(test_content)
        test_file_path = f.name
    
    try:
        # Test ipfs_add_file
        add_result = call_tool("ipfs_add_file", {"path": test_file_path})
        cid = add_result.get("cid")
        
        if not cid:
            print("Error: Could not get CID from add_file result")
            return
        
        # Test ipfs_cat
        cat_result = call_tool("ipfs_cat", {"cid": cid})
        content = cat_result.get("content")
        
        if content != test_content:
            print(f"Error: Content mismatch. Expected '{test_content}', got '{content}'")
        else:
            print("✅ Content verification successful!")
        
        # Test ipfs_pin_add
        call_tool("ipfs_pin_add", {"cid": cid})
        
        # Test ipfs_pin_ls
        pins = call_tool("ipfs_pin_ls", {})
        
        if cid not in pins.get("pins", []):
            print(f"Error: CID {cid} not found in pins")
        else:
            print("✅ Pin verification successful!")
        
        # Test ipfs_pin_rm
        call_tool("ipfs_pin_rm", {"cid": cid})
        
        # Verify pin was removed
        pins_after = call_tool("ipfs_pin_ls", {})
        if cid in pins_after.get("pins", []):
            print(f"Error: CID {cid} still in pins after removal")
        else:
            print("✅ Pin removal verification successful!")
        
        # Test ipfs_files_write
        call_tool("ipfs_files_write", {"path": "/test.txt", "content": "MFS test content"})
        
        # Test ipfs_files_ls
        ls_result = call_tool("ipfs_files_ls", {"path": "/"})
        
        # Test ipfs_files_read
        read_result = call_tool("ipfs_files_read", {"path": "/test.txt"})
        
        if read_result.get("content") != "MFS test content":
            print("Error: MFS content mismatch")
        else:
            print("✅ MFS verification successful!")
        
        # Access resources
        access_resource("ipfs_files")
        access_resource("ipfs_pins")
        access_resource("ipfs_nodes")
        
    finally:
        # Clean up
        os.unlink(test_file_path)
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    run_tests()
