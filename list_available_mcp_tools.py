#!/usr/bin/env python
"""
List Available MCP Tools

This script lists all the available tools registered with the IPFS Accelerate MCP server.
"""

import json
import requests
import sys

def list_available_tools():
    """List all available tools on the MCP server"""
    # Try to automatically find the MCP server port
    ports_to_try = [8002, 8000, 8765, 8001]  # Various possible ports
    
    for port in ports_to_try:
        # Skip OpenAPI and base /mcp endpoint checks and go straight to testing tools
        # since previous tests showed this seems to be a minimal MCP implementation
        
        # Comprehensive list of possible tools to test
        known_tools = [
            # Core hardware tools
            "get_hardware_info",
            "get_system_info",
            
            # IPFS related tools
            "ipfs_node_info",
            "ipfs_gateway_url",
            "ipfs_add_file",
            "ipfs_get_file",
            "ipfs_status",
            
            # Model inference tools
            "model_inference",
            "list_models",
            "get_model_info",
            
            # Additional potential tools
            "ping",
            "echo",
            "version_info",
            "status"
        ]
        
        print(f"\nTesting tools on port {port}:")
        working_tools = []
        found_any = False
        
        for tool in known_tools:
            tool_url = f"http://localhost:{port}/mcp/tool/{tool}"
            try:
                # Empty JSON input since we're just checking if the tool exists
                test_response = requests.post(tool_url, json={}, timeout=2)
                
                # Consider both 200 and 422 (validation error) as tool exists
                # 422 just means we provided incorrect parameters
                if test_response.status_code in [200, 422]:
                    print(f"  ✅ {tool}")
                    working_tools.append(tool)
                    found_any = True
                else:
                    print(f"  ❌ {tool} (Status: {test_response.status_code})")
            except requests.exceptions.Timeout:
                print(f"  ⏱️ {tool} (Timeout)")
            except Exception as e:
                print(f"  ❌ {tool} (Error: {e})")
        
        if found_any:
            print(f"\n✅ Found MCP server on port {port} with {len(working_tools)} tools!")
            return working_tools, port
    
    print("❌ Failed to connect to any MCP server")
    return []

if __name__ == "__main__":
    print("======================================")
    print("IPFS Accelerate MCP Tools Lister")
    print("======================================\n")
    
    # List available tools
    result = list_available_tools()
    
    if result:
        tools, port = result
        print(f"\n✅ Found {len(tools)} available tools on port {port}")
        
        # Try to get more details about the server
        try:
            hardware_url = f"http://localhost:{port}/mcp/tool/get_hardware_info"
            hardware_response = requests.post(hardware_url, json={})
            if hardware_response.status_code == 200:
                hardware_info = hardware_response.json()
                print("\nHardware Information:")
                if isinstance(hardware_info, dict) and "accelerators" in hardware_info:
                    for acc_name, acc_info in hardware_info["accelerators"].items():
                        if isinstance(acc_info, dict) and acc_info.get("available", False):
                            print(f"  - {acc_name}: {acc_info.get('name', 'Unknown')}")
        except Exception as e:
            print(f"Error getting hardware details: {e}")
    else:
        print("\n❌ Could not find any available tools")
    
    print("\n======================================")
