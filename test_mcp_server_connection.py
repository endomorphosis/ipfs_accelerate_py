#!/usr/bin/env python
"""
Test MCP Server Connection

A diagnostic utility to test connections to MCP servers and check available tools.
"""

import os
import sys
import json
import requests
import argparse
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_server_connection(host: str = "localhost", port: int = 8002) -> Dict[str, Any]:
    """Test connection to MCP server and return manifest"""
    base_url = f"http://{host}:{port}/mcp"
    
    try:
        response = requests.get(f"{base_url}/manifest", timeout=5)
        response.raise_for_status()
        manifest = response.json()
        return {
            "success": True,
            "manifest": manifest,
            "error": None
        }
    except requests.RequestException as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        return {
            "success": False,
            "manifest": None,
            "error": str(e)
        }

def test_tool(host: str, port: int, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test a specific MCP tool"""
    base_url = f"http://{host}:{port}/mcp"
    
    try:
        args = args or {}
        response = requests.post(
            f"{base_url}/tools/{tool_name}",
            json=args,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "result": result,
            "error": None
        }
    except requests.RequestException as e:
        logger.error(f"Failed to call tool {tool_name}: {e}")
        return {
            "success": False,
            "result": None,
            "error": str(e)
        }

def print_separator():
    """Print a separator line"""
    print("=" * 80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test MCP server connection")
    parser.add_argument("--host", type=str, default="localhost",
                      help="Host to connect to (default: localhost)")
    parser.add_argument("--port", type=int, default=8002,
                      help="Port to connect to (default: 8002)")
    parser.add_argument("--test-tools", action="store_true",
                      help="Test all available tools")
    parser.add_argument("--tool", type=str,
                      help="Test a specific tool")
    parser.add_argument("--args", type=str, default="{}",
                      help="JSON-encoded arguments for the tool")
    args = parser.parse_args()
    
    print_separator()
    print(f"Testing MCP Server at {args.host}:{args.port}")
    print_separator()
    
    # Test server connection
    print("Testing server connection...")
    result = test_server_connection(args.host, args.port)
    
    if not result["success"]:
        print(f"Failed to connect to MCP server: {result['error']}")
        return 1
    
    manifest = result["manifest"]
    print(f"Connected to MCP server")
    print(f"Server: {manifest.get('server_name', 'Unknown')} v{manifest.get('version', '?')}")
    print(f"MCP Version: {manifest.get('mcp_version', '?')}")
    
    # Print available tools
    tools = manifest.get("tools", {})
    if tools:
        print(f"\nAvailable Tools ({len(tools)}):")
        for tool_name, tool_info in tools.items():
            print(f"  - {tool_name}: {tool_info.get('description', '')}")
    else:
        print("\nWarning: No tools available on the server")
    
    # Test all tools if requested
    if args.test_tools:
        print("\nTesting all tools:")
        success_count = 0
        for tool_name in tools:
            print(f"\n> Testing tool: {tool_name}")
            tool_result = test_tool(args.host, args.port, tool_name)
            if tool_result["success"]:
                print(f"Result: {json.dumps(tool_result['result'], indent=2)}")
                success_count += 1
            else:
                print(f"Error: {tool_result['error']}")
        
        print(f"\nTools test summary: {success_count}/{len(tools)} tools working")
    
    # Test specific tool if requested
    if args.tool:
        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError as e:
            print(f"Error parsing tool arguments: {e}")
            return 1
        
        print(f"\nTesting specific tool: {args.tool}")
        print(f"Arguments: {json.dumps(tool_args, indent=2)}")
        
        tool_result = test_tool(args.host, args.port, args.tool, tool_args)
        if tool_result["success"]:
            print(f"Result: {json.dumps(tool_result['result'], indent=2)}")
        else:
            print(f"Error: {tool_result['error']}")
    
    print_separator()
    return 0

if __name__ == "__main__":
    sys.exit(main())
