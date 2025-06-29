#!/usr/bin/env python3
"""
Test script for IPFS Accelerate MCP server tools.

This script connects to a running MCP server and tests the available tools.
It provides detailed diagnostics to help identify and fix any issues.
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_server_connection(host: str, port: int) -> bool:
    """Test connection to the MCP server"""
    url = f"http://{host}:{port}/mcp"
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Successfully connected to MCP server at {url}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to connect to MCP server at {url}: {str(e)}")
        return False

def get_manifest(host: str, port: int) -> Dict[str, Any]:
    """Get the MCP server manifest"""
    url = f"http://{host}:{port}/mcp/manifest"
    try:
        response = requests.get(url)
        response.raise_for_status()
        manifest = response.json()
        logger.info(f"Retrieved manifest from MCP server")
        return manifest
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Failed to get manifest: {str(e)}")
        return {}

def list_tools(manifest: Dict[str, Any]) -> List[str]:
    """List all tools in the manifest"""
    tools = manifest.get("tools", {})
    logger.info(f"Found {len(tools)} tools in manifest")
    
    for name, info in tools.items():
        description = info.get("description", "No description")
        logger.info(f"Tool: {name} - {description}")
    
    return list(tools.keys())

def test_tool(host: str, port: int, tool_name: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test a specific MCP tool"""
    if args is None:
        args = {}
    
    url = f"http://{host}:{port}/mcp/tools/{tool_name}"
    try:
        logger.info(f"Testing tool {tool_name} with args: {args}")
        response = requests.post(url, json=args)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Tool {tool_name} returned successfully")
        return result
    except requests.RequestException as e:
        logger.error(f"Error calling tool {tool_name}: {str(e)}")
        if hasattr(e, "response") and e.response:
            try:
                error_details = e.response.json()
                logger.error(f"Server error details: {json.dumps(error_details, indent=2)}")
            except:
                logger.error(f"Server response: {e.response.text}")
        return {"error": str(e)}

def test_get_hardware_info(host: str, port: int) -> None:
    """Specifically test the get_hardware_info tool"""
    logger.info("Testing get_hardware_info tool...")
    result = test_tool(host, port, "get_hardware_info")
    if "error" not in result:
        logger.info("get_hardware_info succeeded!")
        
        # Print system info
        if "system" in result:
            system = result["system"]
            print("\nSystem Information:")
            for key, value in system.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        # Print accelerator info
        if "accelerators" in result:
            accelerators = result["accelerators"]
            print("\nHardware Accelerators:")
            for name, info in accelerators.items():
                available = info.get("available", False)
                print(f"\n  {name.upper()}: {'Available' if available else 'Not Available'}")
                if available:
                    for key, value in info.items():
                        if key != "available":
                            print(f"    {key}: {value}")
    else:
        logger.error("get_hardware_info failed")

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Test IPFS Accelerate MCP server tools")
    parser.add_argument("--host", type=str, default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=8002, help="MCP server port")
    parser.add_argument("--tool", type=str, default=None, help="Specific tool to test")
    parser.add_argument("--args", type=str, default="{}", help="JSON args for tool")
    args = parser.parse_args()
    
    # Test server connection
    if not test_server_connection(args.host, args.port):
        logger.error("Failed to connect to MCP server")
        return 1
    
    # Get manifest
    manifest = get_manifest(args.host, args.port)
    if not manifest:
        logger.error("Failed to get manifest")
        return 1
    
    # Print server info
    print(f"Server: {manifest.get('server_name', 'Unknown')} v{manifest.get('version', '?')}")
    print(f"MCP Version: {manifest.get('mcp_version', '?')}")
    
    # List all tools
    tool_names = list_tools(manifest)
    if not tool_names:
        logger.warning("No tools found in manifest")
    
    # Test specific tool if requested
    if args.tool:
        if args.tool in tool_names:
            try:
                tool_args = json.loads(args.args)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing tool arguments: {e}")
                return 1
            
            result = test_tool(args.host, args.port, args.tool, tool_args)
            print(f"\nTool {args.tool} result:")
            print(json.dumps(result, indent=2))
        else:
            logger.error(f"Tool '{args.tool}' not found in manifest")
            return 1
    else:
        # Test the get_hardware_info tool by default
        test_get_hardware_info(args.host, args.port)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
