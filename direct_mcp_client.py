#!/usr/bin/env python
"""
Direct MCP Client for IPFS Accelerate

This client is designed to work with the direct_mcp_server.py implementation.
"""

import json
import logging
import requests
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DirectMCPClient:
    """Client for interacting with the Direct MCP server"""
    
    def __init__(self, host: str = "localhost", port: int = 3000):
        """
        Initialize the client
        
        Args:
            host: Host of the MCP server
            port: Port of the MCP server
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
    
    def is_server_available(self) -> bool:
        """
        Check if the server is available
        
        Returns:
            bool: True if server is available, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/tools")
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_tools(self) -> List[str]:
        """
        Get available tools
        
        Returns:
            List[str]: List of available tool names
        
        Raises:
            requests.RequestException: If request fails
        """
        response = self.session.get(f"{self.base_url}/tools")
        response.raise_for_status()
        data = response.json()
        return data.get("tools", [])
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on the server
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
        
        Returns:
            Dict[str, Any]: Tool result
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        payload = {
            "tool_name": tool_name,
            "arguments": kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/call_tool",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware information
        
        Returns:
            Dict[str, Any]: Hardware information
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        return self.call_tool("get_hardware_info")
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """
        Get detailed hardware capabilities
        
        Returns:
            Dict[str, Any]: Hardware capabilities
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        return self.call_tool("get_hardware_capabilities")

if __name__ == "__main__":
    # When run directly, print hardware info
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct MCP Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=3000, help="Server port")
    
    args = parser.parse_args()
    
    # Create client
    client = DirectMCPClient(host=args.host, port=args.port)
    
    # Check if server is available
    if not client.is_server_available():
        print(f"MCP server is not running at {args.host}:{args.port}")
        exit(1)
    
    # Get available tools
    try:
        tools = client.get_available_tools()
        print(f"Available tools: {tools}")
    except Exception as e:
        print(f"Error getting available tools: {e}")
    
    # Get hardware info
    try:
        hardware_info = client.get_hardware_info()
        print("\nHardware Info:")
        print(json.dumps(hardware_info, indent=2))
    except Exception as e:
        print(f"Error getting hardware info: {e}")
    
    # Get hardware capabilities
    try:
        capabilities = client.get_hardware_capabilities()
        print("\nHardware Capabilities:")
        print(json.dumps(capabilities, indent=2))
    except Exception as e:
        print(f"Error getting hardware capabilities: {e}")
