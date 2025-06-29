#!/usr/bin/env python
"""
IPFS Accelerate MCP Client Example with WebSockets Transport

This example demonstrates how to connect to an IPFS Accelerate MCP server
using WebSockets transport for bidirectional communication.
"""

import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_mcp_client")

# Import WebSockets
try:
    import websockets
except ImportError:
    logger.error("Please install required packages: pip install websockets")
    sys.exit(1)

class IPFSAccelerateMCPWebSocketClient:
    """
    Client for the IPFS Accelerate MCP Server using WebSockets transport
    """
    
    def __init__(self, base_url="ws://localhost:8000", mount_path="/mcp"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the MCP server (ws:// or wss://)
            mount_path: Path where the MCP server is mounted
        """
        self.base_url = base_url
        self.mount_path = mount_path
        self.ws_url = f"{base_url.rstrip('/')}{mount_path}/ws"
        self.websocket = None
        
        logger.info(f"Initialized MCP client with WebSocket URL: {self.ws_url}")
    
    async def connect(self):
        """
        Connect to the WebSocket endpoint
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket endpoint: {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            logger.info("Connected to WebSocket endpoint")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to WebSocket endpoint: {e}")
            return False
    
    async def disconnect(self):
        """
        Disconnect from the WebSocket endpoint
        """
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from WebSocket endpoint")
            self.websocket = None
    
    async def ping(self):
        """
        Send a ping message to the server
        
        Returns:
            Response from the server
        """
        if not self.websocket:
            logger.error("Not connected to WebSocket endpoint")
            return None
        
        try:
            # Send ping message
            ping_message = {
                "type": "ping"
            }
            
            await self.websocket.send(json.dumps(ping_message))
            
            # Wait for response
            response = await self.websocket.recv()
            return json.loads(response)
        
        except Exception as e:
            logger.error(f"Error sending ping message: {e}")
            return None
    
    async def call_tool(self, tool_name: str, tool_input: Dict[str, Any] = None):
        """
        Call a tool on the server
        
        Args:
            tool_name: Name of the tool to call
            tool_input: Input parameters for the tool
            
        Returns:
            Tool result
        """
        if not self.websocket:
            logger.error("Not connected to WebSocket endpoint")
            return None
        
        if tool_input is None:
            tool_input = {}
        
        try:
            # Send tool call message
            tool_message = {
                "type": "call_tool",
                "tool_name": tool_name,
                "input": tool_input
            }
            
            await self.websocket.send(json.dumps(tool_message))
            
            # Wait for response
            response = await self.websocket.recv()
            return json.loads(response)
        
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return None
    
    async def get_resource(self, resource_uri: str):
        """
        Get a resource from the server
        
        Args:
            resource_uri: URI of the resource to get
            
        Returns:
            Resource content
        """
        if not self.websocket:
            logger.error("Not connected to WebSocket endpoint")
            return None
        
        try:
            # Send resource request message
            resource_message = {
                "type": "get_resource",
                "resource_uri": resource_uri
            }
            
            await self.websocket.send(json.dumps(resource_message))
            
            # Wait for response
            response = await self.websocket.recv()
            return json.loads(response)
        
        except Exception as e:
            logger.error(f"Error getting resource {resource_uri}: {e}")
            return None

async def main():
    """
    Main function
    """
    # Create client
    client = IPFSAccelerateMCPWebSocketClient(
        base_url="ws://localhost:8000",  # Change to match your server
        mount_path="/mcp"
    )
    
    # Connect to server
    if not await client.connect():
        logger.error("Failed to connect to WebSocket endpoint")
        return
    
    try:
        # Ping the server
        logger.info("Sending ping to server...")
        ping_response = await client.ping()
        
        if ping_response:
            logger.info(f"Ping response: {json.dumps(ping_response, indent=2)}")
        
        # Get hardware info
        logger.info("Getting hardware info...")
        hardware_info = await client.call_tool("get_hardware_info")
        
        if hardware_info:
            logger.info(f"Hardware info: {json.dumps(hardware_info, indent=2)}")
        
        # Test IPFS tools if available
        try:
            # Get IPFS node info
            logger.info("Getting IPFS node info...")
            ipfs_info = await client.call_tool("ipfs_node_info")
            
            if ipfs_info:
                logger.info(f"IPFS node info: {json.dumps(ipfs_info, indent=2)}")
                
                # Get gateway URL for a test hash
                test_hash = "QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D"  # IPFS readme
                gateway_url = await client.call_tool("ipfs_gateway_url", {"ipfs_hash": test_hash})
                
                if gateway_url:
                    logger.info(f"Gateway URL for {test_hash}: {gateway_url}")
        
        except Exception as e:
            logger.warning(f"IPFS tools test failed: {e}")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    
    finally:
        # Disconnect
        await client.disconnect()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
