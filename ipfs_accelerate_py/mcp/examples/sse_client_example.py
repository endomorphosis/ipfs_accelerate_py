#!/usr/bin/env python
"""
IPFS Accelerate MCP Client Example with SSE Transport

This example demonstrates how to connect to an IPFS Accelerate MCP server
using Server-Sent Events (SSE) transport.
"""

import sys
import asyncio
import json
import logging
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_mcp_client")

# Import requests with SSE support
try:
    import requests
    import sseclient
except ImportError:
    logger.error("Please install required packages: pip install requests sseclient-py")
    sys.exit(1)

class IPFSAccelerateMCPClient:
    """
    Client for the IPFS Accelerate MCP Server using SSE transport
    """
    
    def __init__(self, base_url="http://localhost:8000", mount_path="/mcp"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the MCP server
            mount_path: Path where the MCP server is mounted
        """
        self.base_url = base_url
        self.mount_path = mount_path
        self.sse_url = urljoin(base_url, f"{mount_path}/sse")
        self.api_url = urljoin(base_url, mount_path)
        
        logger.info(f"Initialized MCP client with SSE URL: {self.sse_url}")
    
    def connect_sse(self):
        """
        Connect to the SSE endpoint
        
        Returns:
            SSE client
        """
        try:
            logger.info(f"Connecting to SSE endpoint: {self.sse_url}")
            response = requests.get(self.sse_url, stream=True)
            
            if response.status_code != 200:
                logger.error(f"Failed to connect to SSE endpoint: {response.status_code}")
                if response.status_code == 404:
                    logger.error("404 Not Found: Make sure the server is running with --transport sse")
                return None
            
            return sseclient.SSEClient(response)
        
        except Exception as e:
            logger.error(f"Error connecting to SSE endpoint: {e}")
            return None
    
    def list_tools(self):
        """
        List available tools
        
        Returns:
            List of available tools
        """
        try:
            url = urljoin(self.api_url, "tools")
            response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to list tools: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return None
    
    def call_tool(self, tool_name, tool_input=None):
        """
        Call a tool
        
        Args:
            tool_name: Name of the tool
            tool_input: Input for the tool
            
        Returns:
            Tool output
        """
        if tool_input is None:
            tool_input = {}
        
        try:
            url = urljoin(self.api_url, f"tool/{tool_name}")
            response = requests.post(url, json=tool_input)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to call tool {tool_name}: {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return None
    
    def get_resource(self, resource_uri):
        """
        Get a resource
        
        Args:
            resource_uri: URI of the resource
            
        Returns:
            Resource content
        """
        try:
            url = urljoin(self.api_url, f"resource/{resource_uri}")
            response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get resource {resource_uri}: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting resource {resource_uri}: {e}")
            return None

async def main():
    """
    Main function
    """
    # Initialize client
    client = IPFSAccelerateMCPClient(
        base_url="http://localhost:8000",  # Change this to match your server
        mount_path="/mcp"                  # Change this to match your server
    )
    
    # List tools
    logger.info("Listing available tools...")
    tools = client.list_tools()
    
    if tools:
        logger.info(f"Available tools: {json.dumps(tools, indent=2)}")
    
    # Get hardware info
    logger.info("Getting hardware info...")
    hardware_info = client.call_tool("get_hardware_info")
    
    if hardware_info:
        logger.info(f"Hardware info: {json.dumps(hardware_info, indent=2)}")
    
    # Connect to SSE endpoint
    logger.info("Connecting to SSE endpoint...")
    sse_client = client.connect_sse()
    
    if sse_client:
        logger.info("Connected to SSE endpoint. Listening for events...")
        
        # Listen for events for 10 seconds
        start_time = asyncio.get_event_loop().time()
        
        try:
            for event in sse_client:
                # Process event
                logger.info(f"Received event: {event.event}")
                logger.info(f"Event data: {event.data}")
                
                # Check if we've been listening for more than 10 seconds
                if asyncio.get_event_loop().time() - start_time > 10:
                    logger.info("Listening timeout reached (10 seconds)")
                    break
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping client...")
        
        except Exception as e:
            logger.error(f"Error processing SSE events: {e}")
        
        finally:
            # Close SSE client
            sse_client.close()
            logger.info("SSE client closed")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
