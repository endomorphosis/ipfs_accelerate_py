#!/usr/bin/env python3
"""
IPFS Accelerate MCP Integration Example.

This example demonstrates how to integrate with and use the IPFS Accelerate MCP
server from a Python client application.
"""

import anyio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_example")

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import MCP client (this would typically be from a library)
# For this example, we'll use a simplified mock client
class SimpleMCPClient:
    """Simple mock MCP client for demonstration purposes."""
    
    def __init__(self, server_name: str):
        """Initialize the client.
        
        Args:
            server_name: Name of the MCP server to connect to
        """
        self.server_name = server_name
        logger.info(f"Connecting to MCP server: {server_name}")
        # In a real client, we'd establish a connection here
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        logger.info(f"Calling tool: {tool_name} with args: {kwargs}")
        
        # In a real client, we'd send a request to the MCP server and await the response
        # For this example, we'll just simulate some API calls
        
        if tool_name == "ipfs_add_file":
            # Simulate adding a file to IPFS
            path = kwargs.get("path", "")
            if not path:
                raise ValueError("Missing required argument: path")
            
            # Check if file exists
            if not os.path.exists(path):
                return {
                    "error": "File not found",
                    "path": path,
                    "success": False
                }
            
            # Simulate successful response
            return {
                "cid": "QmSimulatedCidForDemonstrationPurposes12345",
                "size": os.path.getsize(path),
                "name": os.path.basename(path),
                "success": True
            }
        
        elif tool_name == "ipfs_cat":
            # Simulate retrieving content from IPFS
            cid = kwargs.get("cid", "")
            if not cid:
                raise ValueError("Missing required argument: cid")
            
            # Simulate content for demonstration
            return f"Simulated content for CID: {cid}"
        
        elif tool_name == "ipfs_files_write":
            # Simulate writing content to the IPFS MFS
            path = kwargs.get("path", "")
            content = kwargs.get("content", "")
            if not path:
                raise ValueError("Missing required argument: path")
            if not content:
                raise ValueError("Missing required argument: content")
            
            # Simulate successful response
            return {
                "path": path,
                "written": True,
                "size": len(content),
                "cid": "QmSimulatedCidForMfsWrite12345"
            }
        
        # Add more tool simulations as needed
        
        # Default response for unknown tools
        return {
            "error": f"Unknown tool: {tool_name}",
            "success": False
        }


async def basic_file_operations_example(client: SimpleMCPClient) -> None:
    """Demonstrate basic IPFS file operations using the MCP client.
    
    Args:
        client: The MCP client to use
    """
    logger.info("=== Basic File Operations Example ===")
    
    # Create a temporary file for testing
    temp_file_path = "example_temp_file.txt"
    with open(temp_file_path, "w") as f:
        f.write("This is a test file for IPFS MCP example.")
    
    try:
        # Add the file to IPFS
        logger.info("Adding file to IPFS...")
        add_result = await client.call_tool("ipfs_add_file", path=temp_file_path)
        
        if add_result.get("success", False):
            logger.info(f"File added successfully: {add_result}")
            cid = add_result["cid"]
            
            # Retrieve the file content
            logger.info(f"Retrieving content for CID: {cid}")
            content = await client.call_tool("ipfs_cat", cid=cid)
            logger.info(f"Retrieved content: {content}")
            
            # Write to MFS
            logger.info("Writing to IPFS MFS...")
            mfs_path = "/ipfs_accelerate_example/test.txt"
            write_result = await client.call_tool(
                "ipfs_files_write", 
                path=mfs_path, 
                content="This is content written to MFS for testing."
            )
            logger.info(f"MFS write result: {write_result}")
        else:
            logger.error(f"Failed to add file: {add_result}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Removed temporary file: {temp_file_path}")


async def model_acceleration_example(client: SimpleMCPClient) -> None:
    """Demonstrate model acceleration using the MCP client.
    
    Args:
        client: The MCP client to use
    """
    logger.info("=== Model Acceleration Example ===")
    
    # This is a simulated example
    model_cid = "QmExampleModelCid123456789"
    
    # Get hardware information
    logger.info("Getting hardware information...")
    hw_info = await client.call_tool("get_hardware_info")
    logger.info(f"Hardware info: {hw_info}")
    
    # Accelerate the model
    logger.info(f"Accelerating model with CID: {model_cid}")
    acceleration_result = await client.call_tool(
        "accelerate_model",
        model_cid=model_cid,
        target_device="cpu",  # or "gpu", "webgpu", etc.
        optimization_level="medium"
    )
    
    logger.info(f"Acceleration result: {acceleration_result}")
    
    # In a real application, we'd do something with the accelerated model here


async def main() -> None:
    """Run the MCP integration examples."""
    # Create MCP client
    client = SimpleMCPClient("direct-ipfs-kit-mcp")
    
    try:
        # Run examples
        await basic_file_operations_example(client)
        await model_acceleration_example(client)
        
        logger.info("All examples completed successfully")
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    anyio.run(main())
