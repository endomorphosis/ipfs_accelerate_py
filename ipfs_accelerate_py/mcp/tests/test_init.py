#!/usr/bin/env python3
"""
Basic initialization test for IPFS Accelerate MCP.

This script tests basic MCP server initialization without running a server.
"""
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ipfs_mcp_init_test")

# Add the parent directory to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    # Import the required modules
    logger.info("Importing ipfs_accelerate_py...")
    from ipfs_accelerate_py import ipfs_accelerate_py
    
    logger.info("Creating accelerate instance...")
    accelerate = ipfs_accelerate_py()
    
    logger.info("Importing MCP server...")
    from ipfs_accelerate_py.mcp.server import create_mcp_server
    
    logger.info("Creating MCP server...")
    mcp_server = create_mcp_server(
        name="Test MCP Server",
        description="MCP Server for basic testing",
        accelerate_instance=accelerate
    )
    
    # Print basic information
    logger.info(f"MCP Server Name: {mcp_server.name}")
    logger.info(f"MCP Server Description: {mcp_server.description}")
    
    # Check for tools
    logger.info(f"Number of tools: {len(mcp_server.tools)}")
    for tool in mcp_server.tools:
        logger.info(f"Tool: {tool.name}")
    
    # Check for resources
    logger.info(f"Number of resources: {len(mcp_server.resources)}")
    for resource in mcp_server.resources:
        logger.info(f"Resource: {resource.path}")
    
    # Check for prompts
    logger.info(f"Number of prompts: {len(mcp_server.prompts)}")
    for prompt in mcp_server.prompts:
        logger.info(f"Prompt: {prompt.name}")
    
    logger.info("MCP initialization test completed successfully!")

except Exception as e:
    logger.error(f"Error during MCP initialization: {e}")
    import traceback
    traceback.print_exc()
