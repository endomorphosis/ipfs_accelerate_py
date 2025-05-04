#!/usr/bin/env python
"""
Standalone MCP server test

This script creates a simple MCP server without relying on the ipfs_mcp structure.
"""
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("standalone_mcp_test")

# Try to create a simple MCP server
try:
    from fastmcp import FastMCP
    
    # Create a basic MCP server
    mcp = FastMCP("Test MCP Server")
    
    # Add a simple tool
    @mcp.tool()
    def hello(name: str = "World") -> str:
        """Say hello to someone"""
        return f"Hello, {name}!"
    
    # Add a simple resource
    @mcp.resource("test://greeting")
    def get_greeting() -> str:
        """Get a greeting message"""
        return "Welcome to the test MCP server!"
    
    # Print server info
    logger.info(f"MCP Server: {mcp.name}")
    logger.info(f"Number of tools: {len(mcp.tools)}")
    for tool in mcp.tools:
        logger.info(f"Tool: {tool.name}")
    
    logger.info(f"Number of resources: {len(mcp.resources)}")
    for resource in mcp.resources:
        logger.info(f"Resource: {resource.path}")
    
    # Test the tool
    result = mcp.tools[0].function(name="Tester")
    logger.info(f"Tool result: {result}")
    
    # Test the resource
    result = mcp.resources[0].function()
    logger.info(f"Resource result: {result}")
    
    logger.info("MCP server test completed successfully!")
    
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
