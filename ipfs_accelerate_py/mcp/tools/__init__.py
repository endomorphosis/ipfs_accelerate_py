"""
IPFS Accelerate MCP Tools

This package provides tools for the IPFS Accelerate MCP server.
"""

import logging
from typing import Any

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.tools")

def register_all_tools(mcp: Any) -> None:
    """
    Register all tools with the MCP server
    
    This function registers all tools with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering all tools with MCP server")
    
    try:
        # Import tools
        from ipfs_accelerate_py.mcp.tools.hardware import register_hardware_tools
        
        # Register tools
        register_hardware_tools(mcp)
        
        logger.debug("All tools registered with MCP server")
    
    except Exception as e:
        logger.error(f"Error registering tools with MCP server: {e}")
        raise
