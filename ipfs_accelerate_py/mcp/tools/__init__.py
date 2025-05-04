"""
IPFS Accelerate MCP Tools

This module registers all tools with the MCP server.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Union

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.tools")

def register_all_tools(mcp: Any) -> None:
    """
    Register all tools with the MCP server
    
    This function registers all tools with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering all tools")
    
    try:
        # Import hardware tools
        from ipfs_accelerate_py.mcp.tools.hardware import register_hardware_tools
        
        # Register hardware tools
        register_hardware_tools(mcp)
        
        logger.debug("All tools registered")
    
    except Exception as e:
        logger.error(f"Error registering tools: {e}")
        raise
