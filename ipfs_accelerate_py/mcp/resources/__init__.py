"""
IPFS Accelerate MCP Resources

This package provides resources for the IPFS Accelerate MCP server.
"""

import logging
from typing import Any

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.resources")

def register_all_resources(mcp: Any) -> None:
    """
    Register all resources with the MCP server
    
    This function registers all resources with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering all resources with MCP server")
    
    try:
        # Import resources
        from ipfs_accelerate_py.mcp.resources.model_info import register_model_info_resources
        
        # Register resources
        register_model_info_resources(mcp)
        
        logger.debug("All resources registered with MCP server")
    
    except Exception as e:
        logger.error(f"Error registering resources with MCP server: {e}")
        raise
