"""
IPFS Accelerate MCP Resources

This module registers all resources with the MCP server.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Union

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.resources")

def register_all_resources(mcp: Any) -> None:
    """
    Register all resources with the MCP server
    
    This function registers all resources with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering all resources")
    
    try:
        # Import config resources
        from ipfs_accelerate_py.mcp.resources.config import register_config_resources
        
        # Register config resources
        register_config_resources(mcp)
        
        # Import model info resources
        from ipfs_accelerate_py.mcp.resources.model_info import register_model_info_resources
        
        # Register model info resources
        register_model_info_resources(mcp)
        
        logger.debug("All resources registered")
    
    except Exception as e:
        logger.error(f"Error registering resources: {e}")
        raise
