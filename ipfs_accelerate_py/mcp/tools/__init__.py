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
        # Always register hardware tools (supports both Standalone and FastMCP styles)
        from ipfs_accelerate_py.mcp.tools.hardware import register_hardware_tools
        register_hardware_tools(mcp)

        # Register model tools (search, recommendations, details)
        try:
            from ipfs_accelerate_py.mcp.tools.models import register_model_tools
            register_model_tools(mcp)
            logger.debug("Registered model tools")
        except Exception as e:
            logger.warning(f"Model tools not registered: {e}")

        # If FastMCP-style decorators are available, register decorator-based tool modules
        if hasattr(mcp, "tool"):
            try:
                from ipfs_accelerate_py.mcp.tools.inference import register_tools as register_inference_tools
                register_inference_tools(mcp)
                logger.debug("Registered inference tools")
            except Exception as e:
                logger.warning(f"Inference tools not registered: {e}")

            try:
                from ipfs_accelerate_py.mcp.tools.endpoints import register_tools as register_endpoint_tools
                register_endpoint_tools(mcp)
                logger.debug("Registered endpoint tools")
            except Exception as e:
                logger.warning(f"Endpoint tools not registered: {e}")

            try:
                from ipfs_accelerate_py.mcp.tools.status import register_tools as register_status_tools
                register_status_tools(mcp)
                logger.debug("Registered status tools")
            except Exception as e:
                logger.warning(f"Status tools not registered: {e}")
        else:
            logger.warning("FastMCP decorators not available; only hardware and model tools registered in standalone mode")

        logger.debug("All tools registered with MCP server")

    except Exception as e:
        logger.error(f"Error registering tools with MCP server: {e}")
        raise
