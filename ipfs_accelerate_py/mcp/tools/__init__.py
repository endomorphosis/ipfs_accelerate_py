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
        from .hardware import register_hardware_tools
        register_hardware_tools(mcp)

        # Register model tools (search, recommendations, details)
        try:
            from .models import register_model_tools
            register_model_tools(mcp)
            logger.debug("Registered model tools")
        except Exception as e:
            logger.warning(f"Model tools not registered: {e}")

        # If FastMCP-style decorators are available, register decorator-based tool modules
        if hasattr(mcp, "tool"):
            try:
                from .inference import register_tools as register_inference_tools
                register_inference_tools(mcp)
                logger.debug("Registered inference tools")
            except Exception as e:
                logger.warning(f"Inference tools not registered: {e}")

            try:
                from .endpoints import register_tools as register_endpoint_tools
                register_endpoint_tools(mcp)
                logger.debug("Registered endpoint tools")
            except Exception as e:
                logger.warning(f"Endpoint tools not registered: {e}")

            try:
                from .status import register_tools as register_status_tools
                register_status_tools(mcp)
                logger.debug("Registered status tools")
            except Exception as e:
                logger.warning(f"Status tools not registered: {e}")
            
            try:
                from .workflows import register_tools as register_workflow_tools
                register_workflow_tools(mcp)
                logger.debug("Registered workflow tools")
            except Exception as e:
                logger.warning(f"Workflow tools not registered: {e}")
            
            try:
                from .dashboard_data import register_tools as register_dashboard_tools
                register_dashboard_tools(mcp)
                logger.debug("Registered dashboard data tools")
            except Exception as e:
                logger.warning(f"Dashboard data tools not registered: {e}")
            
            try:
                from .github_tools import register_tools as register_github_tools
                register_github_tools(mcp)
                logger.debug("Registered GitHub CLI tools")
            except Exception as e:
                logger.warning(f"GitHub CLI tools not registered: {e}")
        else:
            logger.warning("FastMCP decorators not available; only hardware and model tools registered in standalone mode")

        logger.debug("All tools registered with MCP server")

    except Exception as e:
        logger.error(f"Error registering tools with MCP server: {e}")
        raise
