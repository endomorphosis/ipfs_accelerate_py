"""
MCP Integration with FastAPI Main Application

This module provides integration code to connect the MCP server with the
main FastAPI application in main.py.
"""
import logging
from fastapi import FastAPI
from ipfs_accelerate_py import ipfs_accelerate_py
from ipfs_accelerate_py.mcp.server import create_mcp_server

logger = logging.getLogger("ipfs_accelerate_mcp.fastapi_integration")

def integrate_mcp_with_fastapi(app: FastAPI, model_server) -> None:
    """
    Integrate the MCP server with the main FastAPI application.
    
    Args:
        app: The FastAPI application instance
        model_server: The model server instance containing the ipfs_accelerate_py instance
    """
    # Get access to the ipfs_accelerate_py instance
    accelerate_instance = model_server.resources.get("ipfs_accelerate_py")
    
    if accelerate_instance is None:
        logger.error("Cannot integrate MCP: ipfs_accelerate_py instance not found in model_server.resources")
        return
    
    # Create the MCP server
    logger.info("Creating MCP server for FastAPI integration")
    # When mounting as a sub-application, avoid internally prefixing routes with
    # the mount path (prevents /mcp/mcp/... route duplication).
    mcp_server = create_mcp_server(accelerate_instance=accelerate_instance, mount_path="")
    
    # Mount the MCP application to the FastAPI app
    app.mount("/mcp", mcp_server.app, name="mcp_server")
    
    # Log successful integration
    logger.info("MCP server successfully integrated with FastAPI application at /mcp endpoint")
    
    # Register MCP server with the model_server instance for reference
    model_server.resources["mcp_server"] = mcp_server
