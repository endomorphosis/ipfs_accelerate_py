"""
Integration module for main.py

This module provides the necessary code to integrate the MCP server
with the main FastAPI application in main.py.
"""
import logging
from fastapi import FastAPI
from ipfs_accelerate_py import ipfs_accelerate_py 
from mcp.fastapi_integration import integrate_mcp_with_fastapi

logger = logging.getLogger("ipfs_accelerate_mcp.integration_main")

def add_mcp_to_main_app(app: FastAPI, model_server) -> None:
    """
    Add MCP server functionality to the main FastAPI app.
    
    Args:
        app: The FastAPI application instance
        model_server: The model server instance
    """
    logger.info("Integrating MCP server with main FastAPI application")
    
    # Call the integration function
    integrate_mcp_with_fastapi(app, model_server)
