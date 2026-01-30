"""
MCP Server Integration with FastAPI

This module provides functionality to integrate the MCP server with
the main FastAPI application of IPFS Accelerate.
"""
import logging
from typing import Optional
import os
from fastapi import FastAPI, APIRouter, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse

from ipfs_accelerate_py import ipfs_accelerate_py
from ipfs_accelerate_py.mcp.server import create_mcp_server, get_mcp_server_instance

logger = logging.getLogger("ipfs_accelerate_mcp.integration")

def initialize_mcp_server(app: FastAPI, accelerate_instance: ipfs_accelerate_py) -> None:
    """
    Initialize the MCP server and integrate it with the FastAPI application.
    
    Args:
        app: The FastAPI application instance
        accelerate_instance: The ipfs_accelerate_py instance
    """
    # Create the MCP server with our IPFS Accelerate instance
    mcp_server = create_mcp_server(
        name="IPFS Accelerate MCP",
        description="Hardware-accelerated machine learning inference with IPFS integration",
        accelerate_instance=accelerate_instance
    )
    
    # Create a router for MCP endpoints
    mcp_router = APIRouter(prefix="/mcp", tags=["MCP Server"])
    
    # Mount the MCP server's ASGI application to our router
    # We need to create a catch-all route for all methods
    @mcp_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def handle_mcp_request(request: Request, path: str):
        """
        Handle MCP requests by forwarding them to the MCP server.
        
        This is an ASGI-to-ASGI bridge that passes requests from FastAPI to our MCP server.
        
        Args:
            request: The FastAPI request
            path: The path component of the request
        
        Returns:
            The response from the MCP server
        """
        # Get the FastMCP ASGI application
        mcp_app = mcp_server.app
        
        # Create a new scope with the adjusted path
        scope = dict(request.scope)
        scope["path"] = f"/{path}"  # Adjust path to remove our prefix
        
        # Create response holder
        response_holder = {"response": None}
        
        # Define async response callback
        async def send(message):
            if message.get("type") == "http.response.start":
                response_holder["status"] = message.get("status")
                response_holder["headers"] = message.get("headers")
            elif message.get("type") == "http.response.body":
                response_holder["body"] = message.get("body", b"")
        
        # Call the MCP ASGI application
        await mcp_app(scope, request.receive, send)
        
        # Create a response
        if "status" in response_holder:
            return Response(
                content=response_holder.get("body", b""),
                status_code=response_holder.get("status", 200),
                headers=dict(response_holder.get("headers", []))
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to process MCP request"}
            )
    
    # Include the router in the main app
    app.include_router(mcp_router)
    
    # Log the successful integration
    logger.info("MCP server integrated with FastAPI application")
    
    # Add a health check endpoint
    @mcp_router.get("/health")
    async def mcp_health():
        """Health check endpoint for the MCP server."""
        return {
            "status": "healthy",
            "server": mcp_server.name,
            "version": mcp_server.metadata.get("version", "unknown")
        }
