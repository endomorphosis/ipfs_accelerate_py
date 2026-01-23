"""
IPFS Accelerate MCP server implementation.

This module provides functions for creating and running an MCP server
that exposes IPFS Accelerate functionality.
"""

import argparse
import anyio
import logging
import os
import signal
import sys
from typing import Any, Dict, List, Optional, Union, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Backward-compatible singleton access (used by tests and older integrations)
_mcp_server_instance: Optional["FastMCP"] = None

# Best-effort ensure minimal deps when allowed
try:
    from ipfs_accelerate_py.utils.auto_install import ensure_packages
    ensure_packages({
        "fastmcp": "fastmcp",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
    })
except Exception:
    pass

# Try imports with fallbacks
try:
    # Try to import FastMCP if available
    from fastmcp import FastMCP, Context
    fastmcp_available = True
except ImportError:
    # Fall back to mock implementation if FastMCP is not available
    from mcp.mock_mcp import FastMCP, Context
    fastmcp_available = False
    logger.warning("FastMCP import failed, falling back to mock implementation")

# Import the IPFS context
from mcp.types import IPFSAccelerateContext

# Try to import ipfs_kit_py (be tolerant to any import error)
try:
    import ipfs_kit_py
    from ipfs_kit_py import IPFSApi
    ipfs_available = True
except Exception as e:
    ipfs_available = False
    logger.warning(f"ipfs_kit_py not available or failed to import ({e!s}); some functionality will be limited")

# Import error reporting
try:
    from utils.error_reporter import get_error_reporter, install_global_exception_handler
    error_reporting_available = True
except Exception as e:
    error_reporting_available = False
    logger.warning(f"Error reporting not available: {e}")


def create_ipfs_mcp_server(name: str, description: str = "") -> FastMCP:
    """Create a new IPFS Accelerate MCP server.
    
    Args:
        name: Server name
        description: Server description
        
    Returns:
        The MCP server instance
    """
    mcp_server = FastMCP(name=name, description=description or f"IPFS Accelerate MCP: {name}")
    logger.info(f"Created MCP server: {name}")
    
    # Install global exception handler for automatic error reporting
    if error_reporting_available:
        install_global_exception_handler(source_component='mcp-server')
    
    # Set up lifespan handlers
    @mcp_server.on_lifespan_start()
    async def on_start(ctx: Context) -> IPFSAccelerateContext:
        """Initialize resources when the server starts."""
        logger.info("MCP server starting...")
        
        # Create IPFS Accelerate context for sharing state
        ipfs_context = IPFSAccelerateContext()
        
        # Initialize IPFS client if available
        if ipfs_available:
            try:
                # Create IPFS client
                ipfs_client = ipfs_kit_py.IPFSApi()
                ipfs_context.set_ipfs_client(ipfs_client)
                
                # Test connection
                version = await anyio.to_thread.run_sync(ipfs_client.version)
                await ctx.info(f"Connected to IPFS: {version.get('Version', 'unknown')}")
            except Exception as e:
                await ctx.error(f"Error initializing IPFS client: {str(e)}")
                # Report error if error reporting is available
                if error_reporting_available:
                    get_error_reporter().report_error(
                        exception=e,
                        source_component='mcp-server',
                        context={'operation': 'ipfs_client_initialization'}
                    )
                # Continue without IPFS client
        else:
            # Using mock implementation
            from mcp.tools.mock_ipfs import MockIPFSClient
            mock_client = MockIPFSClient()
            ipfs_context.set_ipfs_client(mock_client)
            await ctx.info("Using mock IPFS client")
        
        return ipfs_context
    
    @mcp_server.on_lifespan_stop()
    async def on_stop(ctx: Context, ipfs_context: IPFSAccelerateContext) -> None:
        """Clean up resources when the server stops."""
        logger.info("MCP server shutting down...")
        
        # Clean up any resources
        # Currently we don't need to do any special cleanup
        await ctx.info("MCP server shutdown complete")
    
    # Return the configured server
    return mcp_server


def create_mcp_server(name: str, description: str = "") -> FastMCP:
    """Backward-compatible alias for creating the MCP server."""
    global _mcp_server_instance
    _mcp_server_instance = create_ipfs_mcp_server(name, description)
    return _mcp_server_instance


def get_mcp_server_instance() -> Optional[FastMCP]:
    """Return the most recently created MCP server instance, if any."""
    return _mcp_server_instance


def register_tools(mcp_server: FastMCP) -> None:
    """Register tools with the MCP server.
    
    Args:
        mcp_server: The MCP server instance
    """
    try:
        # Import the tools module
        from mcp.tools import register_all_tools
        
        # Register all tools
        register_all_tools(mcp_server)
    except Exception as e:
        logger.error(f"Error registering tools: {str(e)}")


def create_and_register(name: str, description: str = "") -> FastMCP:
    """Create an MCP server and register all tools.
    
    Args:
        name: Server name
        description: Server description
        
    Returns:
        The MCP server instance with tools registered
    """
    # Create server
    mcp_server = create_mcp_server(name, description)
    
    # Register tools
    register_tools(mcp_server)
    
    return mcp_server


async def run_server(
    name: str = "IPFS Accelerate MCP",
    description: str = "MCP server for IPFS Accelerate",
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False
) -> None:
    """Run the MCP server.
    
    Args:
        name: Server name
        description: Server description
        transport: Transport type (stdio or sse)
        host: Host to bind to for network transports
        port: Port to bind to for network transports
        debug: Whether to enable debug logging
    """
    # Configure logging based on debug flag
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log server information
    logger.info(f"Starting MCP server: {name}")
    logger.info(f"Transport: {transport}, Host: {host}, Port: {port}")
    
    # Create and configure the server
    mcp_server = create_and_register(name, description)
    
    try:
        # Run the server with the specified transport
        await mcp_server.run(transport=transport, host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        # Report error if error reporting is available
        if error_reporting_available:
            get_error_reporter().report_error(
                exception=e,
                source_component='mcp-server',
                context={'operation': 'server_run', 'transport': transport}
            )
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the IPFS Accelerate MCP server")
    parser.add_argument("--name", default="IPFS Accelerate MCP", help="Server name")
    parser.add_argument("--description", default="", help="Server description")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], help="Transport type")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to for network transports")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to for network transports")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Run the server
    anyio.run(run_server(
        name=args.name,
        description=args.description,
        transport=args.transport,
        host=args.host,
        port=args.port,
        debug=args.debug
    ))
