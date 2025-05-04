#!/usr/bin/env python3
"""
IPFS Accelerate MCP Standalone Server

This module provides a standalone MCP server for IPFS Accelerate.
"""

import os
import sys
import signal
import argparse
import logging
import time
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp.standalone")

def run_server(
    host: str = "localhost",
    port: int = 8080,
    name: str = "ipfs-accelerate",
    description: str = "IPFS Accelerate MCP Server",
    verbose: bool = False
) -> None:
    """
    Run a standalone MCP server
    
    This function creates and starts an MCP server, and keeps it running
    until interrupted.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        name: Name of the server
        description: Description of the server
        verbose: Enable verbose logging
    """
    logger.info(f"Starting standalone MCP server on {host}:{port}")
    
    try:
        # Import MCP modules
        from ipfs_accelerate_py.mcp.server import create_server, register_components, start_server
        
        # Create server
        mcp = create_server(
            host=host,
            port=port,
            name=name,
            description=description,
            verbose=verbose
        )
        
        # Register components
        register_components(mcp)
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal, stopping server...")
            try:
                mcp.stop()
                logger.info("Server stopped")
            except:
                pass
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server (this will block until the server is stopped)
        logger.info(f"Server listening on http://{host}:{port}")
        start_server(mcp, wait=True)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error running standalone MCP server: {e}")
        sys.exit(1)

def run_fastapi_server(
    host: str = "localhost",
    port: int = 8000,
    mount_path: str = "/mcp",
    name: str = "ipfs-accelerate",
    description: str = "IPFS Accelerate MCP Server",
    verbose: bool = False
) -> None:
    """
    Run a standalone FastAPI server with MCP integration
    
    This function creates a FastAPI application with MCP integration and runs it.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        mount_path: Path to mount the MCP server at
        name: Name of the MCP server
        description: Description of the MCP server
        verbose: Enable verbose logging
    """
    logger.info(f"Starting FastAPI server with MCP integration on {host}:{port}")
    
    try:
        # Import FastAPI integration
        from ipfs_accelerate_py.mcp.integration import create_standalone_app, run_standalone_app
        
        # Create standalone FastAPI application
        app = create_standalone_app(
            mount_path=mount_path,
            name=name,
            description=description,
            verbose=verbose
        )
        
        # Run standalone FastAPI application
        logger.info(f"Server listening on http://{host}:{port}")
        run_standalone_app(
            app=app,
            host=host,
            port=port,
            verbose=verbose
        )
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error running FastAPI server with MCP integration: {e}")
        sys.exit(1)

def main() -> None:
    """
    Main entry point for the standalone MCP server
    
    This function parses command-line arguments and runs the server.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Standalone Server")
    
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--name", default="ipfs-accelerate", help="Name of the MCP server")
    parser.add_argument("--description", default="IPFS Accelerate MCP Server", help="Description of the MCP server")
    parser.add_argument("--fastapi", action="store_true", help="Use FastAPI integration")
    parser.add_argument("--mount-path", default="/mcp", help="Path to mount the MCP server at (for FastAPI)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the appropriate server
    if args.fastapi:
        run_fastapi_server(
            host=args.host,
            port=args.port,
            mount_path=args.mount_path,
            name=args.name,
            description=args.description,
            verbose=args.verbose
        )
    else:
        run_server(
            host=args.host,
            port=args.port,
            name=args.name,
            description=args.description,
            verbose=args.verbose
        )

if __name__ == "__main__":
    main()
