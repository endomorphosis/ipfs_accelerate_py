#!/usr/bin/env python3
"""
IPFS Accelerate MCP FastAPI Server Runner

This module provides a command-line interface for running a FastAPI server
with IPFS Accelerate MCP integration.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp.run_fastapi_server")

def main() -> None:
    """
    Main entry point for the FastAPI server runner
    
    This function parses command-line arguments and runs the FastAPI server.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP FastAPI Server Runner")
    
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--mount-path", default="/mcp", help="Path to mount the MCP server at")
    parser.add_argument("--name", default="ipfs-accelerate", help="Name of the MCP server")
    parser.add_argument("--description", default="IPFS Accelerate MCP Server", help="Description of the MCP server")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting FastAPI server with MCP integration on {args.host}:{args.port}")
    
    try:
        # Import integration module
        try:
            from ipfs_accelerate_py.mcp.integration import create_standalone_app, run_standalone_app
        except ImportError:
            logger.error("Could not import create_standalone_app from ipfs_accelerate_py.mcp.integration. Please check your installation.")
            sys.exit(1)
        
        # Create standalone FastAPI application
        app = create_standalone_app(
            mount_path=args.mount_path,
            name=args.name,
            description=args.description,
            verbose=args.verbose
        )
        
        # Run standalone FastAPI application
        logger.info(f"Server listening on http://{args.host}:{args.port}")
        logger.info(f"MCP server mounted at {args.mount_path}")
        logger.info(f"API documentation available at http://{args.host}:{args.port}{args.mount_path}/docs")
        
        run_standalone_app(
            app=app,
            host=args.host,
            port=args.port,
            verbose=args.verbose
        )
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error running FastAPI server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
