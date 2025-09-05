#!/usr/bin/env python3
"""
Run the IPFS Accelerate MCP server.

This script serves as the main entry point for starting the IPFS Accelerate MCP server.
It parses command-line arguments and runs the server with the specified configuration.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp")

# Import the server module
try:
    from mcp.server import run_server
except ImportError as e:
    logger.error(f"Error importing MCP server: {str(e)}")
    logger.error("Make sure the IPFS Accelerate MCP package is installed or in your PYTHONPATH")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the IPFS Accelerate MCP server")
    
    # Server configuration
    parser.add_argument("--name", default="IPFS Accelerate MCP", help="Server name")
    parser.add_argument("--description", default="MCP server for IPFS Accelerate", help="Server description")
    
    # Transport configuration
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], 
                        help="Transport type (stdio or sse)")
    parser.add_argument("--host", default="127.0.0.1", 
                        help="Host to bind to for network transports (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to bind to for network transports (default: 8000)")
    
    # Logging configuration
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()


async def main() -> None:
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Log startup information
    logger.info(f"Starting IPFS Accelerate MCP server: {args.name}")
    logger.info(f"Transport: {args.transport}, Host: {args.host}, Port: {args.port}")
    
    try:
        # Run the server
        await run_server(
            name=args.name,
            description=args.description,
            transport=args.transport,
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        sys.exit(1)
    

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
