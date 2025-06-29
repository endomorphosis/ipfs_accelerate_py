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

# Try to import server components with fallbacks
try:
    # Try to import FastMCP
    import fastmcp
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    logger.warning("FastMCP not found. Using mock implementation.")

try:
    # Try to import server module
    from mcp.server import create_ipfs_mcp_server
    
    async def run_server(
        name: str = "direct-ipfs-kit-mcp",
        description: str = "MCP server for IPFS Accelerate",
        transport: str = "sse",
        host: str = "127.0.0.1",
        port: int = 3000,
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
        mcp_server = create_ipfs_mcp_server(name, description)
        
        # Log tools availability
        tool_count = len(getattr(mcp_server, 'tools', {}))
        logger.info(f"Server initialized with {tool_count} tools")
        
        try:
            # Run the server with the specified transport
            if transport == "sse":
                try:
                    from fastmcp.transports import sse
                    logger.info(f"Starting SSE server on http://{host}:{port}/sse")
                    await sse.run_server(mcp_server, host=host, port=port)
                except ImportError:
                    # Fallback for mock implementation
                    logger.info("Using mock SSE implementation")
                    await mcp_server.run(transport=transport, host=host, port=port)
            else:
                # Use default transport
                await mcp_server.run(transport=transport, host=host, port=port)
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        except Exception as e:
            logger.error(f"Error running server: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("Server shutdown complete")
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
    parser.add_argument("--name", default="direct-ipfs-kit-mcp", help="Server name")
    parser.add_argument("--description", default="MCP server for IPFS Accelerate", help="Server description")
    
    # Transport configuration
    parser.add_argument("--transport", default="sse", choices=["stdio", "sse"], 
                        help="Transport type (stdio or sse)")
    parser.add_argument("--host", default="127.0.0.1", 
                        help="Host to bind to for network transports (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=3000, 
                        help="Port to bind to for network transports (default: 3000)")
    
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
