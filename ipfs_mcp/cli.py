#!/usr/bin/env python3
"""
IPFS Accelerate MCP Server CLI

This script provides a command-line interface for starting and managing
the IPFS Accelerate MCP server.
"""
import argparse
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp_cli")

def main():
    """Main entry point for the MCP server CLI."""
    parser = argparse.ArgumentParser(
        description="IPFS Accelerate Model Context Protocol (MCP) Server"
    )
    
    # Define command-line arguments
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host address to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--dev", 
        action="store_true", 
        help="Run in development mode with auto-reload"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Import MCP components (import here to avoid circular imports)
    from ipfs_accelerate_py import ipfs_accelerate_py
    from mcp.server import create_mcp_server
    
    try:
        # Create IPFS Accelerate instance
        logger.info("Initializing IPFS Accelerate...")
        accelerate = ipfs_accelerate_py()
        
        # Create MCP server
        logger.info("Creating MCP server...")
        mcp_server = create_mcp_server(accelerate_instance=accelerate)
        
        # Start the server
        logger.info(f"Starting MCP server on {args.host}:{args.port}...")
        if args.dev:
            logger.info("Running in development mode with auto-reload enabled")
            mcp_server.run(host=args.host, port=args.port, reload=True)
        else:
            mcp_server.run(host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Error starting MCP server: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
