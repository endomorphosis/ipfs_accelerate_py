#!/usr/bin/env python
"""
IPFS Accelerate MCP Server Runner

This script provides a convenient way to run the MCP server.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="MCP Server Runner")
    parser.add_argument("--port", type=int, default=8002,
                      help="Port to bind to (default: 8002)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true",
                      help="Run in debug mode")
    parser.add_argument("--log-level", type=str, default="info",
                      help="Log level (debug, info, warning, error, critical)")
    parser.add_argument("--find-port", action="store_true",
                      help="Find an available port if the specified port is in use")
    args = parser.parse_args()
    
    try:
        # Add parent directory to path if needed for direct script execution
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Check if we need to find an available port
        if args.find_port:
            try:
                from mcp.client import is_port_in_use, find_available_port
                
                if is_port_in_use(args.port):
                    new_port = find_available_port(start_port=args.port)
                    logger.info(f"Port {args.port} is in use, using port {new_port} instead")
                    args.port = new_port
            except ImportError:
                logger.warning("Could not import port checking functions, will try to use specified port")
        
        # Import server module and start the server
        from mcp.server import start_server
        
        # Start the server
        logger.info(f"Starting MCP server on {args.host}:{args.port}")
        start_server(
            host=args.host,
            port=args.port,
            debug=args.debug,
            log_level=args.log_level,
        )
        
        return 0
    except ImportError as e:
        logger.error(f"Error importing MCP server: {e}")
        logger.error("Make sure you've installed the required packages (pip install -r requirements.txt)")
        return 1
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
