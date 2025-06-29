#!/usr/bin/env python3
"""
IPFS Accelerate MCP Server

This script runs an MCP server specifically for the ipfs-accelerate-py configuration,
using port 8000 as expected by Cline.
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the IPFS Accelerate MCP server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--implementation", choices=["direct", "fastmcp"], default="direct",
                        help="Server implementation to use (direct=Flask, fastmcp=FastMCP)")
    return parser.parse_args()

def run_direct_server():
    """Run the direct Flask-based server implementation."""
    try:
        # Try to import Flask
        import flask
        from flask import Flask, Response
        
        # If we got here, Flask is installed, so run the direct server
        logger.info("Using direct Flask-based MCP server implementation")
        
        # Run the server with the correct port
        subprocess.run([
            sys.executable, 
            "direct_mcp_server.py", 
            "--port", "8000", 
            "--host", "127.0.0.1"
        ])
    except ImportError:
        logger.error("Flask not installed. Please install with: pip install flask flask-cors")
        sys.exit(1)

def run_fastmcp_server():
    """Run the FastMCP-based server implementation."""
    try:
        # Check if FastMCP is available
        import fastmcp
        
        # If we got here, FastMCP is installed, so run the FastMCP server
        logger.info("Using FastMCP-based server implementation")
        
        # Run the server with the correct port
        subprocess.run([
            sys.executable, 
            "run_mcp.py", 
            "--port", "8000", 
            "--host", "127.0.0.1",
            "--name", "ipfs-accelerate-py",
            "--transport", "sse"
        ])
    except ImportError:
        logger.error("FastMCP not installed. Falling back to direct implementation...")
        run_direct_server()

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Print server information
    logger.info("Starting IPFS Accelerate MCP Server")
    logger.info("Server will be available at: http://localhost:8000/sse")
    
    # Run the selected implementation
    if args.implementation == "direct":
        run_direct_server()
    else:
        run_fastmcp_server()

if __name__ == "__main__":
    main()
