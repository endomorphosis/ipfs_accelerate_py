#!/usr/bin/env python3
"""
Integration test for IPFS Accelerate MCP Server FastAPI Integration

This script tests the integration between the MCP server and FastAPI.
"""

import os
import sys
import logging
import requests
import uvicorn
import argparse
from fastapi import FastAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_fastapi_test")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test IPFS Accelerate MCP server integration with FastAPI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--mcp-path", default="/mcp", help="Path where MCP server is mounted (default: /mcp)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("ipfs_accelerate_mcp").setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "="*70)
    print(f"IPFS Accelerate MCP + FastAPI Integration Test")
    print("="*70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"MCP Path: {args.mcp_path}")
    print("="*70 + "\n")
    
    try:
        # Import IPFS Accelerate MCP package
        from ipfs_accelerate_py.mcp import integrate_with_fastapi
        
        # Create FastAPI app
        app = FastAPI(
            title="IPFS Accelerate MCP Test App",
            description="Test application for IPFS Accelerate MCP integration"
        )
        
        # Define root endpoint
        @app.get("/")
        async def root():
            return {
                "status": "ok",
                "message": "IPFS Accelerate MCP + FastAPI Integration Test",
                "mcp_path": args.mcp_path
            }
        
        # Integrate MCP server with FastAPI
        logger.info("Integrating MCP server with FastAPI...")
        mcp_server = integrate_with_fastapi(app, mount_path=args.mcp_path)
        
        # Log MCP server info
        logger.info(f"MCP Server Name: {getattr(mcp_server, 'name', 'unknown')}")
        logger.info(f"MCP Server Tools: {len(getattr(mcp_server, 'tools', []))}")
        logger.info(f"MCP Server Resources: {len(getattr(mcp_server, 'resources', []))}")
        
        # Run the server
        logger.info(f"Starting FastAPI server on {args.host}:{args.port}...")
        uvicorn.run(app, host=args.host, port=args.port)
        
    except ImportError as e:
        logger.error(f"Import Error: {e}")
        logger.error("Make sure IPFS Accelerate MCP package is installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
