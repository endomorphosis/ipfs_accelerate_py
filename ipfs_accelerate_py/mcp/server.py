"""
IPFS Accelerate MCP Server

This module provides the MCP server for IPFS Accelerate.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp.server")

class IPFSAccelerateMCPServer:
    """
    IPFS Accelerate MCP Server
    
    This class provides a Model Context Protocol server for IPFS Accelerate.
    """
    
    def __init__(
        self,
        name: str = "ipfs-accelerate",
        host: str = "localhost",
        port: int = 8000,
        mount_path: str = "/mcp",
        debug: bool = False
    ):
        """
        Initialize the IPFS Accelerate MCP Server
        
        Args:
            name: Name of the server
            host: Host to bind the server to
            port: Port to bind the server to
            mount_path: Path to mount the server at
            debug: Enable debug logging
        """
        self.name = name
        self.host = host
        self.port = port
        self.mount_path = mount_path
        self.debug = debug
        
        # Configure logging
        if debug:
            logging.getLogger("ipfs_accelerate_mcp").setLevel(logging.DEBUG)
        
        # Set up server attributes
        self.mcp = None
        self.fastapi_app = None
        self.server_url = f"http://{host}:{port}{mount_path}"
        
        logger.debug(f"Initialized IPFS Accelerate MCP Server: {self.server_url}")
    
    def setup(self) -> None:
        """
        Set up the MCP server
        
        This function sets up the MCP server with all tools and resources.
        """
        logger.info(f"Setting up IPFS Accelerate MCP Server: {self.name}")
        
        try:
            # Import FastMCP
            from fastmcp import FastMCP
            
            # Create FastMCP instance
            self.mcp = FastMCP(name=self.name)
            
            # Register tools
            self._register_tools()
            
            # Register resources
            self._register_resources()
            
            # Create FastAPI app
            self.fastapi_app = self.mcp.create_fastapi_app(
                title="IPFS Accelerate MCP API",
                description="API for the IPFS Accelerate MCP Server",
                version="0.1.0",
                docs_url="/docs",
                redoc_url="/redoc",
                mount_path=self.mount_path
            )
            
            logger.info(f"IPFS Accelerate MCP Server set up: {self.server_url}")
        
        except ImportError:
            logger.error("Failed to import FastMCP. Please install with 'pip install fastmcp'.")
            raise
        
        except Exception as e:
            logger.error(f"Error setting up MCP server: {e}")
            raise
    
    def run(self) -> None:
        """
        Run the MCP server
        
        This function runs the MCP server using uvicorn.
        """
        if self.fastapi_app is None:
            self.setup()
        
        logger.info(f"Running IPFS Accelerate MCP Server at {self.server_url}")
        
        try:
            import uvicorn
            
            # Run the server
            uvicorn.run(
                self.fastapi_app,
                host=self.host,
                port=self.port,
                log_level="debug" if self.debug else "info"
            )
        
        except ImportError:
            logger.error("Failed to import uvicorn. Please install with 'pip install uvicorn'.")
            raise
        
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise
    
    def _register_tools(self) -> None:
        """
        Register tools with the MCP server
        
        This function registers all tools with the MCP server.
        """
        logger.debug("Registering tools with MCP server")
        
        try:
            # Import tools
            from ipfs_accelerate_py.mcp.tools import register_all_tools
            
            # Register tools
            register_all_tools(self.mcp)
            
            logger.debug("Tools registered with MCP server")
        
        except Exception as e:
            logger.error(f"Error registering tools with MCP server: {e}")
            raise
    
    def _register_resources(self) -> None:
        """
        Register resources with the MCP server
        
        This function registers all resources with the MCP server.
        """
        logger.debug("Registering resources with MCP server")
        
        try:
            # Import resources
            from ipfs_accelerate_py.mcp.resources import register_all_resources
            
            # Register resources
            register_all_resources(self.mcp)
            
            logger.debug("Resources registered with MCP server")
        
        except Exception as e:
            logger.error(f"Error registering resources with MCP server: {e}")
            raise

def main() -> None:
    """
    Main entry point for the IPFS Accelerate MCP Server
    
    This function parses command-line arguments and runs the server.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Server")
    
    parser.add_argument("--name", default="ipfs-accelerate", help="Name of the server")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--mount-path", default="/mcp", help="Path to mount the server at")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Create server
    server = IPFSAccelerateMCPServer(
        name=args.name,
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
        debug=args.debug
    )
    
    # Run server
    try:
        server.run()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
    
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
