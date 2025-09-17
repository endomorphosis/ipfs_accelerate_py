"""
IPFS Accelerate MCP

This package provides a Model Context Protocol (MCP) integration for IPFS Accelerate.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Union

# Package version
__version__ = "0.1.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp")

# Import for external use
try:
    # Ensure minimal deps if allowed
    from ipfs_accelerate_py.utils.auto_install import ensure_packages
    ensure_packages({
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "fastmcp": "fastmcp",
    })
except Exception:
    pass

from ipfs_accelerate_py.mcp.server import IPFSAccelerateMCPServer

def create_server(
    name: str = "ipfs-accelerate",
    host: str = "localhost",
    port: int = 8000,
    mount_path: str = "/mcp",
    debug: bool = False
) -> IPFSAccelerateMCPServer:
    """
    Create a new IPFS Accelerate MCP server
    
    Args:
        name: Name of the server
        host: Host to bind the server to
        port: Port to bind the server to
        mount_path: Path to mount the server at
        debug: Enable debug logging
        
    Returns:
        IPFS Accelerate MCP server instance
    """
    # Create server
    server = IPFSAccelerateMCPServer(
        name=name,
        host=host,
        port=port,
        mount_path=mount_path,
        debug=debug
    )
    
    # Return server instance
    return server

def start_server(
    name: str = "ipfs-accelerate",
    host: str = "localhost",
    port: int = 8000,
    mount_path: str = "/mcp",
    debug: bool = False
) -> None:
    """
    Create and start a new IPFS Accelerate MCP server
    
    Args:
        name: Name of the server
        host: Host to bind the server to
        port: Port to bind the server to
        mount_path: Path to mount the server at
        debug: Enable debug logging
    """
    # Create server
    server = create_server(
        name=name,
        host=host,
        port=port,
        mount_path=mount_path,
        debug=debug
    )
    
    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        raise

# Check required dependencies
def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are installed
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        "fastmcp": False,
        "uvicorn": False,
        "psutil": False,
        "numpy": False,
        "torch": False
    }
    
    # Check dependencies
    try:
        import fastmcp
        dependencies["fastmcp"] = True
    except ImportError:
        pass
    
    try:
        import uvicorn
        dependencies["uvicorn"] = True
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies["psutil"] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    return dependencies

# Perform dependency check
dependencies = check_dependencies()

# Log missing dependencies
missing_dependencies = [dep for dep, installed in dependencies.items() if not installed]
if missing_dependencies:
    logger.warning(f"Missing dependencies: {', '.join(missing_dependencies)}")
    logger.warning("Some features may not be available.")
    logger.warning("Install all dependencies with: pip install fastmcp uvicorn psutil numpy torch")
