"""
IPFS Accelerate MCP - Model Context Protocol integration for IPFS Accelerate

This package provides an MCP server implementation that exposes IPFS Accelerate
functionality to LLMs, allowing them to interact with IPFS operations and hardware
acceleration.
"""

import importlib
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

# Version information
__version__ = "0.1.0"  # Initial version

# Explicitly import and expose key components
try:
    # Try to import FastMCP if available
    from fastmcp import FastMCP, Context
    fastmcp_available = True
    logger.info("Using real FastMCP implementation")
except ImportError:
    # Fall back to mock implementation if FastMCP is not available
    from mcp.mock_mcp import FastMCP, Context
    fastmcp_available = False
    logger.warning("FastMCP import failed, falling back to mock implementation")
    logger.warning("Using mock MCP implementation")

# Check for ipfs_kit_py (tolerate any import error)
try:
    import ipfs_kit_py  # noqa: F401
    ipfs_kit_available = True
    logger.info("IPFS Kit available")
except Exception as e:
    ipfs_kit_available = False
    logger.warning(f"IPFS Kit not available or failed to import ({e!s}); some functionality will be limited")

# Import key modules for easy access
from mcp.types import IPFSAccelerateContext
from mcp.server import create_ipfs_mcp_server, run_server

# Initialize the package
logger.info("IPFS Accelerate MCP package initialized")


def get_version() -> str:
    """Get the package version."""
    return __version__


# Ensure the mock_mcp module is loaded when needed
if not fastmcp_available:
    try:
        import mcp.mock_mcp
    except ImportError:
        logger.error("Failed to import mock_mcp module")
