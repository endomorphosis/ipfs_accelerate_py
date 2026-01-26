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


def _is_pytest() -> bool:
    return (
        os.environ.get("PYTEST_CURRENT_TEST") is not None
        or os.environ.get("PYTEST") is not None
        or "pytest" in sys.modules
    )


def _log_optional_dependency(message: str) -> None:
    if _is_pytest():
        logger.info(message)
    else:
        logger.warning(message)

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
    _log_optional_dependency("FastMCP import failed, falling back to mock implementation")
    _log_optional_dependency("Using mock MCP implementation")

# Check for ipfs_kit_py (tolerate any import error)
try:
    import ipfs_kit_py  # noqa: F401
    ipfs_kit_available = True
    logger.info("IPFS Kit available")
except Exception as e:
    ipfs_kit_available = False
    _log_optional_dependency(f"IPFS Kit not available or failed to import ({e!s}); some functionality will be limited")

from mcp.types import IPFSAccelerateContext
from mcp.server import create_ipfs_mcp_server, run_server

# Initialize the package
logger.info("IPFS Accelerate MCP package initialized")


def get_version_str() -> str:
    """Get the package version as a string (legacy helper)."""
    return __version__


# Compatibility re-exports for the nested implementation used by the modern MCP
# server + the unit tests under ipfs_accelerate_py/mcp/tests.
try:
    _inner_mcp = importlib.import_module("ipfs_accelerate_py.ipfs_accelerate_py.mcp")

    create_server = getattr(_inner_mcp, "create_server")
    register_components = getattr(_inner_mcp, "register_components")
    start_server_thread = getattr(_inner_mcp, "start_server_thread")
    stop_server = getattr(_inner_mcp, "stop_server")
    get_server_info = getattr(_inner_mcp, "get_server_info")
    create_and_start_server = getattr(_inner_mcp, "create_and_start_server")
    get_version = getattr(_inner_mcp, "get_version")
    start_server = getattr(_inner_mcp, "start_server")
    check_dependencies = getattr(_inner_mcp, "check_dependencies")

except Exception as e:
    logger.warning(f"Nested MCP package not available ({e!s}); legacy exports will be limited")


__all__ = [
    "FastMCP",
    "Context",
    "IPFSAccelerateContext",
    "create_ipfs_mcp_server",
    "run_server",
    "get_version_str",
    # nested/compat
    "create_server",
    "register_components",
    "start_server_thread",
    "stop_server",
    "get_server_info",
    "create_and_start_server",
    "get_version",
    "start_server",
    "check_dependencies",
]


# Ensure the mock_mcp module is loaded when needed
if not fastmcp_available:
    try:
        import mcp.mock_mcp
    except ImportError:
        logger.error("Failed to import mock_mcp module")
