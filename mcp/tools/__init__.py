"""
Tools package for IPFS Accelerate MCP.

This package provides tools that can be used by LLMs through the MCP interface.
Tools are organized by functionality into separate modules.
"""

import importlib
import logging
import sys
from typing import Any, Dict, List, Optional, Set, cast

# Configure logging
logger = logging.getLogger("mcp.tools")

# Try imports with fallbacks
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        # Fall back to mock implementation
        from mcp.mock_mcp import FastMCP

import os

# Try to import ipfs_kit_py (tolerate any import error) and allow disabling via env
try:
    if os.environ.get("MCP_DISABLE_IPFS", "0") == "1":
        raise ImportError("IPFS disabled by environment")

    import ipfs_kit_py  # noqa: F401
    # Check if IPFSApi is available
    try:
        from ipfs_kit_py import IPFSApi  # type: ignore
        ipfs_api_available = True
    except Exception:
        logger.warning("IPFSApi not available in ipfs_kit_py; falling back to mock")
        ipfs_api_available = False
    
    ipfs_kit_available = True
except Exception as e:
    ipfs_kit_available = False
    ipfs_api_available = False
    logger.warning(f"ipfs_kit_py not available or failed to import ({e!s}). Using mock implementations.")

# Import the mock IPFS client
from mcp.tools.mock_ipfs import MockIPFSClient


def register_all_tools(mcp: FastMCP) -> None:
    """Register all tools with the MCP server.
    
    Args:
        mcp: The MCP server to register tools with
    """
    logger.info("Registering all tools")
    
    # List of tool modules to import
    tool_modules = [
        "acceleration",
        "ipfs_files", 
        "ipfs_network",
        "shared_tools",
        "github_tools",
        "copilot_tools"
    ]
    
    # Track registered tool modules
    registered_modules = set()
    
    # Check if ipfs_kit_py is available
    if not ipfs_kit_available:
        logger.warning("Error importing tool modules: ipfs_kit_py not available")
    
    # Register tools from each module
    for module_name in tool_modules:
        try:
            logger.info(f"Importing module: {module_name}")
            
            # Import the module
            module = importlib.import_module(f"mcp.tools.{module_name}")
            
            # Register tools from the module
            if module_name == "shared_tools":
                register_function = getattr(module, "register_shared_tools", None)
            elif module_name == "github_tools":
                register_function = getattr(module, "register_github_tools", None)
            elif module_name == "copilot_tools":
                register_function = getattr(module, "register_copilot_tools", None)
            else:
                register_function = getattr(module, f"register_{module_name.replace('ipfs_', '')}_tools", None)
                
            if register_function:
                register_function(mcp)
                registered_modules.add(module_name)
            else:
                logger.warning(f"No registration function found in module {module_name}")
        except Exception as e:
            logger.error(f"Error registering tools from module {module_name}: {str(e)}")
    
    logger.info("Tool registration complete")
    
    # If no modules were registered, log a warning
    if not registered_modules:
        logger.warning("No tool modules were registered")


def get_ipfs_client() -> Any:
    """Get an IPFS client instance.
    
    This function returns either a real IPFS client from ipfs_kit_py if available,
    or a mock client if ipfs_kit_py is not available.
    
    Returns:
        An IPFS client instance
    """
    if ipfs_api_available:
        try:
            # Try to create a real IPFS client
            return ipfs_kit_py.IPFSApi()
        except Exception as e:
            logger.warning(f"Error creating IPFSApi: {str(e)}")
    
    # Fall back to mock implementation
    return MockIPFSClient()
