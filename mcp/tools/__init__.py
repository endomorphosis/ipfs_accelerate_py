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


def _is_pytest() -> bool:
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def _log_optional_dependency(message: str) -> None:
    if _is_pytest():
        logger.info(message)
    else:
        logger.warning(message)

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
    # Check if an IPFS client API is available
    ipfs_client_factory = None
    try:
        from ipfs_kit_py import IPFSApi  # type: ignore
        ipfs_client_factory = IPFSApi
    except Exception:
        try:
            from ipfs_kit_py import IPFSSimpleAPI  # type: ignore
            if IPFSSimpleAPI is not None:
                ipfs_client_factory = IPFSSimpleAPI
        except Exception:
            ipfs_client_factory = None

    if ipfs_client_factory is None:
        try:
            get_high_level_api = getattr(ipfs_kit_py, "get_high_level_api", None)
            if callable(get_high_level_api):
                api_cls, _plugin_base = get_high_level_api()
                if api_cls is not None:
                    ipfs_client_factory = api_cls
        except Exception:
            ipfs_client_factory = None

    if ipfs_client_factory is None:
        try:
            from ipfs_kit_py.ipfs_client import ipfs_py  # type: ignore
            ipfs_client_factory = ipfs_py
        except Exception:
            ipfs_client_factory = None

    if ipfs_client_factory is None:
        _log_optional_dependency("IPFS client API not available in ipfs_kit_py; falling back to mock")
        ipfs_api_available = False
    else:
        ipfs_api_available = True
    
    ipfs_kit_available = True
except Exception as e:
    ipfs_kit_available = False
    ipfs_api_available = False
    ipfs_client_factory = None
    _log_optional_dependency(f"ipfs_kit_py not available or failed to import ({e!s}). Using mock implementations.")

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
        "copilot_tools",
        "copilot_sdk_tools",
        "p2p_workflow_tools"
    ]
    
    # Track registered tool modules
    registered_modules = set()
    
    # Check if ipfs_kit_py is available
    if not ipfs_kit_available:
        _log_optional_dependency("Error importing tool modules: ipfs_kit_py not available")
    
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
            elif module_name == "copilot_sdk_tools":
                register_function = getattr(module, "register_copilot_sdk_tools", None)
            elif module_name == "p2p_workflow_tools":
                register_function = getattr(module, "register_p2p_workflow_tools", None)
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
        _log_optional_dependency("No tool modules were registered")


def get_ipfs_client() -> Any:
    """Get an IPFS client instance.
    
    This function returns either a real IPFS client from ipfs_kit_py if available,
    or a mock client if ipfs_kit_py is not available.
    
    Returns:
        An IPFS client instance
    """
    if ipfs_api_available and ipfs_client_factory is not None:
        try:
            # Try to create a real IPFS client
            return ipfs_client_factory()
        except Exception as e:
            _log_optional_dependency(f"Error creating IPFS client: {str(e)}")
    
    # Fall back to mock implementation
    return MockIPFSClient()
