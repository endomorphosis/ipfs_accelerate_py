"""
Tools for IPFS Accelerate MCP server.

This package contains tool implementations that expose IPFS operations
to LLM clients through the MCP server.
"""

import importlib
import sys
import os
from typing import Any

__all__ = ["register_all_tools"]


def register_all_tools(mcp: Any) -> None:
    """Register all available tools with the MCP server.
    
    This function dynamically imports and registers all tool modules,
    making it easier to add new tool modules without changing this file.
    
    Args:
        mcp: The FastMCP server instance
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Helper function to import and register tools from a module
    def import_and_register(module_name: str) -> None:
        try:
            # Try relative import first
            module_path = f"mcp.tools.{module_name}"
            module = importlib.import_module(module_path)
            
            # Look for register_*_tools function
            register_func_name = f"register_{module_name}_tools"
            if hasattr(module, register_func_name):
                register_func = getattr(module, register_func_name)
                register_func(mcp)
                print(f"Registered tools from {module_path}")
        except (ImportError, AttributeError) as e:
            # Try direct import if relative import fails
            try:
                sys.path.insert(0, current_dir)
                module = importlib.import_module(module_name)
                register_func_name = f"register_{module_name}_tools"
                if hasattr(module, register_func_name):
                    register_func = getattr(module, register_func_name)
                    register_func(mcp)
                    print(f"Registered tools from {module_name} (direct import)")
                sys.path.pop(0)
            except (ImportError, AttributeError) as e2:
                print(f"Could not import tools from {module_name}: {e2}")
    
    # Register tools from each module
    import_and_register("ipfs_files")
    import_and_register("ipfs_network")
    import_and_register("acceleration")
