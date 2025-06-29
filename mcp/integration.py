#!/usr/bin/env python
"""
IPFS Accelerate MCP Integration

This module provides utilities for integrating the MCP server with IPFS Accelerate.
"""

import os
import sys
import json
import logging
import importlib
from typing import Dict, Any, List, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def integrate_with_ipfs_accelerate():
    """
    Integrate MCP with IPFS Accelerate
    
    This function sets up MCP integration with the main IPFS Accelerate package
    by registering tools and resources that expose IPFS Accelerate functionality.
    """
    try:
        # Import MCP server components
        from .server import register_tool, register_resource
        
        # Register IPFS Accelerate tools
        register_ipfs_tools(register_tool)
        
        # Register IPFS Accelerate resources
        register_ipfs_resources(register_resource)
        
        logger.info("Successfully integrated MCP with IPFS Accelerate")
        return True
    
    except ImportError as e:
        logger.error(f"Could not import MCP server components: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Error integrating MCP with IPFS Accelerate: {e}")
        return False

def register_ipfs_tools(register_tool_func: Callable):
    """
    Register IPFS Accelerate-related tools with MCP
    
    Args:
        register_tool_func: Function to register a tool with MCP
    """
    # Import hardware detection tools
    from .tools.hardware import get_hardware_info
    register_tool_func(
        name="get_hardware_info",
        description="Get information about the hardware available for acceleration",
        function=get_hardware_info,
    )
    
    # Try to import and register IPFS tools
    try:
        from .tools.ipfs_files import get_file_from_ipfs, add_file_to_ipfs
        
        register_tool_func(
            name="get_file_from_ipfs",
            description="Get a file from IPFS by its CID",
            function=get_file_from_ipfs,
        )
        
        register_tool_func(
            name="add_file_to_ipfs",
            description="Add a file to IPFS",
            function=add_file_to_ipfs,
        )
    except ImportError:
        logger.warning("IPFS file tools not available")
    
    # Register acceleration tools if available
    try:
        from .tools.acceleration import get_acceleration_options, accelerate_model
        
        register_tool_func(
            name="get_acceleration_options",
            description="Get available acceleration options for a model",
            function=get_acceleration_options,
        )
        
        register_tool_func(
            name="accelerate_model",
            description="Accelerate a model using WebGPU or WebNN",
            function=accelerate_model,
        )
    except ImportError:
        logger.warning("Acceleration tools not available")

def register_ipfs_resources(register_resource_func: Callable):
    """
    Register IPFS Accelerate-related resources with MCP
    
    Args:
        register_resource_func: Function to register a resource with MCP
    """
    # Register system information resource
    try:
        from .resources.system_info import get_system_info
        
        register_resource_func(
            name="system_info",
            description="Information about the system",
            getter=get_system_info,
        )
    except ImportError:
        logger.warning("System information resource not available")
    
    # Register model information resource
    try:
        from .resources.model_info import get_model_info
        
        register_resource_func(
            name="model_info",
            description="Information about available models",
            getter=get_model_info,
        )
    except ImportError:
        logger.warning("Model information resource not available")
    
    # Register hardware resource
    try:
        from .tools.hardware import get_hardware_info
        
        def get_accelerator_info():
            """Get accelerator information"""
            hardware_info = get_hardware_info()
            return hardware_info.get("accelerators", {})
        
        register_resource_func(
            name="accelerator_info",
            description="Information about available hardware accelerators",
            getter=get_accelerator_info,
        )
    except ImportError:
        logger.warning("Hardware information resource not available")

def initialize_mcp_server(start_server: bool = False) -> Tuple[bool, Optional[int]]:
    """
    Initialize the MCP server
    
    Args:
        start_server: Whether to start the server
    
    Returns:
        Tuple[bool, Optional[int]]: Success status and port (if started)
    """
    # Integrate with IPFS Accelerate
    integration_success = integrate_with_ipfs_accelerate()
    
    if not integration_success:
        logger.warning("Could not integrate MCP with IPFS Accelerate")
    
    # Start server if requested
    if start_server:
        try:
            from .client import start_server as start_mcp_server
            success, port = start_mcp_server(find_port=True, wait=2)
            return success, port
        except ImportError:
            logger.error("Could not import MCP client to start server")
            return False, None
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            return False, None
    
    return True, None

if __name__ == "__main__":
    # When run directly, integrate and start the server
    success, port = initialize_mcp_server(start_server=True)
    
    if success and port:
        print(f"MCP server started on port {port}")
    else:
        print("Failed to start MCP server")
        sys.exit(1)
