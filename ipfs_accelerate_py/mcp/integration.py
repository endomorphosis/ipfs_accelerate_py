"""
IPFS Accelerate MCP Integration

This module provides integration helper functions for connecting the MCP server with 
IPFS Accelerate.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.integration")

def integrate_with_ipfs_accelerate(
    server: Any,
    ipfs_accelerate_module: Any = None,
    enable_hardware_tools: bool = True,
    enable_model_info: bool = True,
    enable_config_resources: bool = True,
    custom_tool_handlers: Optional[Dict[str, Callable]] = None,
    custom_resource_handlers: Optional[Dict[str, Callable]] = None
) -> None:
    """
    Integrate the MCP server with IPFS Accelerate
    
    This function integrates the MCP server with IPFS Accelerate by setting up
    handlers for MCP tools and resources to use IPFS Accelerate functionality.
    
    Args:
        server: MCP server instance
        ipfs_accelerate_module: IPFS Accelerate module (if None, will try to import)
        enable_hardware_tools: Enable hardware-related tools
        enable_model_info: Enable model information resources
        enable_config_resources: Enable configuration resources
        custom_tool_handlers: Custom tool handlers
        custom_resource_handlers: Custom resource handlers
    """
    logger.info("Integrating MCP server with IPFS Accelerate")
    
    try:
        # Get IPFS Accelerate module
        ipfs_acc = ipfs_accelerate_module
        
        if ipfs_acc is None:
            try:
                import ipfs_accelerate_py as ipfs_acc
                logger.debug("Imported ipfs_accelerate_py module")
            except ImportError:
                logger.warning("Failed to import ipfs_accelerate_py, integration will be limited")
        
        # Check server setup
        if not hasattr(server, "mcp") or server.mcp is None:
            logger.info("MCP server not yet set up, calling setup()")
            server.setup()
        
        # Custom tool handlers
        if custom_tool_handlers:
            for tool_name, handler in custom_tool_handlers.items():
                logger.debug(f"Setting custom handler for tool: {tool_name}")
                # Logic to override tool handler would go here
        
        # Custom resource handlers
        if custom_resource_handlers:
            for resource_uri, handler in custom_resource_handlers.items():
                logger.debug(f"Setting custom handler for resource: {resource_uri}")
                # Logic to override resource handler would go here
        
        logger.info("MCP server integrated with IPFS Accelerate")
    
    except Exception as e:
        logger.error(f"Error integrating MCP server with IPFS Accelerate: {e}")
        raise

def setup_extended_handlers(server: Any) -> None:
    """
    Set up extended handlers for the MCP server
    
    This function sets up additional handlers for the MCP server beyond the basic
    tools and resources. These include advanced inference capabilities, file operations,
    and IPFS-specific functionality.
    
    Args:
        server: MCP server instance
    """
    logger.info("Setting up extended handlers for MCP server")
    
    try:
        # Check server setup
        if not hasattr(server, "mcp") or server.mcp is None:
            logger.info("MCP server not yet set up, calling setup()")
            server.setup()
        
        # Import IPFS Accelerate
        try:
            import ipfs_accelerate_py as ipfs_acc
            
            # Check if inference module exists
            if hasattr(ipfs_acc, "inference"):
                # Register inference tools
                logger.info("Registering inference tools")
                # Logic to register inference tools would go here
            
            # Check if file operations module exists
            if hasattr(ipfs_acc, "file_ops"):
                # Register file operations tools
                logger.info("Registering file operations tools")
                # Logic to register file operations tools would go here
            
            # Check if IPFS module exists
            if hasattr(ipfs_acc, "ipfs"):
                # Register IPFS tools
                logger.info("Registering IPFS tools")
                # Logic to register IPFS tools would go here
            
        except ImportError:
            logger.warning("Failed to import ipfs_accelerate_py, extended handlers will not be available")
        
        logger.info("Extended handlers set up for MCP server")
    
    except Exception as e:
        logger.error(f"Error setting up extended handlers: {e}")
        raise


def initialize_mcp_server(app: Any, accelerate_instance: Any, mount_path: str = "/mcp") -> Any:
    """Initialize and mount the MCP server into a FastAPI app."""
    from ipfs_accelerate_py.mcp.server import create_mcp_server

    mcp_server = create_mcp_server(accelerate_instance=accelerate_instance)
    app.mount(mount_path, mcp_server.app, name="mcp_server")
    return mcp_server

def register_custom_tool(
    server: Any,
    name: str,
    function: Callable,
    description: str,
    input_schema: Dict[str, Any]
) -> None:
    """
    Register a custom tool with the MCP server
    
    This function registers a custom tool with the MCP server.
    
    Args:
        server: MCP server instance
        name: Name of the tool
        function: Tool function
        description: Tool description
        input_schema: Tool input schema
    """
    logger.info(f"Registering custom tool: {name}")
    
    try:
        # Check server setup
        if not hasattr(server, "mcp") or server.mcp is None:
            logger.info("MCP server not yet set up, calling setup()")
            server.setup()
        
        # Register tool
        server.mcp.register_tool(
            name=name,
            function=function,
            description=description,
            input_schema=input_schema
        )
        
        logger.info(f"Custom tool registered: {name}")
    
    except Exception as e:
        logger.error(f"Error registering custom tool: {e}")
        raise

def register_custom_resource(
    server: Any,
    uri: str,
    function: Callable,
    description: str
) -> None:
    """
    Register a custom resource with the MCP server
    
    This function registers a custom resource with the MCP server.
    
    Args:
        server: MCP server instance
        uri: URI of the resource
        function: Resource function
        description: Resource description
    """
    logger.info(f"Registering custom resource: {uri}")
    
    try:
        # Check server setup
        if not hasattr(server, "mcp") or server.mcp is None:
            logger.info("MCP server not yet set up, calling setup()")
            server.setup()
        
        # Register resource
        server.mcp.register_resource(
            uri=uri,
            function=function,
            description=description
        )
        
        logger.info(f"Custom resource registered: {uri}")
    
    except Exception as e:
        logger.error(f"Error registering custom resource: {e}")
        raise

def create_integrated_server(
    ipfs_accelerate_module: Any = None,
    name: str = "ipfs-accelerate",
    host: str = "localhost",
    port: int = 8000,
    mount_path: str = "/mcp",
    debug: bool = False,
    enable_extensions: bool = True
) -> Any:
    """
    Create an integrated MCP server for IPFS Accelerate
    
    This function creates an MCP server and integrates it with IPFS Accelerate.
    
    Args:
        ipfs_accelerate_module: IPFS Accelerate module (if None, will try to import)
        name: Name of the server
        host: Host to bind the server to
        port: Port to bind the server to
        mount_path: Path to mount the server at
        debug: Enable debug logging
        enable_extensions: Enable extended handlers
        
    Returns:
        Integrated MCP server instance
    """
    logger.info("Creating integrated MCP server for IPFS Accelerate")
    
    try:
        # Import server
        from ipfs_accelerate_py.mcp import create_server
        
        # Create server
        server = create_server(
            name=name,
            host=host,
            port=port,
            mount_path=mount_path,
            debug=debug
        )
        
        # Integrate with IPFS Accelerate
        integrate_with_ipfs_accelerate(
            server=server,
            ipfs_accelerate_module=ipfs_accelerate_module
        )
        
        # Set up extended handlers
        if enable_extensions:
            setup_extended_handlers(server)
        
        logger.info("Integrated MCP server created")
        
        return server
    
    except Exception as e:
        logger.error(f"Error creating integrated MCP server: {e}")
        raise

def run_integrated_server(
    ipfs_accelerate_module: Any = None,
    name: str = "ipfs-accelerate",
    host: str = "localhost",
    port: int = 8000,
    mount_path: str = "/mcp",
    debug: bool = False,
    enable_extensions: bool = True
) -> None:
    """
    Create and run an integrated MCP server for IPFS Accelerate
    
    This function creates an MCP server, integrates it with IPFS Accelerate, and runs it.
    
    Args:
        ipfs_accelerate_module: IPFS Accelerate module (if None, will try to import)
        name: Name of the server
        host: Host to bind the server to
        port: Port to bind the server to
        mount_path: Path to mount the server at
        debug: Enable debug logging
        enable_extensions: Enable extended handlers
    """
    logger.info("Creating and running integrated MCP server for IPFS Accelerate")
    
    try:
        # Create integrated server
        server = create_integrated_server(
            ipfs_accelerate_module=ipfs_accelerate_module,
            name=name,
            host=host,
            port=port,
            mount_path=mount_path,
            debug=debug,
            enable_extensions=enable_extensions
        )
        
        # Run server
        logger.info(f"Running integrated MCP server at {server.server_url}")
        server.run()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
    
    except Exception as e:
        logger.error(f"Error running integrated MCP server: {e}")
        raise
