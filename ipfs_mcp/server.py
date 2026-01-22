"""
Core MCP Server Implementation

This module provides the main FastMCP server implementation that exposes
IPFS Accelerate functionality to language models through the Model Context Protocol.
"""
import logging
from typing import Optional, Dict, Any, List

# Import FastMCP with graceful fallback
try:
    from fastmcp import FastMCP
    HAVE_FASTMCP = True
except ImportError as e:
    HAVE_FASTMCP = False
    FastMCP = None
    print(f"⚠️ FastMCP not available: {e}")
    print("📝 Server will run in mock mode for development purposes")

# Import IPFS Accelerate
from ipfs_accelerate_py import ipfs_accelerate_py

# Configure logging
logger = logging.getLogger("ipfs_accelerate_mcp")

# Global server instance
_mcp_server_instance = None


def create_mcp_server(
    name: str = "IPFS Accelerate MCP",
    description: str = "Hardware-accelerated machine learning inference with IPFS integration",
    accelerate_instance: Optional[Any] = None
) -> FastMCP:
    """
    Create and configure a new FastMCP server instance for IPFS Accelerate.
    
    Args:
        name: The name of the MCP server
        description: A description of what the MCP server does
        accelerate_instance: An existing ipfs_accelerate_py instance to use (creates new if None)
    
    Returns:
        Configured FastMCP server instance
    """
    global _mcp_server_instance
    
    if not HAVE_FASTMCP:
        logger.warning("FastMCP not available, creating mock server for development")
        # Return a simple mock object for development
        class MockMCPServer:
            def __init__(self):
                self.name = name
                self.description = description
                self.metadata = {}
            def run(self, **kwargs):
                print(f"Mock MCP Server '{self.name}' would run here")
                print("Install FastMCP for full functionality")
        return MockMCPServer()
    
    # Create a new server if one doesn't exist
    if _mcp_server_instance is None:
        logger.info(f"Creating new MCP server: {name}")
        
        # Create the MCP server
        mcp = FastMCP(name=name, description=description)
        
        # Set up the IPFS Accelerate instance
        if accelerate_instance is None:
            accelerate_instance = ipfs_accelerate_py()
        
        # Store the accelerate instance with the MCP server
        mcp.state.accelerate = accelerate_instance
        
        # Register the core tools and resources
        _register_core_components(mcp)
        
        _mcp_server_instance = mcp
        
    return _mcp_server_instance


def get_mcp_server_instance() -> Optional[FastMCP]:
    """
    Get the global MCP server instance if it exists.
    
    Returns:
        The FastMCP server instance or None if not created
    """
    return _mcp_server_instance


def _register_core_components(mcp: FastMCP) -> None:
    """
    Register core tools, resources, and prompts with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    try:
        # Import component registrations from their respective modules
        from .tools.hardware import register_hardware_tools
        from .tools.inference import register_inference_tools
        from .resources.system_info import register_system_resources
        from .resources.model_info import register_model_resources
        from .prompts import setup_prompts
        
        # Register all components
        register_hardware_tools(mcp)
        register_inference_tools(mcp)
        register_system_resources(mcp)
        register_model_resources(mcp)
        setup_prompts(mcp)
        
        logger.info("Core MCP components registered")
        
        # Add server metadata
        mcp.metadata.update({
            "version": "0.1.0",
            "type": "IPFS Accelerate MCP Server",
        })
        
    except ImportError as e:
        logger.warning(f"Some MCP components could not be imported: {e}")
    except Exception as e:
        logger.error(f"Error registering MCP components: {e}")
        raise
