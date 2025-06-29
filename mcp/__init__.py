"""
IPFS Accelerate MCP Integration

This package integrates the IPFS Accelerate library with the Model Context Protocol (MCP),
allowing AI models to access hardware information and model inference capabilities.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional, Union, List

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import key components from submodules
try:
    from .client import MCPClient, get_hardware_info, is_server_running, start_server
    from .server import register_tool, register_resource
    from .tools import get_hardware_info
except ImportError as e:
    logger.warning(f"Error importing MCP components: {e}")
    logger.warning("Some functionality may not be available")

# List of exported components
__all__ = [
    "MCPClient",
    "get_hardware_info",
    "is_server_running",
    "start_server",
    "register_tool",
    "register_resource",
]
