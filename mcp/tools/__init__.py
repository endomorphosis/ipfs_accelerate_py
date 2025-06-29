"""
IPFS Accelerate MCP Tools

This package contains tools registered with the MCP server.
"""

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import tools
try:
    from .hardware import get_hardware_info
except ImportError as e:
    logger.warning(f"Error importing hardware tools: {e}")
    
    # Define fallback function
    def get_hardware_info() -> Dict[str, Any]:
        """Fallback function for hardware info"""
        import platform
        
        return {
            "system": {
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
            },
            "accelerators": {
                "cuda": {"available": False},
                "webgpu": {"available": False},
                "webnn": {"available": False},
            }
        }

# Export functions
__all__ = ["get_hardware_info"]
