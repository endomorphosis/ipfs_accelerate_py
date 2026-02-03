"""
IPFS Kit - Core Modules for IPFS Accelerate

This package contains the core functionality modules that are used by both
the unified CLI and the MCP server. All modules here should be pure Python
with minimal dependencies and no CLI-specific code.

Architecture:
    ipfs_kit_py (core modules)
        ↓
    unified_cli (CLI interface)
        ↓
    mcp/unified_tools (MCP wrappers)
        ↓
    mcp/unified_server (MCP server)
        ↓
    JavaScript SDK
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Module registry for dynamic loading
_MODULE_REGISTRY = {}


def register_module(name: str, module):
    """
    Register a module in the kit registry.
    
    Args:
        name: Module name
        module: Module instance or class
    """
    _MODULE_REGISTRY[name] = module
    logger.debug(f"Registered ipfs_kit module: {name}")


def get_module(name: str):
    """
    Get a registered module.
    
    Args:
        name: Module name
        
    Returns:
        Module instance or None
    """
    return _MODULE_REGISTRY.get(name)


def list_modules() -> List[str]:
    """
    List all registered modules.
    
    Returns:
        List of module names
    """
    return list(_MODULE_REGISTRY.keys())


def get_all_modules() -> Dict[str, Any]:
    """
    Get all registered modules.
    
    Returns:
        Dictionary of module name to module
    """
    return _MODULE_REGISTRY.copy()


# Import and auto-register all available modules
def _auto_register_modules():
    """Auto-register all available ipfs_kit modules."""
    modules_to_register = [
        ('github', 'github_kit'),
        ('docker', 'docker_kit'),
        ('inference', 'inference_kit'),
        ('hardware', 'hardware_kit'),
        ('ipfs', 'ipfs_kit'),
        ('network', 'network_kit'),
    ]
    
    for name, module_name in modules_to_register:
        try:
            module = __import__(f'ipfs_kit_py.{module_name}', fromlist=[module_name])
            register_module(name, module)
        except ImportError as e:
            logger.debug(f"Module {module_name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register module {module_name}: {e}")


# Auto-register on import
_auto_register_modules()

__all__ = [
    'register_module',
    'get_module',
    'list_modules',
    'get_all_modules',
]
