"""
IPFS Accelerate MCP Configuration Resources

This module provides configuration resources for the IPFS Accelerate MCP server.
"""

import os
import sys
import json
import logging
import platform
from typing import Dict, Any, Optional, List, Union

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.resources.config")

def register_config_resources(mcp: Any) -> None:
    """
    Register configuration resources with the MCP server
    
    This function registers configuration resources with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering configuration resources")
    
    try:
        # Register version resource
        mcp.register_resource(
            uri="ipfs_accelerate/version",
            function=get_version_info,
            description="IPFS Accelerate version information"
        )
        
        # Register system info resource
        mcp.register_resource(
            uri="ipfs_accelerate/system_info",
            function=get_system_info,
            description="System information"
        )
        
        # Register config resource
        mcp.register_resource(
            uri="ipfs_accelerate/config",
            function=get_config,
            description="IPFS Accelerate configuration"
        )
        
        logger.debug("Configuration resources registered")
    
    except Exception as e:
        logger.error(f"Error registering configuration resources: {e}")
        raise

def get_version_info() -> Dict[str, Any]:
    """
    Get IPFS Accelerate version information
    
    Returns:
        Dictionary with version information
    """
    logger.debug("Getting version information")
    
    try:
        # Try to import ipfs_accelerate_py
        try:
            import ipfs_accelerate_py
            version = getattr(ipfs_accelerate_py, "__version__", "unknown")
        except ImportError:
            version = "unknown"
        
        # Get MCP version
        try:
            from ipfs_accelerate_py.mcp import __version__ as mcp_version
        except ImportError:
            mcp_version = "0.1.0"  # Default version
        
        # Get FastMCP version
        try:
            import fastmcp
            fastmcp_version = getattr(fastmcp, "__version__", "unknown")
        except ImportError:
            fastmcp_version = "unknown"
        
        # Build version info
        version_info = {
            "version": version,
            "mcp_version": mcp_version,
            "fastmcp_version": fastmcp_version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation()
        }
        
        logger.debug("Version information retrieved")
        
        return version_info
    
    except Exception as e:
        logger.error(f"Error getting version information: {e}")
        raise

def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dictionary with system information
    """
    logger.debug("Getting system information")
    
    try:
        # Basic system information
        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine()
        }
        
        # Add CPU information if available
        try:
            import psutil
            
            system_info["cpu_count"] = {
                "physical": psutil.cpu_count(logical=False),
                "logical": psutil.cpu_count(logical=True)
            }
            
            system_info["memory"] = {
                "total_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
                "available_mb": round(psutil.virtual_memory().available / (1024 * 1024)),
                "percent_used": psutil.virtual_memory().percent
            }
            
            system_info["cpu_frequency"] = {
                "current_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else None,
                "min_mhz": round(psutil.cpu_freq().min) if psutil.cpu_freq() and hasattr(psutil.cpu_freq(), "min") else None,
                "max_mhz": round(psutil.cpu_freq().max) if psutil.cpu_freq() and hasattr(psutil.cpu_freq(), "max") else None
            }
            
            system_info["disk"] = {
                "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
                "percent_used": psutil.disk_usage("/").percent
            }
        except ImportError:
            # Fallback to basic information
            system_info["cpu_count"] = {
                "physical": os.cpu_count(),
                "logical": os.cpu_count()
            }
        
        # Add CUDA information if available
        try:
            import torch
            
            if torch.cuda.is_available():
                system_info["cuda"] = {
                    "is_available": True,
                    "version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "devices": [
                        {
                            "name": torch.cuda.get_device_name(i),
                            "total_memory_mb": round(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024))
                        }
                        for i in range(torch.cuda.device_count())
                    ]
                }
            else:
                system_info["cuda"] = {
                    "is_available": False
                }
        except ImportError:
            system_info["cuda"] = {
                "is_available": "unknown"
            }
        
        logger.debug("System information retrieved")
        
        return system_info
    
    except Exception as e:
        logger.error(f"Error getting system information: {e}")
        raise

def get_config() -> Dict[str, Any]:
    """
    Get IPFS Accelerate configuration
    
    Returns:
        Dictionary with configuration
    """
    logger.debug("Getting configuration")
    
    try:
        # Initialize default configuration
        config = {
            "settings": {
                "default_device": "auto",
                "fallback_device": "cpu",
                "enable_logging": True,
                "log_level": "info",
                "cache_dir": "~/.ipfs_accelerate/cache",
                "max_cache_size_gb": 10,
            },
            "features": {
                "hardware_detection": True,
                "automatic_optimization": True,
                "webgpu_support": True,
                "webnn_support": True,
                "ipfs_optimizations": True
            },
            "performance": {
                "max_concurrent_requests": 10,
                "timeout_seconds": 300,
                "retry_count": 3,
                "batch_size": 1,
                "precision": "auto"
            }
        }
        
        # Try to import ipfs_accelerate_py
        try:
            import ipfs_accelerate_py
            
            # Use ipfs_accelerate_py's configuration if available
            if hasattr(ipfs_accelerate_py, "get_config"):
                ipfs_config = ipfs_accelerate_py.get_config()
                
                # Merge with default configuration
                if isinstance(ipfs_config, dict):
                    # Merge settings
                    if "settings" in ipfs_config and isinstance(ipfs_config["settings"], dict):
                        config["settings"].update(ipfs_config["settings"])
                    
                    # Merge features
                    if "features" in ipfs_config and isinstance(ipfs_config["features"], dict):
                        config["features"].update(ipfs_config["features"])
                    
                    # Merge performance
                    if "performance" in ipfs_config and isinstance(ipfs_config["performance"], dict):
                        config["performance"].update(ipfs_config["performance"])
        except ImportError:
            # Use default configuration
            pass
        
        logger.debug("Configuration retrieved")
        
        return config
    
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise
