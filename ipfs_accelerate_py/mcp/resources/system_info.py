"""
System Information Resources

This module provides MCP resources for system information.
"""
import logging
import platform
import os
from typing import Dict, Any
from fastmcp import FastMCP

logger = logging.getLogger("ipfs_accelerate_mcp.resources.system_info")

def register_system_resources(mcp: FastMCP) -> None:
    """
    Register system information resources with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    # Access the ipfs_accelerate_py instance
    accelerate = mcp.state.accelerate
    
    @mcp.resource("system://info")
    def get_system_info() -> Dict[str, Any]:
        """
        Get information about the system running IPFS Accelerate.
        
        Returns a dictionary with details about the operating system,
        Python environment, and hardware configuration.
        
        Returns:
            Dictionary with system information
        """
        logger.info("MCP resource accessed: system://info")
        
        # Basic system info
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node(),
            "cpu_count": os.cpu_count() or 0,
        }
        
        # Add hardware info if available
        try:
            hardware_info = accelerate.hardware_detection.detect_all_hardware()
            system_info["hardware"] = hardware_info
        except Exception as e:
            logger.error(f"Error fetching hardware info: {str(e)}")
            system_info["hardware"] = {"error": str(e)}
        
        return system_info
    
    @mcp.resource("system://capabilities")
    def get_system_capabilities() -> Dict[str, Any]:
        """
        Get information about IPFS Accelerate capabilities on this system.
        
        Returns details about available accelerators, supported model types,
        and enabled features.
        
        Returns:
            Dictionary with capability information
        """
        logger.info("MCP resource accessed: system://capabilities")
        
        # Gather capability information
        capabilities = {
            "accelerators": {
                "cuda": False,
                "rocm": False,
                "mps": False,
                "webnn": False,
                "webgpu": False,
                "openvino": False,
            },
            "networks": {
                "ipfs": True,
                "libp2p": True,
            },
            "features": {
                "distributed_inference": True,
                "model_optimization": True,
                "hardware_acceleration": True,
            }
        }
        
        # Update accelerator info based on hardware detection
        try:
            hardware_info = accelerate.hardware_detection.detect_all_hardware()
            for accel in capabilities["accelerators"]:
                if accel in hardware_info and hardware_info[accel].get("available", False):
                    capabilities["accelerators"][accel] = True
        except Exception as e:
            logger.error(f"Error updating accelerator info: {str(e)}")
        
        return capabilities
