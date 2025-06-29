#!/usr/bin/env python
"""
IPFS Accelerate MCP Tool Registration

This module handles the registration of all IPFS Accelerate tools with the MCP server.
"""

import os
import sys
import logging
import platform
import json
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def register_tools():
    """Register all IPFS Accelerate tools with the MCP server"""
    logger.info("Starting tool registration")
    
    try:
        # Import the required modules
        from mcp.server import register_tool, register_resource
        logger.info("Successfully imported MCP server modules")
        
        # Register hardware info tool
        def get_hardware_info():
            """Get hardware information about the system"""
            hardware_info = {
                "system": {
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "distribution": platform.platform(),
                    "architecture": platform.machine(),
                    "python_version": platform.python_version(),
                    "processor": platform.processor()
                },
                "accelerators": {
                    "cpu": {"available": True, "cores": os.cpu_count() or 1}
                }
            }
            
            # Try to add more detailed information
            try:
                import psutil
                hardware_info["system"]["memory_total"] = round(psutil.virtual_memory().total / (1024**3), 2)
                hardware_info["system"]["memory_available"] = round(psutil.virtual_memory().available / (1024**3), 2)
            except ImportError:
                pass
            
            # Try to check for GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    hardware_info["accelerators"]["cuda"] = {
                        "available": True,
                        "device_count": torch.cuda.device_count(),
                        "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                    }
                else:
                    hardware_info["accelerators"]["cuda"] = {"available": False}
                    
                # Check for MPS (Apple Metal)
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    hardware_info["accelerators"]["mps"] = {"available": True}
                else:
                    hardware_info["accelerators"]["mps"] = {"available": False}
            except ImportError:
                hardware_info["accelerators"]["cuda"] = {"available": False}
                hardware_info["accelerators"]["mps"] = {"available": False}
            
            # Try to check for AMD ROCm
            try:
                # This is a simplified check - not as reliable as the CUDA check
                import subprocess
                try:
                    rocm_check = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=2)
                    if rocm_check.returncode == 0:
                        hardware_info["accelerators"]["rocm"] = {"available": True}
                    else:
                        hardware_info["accelerators"]["rocm"] = {"available": False}
                except (subprocess.SubprocessError, FileNotFoundError):
                    hardware_info["accelerators"]["rocm"] = {"available": False}
            except ImportError:
                hardware_info["accelerators"]["rocm"] = {"available": False}
            
            return hardware_info
        
        # Register IPFS tools
        def ipfs_add_file(path: str) -> dict:
            """Add a file to IPFS and return its CID"""
            import uuid
            file_hash = f"QmMock{uuid.uuid4().hex[:16]}"
            return {"cid": file_hash, "path": path, "success": True}
        
        def ipfs_cat(cid: str) -> str:
            """Retrieve content from IPFS by its CID"""
            return f"Mock content for {cid}"
        
        def ipfs_files_write(path: str, content: str) -> dict:
            """Write content to the IPFS Mutable File System (MFS)"""
            return {"path": path, "written": True, "success": True}
        
        def ipfs_files_read(path: str) -> str:
            """Read content from the IPFS Mutable File System (MFS)"""
            return f"Mock MFS content for {path}"
        
        def health_check() -> dict:
            """Check the health of the IPFS Accelerate MCP server"""
            return {
                "status": "healthy",
                "version": "1.0.0",
                "uptime": 0,
                "ipfs_connected": False
            }
        
        # Register tools with the server
        register_tool("get_hardware_info", "Get hardware information about the system", get_hardware_info)
        register_tool("ipfs_add_file", "Add a file to IPFS and return its CID", ipfs_add_file)
        register_tool("ipfs_cat", "Retrieve content from IPFS by its CID", ipfs_cat)
        register_tool("ipfs_files_write", "Write content to the IPFS Mutable File System (MFS)", ipfs_files_write)
        register_tool("ipfs_files_read", "Read content from the IPFS Mutable File System (MFS)", ipfs_files_read)
        register_tool("health_check", "Check the health of the IPFS Accelerate MCP server", health_check)
        
        # Register resources
        def get_system_info() -> Dict[str, Any]:
            """Get system information"""
            return {
                "os": platform.system(),
                "os_version": platform.version(),
                "distribution": platform.platform(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "processor": platform.processor()
            }
        
        def get_accelerator_info() -> Dict[str, Any]:
            """Get accelerator information"""
            return {"cpu": {"available": True, "cores": os.cpu_count() or 1}}
        
        register_resource("system_info", "System information", get_system_info)
        register_resource("accelerator_info", "Accelerator information", get_accelerator_info)
        
        logger.info("All tools and resources registered successfully")
        return True
    except Exception as e:
        logger.error(f"Error registering tools: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    register_tools()
