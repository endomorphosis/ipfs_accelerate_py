#!/usr/bin/env python
"""
IPFS Accelerate MCP Hardware Tool

This module provides tools for detecting and reporting hardware capabilities.
"""

import os
import sys
import platform
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dict[str, Any]: System information
    """
    import platform
    import psutil
    
    # Basic system information
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "distribution": "",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "memory_total": round(psutil.virtual_memory().total / (1024 ** 3), 2),  # GB
        "memory_available": round(psutil.virtual_memory().available / (1024 ** 3), 2),  # GB
        "cpu": {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        }
    }
    
    # Get distribution info for Linux
    if platform.system() == "Linux":
        try:
            import distro
            system_info["distribution"] = distro.name(pretty=True)
        except ImportError:
            # Fallback for older systems or when distro is not available
            try:
                with open("/etc/os-release") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME="):
                            system_info["distribution"] = line.split('=')[1].strip().strip('"')
                            break
            except Exception as e:
                logger.warning(f"Could not determine Linux distribution: {e}")
    
    return system_info

def detect_cuda() -> Dict[str, Any]:
    """
    Detect CUDA availability and devices
    
    Returns:
        Dict[str, Any]: CUDA information
    """
    # Default result
    cuda_info = {
        "available": False,
        "version": None,
        "devices": []
    }
    
    # Try to import torch to check CUDA
    try:
        import torch
        
        cuda_info["available"] = torch.cuda.is_available()
        
        if cuda_info["available"]:
            cuda_info["version"] = torch.version.cuda
            
            # Get device information
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device = {
                    "name": torch.cuda.get_device_name(i),
                    "index": i,
                    "memory_total": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2),  # GB
                }
                cuda_info["devices"].append(device)
    except ImportError:
        logger.info("PyTorch not available, CUDA detection limited")
        
        # Try direct detection with pycuda if available
        try:
            import pycuda.driver as cuda
            cuda.init()
            
            cuda_info["available"] = True
            cuda_info["version"] = ".".join(map(str, cuda.get_version()))
            
            # Get device information
            device_count = cuda.Device.count()
            for i in range(device_count):
                device = cuda.Device(i)
                device_info = {
                    "name": device.name(),
                    "index": i,
                    "memory_total": round(device.total_memory() / (1024 ** 3), 2),  # GB
                }
                cuda_info["devices"].append(device_info)
        except ImportError:
            # pycuda not available, try nvidia-smi
            try:
                import subprocess
                result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    cuda_info["available"] = True
                    
                    # Parse output
                    for i, line in enumerate(result.stdout.strip().split('\n')):
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 2:
                                name = parts[0].strip()
                                memory = float(parts[1].strip()) / 1024  # Convert to GB
                                
                                cuda_info["devices"].append({
                                    "name": name,
                                    "index": i,
                                    "memory_total": round(memory, 2),
                                })
            except Exception as e:
                logger.debug(f"Could not run nvidia-smi: {e}")
    
    return cuda_info

def detect_webgpu() -> Dict[str, Any]:
    """
    Detect WebGPU availability
    
    Returns:
        Dict[str, Any]: WebGPU information
    """
    # Default result
    webgpu_info = {
        "available": False,
        "version": None,
        "devices": []
    }
    
    # Try to import webgpu packages from ipfs_accelerate_py
    try:
        # Add parent directory to sys.path if needed
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import WebGPU related modules
        from ipfs_accelerate_py import webgpu_platform
        
        # Check if WebGPU is available
        if hasattr(webgpu_platform, "is_webgpu_available"):
            webgpu_info["available"] = webgpu_platform.is_webgpu_available()
            
            # Get version if available
            if hasattr(webgpu_platform, "get_webgpu_version"):
                webgpu_info["version"] = webgpu_platform.get_webgpu_version()
                
            # Get devices if available and function exists
            if webgpu_info["available"] and hasattr(webgpu_platform, "get_webgpu_devices"):
                devices = webgpu_platform.get_webgpu_devices()
                
                if devices and isinstance(devices, list):
                    for i, device in enumerate(devices):
                        device_info = {
                            "name": device.get("name", f"WebGPU Device {i}"),
                            "index": i,
                        }
                        
                        # Add other properties if available
                        for key in ["vendor", "memory_total", "type"]:
                            if key in device:
                                device_info[key] = device[key]
                        
                        webgpu_info["devices"].append(device_info)
    except ImportError as e:
        logger.debug(f"WebGPU detection failed: {e}")
    except Exception as e:
        logger.warning(f"Error detecting WebGPU: {e}")
    
    return webgpu_info

def detect_webnn() -> Dict[str, Any]:
    """
    Detect WebNN availability
    
    Returns:
        Dict[str, Any]: WebNN information
    """
    # Default result
    webnn_info = {
        "available": False,
        "version": None,
        "devices": []
    }
    
    # Try to import WebNN packages from ipfs_accelerate_py
    try:
        # Add parent directory to sys.path if needed
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Try to import WebNN detection functions
        try:
            from ipfs_accelerate_py import hardware_detection
            
            # Check if WebNN is available
            if hasattr(hardware_detection, "is_webnn_available"):
                webnn_info["available"] = hardware_detection.is_webnn_available()
                
                # Get version if available
                if hasattr(hardware_detection, "get_webnn_version"):
                    webnn_info["version"] = hardware_detection.get_webnn_version()
                    
                # Get devices if available
                if webnn_info["available"] and hasattr(hardware_detection, "get_webnn_devices"):
                    devices = hardware_detection.get_webnn_devices()
                    
                    if devices and isinstance(devices, list):
                        for i, device in enumerate(devices):
                            device_info = {
                                "name": device.get("name", f"WebNN Device {i}"),
                                "index": i,
                            }
                            
                            # Add other properties if available
                            for key in ["type", "performance_rating"]:
                                if key in device:
                                    device_info[key] = device[key]
                            
                            webnn_info["devices"].append(device_info)
        except ImportError:
            # Try alternative imports
            from ipfs_accelerate_py import web_platform
            
            if hasattr(web_platform, "is_webnn_supported"):
                webnn_info["available"] = web_platform.is_webnn_supported()
                
                # Get version if available
                if hasattr(web_platform, "get_webnn_version"):
                    webnn_info["version"] = web_platform.get_webnn_version()
    except ImportError as e:
        logger.debug(f"WebNN detection failed: {e}")
    except Exception as e:
        logger.warning(f"Error detecting WebNN: {e}")
    
    return webnn_info

def get_hardware_info() -> Dict[str, Any]:
    """
    Get hardware information
    
    Returns:
        Dict[str, Any]: Hardware information
    """
    # Get system information
    system_info = get_system_info()
    
    # Detect CUDA
    cuda_info = detect_cuda()
    
    # Detect WebGPU
    webgpu_info = detect_webgpu()
    
    # Detect WebNN
    webnn_info = detect_webnn()
    
    # Combine results
    hardware_info = {
        "system": system_info,
        "accelerators": {
            "cuda": cuda_info,
            "webgpu": webgpu_info,
            "webnn": webnn_info,
        }
    }
    
    return hardware_info

if __name__ == "__main__":
    # When run directly, print hardware info
    import json
    
    hardware_info = get_hardware_info()
    print(json.dumps(hardware_info, indent=2))
