#!/usr/bin/env python
"""
IPFS Accelerate MCP System Information Resource

This module provides system information as an MCP resource.
"""

import os
import sys
import platform
import json
import logging
from typing import Dict, Any

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
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "node": platform.node(),
    }
    
    # Add more detailed information based on the OS
    if platform.system() == "Linux":
        system_info["linux_distribution"] = get_linux_distribution()
    
    # Add environment variables (filtered for security)
    env_vars = {}
    safe_vars = ["PATH", "PYTHONPATH", "LANG", "USER", "HOME", "SHELL"]
    for var in safe_vars:
        if var in os.environ:
            env_vars[var] = os.environ[var]
    
    system_info["environment"] = env_vars
    
    return system_info

def get_linux_distribution() -> Dict[str, str]:
    """
    Get Linux distribution information
    
    Returns:
        Dict[str, str]: Linux distribution information
    """
    distro_info = {}
    
    try:
        # Try to use the distro module if available
        import distro
        distro_info = {
            "id": distro.id(),
            "name": distro.name(),
            "version": distro.version(),
            "codename": distro.codename(),
        }
    except ImportError:
        # Fall back to reading /etc/os-release
        try:
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            value = value.strip('"')
                            distro_info[key.lower()] = value
        except Exception as e:
            logger.warning(f"Error reading /etc/os-release: {e}")
    
    return distro_info

def get_hardware_profile() -> Dict[str, Any]:
    """
    Get hardware profile information
    
    Returns:
        Dict[str, Any]: Hardware profile information
    """
    hardware_info = {}
    
    try:
        import psutil
        
        # CPU information
        hardware_info["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "current_frequency": psutil.cpu_freq(),
            "percent_usage": psutil.cpu_percent(interval=0.1, percpu=True),
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        hardware_info["memory"] = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
        }
        
        # Disk information
        disk = psutil.disk_usage("/")
        hardware_info["disk"] = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
        }
        
        # Network information
        network_info = {}
        for name, stats in psutil.net_if_stats().items():
            network_info[name] = {
                "is_up": stats.isup,
                "speed": stats.speed,
                "mtu": stats.mtu,
            }
        hardware_info["network"] = network_info
    
    except ImportError:
        logger.warning("psutil not available, hardware profile will be limited")
        
        # Basic CPU information
        hardware_info["cpu"] = {
            "processor": platform.processor(),
            "architecture": platform.machine(),
        }
    
    except Exception as e:
        logger.warning(f"Error getting hardware profile: {e}")
    
    return hardware_info

if __name__ == "__main__":
    # When run directly, print system info
    system_info = get_system_info()
    hardware_profile = get_hardware_profile()
    
    # Combine information
    info = {
        "system_info": system_info,
        "hardware_profile": hardware_profile,
    }
    
    # Print as formatted JSON
    print(json.dumps(info, indent=2))
