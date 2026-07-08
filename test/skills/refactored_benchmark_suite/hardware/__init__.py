"""
Hardware detection and management for the benchmark suite.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union

from .base import HardwareBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .mps import MPSBackend
from .rocm import ROCmBackend
from .openvino import OpenVINOBackend
from .webnn import WebNNBackend
from .webgpu import WebGPUBackend

logger = logging.getLogger("benchmark.hardware")

# Map of hardware names to backend classes
HARDWARE_BACKENDS = {
    "cpu": CPUBackend,
    "cuda": CUDABackend,
    "mps": MPSBackend,
    "rocm": ROCmBackend,
    "openvino": OpenVINOBackend,
    "webnn": WebNNBackend,
    "webgpu": WebGPUBackend
}

# Map of hardware capabilities to backend classes
CAPABILITY_BACKENDS = {
    "cuda_tensor_cores": CUDABackend
}

def get_available_hardware() -> List[str]:
    """
    Detect available hardware platforms.
    
    Returns:
        List of available hardware platforms
    """
    available = []
    
    # Check each backend
    for name, backend_cls in HARDWARE_BACKENDS.items():
        if backend_cls.is_available():
            available.append(name)
    
    # Check for CUDA capabilities
    if "cuda" in available:
        cuda_capabilities = CUDABackend.get_capabilities()
        for capability in cuda_capabilities:
            if capability != "cuda" and capability not in available:
                available.append(capability)
    
    return available

def get_hardware_info() -> Dict[str, Any]:
    """
    Get detailed information about available hardware.
    
    Returns:
        Dictionary with hardware information
    """
    info = {}
    
    # Get info from each backend
    for name, backend_cls in HARDWARE_BACKENDS.items():
        if backend_cls.is_available():
            info[name] = True
            info[f"{name}_info"] = backend_cls.get_info()
        else:
            info[name] = False
    
    return info

def initialize_hardware(hardware: str, **kwargs) -> Any:
    """
    Initialize a specific hardware platform for benchmarking.
    
    Args:
        hardware: Hardware platform to initialize
        **kwargs: Additional hardware-specific parameters
        
    Returns:
        Device object for the initialized hardware
    """
    # Look up appropriate backend class
    backend_cls = HARDWARE_BACKENDS.get(hardware)
    if backend_cls is None:
        # Check if it's a capability
        backend_cls = CAPABILITY_BACKENDS.get(hardware)
        if backend_cls is None:
            logger.warning(f"Unknown hardware '{hardware}', falling back to CPU")
            backend_cls = CPUBackend
    
    # Check if hardware is available
    if not backend_cls.is_available():
        logger.warning(f"{hardware} not available, falling back to CPU")
        backend_cls = CPUBackend
    
    # Create and initialize backend
    backend = backend_cls(**kwargs)
    return backend.initialize()

def get_hardware_backend(hardware: str, **kwargs) -> HardwareBackend:
    """
    Get a hardware backend instance.
    
    Args:
        hardware: Hardware platform name
        **kwargs: Additional hardware-specific parameters
        
    Returns:
        HardwareBackend instance
    """
    # Look up appropriate backend class
    backend_cls = HARDWARE_BACKENDS.get(hardware)
    if backend_cls is None:
        # Check if it's a capability
        backend_cls = CAPABILITY_BACKENDS.get(hardware)
        if backend_cls is None:
            logger.warning(f"Unknown hardware '{hardware}', falling back to CPU")
            backend_cls = CPUBackend
    
    # Create backend
    return backend_cls(**kwargs)