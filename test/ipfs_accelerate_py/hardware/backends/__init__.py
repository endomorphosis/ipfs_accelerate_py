"""
Hardware backends for IPFS Accelerate SDK.

This module provides hardware-specific backend implementations for different
acceleration platforms, including CPU, GPU, and specialized accelerators.
"""

from typing import Dict, Any, Type

# Import backends
from ipfs_accelerate_py.hardware.backends.cpu_backend import CPUBackend
from ipfs_accelerate_py.hardware.backends.cuda_backend import CUDABackend

# Registry of available backends
BACKEND_REGISTRY: Dict[str, Type] = {
    "cpu": CPUBackend,
    "cuda": CUDABackend,
}

# Add other backends if available
try:
    from ipfs_accelerate_py.hardware.backends.rocm_backend import ROCmBackend
    BACKEND_REGISTRY["rocm"] = ROCmBackend
except ImportError:
    pass

try:
    from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
    BACKEND_REGISTRY["openvino"] = OpenVINOBackend
except ImportError:
    pass

try:
    from ipfs_accelerate_py.hardware.backends.apple_backend import AppleBackend
    BACKEND_REGISTRY["mps"] = AppleBackend
except ImportError:
    pass

try:
    from ipfs_accelerate_py.hardware.backends.qualcomm_backend import QualcommBackend
    BACKEND_REGISTRY["qualcomm"] = QualcommBackend
except ImportError:
    pass

try:
    from ipfs_accelerate_py.hardware.backends.webnn_backend import WebNNBackend
    BACKEND_REGISTRY["webnn"] = WebNNBackend
except ImportError:
    pass

try:
    from ipfs_accelerate_py.hardware.backends.webgpu_backend import WebGPUBackend
    BACKEND_REGISTRY["webgpu"] = WebGPUBackend
except ImportError:
    pass

def get_backend(backend_name: str, config=None) -> Any:
    """
    Get a backend instance.
    
    Args:
        backend_name: Name of the backend.
        config: Configuration instance (optional).
        
    Returns:
        Backend instance.
    """
    backend_class = BACKEND_REGISTRY.get(backend_name)
    if not backend_class:
        raise ValueError(f"Backend {backend_name} not found")
    
    return backend_class(config)

def list_available_backends() -> Dict[str, bool]:
    """
    List available backends.
    
    Returns:
        Dictionary with backend names and availability status.
    """
    availability = {}
    
    for backend_name, backend_class in BACKEND_REGISTRY.items():
        try:
            backend = backend_class()
            is_available = backend.is_available()
        except Exception:
            is_available = False
        
        availability[backend_name] = is_available
    
    return availability