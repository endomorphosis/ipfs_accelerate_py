"""
IPFS Accelerate Python package.

This package provides a framework for hardware-accelerated machine learning inference
with IPFS network-based distribution and acceleration. Key features include:

- Hardware acceleration (CPU, GPU, OpenVINO, WebNN, WebGPU)
- IPFS-based content distribution and caching
- Browser integration for client-side inference
- Model type detection and optimization
- Cross-platform support
"""

# Import original components
try:
    from .container_backends import backends
except ImportError:
    backends = None

try:
    from .install_depends import install_depends
except ImportError:
    install_depends = None

try:
    from .ipfs_accelerate import ipfs_accelerate_py as original_ipfs_accelerate_py
except ImportError:
    original_ipfs_accelerate_py = None

try:
    from .ipfs_multiformats import ipfs_multiformats_py
except ImportError:
    ipfs_multiformats_py = None

try:
    from .worker import worker
except ImportError:
    worker = None

try:
    from .config import config
except ImportError:
    config = None

# Import WebNN/WebGPU integration
try:
    from .webnn_webgpu_integration import (
        accelerate_with_browser,
        WebNNWebGPUAccelerator,
        get_accelerator
    )
    webnn_webgpu_available = True
except ImportError:
    webnn_webgpu_available = False
    
    # Create stubs if not available
    def accelerate_with_browser(*args, **kwargs):
        raise NotImplementedError("WebNN/WebGPU integration is not available")
    
    def get_accelerator(*args, **kwargs):
        raise NotImplementedError("WebNN/WebGPU integration is not available")
    
    class WebNNWebGPUAccelerator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("WebNN/WebGPU integration is not available")

# Import our new implementation
try:
    import sys
    import os
    
    # Add the parent directory to the path to import from top-level module
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from ipfs_accelerate_py import ipfs_accelerate_py, get_instance
except ImportError:
    # Fall back to original implementation if it exists
    if original_ipfs_accelerate_py is not None:
        ipfs_accelerate_py = original_ipfs_accelerate_py
        get_instance = lambda: None
    else:
        # Create stub if neither is available
        def ipfs_accelerate_py(*args, **kwargs):
            raise NotImplementedError("IPFS Accelerate Python is not available")
        
        def get_instance():
            raise NotImplementedError("IPFS Accelerate Python is not available")

# Export all components
export = {
    "backends": backends,
    "config": config,
    "install_depends": install_depends,
    "ipfs_accelerate_py": ipfs_accelerate_py,
    "worker": worker,
    "ipfs_multiformats_py": ipfs_multiformats_py,
    "get_instance": get_instance,
    "accelerate_with_browser": accelerate_with_browser,
    "WebNNWebGPUAccelerator": WebNNWebGPUAccelerator,
    "get_accelerator": get_accelerator,
    "webnn_webgpu_available": webnn_webgpu_available
}

__all__ = [
    'ipfs_accelerate_py', 'get_instance', 'backends', 'config', 
    'install_depends', 'worker', 'ipfs_multiformats_py',
    'accelerate_with_browser', 'WebNNWebGPUAccelerator', 'get_accelerator',
    'webnn_webgpu_available'
]

# Package version
__version__ = "0.4.0"