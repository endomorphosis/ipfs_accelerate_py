"""
IPFS Accelerate Python package.

This package provides a framework for hardware-accelerated machine learning inference
with IPFS network-based distribution and acceleration.
"""

# Import original components
try:
    from ipfs_accelerate_py.container_backends import backends
except ImportError:
    backends = None

try:
    from ipfs_accelerate_py.install_depends import install_depends
except ImportError:
    install_depends = None

try:
    from test.tests.other.ipfs_accelerate_py_tests.ipfs_accelerate import ipfs_accelerate_py as original_ipfs_accelerate_py
except ImportError:
    original_ipfs_accelerate_py = None

try:
    from test.tests.other.ipfs_accelerate_py_tests.ipfs_multiformats import ipfs_multiformats_py
except ImportError:
    ipfs_multiformats_py = None

try:
    from test.tests.other.ipfs_accelerate_py_tests.worker import worker
except ImportError:
    worker = None

try:
    from ipfs_accelerate_py.config import config
except ImportError:
    config = None

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
    "get_instance": get_instance
}

__all__ = ['ipfs_accelerate_py', 'get_instance', 'backends', 'config', 
           'install_depends', 'worker', 'ipfs_multiformats_py']