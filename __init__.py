"""
IPFS Accelerate Python package.

This package provides a framework for hardware-accelerated machine learning inference
with IPFS network-based distribution and acceleration.
"""

# Import from new implementation
from .ipfs_accelerate_py import ipfs_accelerate_py, get_instance

# Try to import from package if available
try:
    from ipfs_accelerate_py import export
except ImportError:
    # Create export if not available from package
    export = {"ipfs_accelerate_py": ipfs_accelerate_py, "get_instance": get_instance}

__all__ = ['ipfs_accelerate_py', 'get_instance', 'export']