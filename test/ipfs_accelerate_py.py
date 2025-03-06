#!/usr/bin/env python
"""
IPFS Accelerate Python Package

This module initializes the IPFS Accelerate Python package by importing components
from the ipfs_accelerate_impl module.
"""

# Import all components from the implementation module
from ipfs_accelerate_impl import (
    config,
    backends,
    ipfs_accelerate,
    load_checkpoint_and_dispatch,
    get_system_info,
    __version__
)

# Package version
__version__ = __version__

# Export the module components
__all__ = [
    'config',
    'backends',
    'ipfs_accelerate',
    'load_checkpoint_and_dispatch',
    'get_system_info',
    '__version__'
]