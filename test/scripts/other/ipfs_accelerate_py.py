#!/usr/bin/env python
"""
IPFS Accelerate Python Package

This module initializes the IPFS Accelerate Python package by importing components
from the ipfs_accelerate_impl module.

Features:
    - IPFS content acceleration with P2P optimization
    - Hardware acceleration ()CPU, GPU, WebNN, WebGPU)
    - Automatic hardware detection and selection
    - Browser-specific optimizations
    - Database integration for storing and analyzing results
    - Cross-platform compatibility
    """

# Import all components from the implementation module
    from ipfs_accelerate_impl import ()
    # Core components
    config,
    backends,
    ipfs_accelerate,
    IPFSAccelerate,
    P2PNetworkOptimizer,
    
    # IPFS functions
    load_checkpoint_and_dispatch,
    get_file,
    add_file,
    get_p2p_network_analytics,
    
    # Hardware acceleration
    HardwareDetector,
    HardwareAcceleration,
    accelerate,
    detect_hardware,
    get_optimal_hardware,
    get_hardware_details,
    is_real_hardware,
    
    # Database functionality
    DatabaseHandler,
    db_handler,
    store_acceleration_result,
    get_acceleration_results,
    generate_report,
    
    # Utility functions
    get_system_info,
    __version__
    )

# Package version
    __version__ = __version__

# Export the module components
    __all__ = [],
    # Core components
    'config',
    'backends',
    'ipfs_accelerate',
    'IPFSAccelerate',
    'P2PNetworkOptimizer',
    
    # IPFS functions
    'load_checkpoint_and_dispatch',
    'get_file',
    'add_file',
    'get_p2p_network_analytics',
    
    # Hardware acceleration
    'HardwareDetector',
    'HardwareAcceleration',
    'accelerate', 
    'detect_hardware',
    'get_optimal_hardware',
    'get_hardware_details',
    'is_real_hardware',
    
    # Database functionality
    'DatabaseHandler',
    'db_handler',
    'store_acceleration_result',
    'get_acceleration_results',
    'generate_report',
    
    # Utility functions
    'get_system_info',
    '__version__'
    ]