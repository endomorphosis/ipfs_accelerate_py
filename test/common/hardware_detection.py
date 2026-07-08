#!/usr/bin/env python3
"""
Hardware detection module for IPFS Accelerate tests.

This module provides utilities for detecting hardware platforms
and skipping tests on unsupported platforms.
"""

import os
import sys
import logging
import platform
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware platforms.
    
    Returns:
        Dictionary with hardware platform information
    """
    result = {
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'platforms': {
            'cpu': {'available': True},
            'cuda': {'available': False},
            'rocm': {'available': False},
            'mps': {'available': False},
            'webgpu': {'available': False},
            'webnn': {'available': False},
            'openvino': {'available': False},
            'qnn': {'available': False}
        }
    }
    
    # Detect CUDA
    try:
        import torch
        result['platforms']['cuda']['available'] = torch.cuda.is_available()
        if result['platforms']['cuda']['available']:
            result['platforms']['cuda']['device_count'] = torch.cuda.device_count()
            result['platforms']['cuda']['devices'] = [
                {
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i)
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        logger.debug("PyTorch not available, skipping CUDA detection")
    
    # Detect ROCm (AMD)
    try:
        import torch
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            result['platforms']['rocm']['available'] = True
            result['platforms']['rocm']['version'] = torch.version.hip
    except ImportError:
        logger.debug("PyTorch not available, skipping ROCm detection")
    
    # Detect MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result['platforms']['mps']['available'] = True
    except ImportError:
        logger.debug("PyTorch not available, skipping MPS detection")
    
    # Detect OpenVINO
    try:
        import openvino
        result['platforms']['openvino']['available'] = True
        result['platforms']['openvino']['version'] = openvino.__version__
    except ImportError:
        logger.debug("OpenVINO not available")
    
    # Detect QNN (Qualcomm Neural Network)
    try:
        import qti
        result['platforms']['qnn']['available'] = True
    except ImportError:
        try:
            import qnn_wrapper
            result['platforms']['qnn']['available'] = True
        except ImportError:
            logger.debug("QNN not available")
    
    # WebGPU and WebNN require browser detection
    # For this simplified version, assume they're not available natively
    result['platforms']['webgpu']['available'] = False
    result['platforms']['webnn']['available'] = False
    
    return result


def setup_platform(platform_name: str) -> Any:
    """
    Set up a specific hardware platform.
    
    Args:
        platform_name: Name of the platform to set up
        
    Returns:
        Platform-specific object or None if setup fails
    """
    if platform_name == 'cuda':
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device('cuda')
        except ImportError:
            logger.warning("PyTorch not available, cannot set up CUDA")
    
    elif platform_name == 'rocm':
        try:
            import torch
            if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return torch.device('cuda')  # ROCm uses CUDA device type in PyTorch
        except ImportError:
            logger.warning("PyTorch not available, cannot set up ROCm")
    
    elif platform_name == 'mps':
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
        except ImportError:
            logger.warning("PyTorch not available, cannot set up MPS")
    
    elif platform_name == 'cpu':
        try:
            import torch
            return torch.device('cpu')
        except ImportError:
            logger.warning("PyTorch not available, returning 'cpu' string")
            return 'cpu'
    
    elif platform_name == 'openvino':
        try:
            import openvino as ov
            return ov.Core()
        except ImportError:
            logger.warning("OpenVINO not available, cannot set up OpenVINO")
    
    elif platform_name == 'qnn':
        try:
            import qti
            return qti
        except ImportError:
            try:
                import qnn_wrapper
                return qnn_wrapper
            except ImportError:
                logger.warning("QNN not available, cannot set up QNN")
    
    # WebGPU and WebNN require browser integration
    # For this simplified version, return None
    elif platform_name in ['webgpu', 'webnn']:
        logger.warning(f"{platform_name} requires browser integration")
    
    else:
        logger.warning(f"Unknown platform: {platform_name}")
    
    return None


# Pytest skip decorators
def skip_if_no_cuda(func):
    """
    Skip test if CUDA is not available.
    
    Args:
        func: Test function
        
    Returns:
        Decorated test function
    """
    import pytest
    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info['platforms']['cuda']['available'],
        reason="CUDA not available"
    )(func)


def skip_if_no_rocm(func):
    """
    Skip test if ROCm is not available.
    
    Args:
        func: Test function
        
    Returns:
        Decorated test function
    """
    import pytest
    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info['platforms']['rocm']['available'],
        reason="ROCm not available"
    )(func)


def skip_if_no_mps(func):
    """
    Skip test if MPS is not available.
    
    Args:
        func: Test function
        
    Returns:
        Decorated test function
    """
    import pytest
    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info['platforms']['mps']['available'],
        reason="MPS not available"
    )(func)


def skip_if_no_openvino(func):
    """
    Skip test if OpenVINO is not available.
    
    Args:
        func: Test function
        
    Returns:
        Decorated test function
    """
    import pytest
    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info['platforms']['openvino']['available'],
        reason="OpenVINO not available"
    )(func)


def skip_if_no_qnn(func):
    """
    Skip test if QNN is not available.
    
    Args:
        func: Test function
        
    Returns:
        Decorated test function
    """
    import pytest
    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info['platforms']['qnn']['available'],
        reason="QNN not available"
    )(func)


def skip_if_no_webgpu(func):
    """
    Skip test if WebGPU is not available.
    
    Args:
        func: Test function
        
    Returns:
        Decorated test function
    """
    import pytest
    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info['platforms']['webgpu']['available'],
        reason="WebGPU not available"
    )(func)


def skip_if_no_webnn(func):
    """
    Skip test if WebNN is not available.
    
    Args:
        func: Test function
        
    Returns:
        Decorated test function
    """
    import pytest
    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info['platforms']['webnn']['available'],
        reason="WebNN not available"
    )(func)


# Detect hardware on module import
HARDWARE_INFO = detect_hardware()

if __name__ == "__main__":
    # Print hardware information when run directly
    import json
    print(json.dumps(detect_hardware(), indent=2))