#!/usr/bin/env python3
"""
Hardware detection module for the test framework.
Provides reliable detection of hardware capabilities.

This package provides hardware detection and support for various hardware platforms:
    - CPU (x86, ARM)
    - CUDA (NVIDIA GPUs)
    - ROCm (AMD GPUs)
    - MPS (Apple Metal Performance Shaders)
    - OpenVINO (Intel Neural Compute Stick and CPUs)
    - QNN (Qualcomm Neural Networks) - Added March 2025
    - WebNN (Browser Neural Networks API)
    - WebGPU (Browser Graphics API)
"""

from .capabilities import (
    detect_all_hardware,
    HardwareDetector,
    HAS_CUDA,
    HAS_ROCM,
    HAS_OPENVINO,
    HAS_MPS,
    HAS_QNN,
    HAS_WEBNN,
    HAS_WEBGPU
)

# Optional imports for specific hardware platforms
try:
    from .qnn_support import (
        QNNCapabilityDetector,
        QNNPowerMonitor,
        QNNModelOptimizer
    )
    HAS_QNN = True
except ImportError:
    HAS_QNN = False