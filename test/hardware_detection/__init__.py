#!/usr/bin/env python3
"""
Hardware detection module for the test framework.
Provides reliable detection of hardware capabilities.
"""

from .capabilities import (
    detect_all_hardware,
    HAS_CUDA,
    HAS_ROCM,
    HAS_OPENVINO,
    HAS_MPS,
    HAS_WEBNN,
    HAS_WEBGPU
)