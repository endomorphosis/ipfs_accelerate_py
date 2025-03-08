"""
IPFS Accelerate Python SDK - High-level access to hardware acceleration for models

This module provides a comprehensive set of tools for accelerating models across different
hardware platforms, including CPU, GPU, specialized accelerators, and web platforms.

Features:
- Hardware detection and selection
- Model optimization and acceleration
- Benchmarking and performance analysis
- Cross-platform compatibility
- Advanced quantization and memory optimization
"""

__version__ = "0.5.0"  # Enhanced SDK version

# Import core components 
from ipfs_accelerate_py.hardware.hardware_profile import HardwareProfile
from ipfs_accelerate_py.hardware.hardware_detector import HardwareDetector
from ipfs_accelerate_py.worker.worker import Worker
from ipfs_accelerate_py.model.model_manager import ModelManager
from ipfs_accelerate_py.model.model_accelerator import ModelAccelerator
from ipfs_accelerate_py.benchmark.benchmark import Benchmark, BenchmarkConfig
from ipfs_accelerate_py.quantization.quantization_engine import QuantizationEngine

# Export all components
__all__ = [
    # New SDK components
    'HardwareProfile',
    'HardwareDetector',
    'Worker',
    'ModelManager',
    'ModelAccelerator',
    'Benchmark',
    'BenchmarkConfig',
    'QuantizationEngine',
]