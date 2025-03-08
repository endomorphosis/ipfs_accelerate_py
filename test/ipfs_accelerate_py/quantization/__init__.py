"""
Quantization module for IPFS Accelerate SDK.

This module provides tools for model quantization across different
hardware platforms and precision levels.
"""

from ipfs_accelerate_py.quantization.quantization_engine import (
    QuantizationEngine,
    QuantizationConfig,
    CalibrationDataset
)

__all__ = [
    'QuantizationEngine',
    'QuantizationConfig',
    'CalibrationDataset'
]