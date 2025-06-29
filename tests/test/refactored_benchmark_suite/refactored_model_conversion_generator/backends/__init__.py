"""
Model Conversion Backend Package

This package provides implementations for various model conversion backends.
"""

# Import backend implementations so they register themselves
from .pytorch_to_onnx import PyTorchToOnnxConverter
from .onnx_to_openvino import OnnxToOpenvinoConverter
from .onnx_to_webnn import OnnxToWebNNConverter
from .onnx_to_webgpu import OnnxToWebGPUConverter

__all__ = [
    'PyTorchToOnnxConverter',
    'OnnxToOpenvinoConverter',
    'OnnxToWebNNConverter',
    'OnnxToWebGPUConverter'
]