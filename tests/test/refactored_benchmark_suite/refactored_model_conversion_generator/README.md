# Refactored Model Conversion Generator

A comprehensive tool for converting AI models between different formats, with support for optimizations and hardware-specific implementations.

## Overview

The Model Conversion Generator provides utilities for converting models between various formats:

- **PyTorch → ONNX**: Convert PyTorch models to ONNX format
- **ONNX → OpenVINO**: Convert ONNX models to OpenVINO IR format
- **ONNX → WebNN**: Convert ONNX models to WebNN-compatible JavaScript modules
- **ONNX → WebGPU**: Convert ONNX models to WebGPU-compatible JavaScript modules

The tool is designed to be extensible and can be easily expanded to support additional formats and hardware platforms.

## Directory Structure

```
refactored_model_conversion_generator/
├── README.md                 # This file
├── __init__.py               # Package exports
├── __main__.py               # Command line interface
├── core/                     # Core components
│   ├── __init__.py           # Package exports
│   ├── converter.py          # Base model converter
│   └── registry.py           # Converter registry
├── backends/                 # Converter implementations
│   ├── __init__.py           # Package exports
│   ├── pytorch_to_onnx.py    # PyTorch to ONNX converter
│   ├── onnx_to_openvino.py   # ONNX to OpenVINO converter
│   ├── onnx_to_webnn.py      # ONNX to WebNN converter
│   └── onnx_to_webgpu.py     # ONNX to WebGPU converter
├── templates/                # Template files for model conversion
├── utils/                    # Utility functions
│   ├── __init__.py           # Package exports
│   ├── hardware_detection.py # Hardware detection utilities
│   ├── file_management.py    # File management utilities
│   ├── logging_utils.py      # Logging utilities
│   └── verification.py       # Model verification utilities
└── tests/                    # Test cases
```

## Features

- **Multiple Format Support**: Convert between various model formats (PyTorch, ONNX, OpenVINO, WebNN, WebGPU)
- **Hardware Detection**: Automatically detect available hardware (CPU, CUDA, ROCm, MPS, etc.)
- **Model Verification**: Verify model structure and compatibility before conversion
- **Caching**: Cache converted models for improved performance
- **Ultra-low Precision Quantization**: Comprehensive precision options from float32 down to 2-bit quantization
- **Memory Optimization**: Up to 87.5% memory reduction with ultra-low precision techniques
- **Mixed Precision Support**: Different precision levels for different parts of the model
- **Command-line Interface**: Easy-to-use CLI for model conversion and management

## Usage

### Command Line Interface

```bash
# Convert a PyTorch model to ONNX
python -m refactored_model_conversion_generator convert \
    --source model.pth \
    --source-format pytorch \
    --target-format onnx \
    --model-type bert \
    --output model.onnx

# Convert an ONNX model to OpenVINO
python -m refactored_model_conversion_generator convert \
    --source model.onnx \
    --source-format onnx \
    --target-format openvino \
    --model-type bert \
    --output model.xml
    
# Convert with mixed precision quantization
python -m refactored_model_conversion_generator convert \
    --source model.onnx \
    --source-format onnx \
    --target-format webgpu \
    --precision 4bit \
    --mixed-precision-config mixed_precision.json \
    --output model.webgpu.js

# Generate a quantization configuration
python -m refactored_model_conversion_generator quantize \
    --model model.onnx \
    --format onnx \
    --precision mixed \
    --output mixed_precision.json

# List available converters
python -m refactored_model_conversion_generator list

# Detect available hardware
python -m refactored_model_conversion_generator detect

# Find model files in a directory
python -m refactored_model_conversion_generator find \
    --directory /path/to/models \
    --formats onnx,pytorch \
    --recursive

# Verify a model
python -m refactored_model_conversion_generator verify \
    --model model.onnx \
    --format onnx
```

### Python API

```python
from refactored_model_conversion_generator import ModelConverterRegistry, HardwareDetector

# Get available hardware
hardware_info = HardwareDetector.get_available_hardware()
print(f"CUDA available: {hardware_info['cuda']['available']}")

# Convert model
converter_class = ModelConverterRegistry.get_converter(
    source_format='pytorch', 
    target_format='onnx',
    model_type='bert'
)
converter = converter_class()
result = converter.convert(
    model_path='model.pth', 
    output_path='model.onnx',
    model_type='bert'
)

if result.success:
    print(f"Conversion successful: {result.output_path}")
else:
    print(f"Conversion failed: {result.error}")
```

## Supported Hardware

- **CPU**: Intel, AMD, ARM
- **GPU**:
  - NVIDIA (CUDA)
  - AMD (ROCm)
  - Apple (Metal Performance Shaders)
- **Neural Processing Units**:
  - Intel Neural Compute Stick (OpenVINO)
  - Qualcomm Neural Processing SDK
- **Web**:
  - WebNN (browser neural network acceleration)
  - WebGPU (browser GPU acceleration)

## Supported Models

- **Transformer Models**: BERT, GPT-2, DistilBERT, RoBERTa, T5, etc.
- **Vision Models**: ViT, CLIP, ResNet, DenseNet, EfficientNet, MobileNet, etc.
- **Audio Models**: Whisper, Wav2Vec2, HuBERT, etc.

## Quantization Options

### Precision Levels

The framework supports multiple precision levels for optimized model conversion:

| Precision | Bits per Value | Memory Reduction | Note |
|-----------|---------------|-----------------|------|
| default (float32) | 32 | 0% | Full precision for maximum accuracy |
| float16 | 16 | 50% | Good balance of precision and size |
| int8/8bit | 8 | 75% | Standard quantization |
| 4bit | 4 | 87.5% | Ultra-low precision |
| 3bit | 3 | 90.6% | Advanced ultra-low precision |
| 2bit | 2 | 93.75% | Extreme memory optimization |

### Mixed Precision

The framework supports mixed precision quantization, where different parts of the model use different precision levels for optimal balance between size and accuracy.

```bash
# Convert model with 4-bit precision
python -m refactored_model_conversion_generator convert \
    --source model.onnx \
    --source-format onnx \
    --target-format webgpu \
    --precision 4bit \
    --output model.webgpu.js

# Convert model with 2-bit precision (maximum compression)
python -m refactored_model_conversion_generator convert \
    --source model.onnx \
    --source-format onnx \
    --target-format webgpu \
    --precision 2bit \
    --output model.webgpu.js
```

## Extending the Framework

To add support for a new converter:

1. Create a new file in the `backends` directory
2. Implement a converter class that inherits from `ModelConverter`
3. Register the converter using the `@register_converter` decorator

Example:

```python
from refactored_model_conversion_generator.core import ModelConverter, ConversionResult
from refactored_model_conversion_generator.core import register_converter

@register_converter(source_format='tensorflow', target_format='onnx')
class TensorflowToOnnxConverter(ModelConverter):
    def _get_source_format(self) -> str:
        return 'tensorflow'
        
    def _get_target_format(self) -> str:
        return 'onnx'
        
    def _get_supported_model_types(self) -> List[str]:
        return ['bert', 'vit', 'resnet']
        
    def _execute_conversion(self, model_path: str, output_path: str, 
                           model_type: Optional[str] = None, **kwargs) -> ConversionResult:
        # Implement conversion logic here
        pass
```

## Requirements

- Python 3.8+
- PyTorch (for PyTorch conversion)
- ONNX (for ONNX conversion)
- OpenVINO (for OpenVINO conversion)
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.