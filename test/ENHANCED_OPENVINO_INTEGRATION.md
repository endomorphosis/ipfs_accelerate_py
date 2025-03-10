# Enhanced OpenVINO Integration Guide

This guide covers the enhanced OpenVINO integration for the IPFS Accelerate SDK. The implementation provides advanced features for model acceleration using Intel's OpenVINO toolkit with improved compatibility with legacy code.

## Overview

The enhanced OpenVINO backend provides the following key features:

- **Advanced optimum.intel Integration**: Seamless integration with HuggingFace Optimum for better performance and compatibility.
- **Enhanced INT8 Quantization**: Improved INT8 quantization with calibration data support for better performance.
- **Precision Controls**: Support for FP32, FP16, and INT8 precision with easy model conversion.
- **Model Format Conversion**: Built-in conversion between PyTorch, ONNX, and OpenVINO IR formats.
- **Legacy Code Integration**: Better compatibility with existing systems and workflows.
- **Comprehensive Device Support**: Support for CPU, GPU, VPU (Visual Processing Unit), and other Intel hardware.

## Key Enhancements

### 1. optimum.intel Integration

The backend now includes comprehensive integration with [optimum.intel](https://huggingface.co/docs/optimum/intel/inference), providing automatic detection and loading of HuggingFace models:

```python
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend

# Initialize backend
backend = OpenVINOBackend()

# Get optimum.intel integration status
optimum_info = backend.get_optimum_integration()
print(f"optimum.intel available: {optimum_info['available']}")
print(f"optimum.intel version: {optimum_info['version']}")

# Load a model with optimum.intel
config = {
    "device": "CPU",
    "model_type": "text",
    "precision": "FP32",
    "use_optimum": True  # Enable optimum.intel integration
}

# The backend will automatically use optimum.intel for HuggingFace models
result = backend.load_model("bert-base-uncased", config)

# Run inference
inference_result = backend.run_inference(
    "bert-base-uncased",
    "This is a test sentence for inference.",
    {"device": "CPU", "model_type": "text"}
)

# Print results
print(f"Latency: {inference_result['latency_ms']:.2f} ms")
print(f"Throughput: {inference_result['throughput_items_per_sec']:.2f} items/sec")
```

The improved integration automatically selects the appropriate model class based on the model type, architecture, and available classes in optimum.intel.

### 2. Enhanced INT8 Quantization

The backend now includes improved INT8 quantization with calibration data support:

```python
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
import numpy as np

# Initialize backend
backend = OpenVINOBackend()

# Generate calibration data
calibration_data = [
    {
        "input_ids": np.array([[101, 2054, 2003, 1996, 2034, 2035, 102]]),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1]])
    },
    {
        "input_ids": np.array([[101, 2430, 2001, 1996, 2034, 2035, 102]]),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1]])
    }
]

# Load model with INT8 quantization
config = {
    "device": "CPU",
    "model_type": "text",
    "precision": "INT8",
    "model_path": "model.onnx",
    "model_format": "ONNX",
    "calibration_data": calibration_data  # Provide calibration data for better quantization
}

result = backend.load_model("quant_model", config)
```

The backend supports two quantization paths:
- Basic INT8 quantization without calibration data (legacy compatibility)
- Advanced INT8 quantization with calibration data using POT (Post-training Optimization Tool) API

### 3. Multi-Precision Support

The backend supports multiple precision formats with easy conversion:

```python
# FP32 precision (default)
fp32_config = {
    "device": "CPU",
    "model_type": "text",
    "precision": "FP32",
    "model_path": "model.onnx"
}

# FP16 precision
fp16_config = {
    "device": "CPU",
    "model_type": "text",
    "precision": "FP16",
    "model_path": "model.onnx"
}

# INT8 precision
int8_config = {
    "device": "CPU",
    "model_type": "text",
    "precision": "INT8",
    "model_path": "model.onnx"
}

# Load models with different precisions
backend.load_model("fp32_model", fp32_config)
backend.load_model("fp16_model", fp16_config)
backend.load_model("int8_model", int8_config)
```

### 4. Model Format Conversion

The backend includes built-in model conversion from PyTorch and ONNX to OpenVINO IR format:

```python
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
import torch
from transformers import AutoModel

# Initialize backend
backend = OpenVINOBackend()

# Load PyTorch model
pt_model = AutoModel.from_pretrained("bert-base-uncased")

# Sample inputs for tracing
example_inputs = {"input_ids": torch.ones(1, 128, dtype=torch.long)}

# Convert to OpenVINO IR
result = backend.convert_from_pytorch(
    pt_model,
    example_inputs,
    "bert_openvino.xml",
    {"precision": "FP16"}
)

# Convert ONNX to OpenVINO IR
onnx_result = backend.convert_from_onnx(
    "model.onnx",
    "model_openvino.xml",
    {"precision": "FP16"}
)
```

## Integration with Legacy Code

The enhanced OpenVINO backend maintains full compatibility with legacy code:

```python
# Legacy code using standard OpenVINO
backend.load_model("legacy_model", {
    "device": "CPU",
    "model_path": "model.xml",
    "model_format": "IR"
})

# Enhanced code with optimum.intel integration
backend.load_model("new_model", {
    "device": "CPU",
    "model_type": "text",
    "use_optimum": True
})
```

## Testing the Enhanced Backend

A comprehensive test script is provided to validate the enhanced OpenVINO integration:

```bash
# Test optimum.intel integration
python generators/models/test_enhanced_openvino.py --test-optimum

# Test INT8 quantization
python generators/models/test_enhanced_openvino.py --test-int8

# Compare FP32, FP16, and INT8 precision performance
python generators/models/test_enhanced_openvino.py --compare-precisions

# Run all tests
python generators/models/test_enhanced_openvino.py --run-all

# Test with a specific model and device
python generators/models/test_enhanced_openvino.py --run-all --model bert-base-uncased --device CPU
```

## Performance Considerations

Different precision formats have different performance characteristics:

| Precision | Performance | Accuracy | Memory Usage | Use Case |
|-----------|-------------|----------|--------------|----------|
| FP32 | Baseline | Highest | Highest | High-precision requirements |
| FP16 | 1.5-2x faster | Slightly lower | ~50% less | General inference |
| INT8 | 2-4x faster | Lower | ~75% less | Production deployment |

### Best Practices for Performance

1. Use **FP16** for most use cases as it provides a good balance between performance and accuracy
2. Use **INT8** with calibration data for production deployment to maximize performance
3. Use **optimum.intel** integration for HuggingFace models to benefit from model-specific optimizations

## Device Selection

OpenVINO supports multiple device types:

- **CPU**: Available on all Intel CPUs (Xeon, Core, etc.)
- **GPU**: Available on Intel integrated and discrete GPUs
- **MYRIAD**: Available on Intel Neural Compute Stick 2 (NCS2)
- **HDDL**: Available on Intel Vision Accelerator Design with Intel Movidius VPUs
- **GNA**: Available on Intel Gaussian & Neural Accelerator
- **AUTO**: Automatically selects the best available device

To use a specific device:

```python
config = {
    "device": "GPU",  # Use Intel GPU
    "model_type": "text",
    "precision": "FP16"
}

backend.load_model("gpu_model", config)
```

## Template Integration

The enhanced backend includes template support for easy integration with the template system:

```python
# Generate a test with OpenVINO support
python generators/test_generators/simple_test_generator.py -g bert -p openvino -o test_bert_openvino.py

# Generate a test with multiple hardware support including OpenVINO
python generators/test_generators/simple_test_generator.py -g bert -p cpu,cuda,openvino -o test_bert_multi.py
```

## Additional Examples

### Using OpenVINO with Different Model Types

```python
# Text models (BERT, T5, etc.)
text_config = {
    "device": "CPU",
    "model_type": "text",
    "precision": "FP16",
    "use_optimum": True
}
backend.load_model("bert-base-uncased", text_config)

# Vision models (ViT, ResNet, etc.)
vision_config = {
    "device": "GPU",
    "model_type": "vision",
    "precision": "FP16",
    "use_optimum": True
}
backend.load_model("google/vit-base-patch16-224", vision_config)

# Audio models (Whisper, Wav2Vec2, etc.)
audio_config = {
    "device": "CPU",
    "model_type": "audio",
    "precision": "INT8",
    "use_optimum": True
}
backend.load_model("openai/whisper-small", audio_config)
```

### Advanced Configuration Options

```python
# CPU threads control
cpu_config = {
    "device": "CPU",
    "cpu_threads": 8,  # Use 8 CPU threads for inference
    "precision": "FP16"
}
backend.load_model("cpu_model", cpu_config)

# Cache directory for compiled models
cache_config = {
    "device": "CPU",
    "cache_dir": "/tmp/openvino_cache",  # Cache compiled models
    "precision": "FP16"
}
backend.load_model("cached_model", cache_config)

# Dynamic shapes control
shapes_config = {
    "device": "CPU",
    "dynamic_shapes": True,  # Enable dynamic shapes support
    "precision": "FP16"
}
backend.load_model("dynamic_model", shapes_config)
```

## Conclusion

The enhanced OpenVINO backend provides comprehensive support for Intel hardware with improved integration, quantization, and compatibility features. It maintains full compatibility with legacy code while providing advanced features for new deployments.

For detailed information, see the following resources:

- [OpenVINO Official Documentation](https://docs.openvino.ai/)
- [optimum.intel Documentation](https://huggingface.co/docs/optimum/intel/inference)
- [IPFS Accelerate SDK Documentation](README.md)