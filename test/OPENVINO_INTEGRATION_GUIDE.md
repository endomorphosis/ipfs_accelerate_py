# OpenVINO Integration Guide for IPFS Accelerate

This guide describes the implementation and usage of the OpenVINO backend for the IPFS Accelerate framework. OpenVINO provides hardware acceleration for Intel CPUs, GPUs, and specialized hardware like VPUs (Vision Processing Units).

## Overview

The OpenVINO backend allows the IPFS Accelerate framework to leverage Intel's hardware acceleration capabilities, including:

- Intel CPUs with optimized operations
- Intel integrated and discrete GPUs
- Intel Neural Compute Stick (NCS) and other VPUs
- Multiple device support with heterogeneous execution

The implementation provides a seamless interface that follows the same patterns as other hardware backends in the framework, making it easy to switch between different acceleration options based on availability and performance requirements.

## Implementation

The OpenVINO backend consists of the following components:

1. `OpenVINOBackend` class in `ipfs_accelerate_py/hardware/backends/openvino_backend.py`
2. Integration with the hardware detection system
3. Support for different model types and precision formats
4. Model conversion capabilities from PyTorch and ONNX formats

### Key Features

- **Comprehensive device detection**: Detects all available OpenVINO-compatible devices
- **Multiple device support**: Can use CPU, GPU, VPU, or AUTO devices
- **Precision control**: Supports FP32, FP16, and INT8 precisions
- **Model conversion**: Converts models from PyTorch and ONNX to OpenVINO format
- **Optimum.intel integration**: Optional integration with HuggingFace Optimum for easier model loading
- **Performance metrics**: Provides detailed performance metrics for benchmarking

## Usage

### Basic Usage

```python
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend

# Initialize the backend
backend = OpenVINOBackend()

# Check if OpenVINO is available
if backend.is_available():
    # Get available devices
    devices = backend.get_all_devices()
    print(f"Available devices: {devices}")
    
    # Load a model
    result = backend.load_model(
        "bert-base-uncased",
        {
            "device": "CPU",  # or "GPU", "MYRIAD", "AUTO", etc.
            "model_type": "text",
            "precision": "FP32"  # or "FP16", "INT8"
        }
    )
    
    # Run inference
    inputs = {
        "input_ids": [101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    inference_result = backend.run_inference(
        "bert-base-uncased",
        inputs,
        {"device": "CPU", "model_type": "text"}
    )
    
    # Print inference metrics
    print(f"Latency: {inference_result.get('latency_ms', 0):.2f} ms")
    print(f"Throughput: {inference_result.get('throughput_items_per_sec', 0):.2f} items/sec")
    print(f"Memory usage: {inference_result.get('memory_usage_mb', 0):.2f} MB")
    
    # Unload the model
    backend.unload_model("bert-base-uncased", "CPU")
```

### Integration with IPFS Accelerate Framework

The OpenVINO backend integrates with the IPFS Accelerate framework through the hardware detection and selection system:

```python
from ipfs_accelerate_py.hardware import HardwareDetector, HardwareProfile
from ipfs_accelerate_py.model import ModelAccelerator

# Initialize hardware detector
detector = HardwareDetector()
available_hardware = detector.get_available_hardware()

# Create hardware profile with OpenVINO preference
if available_hardware.get("openvino", False):
    profile = HardwareProfile(
        preferred_hardware=["openvino", "cpu"],
        device_map={"openvino": "CPU"}  # or "GPU", "AUTO", etc.
    )
else:
    # Fallback to CPU
    profile = HardwareProfile(preferred_hardware=["cpu"])

# Create model accelerator with profile
accelerator = ModelAccelerator(hardware_profile=profile)

# Load and accelerate a model
model = accelerator.load("bert-base-uncased", model_type="text")

# Run inference
result = model.run({"input_text": "Sample text to process"})
```

## Model Conversion

The OpenVINO backend provides model conversion capabilities to optimize models for Intel hardware:

### PyTorch to OpenVINO Conversion

```python
import torch
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend

# Create backend
backend = OpenVINOBackend()

# Create a PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# Create example inputs
example_inputs = torch.zeros(1, 10)

# Convert to OpenVINO format
result = backend.convert_from_pytorch(
    model,
    example_inputs,
    "converted_model.xml",
    {"precision": "FP16"}
)

print(f"Conversion result: {result}")
```

### ONNX to OpenVINO Conversion

```python
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend

# Create backend
backend = OpenVINOBackend()

# Convert ONNX model to OpenVINO format
result = backend.convert_from_onnx(
    "model.onnx",
    "converted_model.xml",
    {"precision": "FP16"}
)

print(f"Conversion result: {result}")
```

## Advanced Features

### Optimum.intel Integration

The OpenVINO backend can detect and use [optimum.intel](https://huggingface.co/docs/optimum/intel/inference) for easier integration with HuggingFace models:

```python
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend

# Create backend
backend = OpenVINOBackend()

# Check for optimum.intel integration
optimum_info = backend.get_optimum_integration()

if optimum_info.get("available", False):
    print(f"optimum.intel is available (version: {optimum_info.get('version', 'Unknown')})")
    
    # Check for specific model type support
    print(f"Sequence Classification: {optimum_info.get('sequence_classification_available', False)}")
    print(f"Causal LM: {optimum_info.get('causal_lm_available', False)}")
    print(f"Seq2Seq LM: {optimum_info.get('seq2seq_lm_available', False)}")
```

### Multiple Device Support

OpenVINO supports multiple execution devices and heterogeneous execution:

```python
from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend

# Create backend
backend = OpenVINOBackend()

# Load a model on AUTO device (OpenVINO selects the best available device)
backend.load_model("bert-base-uncased", {"device": "AUTO", "model_type": "text"})

# Load a model with heterogeneous execution (run on GPU with CPU fallback)
backend.load_model("bert-base-uncased", {"device": "HETERO:GPU,CPU", "model_type": "text"})

# Load a model with multi-device execution (run on both GPU and CPU)
backend.load_model("bert-base-uncased", {"device": "MULTI:GPU,CPU", "model_type": "text"})
```

## Performance Considerations

### Precision Selection

OpenVINO supports different precision formats that affect both performance and accuracy:

- **FP32**: Full precision, highest accuracy but lowest performance
- **FP16**: Half precision, good accuracy with improved performance
- **INT8**: Quantized precision, further improved performance with some accuracy loss

Choose the precision based on your accuracy requirements and hardware capabilities.

### Device Selection

Different OpenVINO devices have different performance characteristics:

- **CPU**: Available on all Intel CPUs, good for any model type
- **GPU**: Available on Intel integrated and discrete GPUs, good for vision models
- **MYRIAD/HDDL**: Available on Intel Neural Compute Stick and VPUs, good for vision models
- **AUTO**: OpenVINO automatically selects the best available device
- **HETERO**: Heterogeneous execution across multiple devices
- **MULTI**: Multi-device execution for improved throughput

### Benchmark Results

Typical performance improvement over standard CPU execution:

| Model Type | CPU Speedup | GPU Speedup | Power Efficiency |
|------------|-------------|-------------|------------------|
| Text Models | 1.3-1.5x | 1.5-2.5x | Moderate |
| Vision Models | 1.5-2.0x | 2.0-4.0x | High |
| Audio Models | 1.2-1.5x | 1.5-2.5x | Moderate |
| Multimodal Models | 1.2-1.8x | 1.5-3.0x | Moderate |

*Note: Performance can vary significantly based on specific hardware and model characteristics.*

## Troubleshooting

### Common Issues

1. **OpenVINO not detected**: Ensure OpenVINO is properly installed and the environment is configured

2. **Device not available**: Check that the requested device (GPU, MYRIAD, etc.) is connected and recognized

3. **Model loading fails**: Verify the model format and structure is compatible with OpenVINO

4. **Low performance**: Try different devices, precision formats, or OpenVINO configuration options

### Checking OpenVINO Installation

You can verify your OpenVINO installation with:

```python
import openvino
print(f"OpenVINO version: {openvino.__version__}")

from openvino.runtime import Core
core = Core()
print(f"Available devices: {core.available_devices}")
```

### Testing OpenVINO Backend

Use the provided test script to verify the OpenVINO backend:

```bash
# Test backend initialization
python generators/models/test_openvino_backend.py --test-init

# Test model operations
python generators/models/test_openvino_backend.py --test-model --model bert-base-uncased --device CPU

# Run benchmarks
python generators/models/test_openvino_backend.py --run-benchmarks --model bert-base-uncased --device CPU --iterations 10

# Compare with CPU performance
python generators/models/test_openvino_backend.py --compare-cpu --model bert-base-uncased --iterations 5

# Run all tests
python generators/models/test_openvino_backend.py --run-all
```

## Example Applications

Check the `ipfs_openvino_example.py` script for a complete example of using the OpenVINO backend with the IPFS Accelerate framework.

```bash
# Run simple inference example
python ipfs_openvino_example.py --model bert-base-uncased --model-type text --device CPU

# Run benchmark with specific model and device
python ipfs_openvino_example.py --benchmark --model bert-base-uncased --model-type text --device CPU --iterations 10

# Save benchmark results to JSON
python ipfs_openvino_example.py --benchmark --model bert-base-uncased --model-type text --device CPU --iterations 10 --output-json results.json

# Test different precisions
python ipfs_openvino_example.py --benchmark --model bert-base-uncased --model-type text --device CPU --precision FP16
```

## Integration Status

The OpenVINO backend has been successfully integrated with the IPFS Accelerate framework with the following components:

- ✅ Backend implementation complete
- ✅ Hardware detection integration
- ✅ Model loading and inference
- ✅ Multiple device support
- ✅ Precision control
- ✅ Performance metrics
- ✅ Model conversion utilities
- ✅ Example applications

Future enhancements:
- Better integration with optimum.intel
- Support for more quantization options
- Advanced model optimization techniques
- Direct integration with IPFS acceleration pipeline

## Resources

- [OpenVINO Official Documentation](https://docs.openvino.ai/)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum/intel/inference)
- [OpenVINO GitHub Repository](https://github.com/openvinotoolkit/openvino)