# Enhanced Model Registry with AMD and Precision Support

This guide provides quick instructions for using the enhanced model registry with AMD hardware support and comprehensive precision types in the IPFS Accelerate Python framework.

## Overview

The enhanced model registry provides:

1. **Support for All Major Hardware Platforms**:
   - CPU
   - NVIDIA CUDA
   - AMD ROCm
   - Apple Silicon (MPS)
   - Intel OpenVINO
   - Qualcomm AI

2. **Comprehensive Precision Types**:
   - fp32: Standard 32-bit floating point
   - fp16: 16-bit half precision
   - bf16: 16-bit Brain Floating Point
   - int8: 8-bit integer quantization
   - int4: 4-bit integer quantization
   - uint4: 4-bit unsigned integer quantization
   - fp8: 8-bit floating point (emerging standard)
   - fp4: 4-bit floating point (experimental)

3. **Hardware-Specific Optimization Information**:
   - Hardware-specific dependencies
   - Precision compatibility matrices
   - Initialization methods for each hardware type

## Quick Start

### 1. Basic Usage

```python
# Import the enhanced model registry
from model_registry_with_amd_precision import MODEL_REGISTRY, detect_hardware

# Detect available hardware
hardware_capabilities = detect_hardware()

# Check if specific hardware is available
if hardware_capabilities.get("amd", False):
    print(f"AMD ROCm detected with {hardware_capabilities['amd_devices']} devices")
    
# Check model information
model_info = MODEL_REGISTRY["bert"]
print(f"Model description: {model_info['description']}")
```

### 2. Checking Precision Compatibility

```python
def is_precision_supported(model_name, hardware, precision):
    """Check if a precision is supported for a model on specific hardware."""
    if model_name not in MODEL_REGISTRY:
        return False
        
    model_info = MODEL_REGISTRY[model_name]
    
    # Check hardware compatibility first
    if not model_info["hardware_compatibility"].get(hardware, False):
        return False
        
    # Check precision compatibility
    return model_info["precision_compatibility"][hardware].get(precision, False)
    
# Example usage:
supported = is_precision_supported("bert", "amd", "bf16")
print(f"bf16 precision on AMD ROCm is supported: {supported}")
```

### 3. Getting Dependencies

```python
def get_dependencies(model_name, hardware, precision):
    """Get all dependencies for a model on given hardware and precision."""
    if model_name not in MODEL_REGISTRY:
        return []
        
    model_info = MODEL_REGISTRY[model_name]
    
    # Start with base dependencies
    dependencies = list(model_info["dependencies"]["pip"])
    
    # Add hardware-specific dependencies
    if hardware in model_info["dependencies"]["optional"]:
        dependencies.extend(model_info["dependencies"]["optional"][hardware])
    
    # Add precision-specific dependencies
    if precision in model_info["dependencies"]["precision"]:
        dependencies.extend(model_info["dependencies"]["precision"][precision])
    
    return dependencies
    
# Example usage:
deps = get_dependencies("bert", "amd", "int8")
print(f"Dependencies for BERT on AMD with INT8: {deps}")
```

### 4. Testing with Different Precision Types

```python
# Initialize model
model = initialize_model_from_registry("bert")

# Specify hardware and precision to test
results = model.test_with_precision(
    hardware="amd",
    precision="fp16",
    input_data="This is a test input"
)

print(f"Test results with AMD ROCm and FP16: {results}")
```

### 5. Running the Demonstration Script

You can run the demonstration script to see the enhanced model registry in action:

```bash
# Run basic demo with default model
python sample_tests/demonstrate_amd_precision.py

# Test specific model
python sample_tests/demonstrate_amd_precision.py --model bert-base-uncased

# Test specific hardware types
python sample_tests/demonstrate_amd_precision.py --hardware cpu,cuda,amd

# Test specific precision types
python sample_tests/demonstrate_amd_precision.py --precision fp32,fp16,int8
```

## Files and Components

1. **model_registry_with_amd_precision.py**: Standalone implementation of the enhanced model registry
2. **test_hf_bert_base_uncased_with_amd.py**: Test implementation for BERT with AMD and precision support
3. **demonstrate_amd_precision.py**: Demonstration script showing usage of the enhanced model registry

For detailed implementation details, see [AMD_PRECISION_README.md](AMD_PRECISION_README.md)

## Adding Support for New Models

To add support for a new model with AMD and precision information:

```python
# Define model information
model_info = {
    "description": "My custom model",
    "embedding_dim": 1024,
    "sequence_length": 768,
    "model_precision": "float32",
    "default_batch_size": 1,
    
    # Hardware compatibility
    "hardware_compatibility": {
        "cpu": True,
        "cuda": True,
        "openvino": True,
        "apple": True,
        "qualcomm": False,
        "amd": True  # AMD ROCm support
    },
    
    # Precision support by hardware
    "precision_compatibility": {
        "cpu": {
            "fp32": True,
            "fp16": False,
            "bf16": True,
            "int8": True,
            "int4": False,
            "uint4": False,
            "fp8": False,
            "fp4": False
        },
        # Define for other hardware...
    },
    
    # Input/Output specifications
    "input": { ... },
    "output": { ... },
    
    # Dependencies
    "dependencies": { ... }
}

# Add to model registry
MODEL_REGISTRY["my-custom-model"] = model_info
```

## Conclusion

The enhanced model registry provides a comprehensive framework for managing model compatibility across different hardware platforms and precision types. By using this registry, you can:

1. Detect available hardware on the system
2. Check which precision formats are supported for each model on different hardware
3. Get the necessary dependencies for specific hardware and precision combinations
4. Test models across multiple hardware platforms and precision types

This enables maximum flexibility and performance optimization for deploying models across diverse hardware environments.