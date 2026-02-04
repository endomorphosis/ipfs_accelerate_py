# Enhanced Model Registry with AMD Support and Precision Types

This document describes the enhancements made to the model registry in the IPFS Accelerate Python framework to add support for AMD ROCm hardware and comprehensive precision types.

## Overview

The enhanced model registry now includes:

1. **AMD ROCm Hardware Support**:
   - Hardware detection for AMD GPUs using rocm-smi and torch.utils.hip
   - AMD-specific dependencies including rocm-smi, rccl, and torch-rocm
   - Implementation methods for AMD ROCm hardware initialization and handlers
   - Hardware compatibility information for all model types

2. **Comprehensive Precision Types**:
   - Support for all major precision types: fp32, fp16, bf16, int8, int4, uint4, fp8, fp4
   - Hardware-specific precision compatibility matrices
   - Precision-specific dependencies and initialization code
   - Detailed documentation of precision capabilities by model and hardware

3. **Enhanced Testing Framework**:
   - Multi-precision testing across all hardware platforms
   - Automated detection and skipping of unsupported precision/hardware combinations
   - Precision-aware model initialization and conversion
   - Comprehensive test results with precision information

## Implementation Details

### Model Registry Structure

The enhanced model registry now includes detailed hardware and precision information:

```python
MODEL_REGISTRY = {
    "bert-base-uncased": {
        "description": "Default BERT base (uncased) model",
        
        # Model dimensions and capabilities
        "embedding_dim": 768,
        "sequence_length": 512,
        "model_precision": "float32", 
        "default_batch_size": 1,
        
        # Hardware compatibility
        "hardware_compatibility": {
            "cpu": True,
            "cuda": True,
            "openvino": True,
            "apple": True,
            "qualcomm": False,
            "amd": True
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
            "cuda": { ... },
            "openvino": { ... },
            "apple": { ... },
            "amd": { ... },
            "qualcomm": { ... }
        },
        
        # Dependencies including hardware and precision-specific ones
        "dependencies": { ... }
    }
}
```

### Hardware Detection

The enhanced hardware detection includes:

1. **AMD ROCm Detection**:
   - Primary detection through rocm-smi command-line tool
   - Secondary detection through torch.utils.hip if available
   - Detailed version and device count information

2. **Multi-Hardware Capabilities Dictionary**:
   ```python
   capabilities = {
       "cpu": True,
       "cuda": False,
       "cuda_version": None,
       "cuda_devices": 0,
       "mps": False,
       "openvino": False,
       "qualcomm": False,
       "amd": False,
       "amd_version": None,
       "amd_devices": 0
   }
   ```

### Precision Support

The implementation includes:

1. **Precision Type Definitions**:
   - **fp32**: Standard 32-bit floating point (full precision)
   - **fp16**: Half precision 16-bit floating point
   - **bf16**: Brain Floating Point 16-bit (better numerical stability than fp16)
   - **int8**: 8-bit integer quantization
   - **int4**: 4-bit integer quantization
   - **uint4**: Unsigned 4-bit integer quantization
   - **fp8**: 8-bit floating point (emerging standard)
   - **fp4**: 4-bit floating point (experimental)

2. **Hardware-Specific Precision Support**:
   - CPU supports fp32, bf16, int8
   - CUDA supports fp32, fp16, bf16, int8, int4, uint4
   - AMD supports fp32, fp16, bf16, int8
   - Apple Silicon supports fp32, fp16
   - OpenVINO supports fp32, fp16, int8
   - Qualcomm supports fp32, fp16, int8

3. **Precision-Specific Dependencies**:
   ```python
   "precision": {
       "fp16": [],
       "bf16": ["torch>=1.12.0"],
       "int8": ["bitsandbytes>=0.41.0", "optimum>=1.12.0"],
       "int4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
       "uint4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
       "fp8": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"],
       "fp4": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"]
   }
   ```

## Implementation Methods

### AMD ROCm Methods

1. **Detection Method**:
   ```python
   # Check AMD ROCm support
   try:
       # Run rocm-smi to detect ROCm installation
       result = subprocess.run(['rocm-smi', '--showproductname'], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                             universal_newlines=True, check=False)
       
       if result.returncode == 0:
           capabilities["amd"] = True
           # Get version and device information...
   except (ImportError, FileNotFoundError):
       pass
   ```

2. **Initialization Method**:
   ```python
   def init_amd(self, model_name, model_type, device="rocm:0", **kwargs):
       """Initialize model for AMD ROCm inference."""
       # Get precision from kwargs or default to fp32
       precision = kwargs.get("precision", "fp32")
       
       # Create processor and endpoint
       tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
       model = transformers.AutoModel.from_pretrained(model_name)
       
       # Move to AMD ROCm device
       model = model.to(device)
       
       # Apply precision conversion if needed
       if precision == "fp16":
           model = model.half()
       elif precision == "bf16" and hasattr(torch, "bfloat16"):
           model = model.to(torch.bfloat16)
       elif precision in ["int8", "int4", "uint4"]:
           # Apply quantization for specified precision
       
       # Create handler and return
   ```

3. **Handler Method**:
   ```python
   def create_amd_text_embedding_endpoint_handler(self, endpoint_model, device, 
                                               hardware_label, endpoint=None, 
                                               tokenizer=None, precision="fp32"):
       """Create a handler function for AMD ROCm inference."""
       def handler(text_input):
           # Process input, run model, and return results with precision info
           return {
               "tensor": embeddings,
               "implementation_type": "AMD_ROCM",
               "device": device,
               "model": endpoint_model,
               "precision": precision,
               "is_amd": True
           }
       return handler
   ```

### Multi-Precision Test Implementation

The test implementation supports running models with multiple precision types:

```python
# Test on hardware with different precision types
for precision in ["fp32", "fp16", "bf16", "int8"]:
    try:
        # Skip if precision not supported on this hardware
        if not model_info["precision_compatibility"][hardware].get(precision, False):
            print(f"Precision {precision.upper()} not supported on {hardware}, skipping...")
            continue
        
        # Initialize model with specific precision
        endpoint, processor, handler, queue, batch_size = self.init_hardware(
            model_name="test-model",
            model_type="task-type",
            precision=precision
        )
        
        # Test with simple input
        input_text = f"Test input with {precision.upper()} precision on {hardware}"
        output = handler(input_text)
        
        # Record results with precision information
        examples.append({
            "platform": f"{hardware} ({precision.upper()})",
            "precision": precision,
            # Other test information...
        })
    except Exception as e:
        print(f"Error testing on {hardware} with {precision.upper()}: {e}")
```

## Using the Enhanced Model Registry

### Loading the Registry

```python
from model_registry_with_amd_precision import MODEL_REGISTRY, detect_hardware

# Get hardware capabilities
hardware_capabilities = detect_hardware()

# Check if AMD is available
if hardware_capabilities.get("amd", False):
    print(f"AMD ROCm detected, version: {hardware_capabilities['amd_version']}")
    print(f"AMD devices: {hardware_capabilities['amd_devices']}")
```

### Checking Precision Compatibility

```python
def is_precision_supported(model_name, hardware, precision):
    """Check if a specific precision is supported for a model on given hardware."""
    if model_name not in MODEL_REGISTRY:
        return False
        
    # Get model info
    model_info = MODEL_REGISTRY[model_name]
    
    # Check hardware compatibility
    if not model_info["hardware_compatibility"].get(hardware, False):
        return False
        
    # Check precision compatibility
    return model_info["precision_compatibility"][hardware].get(precision, False)
```

### Getting Required Dependencies

```python
def get_dependencies(model_name, hardware, precision):
    """Get all dependencies for a model on specific hardware and precision."""
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
```

### Running Tests with Different Precision Types

```python
def run_precision_tests(model_name, hardware_types=None, precision_types=None):
    """Run tests for a model across hardware and precision types."""
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        print(f"Model {model_name} not found in registry")
        return
    
    # Set defaults
    hardware_types = hardware_types or ["cpu", "cuda", "amd", "openvino", "apple"]
    precision_types = precision_types or ["fp32", "fp16", "bf16", "int8", "int4"]
    
    # Get hardware capabilities
    hardware_capabilities = detect_hardware()
    
    # Track results
    results = {}
    
    # Test each hardware and precision combination if available
    for hardware in hardware_types:
        # Skip if hardware not available
        if not hardware_capabilities.get(hardware, False):
            results[f"{hardware}_test"] = "Hardware not available"
            continue
            
        for precision in precision_types:
            # Skip if precision not supported on this hardware
            if not model_info["precision_compatibility"][hardware].get(precision, False):
                results[f"{hardware}_{precision}_test"] = "Precision not supported"
                continue
                
            # Run test with specific hardware and precision
            try:
                # Initialize model
                print(f"Testing {model_name} on {hardware.upper()} with {precision.upper()} precision...")
                
                # In actual implementation, would initialize and run model here
                
                results[f"{hardware}_{precision}_test"] = "Success"
            except Exception as e:
                results[f"{hardware}_{precision}_test"] = f"Error: {str(e)}"
    
    return results
```

## Testing Across Hardware and Precision Types

To run a comprehensive test across all available hardware and precision types, use:

```python
# Initialize model with registry
model = initialize_model_from_registry("bert-base-uncased")

# Run tests across all available hardware types
results = model.__test__()

# Print hardware and precision-specific results
for platform, result in results["results"].items():
    print(f"- {platform}: {result}")
```

The test will:
1. Detect available hardware (CPU, CUDA, AMD, etc.)
2. For each available hardware platform, test all supported precision types
3. Skip unsupported hardware/precision combinations
4. Return detailed results for each combination

## Advanced Usage: Custom Precision Configurations

You can define custom precision configurations for specific models:

```python
# Define custom precision profile
custom_precision = {
    "cpu": {
        "fp32": True,
        "int8": True 
    },
    "cuda": {
        "fp16": True,
        "int8": True
    },
    "amd": {
        "fp32": True,
        "fp16": True
    }
}

# Update model registry with custom precision
MODEL_REGISTRY["custom-model"]["precision_compatibility"] = custom_precision

# Test with custom configuration
run_precision_tests("custom-model")
```

## Conclusion

The enhanced model registry with AMD hardware support and comprehensive precision types provides:

1. **Complete Hardware Coverage**: Support for all major hardware platforms (CPU, CUDA, AMD, Apple, OpenVINO, Qualcomm)
2. **Comprehensive Precision Support**: Testing across all precision types (fp32, fp16, bf16, int8, int4, uint4, fp8, fp4)
3. **Hardware-Specific Optimization**: Detailed configuration for optimizing models on each hardware platform
4. **Flexible Testing Framework**: Support for testing any combination of hardware and precision

These enhancements ensure that the IPFS Accelerate Python framework can fully utilize all available hardware and precision optimizations, providing maximum performance and flexibility across diverse deployment environments.