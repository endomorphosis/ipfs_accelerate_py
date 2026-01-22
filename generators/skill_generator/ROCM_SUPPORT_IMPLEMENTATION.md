# ROCm Support Implementation

This document details the implementation of AMD GPU (ROCm) support in the HuggingFace Model Test Generator Suite.

## Overview

ROCm (Radeon Open Compute) is AMD's open-source software platform for GPU computing. It provides a CUDA-compatible programming model, allowing many CUDA applications to run on AMD GPUs with minimal modification. The test generator suite now includes comprehensive support for ROCm, enabling hardware-aware testing on AMD GPUs.

## Implementation Details

### 1. Dual-Path Detection

ROCm support is detected through two mechanisms to maximize compatibility:

#### HIP API Detection
```python
if hasattr(torch, 'hip') and torch.hip.is_available():
    rocm_available = True
```

#### CUDA Compatibility Layer Detection
```python
elif torch.cuda.is_available():
    # Could be ROCm using CUDA API
    device_name = torch.cuda.get_device_name(0)
    if "AMD" in device_name or "Radeon" in device_name:
        rocm_available = True
```

This dual-path approach ensures detection works regardless of whether the ROCm installation exposes the HIP API directly or operates through the CUDA compatibility layer.

### 2. Environment Variable Support

The implementation checks for and respects ROCm-specific environment variables:

```python
# Check for HIP_VISIBLE_DEVICES environment variable
visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", None) or os.environ.get("CUDA_VISIBLE_DEVICES", None)
if visible_devices is not None:
    print(f"Using ROCm visible devices: {visible_devices}")
```

This provides compatibility with both ROCm-specific environment variables and CUDA environment variables when ROCm is used through the CUDA compatibility layer.

### 3. Half-Precision Support Detection

The implementation includes automatic detection and handling of half-precision (FP16) support:

```python
# Determine if we should use half precision
use_half = True
try:
    # Try to create a small tensor in half precision as a test
    test_tensor = torch.ones((10, 10), dtype=torch.float16, device="cuda")
    del test_tensor
    print("Half precision is supported on this AMD GPU")
except Exception as e:
    use_half = False
    print(f"Half precision not supported on this AMD GPU: {e}")
```

This ensures optimal performance on AMD GPUs that support half-precision, while gracefully falling back to full precision when not supported.

### 4. Memory Management

The implementation includes functionality to report and manage GPU memory:

```python
# Get the total GPU memory for logging purposes
try:
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
    print(f"AMD GPU memory: {total_mem:.2f} GB")
except Exception as e:
    print(f"Could not query AMD GPU memory: {e}")
```

Additionally, proper memory cleanup is implemented after operations:

```python
# Clean up memory
with torch.no_grad():
    if "cuda" in dir(torch):
        torch.cuda.empty_cache()
```

### 5. AMD-Specific Error Handling

The implementation includes robust error handling specific to AMD GPUs:

```python
try:
    # AMD GPU-specific operations
    # ...
except Exception as e:
    print(f"Error initializing ROCm endpoint: {e}")
    print("Creating mock implementation instead")
    return self._create_mock_endpoint(model_name, rocm_label)
```

This ensures graceful degradation when errors occur, with informative error messages and fallback to mock implementations.

## Integration with Templates

ROCm support is fully integrated into all template types:

1. **Base Hardware Template**: Common ROCm detection logic
2. **Architecture-Specific Templates**: ROCm-specific optimizations for different model architectures
3. **Reference Template**: Complete implementation for reference models

Each template includes:

- `init_rocm` method for ROCm initialization
- `create_rocm_*_endpoint_handler` methods for task-specific handlers
- ROCm-specific optimizations and fallbacks

## Testing and Verification

The implementation includes comprehensive testing and verification:

1. **ROCm Detection Test**: Verifies that ROCm is correctly detected
2. **Model Loading Test**: Tests loading and running a model on ROCm
3. **Template Verification**: Ensures all templates include proper ROCm support

These tests can be run using the included test scripts:

```bash
python test_rocm_detection.py
python test_rocm_detection.py --run-model
python verify_templates.py
```

## ROCm vs CUDA Performance Considerations

When generating tests for ROCm, several performance considerations are taken into account:

1. **Memory Optimization**: ROCm may require different memory optimization strategies than CUDA
2. **Half Precision Support**: Some AMD GPUs may have different half-precision performance characteristics
3. **Device Mapping**: The "auto" device mapping may work differently on ROCm
4. **AMD GPU Architecture**: Different AMD GPU architectures may have different performance characteristics

The implementation includes logging and diagnostics to help identify and address these considerations.

## Example Generated Code

Here's an example of ROCm-specific code generated by the test generator:

```python
def init_rocm(self, model_name, device, rocm_label):
    """Initialize model for ROCm (AMD GPU) inference."""
    self.init()
    
    # Check for ROCm availability
    rocm_available = False
    try:
        if hasattr(self.torch, 'hip') and self.torch.hip.is_available():
            rocm_available = True
        elif self.torch.cuda.is_available():
            # Could be ROCm using CUDA API
            device_name = self.torch.cuda.get_device_name(0)
            if "AMD" in device_name or "Radeon" in device_name:
                rocm_available = True
    except Exception as e:
        print(f"Error checking ROCm availability: {e}")
    
    if not rocm_available:
        print(f"ROCm not available, falling back to CPU")
        return self.init_cpu(model_name, "cpu", rocm_label.replace("rocm", "cpu"))
    
    print(f"Loading {model_name} for ROCm (AMD GPU) inference...")
    
    try:
        # Load model with appropriate precision for AMD GPU
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.torch.float16 if use_half else self.torch.float32,
            device_map="auto"
        )
        
        model.eval()
        
        # Create handler function
        handler = self.create_rocm_endpoint_handler(
            endpoint_model=model_name,
            device=device,
            hardware_label=rocm_label,
            endpoint=model,
            tokenizer=tokenizer
        )
        
        return model, tokenizer, handler, asyncio.Queue(32), 0
        
    except Exception as e:
        print(f"Error initializing ROCm endpoint: {e}")
        print("Creating mock implementation instead")
        return self._create_mock_endpoint(model_name, rocm_label)
```

## Future Enhancements

Future enhancements to ROCm support may include:

1. **ROCm-Specific Optimizations**: More fine-grained optimizations for different AMD GPU architectures
2. **Multiple GPU Support**: Improved support for multi-GPU setups with AMD GPUs
3. **Mixed Precision**: Better support for mixed precision on AMD GPUs
4. **ROCm Version Detection**: Detection and adaptation based on ROCm version
5. **HIP Kernel Fusion**: Optimizations using HIP kernel fusion when available

## Conclusion

The implementation of ROCm support in the HuggingFace Model Test Generator Suite enables comprehensive hardware-aware testing on AMD GPUs. By supporting both the HIP API and CUDA compatibility layer, the implementation maximizes compatibility with different ROCm installations and configurations. The robust error handling and graceful degradation ensure that tests remain functional even when ROCm-specific features are not available.