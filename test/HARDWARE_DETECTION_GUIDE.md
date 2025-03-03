# Hardware Detection Guide

## Overview

The Hardware Detection module provides comprehensive detection of available hardware capabilities for optimal resource allocation and device selection. It supports a wide range of hardware platforms including CPU, CUDA (NVIDIA), ROCm (AMD), MPS (Apple Silicon), OpenVINO, WebNN, WebGPU, and Qualcomm AI.

## Features

- **Multi-Platform Support**: Detects and classifies all major hardware acceleration platforms
- **Detailed Capability Analysis**: Provides detailed information about hardware capabilities and limitations
- **Comprehensive Error Handling**: Resilient detection with graceful fallbacks and error recovery
- **Memory Information**: Reports available memory across different hardware platforms
- **Cache Support**: Optional caching of detection results for performance optimization
- **Hardware Priority System**: Configurable priority lists for hardware selection
- **Device Index Selection**: Support for selecting specific devices in multi-GPU environments
- **Hardware Compatibility Analysis**: Hardware-model compatibility checking for intelligent allocation
- **Multiple Detection Levels**: Standard and comprehensive detection options with different detail levels
- **Extensible Architecture**: Easily extensible for new hardware platforms
- **Thread Safety**: Thread-safe design for concurrent environments

## Usage

### Basic Usage

```python
from hardware_detection import detect_available_hardware

# Get hardware information with standard detection
hardware_info = detect_available_hardware()

# Access detected hardware types
available_hardware = hardware_info["hardware"]
for hw_type, available in available_hardware.items():
    if available:
        print(f"{hw_type} is available")

# Get the best available hardware
best_hardware = hardware_info["best_available"]
print(f"Best available hardware: {best_hardware}")

# Get recommended PyTorch device
torch_device = hardware_info["torch_device"]
print(f"Recommended PyTorch device: {torch_device}")

# Access detailed hardware information
details = hardware_info["details"]
if "cuda" in details and details["cuda"]["device_count"] > 1:
    print(f"Multiple CUDA devices available: {details['cuda']['device_count']}")
```

### Comprehensive Detection

```python
from hardware_detection import detect_hardware_with_comprehensive_checks

# Get comprehensive hardware information with detailed capabilities
hardware_info = detect_hardware_with_comprehensive_checks()

# Check system information
system_info = hardware_info["system"]
print(f"Platform: {system_info['platform']}")
print(f"CPU cores: {system_info['cpu_count']}")
print(f"Available memory: {system_info.get('available_memory', 'Unknown')} GB")

# Check CUDA capabilities if available
if hardware_info.get("cuda", False):
    cuda_devices = hardware_info.get("cuda_devices", [])
    for i, device in enumerate(cuda_devices):
        print(f"CUDA device {i}: {device['name']}")
        print(f"  Memory: {device['total_memory']:.2f} GB")
        print(f"  Compute capability: {device.get('compute_capability', 'Unknown')}")
```

### Custom Hardware Priority

```python
from hardware_detection import detect_available_hardware, CUDA, MPS, OPENVINO, CPU

# Define a custom hardware priority list
priority_list = [MPS, CUDA, OPENVINO, CPU]  # Prioritize Apple Silicon

# Get hardware information with custom priority
hardware_info = detect_available_hardware(priority_list=priority_list)

# The best available hardware will be chosen based on the priority list
best_hardware = hardware_info["best_available"]
print(f"Best available hardware based on priority list: {best_hardware}")

# The recommended device reflects the priority settings
device = hardware_info["torch_device"]
print(f"Recommended PyTorch device: {device}")
```

### Device Selection in Multi-GPU Systems

```python
from hardware_detection import detect_available_hardware

# Get hardware information with specific GPU device preference
hardware_info = detect_available_hardware(preferred_device_index=1)  # Use second GPU if available

# Get the recommended PyTorch device with index
device = hardware_info["torch_device"]
print(f"Recommended device with index preference: {device}")  # Will show cuda:1 if available
```

### Caching Detection Results

```python
from hardware_detection import detect_available_hardware

# Get hardware information with caching
cache_file = "hardware_cache.json"
hardware_info = detect_available_hardware(cache_file=cache_file)

# Subsequent calls will use the cached results for faster performance
hardware_info2 = detect_available_hardware(cache_file=cache_file)  # Uses cache

# Force refresh when needed
hardware_info3 = detect_available_hardware(cache_file=cache_file, force_refresh=True)  # Ignores cache
```

## HardwareDetector Class API

The `HardwareDetector` class provides the foundation for hardware detection:

```python
from hardware_detection import HardwareDetector

# Create detector instance
detector = HardwareDetector(cache_file="hardware_cache.json")

# Get available hardware types
available_hardware = detector.get_available_hardware()

# Get detailed hardware information
hardware_details = detector.get_hardware_details()

# Get errors encountered during detection
errors = detector.get_errors()

# Check if specific hardware type is available
is_cuda_available = detector.is_available("cuda")

# Get best hardware type
best_hardware = detector.get_best_available_hardware()

# Get appropriate PyTorch device
torch_device = detector.get_torch_device()

# Get device with specific index for multi-GPU systems
torch_device_with_index = detector.get_device_with_index(preferred_index=1)

# Check hardware compatibility with specific model requirements
model_requirements = {
    "cuda": {"memory_required": 4000},  # MB
    "mps": {"compatible": True}
}
compatible_hw = detector.get_compatible_hardware_types(model_requirements)

# Select hardware based on priority list
priority_list = ["cuda", "mps", "cpu"]
selected_hw = detector.get_hardware_by_priority(priority_list)

# Get PyTorch device string with priority and index
device_str = detector.get_torch_device_with_priority(
    priority_list=priority_list,
    preferred_index=0
)

# Print summary of hardware detection
detector.print_summary(detailed=True)
```

## Hardware Types

The module provides constants for hardware types to ensure consistent usage:

```python
from hardware_detection import CPU, CUDA, ROCM, MPS, OPENVINO, WEBNN, WEBGPU, QUALCOMM

# These constants can be used in priority lists and compatibility checks
priority_list = [CUDA, ROCM, MPS, OPENVINO, CPU]
```

## Hardware Compatibility Matrix

The following table summarizes general compatibility across different hardware platforms:

| Hardware Type | Platform | Performance | Memory Model | Precision Support | Special Considerations |
|---------------|----------|-------------|--------------|-------------------|------------------------|
| CUDA | NVIDIA GPUs | High | Unified | FP32, FP16, INT8 | Most widely supported |
| ROCm | AMD GPUs | High | Unified | FP32, FP16, INT8 | Limited model support |
| MPS | Apple Silicon | Medium-High | Unified | FP32, FP16 | macOS only, some model limitations |
| OpenVINO | Intel CPUs/GPUs/VPUs | Medium | Varied | FP32, FP16, INT8 | Specialized for inference |
| WebNN | Browsers | Low-Medium | Limited | FP32, FP16 | Web deployment only |
| WebGPU | Modern browsers | Medium | Limited | FP32, FP16 | Limited model support |
| CPU | All platforms | Low | System RAM | FP32 | Universal fallback |

## Advanced Feature: Hardware-Model Integration

The hardware detection module can be integrated with the model family classifier for intelligent device selection:

```python
from hardware_detection import detect_available_hardware
from model_family_classifier import classify_model

# Classify a model
model_classification = classify_model("bert-base-uncased")
model_family = model_classification["family"]  # e.g., "embedding"

# Get hardware information
hardware_info = detect_available_hardware()
available_hardware = hardware_info["hardware"]

# Create model-specific hardware priority based on model family
if model_family == "embedding":
    # Embedding models work well on various hardware
    priority_list = ["cuda", "mps", "openvino", "cpu"]
elif model_family == "text_generation":
    # LLMs often need GPU memory
    priority_list = ["cuda", "cpu"]
elif model_family == "vision":
    # Vision models work well with OpenVINO
    priority_list = ["cuda", "openvino", "mps", "cpu"]
else:
    # Default priority
    priority_list = ["cuda", "rocm", "mps", "openvino", "cpu"]

# Filter by actually available hardware
filtered_priority = [hw for hw in priority_list if available_hardware.get(hw, False)]

# Get hardware with model-specific priority
model_specific_hw = detect_available_hardware(priority_list=filtered_priority)
recommended_device = model_specific_hw["torch_device"]
```

## Troubleshooting

### Hardware Not Detected

If hardware is not properly detected:

1. **Driver Issues**: Verify hardware drivers are properly installed (especially for CUDA/ROCm)
2. **Missing Libraries**: Ensure required libraries are installed (PyTorch, OpenVINO, etc.)
3. **Version Compatibility**: Check version compatibility between PyTorch and CUDA
4. **Environment Variables**: Verify no conflicting environment variables are set
5. **Permission Issues**: Check hardware access permissions, especially on Linux
6. **Check Detection Errors**: Examine the errors returned by `detector.get_errors()`
7. **Run Manual Tests**: Run `torch.cuda.is_available()` or equivalent direct detection
8. **Use Comprehensive Detection**: Try `detect_hardware_with_comprehensive_checks()` for more detailed analysis
9. **System Resources**: Ensure sufficient system resources are available
10. **Platform-Specific Setup**: Follow platform-specific setup requirements

### Incorrect Device Selection

If the wrong device is being selected:

1. **Priority Lists**: Verify your priority list includes the desired hardware
2. **Hardware Availability**: Confirm the desired hardware is actually available
3. **Device Indices**: Check device indices in multi-GPU systems
4. **Model Requirements**: Verify model requirements are correctly specified
5. **Cached Results**: Clear cache if hardware configuration has changed
6. **Run with Debug Logging**: Enable debug logging for more detailed information
7. **Force Device Selection**: Use explicit device specification to override automatic selection

## Testing Hardware Detection

To thoroughly test hardware detection, use the provided test script:

```bash
# Run the basic test
python test_comprehensive_hardware.py

# Run specific test components
python test_comprehensive_hardware.py --test detection
python test_comprehensive_hardware.py --test comparison
python test_comprehensive_hardware.py --test integration

# Run all tests
python test_comprehensive_hardware.py --test all
```

## Version History

### v2.0 (March 2025)
- Added WebNN/WebGPU comprehensive detection
- Enhanced ROCm detection with detailed capabilities
- Added Qualcomm AI Engine Direct detection
- Improved OpenVINO detection with device-specific details
- Enhanced resilient error handling with multi-level fallbacks
- Added TPU detection capabilities
- Improved multi-GPU support with enhanced device selection
- Added comprehensive system memory monitoring
- Enhanced cache system with validation

### v1.5 (February 2025)
- Added ROCm (AMD) detailed capability detection
- Enhanced MPS (Apple Silicon) support
- Added hardware compatibility matrix
- Implemented device index selection for multi-GPU systems
- Added custom priority lists for hardware selection
- Enhanced OpenVINO integration
- Improved error reporting and recovery

### v1.0 (January 2025)
- Initial implementation with basic detection
- Support for CPU, CUDA, MPS, and OpenVINO
- Basic error handling and reporting
- Simple caching mechanism
- Basic hardware selection logic