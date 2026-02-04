# ResourcePool Guide

## Overview

The ResourcePool is a centralized resource management system for efficiently sharing computational resources, models, and tokenizers across test execution and implementation validation. It helps avoid duplicate model loading, optimizes memory usage, and provides a clean interface for resource management.

## Latest Updates (March 2025)

The ResourcePool and test generation system now includes comprehensive improvements for model family detection, hardware compatibility, and error handling:

1. **WebNN Integration**: Full support for Web Neural Network API with hardware-aware resource allocation
2. **WebGPU Support**: Integration with WebGPU and transformers.js for browser-based acceleration
3. **Web Deployment Subfamilies**: Special subfamily handling for web deployment scenarios
4. **Extended Hardware Compatibility Matrix**: Improved compatibility information for web platforms
5. **Resilient Error Handling**: Robust error recovery for web platform detection
6. **Comprehensive Testing**: Extended test suite for WebNN and WebGPU platforms
7. **Simulation Mode**: Testing capability for web platforms in non-browser environments
8. **Enhanced Hardware Preferences**: Specialized configurations for web deployment scenarios
9. **Family-Based Optimizations**: Model family-specific settings for web platform deployment
10. **Browser-Specific Settings**: Optimizations specifically for browser environments
11. **Hardware Compatibility Error Reporting**: Intelligent error analysis with context-specific recommendations
12. **Error Recommendation System**: Automatic suggestion of alternative hardware when compatibility issues occur
13. **Persistent Error Reports**: Optional saving of structured error reports for offline analysis
14. **Model Family Error Integration**: Error reporting integrated with model family classification
15. **Enhanced Model Family Detection**: Improved heuristics for accurately classifying models by family
16. **Subfamily Support**: Specialized handling for model subfamilies like speech recognition models
17. **Multimodal Model Handling**: Special support for complex multimodal models like LLaVA, CLIP, and BLIP
18. **Fallback Mechanisms**: Graceful degradation when specialized model handling fails
19. **Test Generation Reliability**: Fixed common issues affecting test generation for key model types
20. **Robust Output Shape Analysis**: Improved model output analysis for consistent test case generation

## Features

- **Centralized Resource Management**: Share resources across tests and implementations
- **Model Caching**: Load models once and reuse them across multiple tests
- **Device-Specific Model Instances**: Cache separate models for different devices (CPU, CUDA, MPS)
- **Tokenizer Caching**: Efficiently share tokenizers for text models
- **Enhanced Memory Tracking**: Multiple methods for accurate memory usage estimation
- **Thread Safety**: Robust locking mechanisms for concurrent access
- **Intelligent Resource Cleanup**: Smart cleanup of unused resources based on access patterns
- **Detailed Statistics Tracking**: Comprehensive usage statistics with per-model metrics
- **Hardware-Aware Management**: Integration with hardware detection for optimal resource allocation
- **Model Family Integration**: Works with model family classifier for intelligent device selection
- **Multi-GPU Support**: Smart distribution of models across available GPUs
- **Low-Memory Mode**: Automatic adjustment for resource-constrained environments
- **Automatic Resource Monitoring**: Monitor resource usage and provide warnings
- **Hardware Compatibility Analysis**: Checks model compatibility with available hardware
- **Dynamic Device Selection**: Intelligently assigns devices based on model requirements
- **Cross-Platform Optimization**: Ensures efficient resource usage across different hardware platforms
- **WebNN/WebGPU Support**: Compatibility with web-based deployment platforms
- **Resilient Error Handling**: Graceful degradation when optional components are missing

## Usage

### Basic Usage

```python
from resource_pool import get_global_resource_pool

# Get the global resource pool instance
pool = get_global_resource_pool()

# Get or create a resource 
torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))

# Get or create a model with hardware awareness
model = pool.get_model(
    "embedding", 
    "bert-base-uncased", 
    constructor=lambda: transformers.AutoModel.from_pretrained("bert-base-uncased"),
    hardware_preferences={"device": "auto"}  # Automatically select optimal device
)

# Get or create a tokenizer
tokenizer = pool.get_tokenizer(
    "embedding", 
    "bert-base-uncased",
    constructor=lambda: transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
)

# Get resource pool usage statistics
stats = pool.get_stats()
print(f"Resource pool stats: {stats}")

# Clean up unused resources (resources not accessed in the last 30 minutes)
pool.cleanup_unused_resources(max_age_minutes=30)

# Or clear all resources
pool.clear()
```

### Integration with Test Generator

The ResourcePool is deeply integrated with the test generator workflow, providing intelligent resource allocation and hardware selection during test generation and execution. The test generator creates test files that automatically use the ResourcePool with hardware-aware model loading.

#### Generating Tests with Hardware Awareness

```bash
# Generate a hardware-aware test file for bert-base-uncased
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --output-dir ./skills

# Generated tests include hardware-aware resource handling:
# - Automatic device selection based on model family
# - Memory-efficient model loading with hardware compatibility checks
# - Proper resource cleanup and memory management
# - Hardware-specific test cases optimized for available hardware
# - WebNN/WebGPU compatibility for web-based deployment scenarios
```

#### Advanced Test Generator Options

```bash
# Generate with debug logging to see detailed resource allocation
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --debug

# Clear the resource cache before generating
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --clear-cache

# Set a custom timeout for resource cleanup
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --timeout 60

# Use the model family classifier to optimize test generation
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --use-model-family

# Force a specific hardware backend
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --device cuda

# Use hardware detection cache to speed up generation
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --hw-cache ./hardware_cache.json

# Specify model database for improved classification
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --model-db ./model_database.json
```

#### How Tests Use ResourcePool

The generated test files use the ResourcePool in the following ways:

1. **Efficient Resource Sharing**: Models and tokenizers are loaded only once per test class
2. **Hardware-Aware Loading**: Automatic hardware detection and optimal device selection based on model family
3. **Memory-Efficient Execution**: Models are moved to the appropriate device only when needed
4. **Proper Cleanup**: Test classes include proper cleanup in teardown methods
5. **Comprehensive Statistics**: Tests output resource usage statistics when complete
6. **Multi-GPU Management**: Distribution of models across available GPUs for parallel processing
7. **Dynamic Hardware Fallback**: Graceful fallback to alternative hardware when preferred device is unavailable
8. **Memory Monitoring**: Real-time monitoring of memory usage with automatic optimizations
9. **Model Family Optimization**: Specialized handling based on model family characteristics
10. **WebNN/WebGPU Support**: Integration with web-based deployment platforms for cross-platform testing

## Resource Pool API

### `ResourcePool` Class

The main class for resource management.

#### Core Methods

- `get_resource(resource_type, resource_id=None, constructor=None)`: Get or create a general resource
- `get_model(model_type, model_name, constructor=None, hardware_preferences=None)`: Get or create a model with intelligent hardware selection
- `get_tokenizer(model_type, model_name, constructor=None)`: Get or create a tokenizer
- `cleanup_unused_resources(max_age_minutes=30)`: Clean up resources that haven't been used recently
- `get_stats()`: Get detailed usage statistics including per-model memory usage and system information
- `clear()`: Clear all cached resources with proper memory handling
- `get_resource_details(resource_key)`: Get detailed information about a specific resource
- `move_model_to_device(model_type, model_name, device)`: Move a cached model to a different device
- `check_memory_pressure()`: Check if the system is under memory pressure
- `enable_low_memory_mode()`: Enable optimizations for resource-constrained environments
- `get_recommended_device(model_type, model_family=None)`: Get recommended device for a specific model type
- `generate_error_report(model_name, hardware_type, error_message, stack_trace=None)`: Generate a detailed error report with hardware compatibility analysis and recommendations
- `save_error_report(report, output_dir="./hardware_reports")`: Save error report to file for offline analysis

### `get_global_resource_pool()`

A function that returns the global ResourcePool instance for sharing across modules.

### Hardware Preferences Dictionary

When calling `get_model()`, you can pass a hardware_preferences dictionary with the following options:

```python
hardware_preferences = {
    # Basic device selection
    "device": "auto",  # or "cuda", "cpu", "mps", "cuda:1", "webnn", "webgpu", etc.
    
    # Advanced hardware selection
    "priority_list": ["cuda", "mps", "webnn", "webgpu", "cpu"],  # Hardware types in order of preference
    "preferred_index": 0,  # For multi-GPU systems, which GPU to prefer
    
    # Hardware compatibility information
    "hw_compatibility": {
        "cuda": {"compatible": True, "memory_usage": {"peak": 1200}},
        "mps": {"compatible": True},
        "openvino": {"compatible": True},
        "webnn": {"compatible": True},
        "webgpu": {"compatible": True}
    },
    
    # Memory management
    "force_low_memory": False,  # Force low-memory optimizations
    "precision": "fp16",  # Request mixed precision if available
    
    # Web deployment specific options
    "model_family": "embedding",  # Used for web platform optimization
    "subfamily": "web_deployment",  # Special handling for web deployment
    "fallback_to_simulation": True,  # Allow fallback to simulation mode for web platforms
    "browser_optimized": True  # Enable browser-specific optimizations
}
```

### Web Platform Specific Preferences

```python
# Embedding model optimized for WebNN
webnn_preferences = {
    "priority_list": ["webnn", "webgpu", "cpu"],
    "model_family": "embedding",
    "subfamily": "web_deployment",
    "description": "Web deployment optimized for embedding models"
}

# Vision model optimized for WebGPU
webgpu_preferences = {
    "priority_list": ["webgpu", "webnn", "cpu"],
    "model_family": "vision",
    "subfamily": "web_deployment",
    "description": "Web deployment optimized for vision models"
}
```

## Memory Management Benefits

The ResourcePool provides several memory management benefits:

1. **Reduced Memory Usage**: By sharing resources across tests, memory usage is significantly reduced
2. **Optimized Model Loading**: Models are only loaded once, even when used in multiple tests
3. **CUDA Optimization**: Proper handling of CUDA resources with automatic cache clearing
4. **Garbage Collection**: Automatic cleanup of unused resources to free memory
5. **Resource Sharing**: Efficient sharing of common libraries like PyTorch and Transformers
6. **Memory Pressure Detection**: Automatic detection of system memory pressure with mitigation strategies
7. **Multiple GPU Management**: Distribution of models across available GPUs for optimal memory usage
8. **Precision Control**: Support for mixed precision to reduce memory footprint when appropriate
9. **Per-Device Optimization**: Device-specific memory management strategies for CPU, CUDA, MPS, etc.
10. **Memory Monitoring**: Comprehensive memory statistics for both system and device memory
11. **Low-Memory Mode**: Automatic or manual enabling of optimizations for resource-constrained environments
12. **Intelligent Fallbacks**: Graceful degradation when memory requirements exceed available resources
13. **WebNN/WebGPU Support**: Memory-efficient deployment options for web-based platforms

## Integration with Test-Driven Development Workflow

The ResourcePool integrates with the test-driven development workflow in the following ways:

1. **Test Generation**: Used in test generators to share resources during test generation
2. **Test Execution**: Used in tests to share models and tokenizers across test cases
3. **Implementation Validation**: Used in implementation validators to efficiently test implementations
4. **Continuous Integration**: Optimizes resource usage in CI/CD pipelines with intelligent caching
5. **Hardware Detection Integration**: Works with hardware_detection module for platform-aware resource allocation
6. **Model Family Classification**: Integrates with model_family_classifier for optimized template selection
7. **Cross-Platform Testing**: Enables testing across different hardware platforms with consistent interfaces
8. **Multi-GPU Optimization**: Distributes workloads efficiently in multi-GPU environments
9. **Web Platform Support**: Facilitates testing for WebNN/WebGPU deployment scenarios
10. **Unified Hardware Interface**: Provides a consistent API across CPU, CUDA, MPS, ROCm, and OpenVINO backends
11. **Memory-Aware Test Scaling**: Adapts test complexity based on available system resources
12. **Distributed Test Execution**: Supports parallel testing with efficient resource sharing
13. **Automated Hardware Selection**: Chooses optimal hardware based on model characteristics

## Thread Safety

The ResourcePool is thread-safe and can be used in multi-threaded environments. All resource access is protected by a thread lock to ensure consistency.

## Hardware Awareness and Intelligent Device Selection

The ResourcePool integrates with the hardware_detection module to automatically determine the optimal device for models based on their characteristics and available hardware. This integration uses model family classification to make intelligent decisions about resource allocation.

### Automatic Hardware Selection

```python
from resource_pool import get_global_resource_pool
from transformers import AutoModel

# Get resource pool - hardware detection happens automatically
pool = get_global_resource_pool()

# Define model constructor - the pool will decide optimal device
def create_model():
    return AutoModel.from_pretrained("bert-base-uncased")

# The ResourcePool automatically detects hardware capabilities,
# classifies the model family, and chooses the optimal device
model = pool.get_model(
    "embedding",  # Model family helps optimize hardware selection
    "bert-base-uncased", 
    constructor=create_model
)

# The selected device is logged during model loading:
# "Selected optimal device for embedding:bert-base-uncased: cuda:0"
```

### Advanced Hardware Selection

```python
from resource_pool import get_global_resource_pool
from hardware_detection import detect_available_hardware

# Get comprehensive hardware information
hardware_info = detect_available_hardware()

# Create hardware-aware preferences with specific needs
hardware_preferences = {
    "priority_list": ["cuda", "mps", "cpu"],  # Hardware priority order
    "preferred_index": 0,                     # Use primary GPU
    "precision": "fp16",                     # Use mixed precision when available
    "hw_compatibility": {
        "cuda": {"compatible": True, "memory_usage": {"peak": 500}},
        "mps": {"compatible": True},
        "openvino": {"compatible": True}
    }
}

# Load model with these specific hardware preferences
pool = get_global_resource_pool()
model = pool.get_model(
    "embedding",
    "bert-base-uncased",
    constructor=lambda: AutoModel.from_pretrained("bert-base-uncased"),
    hardware_preferences=hardware_preferences
)
```

### Manual Hardware Preferences

```python
from hardware_detection import detect_available_hardware, CUDA, CPU, MPS, OPENVINO
from resource_pool import get_global_resource_pool
from transformers import AutoModel

# Get comprehensive hardware information
hw_info = detect_available_hardware()
best_device = hw_info["torch_device"]

# Get resource pool
pool = get_global_resource_pool()

# Set hardware preferences to override automatic selection
hardware_preferences = {
    "device": best_device,
    "force_device": True,  # Skip compatibility checks
    "precision": "fp16"    # Request mixed precision if available
}

# Pass hardware preferences to override automatic device selection
model = pool.get_model(
    "embedding", 
    "bert-base-uncased", 
    constructor=lambda: AutoModel.from_pretrained("bert-base-uncased"),
    hardware_preferences=hardware_preferences
)

# You can also request device-specific versions:
cpu_model = pool.get_model(
    "embedding", 
    "bert-base-uncased", 
    constructor=lambda: AutoModel.from_pretrained("bert-base-uncased"),
    hardware_preferences={"device": "cpu"}
)

# Or use the hardware constants for more reliable selection:
cuda_model = pool.get_model(
    "embedding", 
    "bert-base-uncased", 
    constructor=lambda: AutoModel.from_pretrained("bert-base-uncased"),
    hardware_preferences={"priority_list": [CUDA, CPU]}
)

# The pool maintains separate instances for each unique hardware configuration
```

### Smart Model Family Based Decisions

The ResourcePool's device selection logic considers model family characteristics with comprehensive hardware compatibility matrix:

| Model Family | CUDA | ROCm (AMD) | MPS (Apple) | OpenVINO | WebNN | WebGPU | Special Handling |
|--------------|------|------------|-------------|----------|-------|--------|------------------|
| Text Generation (LLMs) | ✅ High | ✅ Medium | ✅ Medium | ✅ Low | ✅ Low* | ✅ Low* | Memory requirements checked against GPU VRAM. *Only tiny models in browsers |
| Embedding (BERT, etc.) | ✅ High | ✅ High | ✅ High | ✅ Medium | ✅ High | ✅ Medium | Efficient on all hardware. Best on WebNN for browsers |
| Vision (ViT, CLIP, etc.) | ✅ High | ✅ Medium | ✅ High | ✅ High | ✅ Medium | ✅ High | OpenVINO for servers, WebGPU for browsers |
| Audio (Whisper, etc.) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ❌ N/A | ❌ N/A | CUDA preferred for performance |
| Multimodal (LLaVA, etc.) | ✅ High | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | Typically only fully supported on CUDA |

The ResourcePool's advanced decision system checks each model against this compatibility matrix and makes intelligent decisions about device placement based on:
- Available hardware on the current system
- Model size and memory requirements
- Known hardware compatibility issues
- Optimal performance characteristics
- Model family-specific optimizations
- Web platform deployment needs

You can see this comprehensive system in action by running the enhanced hardware test:

```bash
python scripts/generators/models/test_comprehensive_hardware.py --test all
```

This will show detailed hardware-aware model classification results like:

```
Model: bert-base-uncased
  Family: embedding (confidence: 0.75)
  Optimal Hardware: cuda
  PyTorch Device: cuda:0
  Hardware Priority: cuda > mps > cpu
  Recommended Template: hf_embedding_template.py
  Hardware Compatibility:
    cuda: ✅ (System: ✅, Effective: ✅)
    mps: ✅ (System: ❌, Effective: ❌)
    openvino: ✅ (System: ✅, Effective: ✅)
    webnn: ✅ (System: ❌, Effective: ❌)
    webgpu: ✅ (System: ❌, Effective: ❌)
```

### Enhanced Memory Tracking

The ResourcePool uses multiple methods to accurately track memory usage across all system and device resources:

```python
pool = get_global_resource_pool()
stats = pool.get_stats()

# See overall memory usage
print(f"Total memory usage: {stats['memory_usage_mb']:.2f} MB")

# Get system memory status
system_memory = stats.get("system_memory", {})
if system_memory:
    available_mb = system_memory.get("available_mb", 0)
    percent_used = system_memory.get("percent_used", 0)
    print(f"System memory: {available_mb:.2f} MB available, {percent_used}% used")
    
    # Check for memory pressure
    under_pressure = system_memory.get("under_pressure", False)
    if under_pressure:
        print("System is under memory pressure - consider clearing resources")

# Get CUDA memory status
cuda_memory = stats.get("cuda_memory", {})
if cuda_memory and cuda_memory.get("device_count", 0) > 0:
    print("CUDA memory stats:")
    for device in cuda_memory.get("devices", []):
        print(f"  Device {device['id']} ({device['name']}): {device['free_mb']:.2f} MB free, {device['percent_used']:.1f}% used")

# Get detailed per-model memory information
for key in pool.models:
    if key in pool._stats:
        memory_mb = pool._stats[key].get('memory_usage', 0) / (1024 * 1024)
        method = pool._stats[key].get('memory_estimation_method', 'unknown')
        device = pool._stats[key].get('device', 'unknown')
        print(f"Model {key}: {memory_mb:.2f} MB (method: {method}, device: {device})")
```

## Example Test Implementation

Here's an example of a test implementation that uses the ResourcePool with all the advanced features:

```python
import os
import sys
import logging
import unittest
from typing import Dict, Any, Optional

# Import the resource pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestBertEmbedding(unittest.TestCase):
    """Test case for BERT embedding model with hardware-aware resource management"""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class - load model once for all tests"""
        # Use resource pool for efficient resource sharing
        pool = get_global_resource_pool()
        
        # Load dependencies
        cls.torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Define model constructor with hardware awareness
        def create_model():
            from transformers import AutoModel
            return AutoModel.from_pretrained("bert-base-uncased")
        
        # Set hardware preferences based on model family
        hardware_preferences = {
            "priority_list": ["cuda", "mps", "openvino", "cpu"],
            "preferred_index": 0,
            "precision": "fp16" if cls.torch.cuda.is_available() else None
        }
        
        # Get or create model with hardware awareness
        cls.model = pool.get_model(
            "embedding",  # Model family helps with hardware selection
            "bert-base-uncased", 
            constructor=create_model,
            hardware_preferences=hardware_preferences
        )
        
        # Define tokenizer constructor
        def create_tokenizer():
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Get or create tokenizer
        cls.tokenizer = pool.get_tokenizer(
            "embedding", 
            "bert-base-uncased",
            constructor=create_tokenizer
        )
        
        # Get device from model (ResourcePool already placed it on optimal device)
        if hasattr(cls.model, "device"):
            cls.device = cls.model.device
        else:
            # Fallback to parameter device detection
            try:
                cls.device = next(cls.model.parameters()).device
            except (StopIteration, AttributeError):
                # Final fallback
                cls.device = cls.torch.device("cpu")
        
        logger.info(f"Model loaded and ready on device: {cls.device}")
    
    def test_model_embedding(self):
        """Test the model embedding functionality"""
        # Run inference with proper device handling
        inputs = self.tokenizer("Hello, world!", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Validate outputs with proper assertions
        self.assertIsNotNone(outputs, "Model outputs should not be None")
        self.assertTrue(hasattr(outputs, "last_hidden_state"), 
                       "Output should have last_hidden_state attribute")
        self.assertEqual(outputs.last_hidden_state.shape[0], 1, 
                        "Batch size should be 1")
    
    def test_device_compatibility(self):
        """Test that the model is on the correct device"""
        # Check model device matches expected device
        if hasattr(self.model, "device"):
            model_device = str(self.model.device)
            device_str = str(self.device)
            self.assertEqual(model_device, device_str, 
                          f"Model should be on {device_str}, but is on {model_device}")
        
        # Check parameter device
        param_device = str(next(self.model.parameters()).device)
        device_str = str(self.device)
        self.assertEqual(param_device, device_str,
                       f"Parameters should be on {device_str}, but are on {param_device}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        # Get resource pool stats for reporting
        pool = get_global_resource_pool()
        stats = pool.get_stats()
        logger.info(f"Resource pool stats: {stats}")
        
        # Clean up unused resources to prevent memory leaks
        # Short timeout for cleanup in test environment
        pool.cleanup_unused_resources(max_age_minutes=0.1)  # 6 seconds

if __name__ == "__main__":
    unittest.main()
```

## Monitoring Resource Usage

The ResourcePool provides comprehensive usage statistics that can be accessed through the `get_stats()` method:

```python
pool = get_global_resource_pool()
stats = pool.get_stats()
print(f"Resource pool stats: {stats}")
```

The comprehensive stats include:
- **Cache Statistics**:
  - Hit count (number of cache hits)
  - Miss count (number of cache misses)
  - Hit ratio (ratio of hits to total requests)
  - Resource counts (total resources, models, tokenizers)

- **Memory Usage**:
  - Total memory usage (in MB)
  - Per-model memory usage with estimation method
  - Per-device memory usage

- **System Resources**:
  - System memory availability and usage
  - Memory pressure detection
  - CPU utilization

- **GPU Resources** (when available):
  - Per-GPU memory allocation
  - Per-GPU utilization
  - Tensor core availability
  - CUDA/ROCm version information
  
- **Other Hardware**:
  - MPS availability and usage (Apple Silicon)
  - OpenVINO device information
  - WebNN/WebGPU compatibility
  
- **Resource Management**:
  - Low memory mode status
  - Cleanup statistics
  - Resource age tracking

## Best Practices

1. **Use Model Family Information**: Always specify the model family when requesting models for optimal hardware selection
2. **Provide Constructors**: Always provide a constructor function when requesting resources
3. **Clean Up Regularly**: Call `cleanup_unused_resources()` periodically to free memory
4. **Share Dependencies**: Use the ResourcePool to share common dependencies like PyTorch and Transformers
5. **Monitor Memory Usage**: Check the resource pool stats to monitor memory usage and system pressure
6. **Hardware-Aware Test Design**: Design tests to leverage the hardware-aware resource allocation system
7. **Use Hardware Preferences**: Provide hardware_preferences to specify device selection for different model types
8. **Leverage Multi-GPU Systems**: Use preferred_index in hardware_preferences to distribute models across multiple GPUs
9. **Define Custom Priority Lists**: Create model-specific hardware priority lists for optimal performance
10. **Integrate Hardware Detection**: Use the hardware_detection module for automatic device selection
11. **Use Model Family Classification**: Leverage model family classification for optimized resource allocation
12. **Enable Low-Memory Mode**: Use low-memory mode on resource-constrained systems or set environment variable RESOURCE_POOL_LOW_MEMORY=1
13. **Consider Model Compatibility**: Be aware of model-specific hardware compatibility issues across platforms
14. **Implement Robust Error Handling**: Use file existence checks and try/except blocks for optional components
15. **Test Integration**: Use test_comprehensive_hardware.py to verify all components work together
16. **Use Modern Unittest Framework**: Leverage Python unittest for test organization and proper cleanup
17. **Centralize Resource Pool Access**: Always use get_global_resource_pool() to ensure synchronized resource access
18. **Platform-Specific Optimizations**: Consider platform-specific needs in hardware_preferences
19. **Web Platform Support**: Use WebNN/WebGPU compatibility options for web deployment scenarios:
   ```python
   # Example of web platform preferences for browser deployment
   web_preferences = {
       "priority_list": ["webnn", "webgpu", "cpu"],
       "model_family": "embedding",
       "subfamily": "web_deployment",
       "fallback_to_simulation": True
   }
   ```
20. **Memory Pressure Monitoring**: Check for system memory pressure and trigger cleanup when needed

## Limitations

1. **Memory Estimation**: Memory usage estimation is approximate, especially for complex models
2. **Resource Dependencies**: Resources with complex dependencies may not be properly tracked
3. **Non-Python Resources**: The ResourcePool is designed for Python resources and may not work well with non-Python resources
4. **WebNN/WebGPU Limitations**: Web platform integration requires additional setup and has limited hardware access
5. **Deep Module Dependencies**: Resources with complex import hierarchies may need manual dependency management
6. **Mixed Precision Limitations**: Not all models support efficient mixed precision usage
7. **Cross-Backend Optimizations**: Some optimizations are specific to particular hardware backends
8. **Dynamic Graph Models**: Models using dynamic computation graphs may have less accurate memory estimation
9. **Custom CUDA Kernels**: Models with custom CUDA kernels may require special handling
10. **Distributed Training**: ResourcePool focuses on inference cases, distributed training requires additional consideration

## Resilient Error Handling

The ResourcePool system is designed with robust error handling to function gracefully even when optional components are missing:

1. **Modular Dependencies**: Each component checks for the existence of its dependencies before attempting to use them
2. **Graceful Degradation**: When scripts/generators/hardware/hardware_detection.py or model_family_classifier.py are missing, the system falls back to basic functionality
3. **Runtime Feature Detection**: Components automatically detect available features at runtime rather than failing during import
4. **Comprehensive Error Logging**: Clear error messages explain what features are missing and how the system is adapting
5. **Self-Testing Capability**: The run_integrated_hardware_model_test.py script verifies all components work together correctly
6. **Dynamic Import System**: Optional components are imported only when needed to minimize potential import errors
7. **Cross-Platform Compatibility**: Fallback mechanisms ensure functionality across different platforms
8. **Contextual Error Handling**: Error handling strategies are tailored to specific usage contexts
9. **Hardware Fallback Chain**: Well-defined fallback path from specialized hardware to general-purpose hardware
10. **Automatic Memory Recovery**: System can detect and recover from out-of-memory situations
11. **Component Status Monitoring**: Continuous monitoring of component status during operation
12. **WebNN/WebGPU Adaptation**: Special handling for web platform deployment scenarios
13. **File Existence Checks**: System checks for the presence of optional module files before attempting imports
14. **Component-Aware Operation**: ResourcePool adapts its behavior based on available components at runtime
15. **Multi-Level Fallbacks**: Progressive fallback mechanisms with detailed logging at each step
16. **Isolated Component Failures**: Failures in one component do not prevent other components from functioning
17. **Web Platform Error Management**: Robust detection and handling of WebNN and WebGPU errors
18. **Simulation Mode Fallbacks**: Automatic fallback to simulation mode when real web platform implementations are unavailable
19. **Subfamily-Based Error Handling**: Specialized error handling strategies for web deployment subfamilies
20. **Hardware Compatibility Reporting**: Comprehensive error reporting system for hardware compatibility issues
21. **Error Analysis and Recommendations**: Intelligent analysis of errors with context-specific recommendations
22. **Alternative Hardware Suggestions**: Automatic suggestion of alternative hardware when compatibility issues occur
23. **Persistent Error Reports**: Optional saving of structured error reports for offline analysis
24. **Memory-Specific Recommendations**: Special error handling for out-of-memory conditions with tailored suggestions
25. **Operation Compatibility Analysis**: Identification of unsupported operations with platform-specific alternatives
26. **Driver Version Management**: Suggestions for driver updates and version compatibility checks
27. **Model Family-Aware Error Handling**: Integration of model family information into error analysis and recommendations

## Version History

### v2.3 (March 2025)
- Added WebNN and WebGPU simulation mode for testing without browsers
- Created specialized web subfamily configurations for various model families
- Implemented browser-specific optimizations for web deployment
- Added model size limitation controls for browser deployment
- Created comprehensive WEB_PLATFORM_INTEGRATION_GUIDE.md
- Enhanced web platform integration test suite with new commands
- Added browser compatibility information to hardware matrices
- Implemented fallback mechanisms for web platform unavailability

### v2.2 (March 2025)
- Added comprehensive WebNN/WebGPU support for web platform deployment
- Enhanced resilient error handling with multi-level fallback mechanisms
- Implemented intelligent model family-based hardware compatibility matrix
- Added advanced memory pressure detection and mitigation strategies
- Improved hardware-model integration with comprehensive template system
- Enhanced documentation with updated best practices and troubleshooting
- Added cross-platform optimization strategies for consistent model behavior
- Implemented automatic mixed precision selection based on hardware capabilities
- Improved hardware detection with Qualcomm AI and TPU support
- Fixed CPU-only mode issues for environments without GPU
- Added comprehensive hardware compatibility error reporting system
- Implemented intelligent error analysis with context-specific recommendations
- Added persistent error report saving for offline troubleshooting

### v2.1 (February 2025)
- Added integration with enhanced hardware detection system for automatic device selection
- Implemented intelligent model family-based device assignment
- Added support for multi-GPU environments with customizable device distribution
- Enhanced memory tracking and monitoring with better pressure detection
- Improved compatibility checking for model-hardware combinations
- Added experimental support for TPU detection and allocation
- Fixed issues with model relocalization for device transitions
- Implemented ROCm/AMD GPU detection and optimization

### v2.0 (January 2025)
- Added device-specific caching mechanism for model instances
- Implemented hardware priority configuration for device selection
- Added detailed hardware compatibility matrix for model families
- Enhanced cleanup mechanism with memory pressure detection
- Added extensive statistics tracking for resource usage
- Implemented thread-safe operations for concurrent access
- Added OpenVINO integration for specialized hardware acceleration
- Significantly improved documentation and usage examples
- Implemented MPS (Apple Silicon) support for macOS platforms

### v1.0 (December 2024)
- Initial release with basic resource sharing capabilities
- Support for model and tokenizer caching
- Basic memory tracking and resource cleanup
- Integration with simple hardware detection
- Foundational test generator integration