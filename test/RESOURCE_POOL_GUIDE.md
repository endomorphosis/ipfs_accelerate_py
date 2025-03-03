# ResourcePool Guide

## Overview

The ResourcePool is a centralized resource management system for efficiently sharing computational resources, models, and tokenizers across test execution and implementation validation. It helps avoid duplicate model loading, optimizes memory usage, and provides a clean interface for resource management.

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
python test_generator_with_resource_pool.py --model bert-base-uncased --output-dir ./skills

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
python test_generator_with_resource_pool.py --model bert-base-uncased --debug

# Clear the resource cache before generating
python test_generator_with_resource_pool.py --model bert-base-uncased --clear-cache

# Set a custom timeout for resource cleanup
python test_generator_with_resource_pool.py --model bert-base-uncased --timeout 60

# Use the model family classifier to optimize test generation
python test_generator_with_resource_pool.py --model bert-base-uncased --use-model-family

# Force a specific hardware backend
python test_generator_with_resource_pool.py --model bert-base-uncased --device cuda

# Use hardware detection cache to speed up generation
python test_generator_with_resource_pool.py --model bert-base-uncased --hw-cache ./hardware_cache.json

# Specify model database for improved classification
python test_generator_with_resource_pool.py --model bert-base-uncased --model-db ./model_database.json
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

### `get_global_resource_pool()`

A function that returns the global ResourcePool instance for sharing across modules.

### Hardware Preferences Dictionary

When calling `get_model()`, you can pass a hardware_preferences dictionary with the following options:

```python
hardware_preferences = {
    # Basic device selection
    "device": "auto",  # or "cuda", "cpu", "mps", "cuda:1", etc.
    
    # Advanced hardware selection
    "priority_list": ["cuda", "mps", "cpu"],  # Hardware types in order of preference
    "preferred_index": 0,  # For multi-GPU systems, which GPU to prefer
    
    # Hardware compatibility information
    "hw_compatibility": {
        "cuda": {"compatible": True, "memory_usage": {"peak": 1200}},
        "mps": {"compatible": True},
        "openvino": {"compatible": True}
    },
    
    # Memory management
    "force_low_memory": False,  # Force low-memory optimizations
    "precision": "fp16"  # Request mixed precision if available
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
| Text Generation (LLMs) | ✅ High | ✅ Medium | ✅ Medium | ✅ Low | ❌ | ✅ Low | Memory requirements checked against GPU VRAM |
| Embedding (BERT, etc.) | ✅ High | ✅ High | ✅ High | ✅ Medium | ✅ High | ✅ Medium | Efficient on all hardware |
| Vision (ViT, CLIP, etc.) | ✅ High | ✅ Medium | ✅ High | ✅ High | ✅ Medium | ✅ Medium | OpenVINO prioritized when available |
| Audio (Whisper, etc.) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ❌ | ❌ | CUDA preferred for performance |
| Multimodal (LLaVA, etc.) | ✅ High | ❌ Low | ❌ Low | ❌ Low | ❌ | ❌ | Typically only fully supported on CUDA |

The ResourcePool's advanced decision system checks each model against this compatibility matrix and makes intelligent decisions about device placement based on:
- Available hardware on the current system
- Model size and memory requirements
- Known hardware compatibility issues
- Optimal performance characteristics
- Model family-specific optimizations
- Web platform deployment needs

You can see this comprehensive system in action by running the enhanced hardware test:

```bash
python test_comprehensive_hardware.py --test all
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
19. **Web Platform Support**: Use WebNN/WebGPU compatibility options for web deployment scenarios
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
2. **Graceful Degradation**: When hardware_detection.py or model_family_classifier.py are missing, the system falls back to basic functionality
3. **Runtime Feature Detection**: Components automatically detect available features at runtime rather than failing during import
4. **Comprehensive Error Logging**: Clear error messages explain what features are missing and how the system is adapting
5. **Self-Testing Capability**: The test_comprehensive_hardware.py script can verify that all components work together correctly
6. **Dynamic Import System**: Optional components are imported only when needed to minimize potential import errors
7. **Cross-Platform Compatibility**: Fallback mechanisms ensure functionality across different platforms
8. **Contextual Error Handling**: Error handling strategies are tailored to specific usage contexts
9. **Hardware Fallback Chain**: Well-defined fallback path from specialized hardware to general-purpose hardware
10. **Automatic Memory Recovery**: System can detect and recover from out-of-memory situations
11. **Component Status Monitoring**: Continuous monitoring of component status during operation
12. **WebNN/WebGPU Adaptation**: Special handling for web platform deployment scenarios

Example of resilient hardware_detection integration:

```python
def get_best_device():
    """Get the best available device with comprehensive error handling"""
    # Try the most comprehensive approach first
    try:
        # First check if the advanced hardware detection file exists
        import os.path
        advanced_detector_path = os.path.join(os.path.dirname(__file__), "hardware_detection.py")
        
        if os.path.exists(advanced_detector_path):
            try:
                # Try to import and use comprehensive hardware detection
                from hardware_detection import detect_hardware_with_comprehensive_checks
                
                # Get detailed hardware information
                hardware_info = detect_hardware_with_comprehensive_checks()
                if "torch_device" in hardware_info:
                    logger.info(f"Using comprehensive hardware detection: {hardware_info['torch_device']}")
                    return hardware_info["torch_device"]
            except ImportError as e:
                logger.warning(f"Comprehensive hardware detection not available: {e}")
            except Exception as e:
                logger.warning(f"Error during comprehensive hardware detection: {e}")
        
        # Fall back to standard hardware detection
        try:
            if os.path.exists(advanced_detector_path):
                from hardware_detection import detect_available_hardware
                hardware_info = detect_available_hardware()
                if "torch_device" in hardware_info:
                    logger.info(f"Using standard hardware detection: {hardware_info['torch_device']}")
                    return hardware_info["torch_device"]
        except ImportError as e:
            logger.warning(f"Standard hardware detection not available: {e}")
        except Exception as e:
            logger.warning(f"Error during standard hardware detection: {e}")
        
        # Fall back to basic PyTorch detection
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Using basic CUDA detection: cuda")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Using basic MPS detection: mps")
                return "mps"
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
        except Exception as e:
            logger.warning(f"Error during basic PyTorch detection: {e}")
        
        # Ultimate fallback to CPU
        logger.info("Fallback to CPU")
        return "cpu"
    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in device detection: {e}")
        # Return the most conservative option
        return "cpu"
```

## Troubleshooting

### High Memory Usage

If memory usage is too high:
1. Call `cleanup_unused_resources()` more frequently
2. Use a shorter timeout value (e.g., 5 minutes instead of 30)
3. Call `clear()` to release all resources completely
4. Enable low-memory mode by setting the environment variable `RESOURCE_POOL_LOW_MEMORY=1`
5. Check memory usage with `pool.get_stats()` to identify memory-intensive resources
6. Use more specific resource types to avoid redundant caching
7. Consider using mixed precision with `"precision": "fp16"` in hardware preferences
8. Implement a regular cleanup schedule in long-running applications
9. Monitor system memory pressure with `stats.get("system_memory", {}).get("under_pressure", False)`
10. Use device-specific model instances only when necessary

### Resource Not Found

If a resource is not found:
1. Check that the constructor function is working correctly
2. Verify that the model type, resource type, and ID are correct
3. Check the resource pool stats to see if the resource was ever created
4. Ensure resource keys are consistent (same model type and name pairs)
5. Verify resource was not cleared by another process or by cleanup
6. Check error logs for any import or initialization errors
7. Verify the model exists and is accessible (for HuggingFace models)
8. Check for network connectivity if loading from external sources
9. Try explicit device specification if automatic selection is failing
10. Test your constructor function separately to ensure it works

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
1. Reduce batch size in your tests
2. Move models to CPU when not in use with `model.to("cpu")` 
3. Call `torch.cuda.empty_cache()` after model usage
4. Use `cleanup_unused_resources()` with a short timeout
5. Enable low-memory mode to automatically manage GPU resources
6. Consider using smaller models for testing purposes
7. Distribute models across multiple GPUs if available
8. Use model offloading techniques for very large models
9. Monitor GPU memory with `nvidia-smi` or through ResourcePool stats
10. Consider using gradient checkpointing for training models
11. Try mixed precision training/inference to reduce memory footprint

### Hardware Detection Issues

If hardware detection is not working as expected:
1. Verify hardware detection modules are available
2. Run `test_comprehensive_hardware.py` to diagnose hardware issues
3. Check for hardware driver updates (especially CUDA, ROCm)
4. Manually specify hardware preferences if automatic detection fails
5. Verify platform compatibility with specific hardware backends
6. Check for conflicting environment variables affecting hardware selection
7. Use the most recent version of hardware_detection.py
8. Run hardware detection with increased logging level for detailed diagnostics
9. Verify tensor operations work correctly on the selected device
10. Consider platform-specific setup requirements (e.g., MPS activation)

## Advanced Usage

### Multi-GPU Support with Device Selection

ResourcePool now integrates with hardware detection to support device selection in multi-GPU environments:

```python
from resource_pool import get_global_resource_pool
from hardware_detection import CUDA, MPS, CPU

# Get the resource pool
pool = get_global_resource_pool()

# Define hardware preferences for different models
llm_preferences = {
    "priority_list": [CUDA, CPU],  # LLMs often need CUDA
    "preferred_index": 0  # Use primary GPU (typically has most memory)
}

vision_preferences = {
    "priority_list": [CUDA, MPS, CPU],
    "preferred_index": 1  # Use secondary GPU for vision models
}

# Load models with different hardware preferences
llm_model = pool.get_model(
    "text_generation", 
    "gpt2",
    constructor=lambda: AutoModelForCausalLM.from_pretrained("gpt2"),
    hardware_preferences=llm_preferences  # Uses CUDA:0
)

vision_model = pool.get_model(
    "vision", 
    "vit-base-patch16-224",
    constructor=lambda: AutoModelForImageClassification.from_pretrained("vit-base-patch16-224"),
    hardware_preferences=vision_preferences  # Uses CUDA:1 if available
)
```

### Custom Hardware Priority Lists

You can customize hardware selection based on specific model requirements:

```python
from resource_pool import get_global_resource_pool
from hardware_detection import CUDA, ROCM, MPS, OPENVINO, CPU

# Define custom hardware priority for different model types
embedding_priority = [MPS, CUDA, CPU]  # Embeddings work well on MPS
text_gen_priority = [CUDA, CPU]  # LLMs need CUDA's memory
vision_priority = [CUDA, OPENVINO, CPU]  # Vision can use OpenVINO as fallback

# Create hardware preferences
hardware_preferences = {
    "priority_list": embedding_priority,
    "preferred_index": 0
}

# The ResourcePool will select hardware based on your priority list
pool = get_global_resource_pool()
model = pool.get_model(
    "embedding", 
    "bert-base-uncased",
    constructor=lambda: AutoModel.from_pretrained("bert-base-uncased"),
    hardware_preferences=hardware_preferences
)
```

### Low-Memory Mode and Automatic Memory Detection

The ResourcePool now includes automatic memory detection and can enable low-memory mode automatically when system resources are constrained:

```python
from resource_pool import get_global_resource_pool

# ResourcePool automatically detects available memory and enables low-memory mode if needed
pool = get_global_resource_pool()

# Check the log for a message like:
# "Low memory detected (4200.5 MB). Enabling low memory mode."

# You can also manually enable low-memory mode by setting an environment variable:
import os
os.environ["RESOURCE_POOL_LOW_MEMORY"] = "1"

# Then create a new pool instance that will use low-memory settings:
pool = get_global_resource_pool()
```

#### Effects of Low-Memory Mode:

1. **Aggressive Resource Cleanup**: Shorter timeouts for resource cleanup (10 minutes instead of 30)
2. **Model Device Management**: GPU models are moved to CPU after initialization if memory usage is high
3. **Memory Usage Warnings**: More frequent warnings about memory pressure
4. **Careful CUDA Management**: More aggressive CUDA cache clearing
5. **Reduced Cache Sizes**: Some internal caches are limited in size

#### Checking Memory Status:

```python
# Get detailed memory stats including system and CUDA memory status
stats = pool.get_stats()

# System memory status
system_memory = stats.get("system_memory", {})
available_mb = system_memory.get("available_mb", 0)
percent_used = system_memory.get("percent_used", 0)
print(f"System memory: {available_mb:.2f} MB available, {percent_used}% used")

# Check if the system is under memory pressure
under_pressure = system_memory.get("under_pressure", False)
if under_pressure:
    print("System is under memory pressure - consider clearing resources")

# CUDA memory status
cuda_memory = stats.get("cuda_memory", {})
if cuda_memory.get("device_count", 0) > 0:
    for device in cuda_memory.get("devices", []):
        print(f"CUDA device {device['id']} ({device['name']}): {device['free_mb']:.2f} MB free, {device['percent_used']:.2f}% used")
```

### Integration with Model Family Classifier and Hardware Detection

The ResourcePool integrates with both `model_family_classifier` and `hardware_detection` to make intelligent decisions about resource allocation and hardware selection:

```python
from model_family_classifier import classify_model
from hardware_detection import detect_available_hardware
from resource_pool import get_global_resource_pool

# The integration happens automatically in ResourcePool.get_model()
pool = get_global_resource_pool()

# ResourcePool will automatically:
# 1. Detect available hardware using hardware_detection
# 2. Classify the model family using model_family_classifier
# 3. Determine the optimal device for the specific model family
# 4. Select appropriate precision and memory settings
model = pool.get_model("bert", "bert-base-uncased", constructor=lambda: ...)

# This happens behind the scenes:
# - Embedding models like BERT are analyzed for size and hardware compatibility
# - Text generation models are checked against available GPU memory
# - Multimodal models get special handling for specific hardware platforms
# - Hardware-specific optimizations are applied based on model family
```

#### Hardware-Aware Model Loading

The ResourcePool uses comprehensive hardware detection to make smart decisions about device placement:

```python
# The ResourcePool automatically detects:
# - CPU capabilities (cores, architecture, extensions like AVX/SSE)
# - CUDA availability (device count, memory, compute capability)
# - MPS availability (Apple Silicon)
# - ROCm availability (AMD GPUs)
# - OpenVINO availability
# - WebNN/WebGPU availability
# - System memory availability

# Then it analyzes the model type and makes intelligent decisions:
from resource_pool import get_global_resource_pool

pool = get_global_resource_pool()

# Example: Loading an LLM
# - Will automatically check GPU memory requirements
# - Will fall back to CPU if not enough CUDA memory is available
# - Will handle quantization options if available
model = pool.get_model(
    "text_generation", 
    "llama-7b",
    constructor=lambda: create_llama_model()
)

# Example: Loading an embedding model
# - Will prioritize MPS on Apple Silicon for efficiency
# - Will use CUDA for higher throughput if available
# - Will use available CPU optimizations if no GPU is available
model = pool.get_model(
    "embedding", 
    "bert-base",
    constructor=lambda: create_bert_model()
)

# Example: Loading a vision model
# - Will check for specialized vision hardware support
# - Will leverage OpenVINO if available
# - Will optimize batch processing for input images
model = pool.get_model(
    "vision", 
    "vit-base",
    constructor=lambda: create_vision_model()
)
```

#### Model Family-Based Hardware Compatibility Matrix

The ResourcePool leverages a comprehensive hardware compatibility matrix based on model families:

| Model Type | Family | CUDA | AMD | MPS | OpenVINO | Memory Priority | Special Handling |
|------------|--------|------|-----|-----|----------|----------------|------------------|
| BERT       | Embedding | ✅ High | ✅ High | ✅ High | ✅ Medium | Low | Works well on all hardware |
| RoBERTa    | Embedding | ✅ High | ✅ High | ✅ High | ✅ Medium | Low | Works well on all hardware |
| T5         | Text Gen | ✅ High | ✅ Medium | ✅ Medium | ❌ Low | Medium | Needs more memory than embeddings |
| GPT-2      | Text Gen | ✅ High | ✅ Medium | ✅ Low | ❌ Low | Medium | MPS has performance issues with some operations |
| LLaMA      | Text Gen | ✅ High | ✅ Medium | ❌ Low | ❌ Low | High | Requires significant VRAM for larger variants |
| ViT        | Vision | ✅ High | ✅ Medium | ✅ High | ✅ High | Low | OpenVINO optimized for vision models |
| CLIP       | Multimodal | ✅ High | ✅ Medium | ✅ Medium | ✅ Low | Medium | Vision component works well on MPS |
| Whisper    | Audio | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | Medium | Special handling for audio processing |
| LLaVA      | Multimodal | ✅ High | ❌ Low | ❌ Low | ❌ Low | High | Typically only fully supported on CUDA |

This comprehensive matrix is used by the ResourcePool in conjunction with the hardware detection system to make optimal device placement decisions for each model family.

#### Manual Hardware and Model Classification Integration

For advanced use cases, you can manually integrate these components:

```python
from model_family_classifier import classify_model
from hardware_detection import detect_available_hardware
from resource_pool import get_global_resource_pool

# Get hardware information
hw_info = detect_available_hardware()
available_devices = []
if hw_info["hardware"].get("cuda", False):
    available_devices.append("cuda")
if hw_info["hardware"].get("mps", False):
    available_devices.append("mps")
available_devices.append("cpu")  # CPU is always available

# Classify model to determine family and hardware compatibility
model_info = classify_model("bert-base-uncased")
model_family = model_info["family"]
subfamily = model_info.get("subfamily")

# Check for hardware compatibility issues
hw_compatibility = {}
if model_family == "multimodal" and "mps" in available_devices:
    # Multimodal models may have issues on MPS (Apple Silicon)
    hw_compatibility["mps"] = {"compatible": False, "reason": "Multimodal models not fully supported on MPS"}

# Use all this information for optimal resource allocation
pool = get_global_resource_pool()
model = pool.get_model(
    model_type=model_family,
    model_name="bert-base-uncased", 
    constructor=lambda: ...,
    hardware_preferences={
        "compatible_devices": available_devices,
        "hw_compatibility": hw_compatibility
    }
)
```

## Creating Hardware-Aware Tests

The ResourcePool system includes the `test_generator_with_resource_pool.py` script, which generates hardware-aware test files for Hugging Face models. These tests automatically leverage the integrated ResourcePool, hardware_detection, and model_family_classifier systems.

### Testing the Integration

To verify that all components work together correctly, you can use the `run_integrated_hardware_model_test.py` script:

```bash
# Just check if all required files exist
python run_integrated_hardware_model_test.py --check-only

# Run a full integration test with all components
python run_integrated_hardware_model_test.py

# Enable debug logging for more detailed information
python run_integrated_hardware_model_test.py --debug
```

This script performs the following checks:

1. Verifies that all required files exist
2. Tests the ResourcePool instantiation
3. Tests hardware detection (if available)
4. Tests model classification (if available)
5. Tests the integration of all components together

The integration test script is designed to gracefully handle missing components and provide clear information about what's available and what's missing. You can use this to verify that your setup is working correctly or to diagnose issues.

### Test Generator Walkthrough

Here's a complete walkthrough of generating and running hardware-aware tests:

1. **Generate a test for BERT**:
   ```bash
   python test_generator_with_resource_pool.py --model bert-base-uncased --output-dir ./skills
   ```

2. **Run the generated test**:
   ```bash
   python ./skills/test_hf_bert_base_uncased.py
   ```

3. **View resource usage statistics**:
   The test will output resource usage statistics at the end:
   ```
   Resource pool stats: {
     'hits': 4, 
     'misses': 4, 
     'memory_usage_mb': 417.6, 
     'cached_models': 1, 
     'low_memory_mode': False, 
     'system_memory': {'available_mb': 48009.7, 'percent_used': 24.6}
   }
   ```

### Testing Multiple Models with Shared Resources

When testing multiple models, the ResourcePool ensures efficient resource sharing:

```bash
# Generate tests for multiple models
python test_generator_with_resource_pool.py --model bert-base-uncased --output-dir ./skills
python test_generator_with_resource_pool.py --model t5-small --output-dir ./skills
python test_generator_with_resource_pool.py --model gpt2 --output-dir ./skills

# Run the tests sequentially - resources will be shared
python ./skills/test_hf_bert_base_uncased.py
python ./skills/test_hf_t5_small.py
python ./skills/test_hf_gpt2.py
```

### Multi-GPU Device Distribution

In a multi-GPU environment, you can distribute models across available GPUs:

```python
from resource_pool import get_global_resource_pool
from hardware_detection import HardwareDetector, CUDA, CPU

# Get detector to check GPU availability
detector = HardwareDetector()
has_multiple_gpus = (detector.is_available(CUDA) and 
                     detector.get_hardware_details()[CUDA]["device_count"] > 1)

# Get resource pool
pool = get_global_resource_pool()

# Define model constructors
def create_bert():
    from transformers import AutoModel
    return AutoModel.from_pretrained("bert-base-uncased")

def create_t5():
    from transformers import AutoModelForSeq2SeqLM
    return AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def create_gpt2():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained("gpt2")

# Distribute models across GPUs if available
if has_multiple_gpus:
    # GPU 0 for primary LLM (highest memory requirements)
    gpt2_prefs = {"priority_list": [CUDA, CPU], "preferred_index": 0}
    
    # GPU 1 for encoder-decoder model
    t5_prefs = {"priority_list": [CUDA, CPU], "preferred_index": 1}
    
    # GPU 1 for embedding model (can share with encoder-decoder)
    bert_prefs = {"priority_list": [CUDA, CPU], "preferred_index": 1}
    
    # Load models on different GPUs
    gpt2_model = pool.get_model("text_generation", "gpt2", 
                                constructor=create_gpt2, 
                                hardware_preferences=gpt2_prefs)
    
    t5_model = pool.get_model("text_generation", "t5-small", 
                             constructor=create_t5, 
                             hardware_preferences=t5_prefs)
    
    bert_model = pool.get_model("embedding", "bert-base-uncased", 
                               constructor=create_bert, 
                               hardware_preferences=bert_prefs)
else:
    # Single GPU fallback - use standard loading
    gpt2_model = pool.get_model("text_generation", "gpt2", constructor=create_gpt2)
    t5_model = pool.get_model("text_generation", "t5-small", constructor=create_t5)
    bert_model = pool.get_model("embedding", "bert-base-uncased", constructor=create_bert)

### Creating a Test Runner Script with Hardware-Aware Resource Allocation

You can create a script to run multiple tests with hardware-aware resource allocation:

```python
#!/usr/bin/env python
import os
import sys
import importlib
import glob
from resource_pool import get_global_resource_pool
from hardware_detection import HardwareDetector, CUDA, MPS, OPENVINO, CPU

def get_hardware_preferences(model_name):
    """Get hardware preferences based on model name"""
    # Check if multi-GPU system
    detector = HardwareDetector()
    multi_gpu = (detector.is_available(CUDA) and 
                detector.get_hardware_details()[CUDA]["device_count"] > 1)
    
    # Define model-specific preferences
    if "gpt2" in model_name or "llama" in model_name or "llm" in model_name:
        # LLMs need the most GPU memory - primary GPU
        if multi_gpu:
            return {"priority_list": [CUDA, CPU], "preferred_index": 0}
        else:
            return {"priority_list": [CUDA, CPU]}
    
    elif "t5" in model_name or "bart" in model_name:
        # Encoder-decoder models - secondary GPU if available
        if multi_gpu:
            return {"priority_list": [CUDA, CPU], "preferred_index": 1}
        else:
            return {"priority_list": [CUDA, CPU]}
    
    elif "bert" in model_name or "roberta" in model_name or "distil" in model_name:
        # Embedding models - Apple Silicon does well with embeddings
        return {"priority_list": [MPS, CUDA, CPU]}
    
    elif "vit" in model_name or "resnet" in model_name or "yolo" in model_name:
        # Vision models - can use OpenVINO
        return {"priority_list": [CUDA, OPENVINO, CPU]}
    
    elif "whisper" in model_name or "wav2vec" in model_name:
        # Audio models - often need GPU
        return {"priority_list": [CUDA, CPU]}
    
    # Default case
    return {"priority_list": [CUDA, MPS, OPENVINO, CPU]}

def run_test_file(test_file_path):
    """Run a test file using the global resource pool"""
    print(f"Running {test_file_path}...")
    
    # Get the module name from the file path
    module_name = os.path.basename(test_file_path).replace('.py', '')
    
    # Set hardware preferences based on model name
    model_name = module_name.replace('test_hf_', '')
    hw_preferences = get_hardware_preferences(model_name)
    print(f"Using hardware preferences for {model_name}: {hw_preferences}")
    
    # Set environment variable to pass hardware preferences
    os.environ["TEST_HARDWARE_PREFERENCES"] = str(hw_preferences)
    
    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, test_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Run the test if it has a main function
    if hasattr(module, 'main'):
        module.main()
    else:
        print(f"No main function found in {test_file_path}")
    
    print(f"Completed {test_file_path}")

def main():
    # Get the global resource pool
    pool = get_global_resource_pool()
    
    # Get list of test files
    test_dir = "./skills"
    test_files = glob.glob(os.path.join(test_dir, "test_hf_*.py"))
    
    # Run all tests
    for test_file in test_files:
        run_test_file(test_file)
        # Clean up after each test to prevent memory buildup
        pool.cleanup_unused_resources(max_age_minutes=0.5)  # 30 second timeout
    
    # Show resource pool stats
    stats = pool.get_stats()
    print(f"ResourcePool stats: {stats}")
    
    # Clean up resources
    pool.clear()

if __name__ == "__main__":
    main()
# Resource sharing happens automatically through the global ResourcePool

# Full System Integration

The ResourcePool forms a critical part of the hardware-aware workflow in the framework. All components work together to provide optimal resource management:

```
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│                     │      │                      │      │                      │
│  hardware_detection ├──────►  resource_pool       ◄──────┤  model_family        │
│  (device selection) │      │  (memory management) │      │  (model classification)|
│                     │      │                      │      │                      │
└─────────────────────┘      └──────────────────────┘      └──────────────────────┘
          │                            │                             │
          │                            │                             │
          │                            ▼                             │
          │                   ┌──────────────────────┐              │
          │                   │                      │              │
          └───────────────────►  test_generator      ◄──────────────┘
                              │  (template selection)│
                              │                      │
                              └──────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │                      │
                              │  generated_tests     │
                              │  (optimized for HW)  │
                              │                      │
                              └──────────────────────┘
```

## Notes for Added Directories in Your Path
When using the ResourcePool in your tests, make sure to add the parent directory to your Python path:

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool
```

## Version History

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