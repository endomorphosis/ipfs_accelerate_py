# WebNN/WebGPU Resource Pool Recovery Integration Guide

This document provides comprehensive information about the WebNN/WebGPU Resource Pool integration with the Recovery System, which enhances reliability and fault tolerance when using browser-based hardware acceleration.

## Overview

The WebNN/WebGPU Resource Pool Recovery integration provides a robust, fault-tolerant layer that enables resilient operation with browser-based hardware acceleration. The integration bridges three key components:

1. **ResourcePool**: Central resource management system for model sharing and caching
2. **WebNN/WebGPU Resource Pool**: Manages browser-based hardware acceleration
3. **Recovery System**: Provides fault tolerance with error detection, categorization, and recovery

This integration enables applications to leverage browser-based WebNN and WebGPU hardware acceleration with automatic error recovery, fallbacks, and performance monitoring.

## Key Features

- **Automatic Error Recovery**: Detects and recovers from common browser errors
- **Smart Fallbacks**: Intelligently switches between WebNN, WebGPU, and CPU simulation
- **Browser Optimization**: Uses the most appropriate browser for each model type
  - Firefox for audio models (optimized compute shaders)
  - Edge for text models (superior WebNN support)
  - Chrome for vision models (solid WebGPU support)
- **Performance Monitoring**: Tracks browser and platform health for smart decisions
- **Graceful Degradation**: Falls back to simulation when hardware is unavailable
- **Comprehensive Metrics**: Provides detailed statistics on recovery and performance
- **Seamless Integration**: Works transparently through the ResourcePool interface

## Architecture

The integration uses a layered architecture:

```
┌─────────────────────────┐
│      ResourcePool       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ ResourcePoolBridge with │
│     Recovery System     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    Browser Automation    │
│    & WebSocket Bridge   │
└─────────────────────────┘
```

- **ResourcePool**: Provides the main interface for model loading and caching
- **ResourcePoolBridge with Recovery**: Handles error detection, categorization, and recovery
- **Browser Automation & WebSocket**: Manages browser instances and communication

## Usage

### Loading Models with WebNN/WebGPU Acceleration

```python
from resource_pool import global_resource_pool

# For WebGPU acceleration
model = global_resource_pool.get_model(
    model_type="text",
    model_name="bert-base-uncased",
    constructor=my_constructor_function,
    hardware_preferences={
        "priority_list": ["webgpu", "cpu"],
        "browser": "chrome"
    }
)

# For WebNN acceleration
model = global_resource_pool.get_model(
    model_type="text",
    model_name="bert-base-uncased",
    constructor=my_constructor_function,
    hardware_preferences={
        "priority_list": ["webnn", "cpu"],
        "browser": "edge"
    }
)
```

### Hardware Preferences

You can specify various hardware preferences:

```python
hardware_preferences = {
    # Prioritize hardware platforms in order
    "priority_list": ["webgpu", "webnn", "cpu"],
    
    # Specify browser (firefox optimized for audio models)
    "browser": "firefox",
    
    # Set precision for quantization (16, 8, 4)
    "precision": 8,
    
    # Enable mixed precision for better performance
    "mixed_precision": True,
    
    # Enable optimizations
    "compute_shaders": True,  # Especially for audio models
    "precompile_shaders": True,  # For faster startup
    "parallel_loading": True,  # For multimodal models
}
```

### Concurrent Execution

```python
# Get models
model1 = global_resource_pool.get_model("text", "bert-base-uncased", ...)
model2 = global_resource_pool.get_model("vision", "vit-base-patch16-224", ...)

# Prepare inputs
models_and_inputs = [
    (model1, {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}),
    (model2, {"pixel_values": [...]})
]

# Execute concurrently with automatic WebNN/WebGPU acceleration
results = global_resource_pool.execute_concurrent(models_and_inputs)
```

### Getting Statistics and Metrics

```python
# Get comprehensive stats including recovery metrics
stats = global_resource_pool.get_stats()

# Access WebNN/WebGPU stats
web_stats = stats["web_resource_pool"]
print(f"WebNN/WebGPU available: {web_stats['available']}")
print(f"WebNN/WebGPU initialized: {web_stats['initialized']}")

# Access recovery statistics if available
if "recovery_stats" in web_stats:
    recovery_stats = web_stats["recovery_stats"]
    print(f"Total recovery attempts: {recovery_stats['total_recovery_attempts']}")
    print(f"Error categories: {recovery_stats['error_categories']}")
```

## Error Categories and Recovery Strategies

The recovery system categorizes errors and applies appropriate strategies:

| Error Category | Description | Recovery Strategies |
|----------------|-------------|---------------------|
| CONNECTION | WebSocket or browser connection issues | Retry, restart browser, try another browser |
| BROWSER_CRASH | Browser process crashed | Restart browser, try another browser |
| OUT_OF_MEMORY | Out of memory errors | Reduce model size, reduce precision, try another browser |
| TIMEOUT | Operation timed out | Retry with delay, restart browser |
| UNSUPPORTED_OPERATION | Operation not supported on platform | Try another platform (WebNN/WebGPU/CPU), try another browser |
| BROWSER_CAPABILITY | Browser lacks required capability | Try another browser, try another platform |
| MODEL_INCOMPATIBLE | Model not compatible with backend | Try another platform, reduce precision, reduce model size |

## Environment Variables

Configure behavior with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| INIT_WEB_RESOURCE_POOL | 1 | Auto-initialize WebNN/WebGPU resource pool |
| FORCE_WEB_RESOURCE_POOL | 0 | Force use of WebNN/WebGPU resource pool |
| RESOURCE_POOL_LOW_MEMORY | 0 | Enable low memory mode |

## Browser-Specific Optimizations

Different browsers excel at different model types:

| Browser | Best For | Optimization | Improvement |
|---------|----------|--------------|-------------|
| Firefox | Audio models | Compute shader optimization | 20-25% faster for Whisper, CLAP |
| Edge | Text models | WebNN optimization | More efficient for BERT, T5 models |
| Chrome | Vision models | Shader precompilation | Better for ViT, CLIP models |

## Integration with Existing Code

The integration is designed to work transparently with existing code that uses the ResourcePool:

```python
# Existing code remains unchanged
model = global_resource_pool.get_model("text", "bert-base-uncased", constructor)

# Add hardware preferences to enable WebNN/WebGPU with recovery
model = global_resource_pool.get_model(
    "text", 
    "bert-base-uncased", 
    constructor,
    hardware_preferences={"priority_list": ["webgpu", "cpu"]}
)
```

## Recovery Process

When an error occurs during model loading or inference, the recovery system:

1. **Categorizes the error** into one of the error categories
2. **Selects a recovery strategy** based on the error category and context
3. **Applies the strategy** (e.g., restart browser, switch browser, reduce precision)
4. **Retries the operation** with the modified context
5. **Falls back** to simulation mode if all recovery attempts fail

## Implementation Details

The integration consists of three key components:

1. **ResourcePoolBridgeRecovery**: Provides error categorization and recovery strategies
2. **ResourcePoolBridgeWithRecovery**: Wraps the base bridge with recovery capabilities
3. **ResourcePoolBridgeIntegrationWithRecovery**: Integrates the recovery system with the resource pool

## Advanced Features

### Cross-Model Tensor Sharing

The integration supports cross-model tensor sharing for improved efficiency:

```python
# Setup tensor sharing with memory limit
manager = global_resource_pool.web_resource_pool.setup_tensor_sharing(max_memory_mb=2048)

# Models using the same resource pool can share tensors
model1 = global_resource_pool.get_model("text", "bert-base-uncased", ...)
model2 = global_resource_pool.get_model("text", "t5-small", ...)

# Tensors will be automatically shared when possible
```

### Health Monitoring

The integration includes comprehensive health monitoring for browsers and platforms:

```python
# Get health status
health = global_resource_pool.web_resource_pool.get_health_status()
```

## Troubleshooting

### Common Issues

1. **Browser Connection Failures**
   - Ensure that the browser (Chrome, Firefox, Edge) is installed and working
   - Check if Selenium WebDriver is installed and properly configured

2. **Model Loading Errors**
   - Verify that the model is compatible with WebNN/WebGPU
   - Try with a smaller model or reduced precision

3. **Slow Performance**
   - Check browser health metrics
   - Consider using browser-specific optimizations

### Logs and Diagnostics

Enable detailed logging for diagnostics:

```python
import logging
logging.getLogger("ResourcePool").setLevel(logging.DEBUG)
```

## Conclusion

The WebNN/WebGPU Resource Pool Recovery integration provides a robust, fault-tolerant solution for using browser-based hardware acceleration in your applications. By automatically handling errors, providing smart fallbacks, and optimizing for different browsers and model types, it enables reliable and efficient inference with WebNN and WebGPU.