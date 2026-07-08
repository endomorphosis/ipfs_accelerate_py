# Unified Framework API Reference

**Version:** 1.0.0  
**Last Updated:** March 6, 2025

## Overview

The Unified Framework API provides a consistent interface for interacting with all hardware platforms and model types supported by the framework. This document provides a comprehensive reference for all components, classes, and methods in the API.

## Core Components

The Unified Framework is organized into the following core components:

1. **UnifiedWebPlatform**: Main entry point for web platform integration
2. **WebGPUStreamingInference**: API for streaming token generation with WebGPU
3. **HardwareDetection**: Utilities for detecting and selecting hardware platforms
4. **ModelRegistry**: Registry of supported models and their capabilities
5. **ConfigurationManager**: Validation and management of framework configurations
6. **ErrorHandler**: Cross-component error handling and recovery

## UnifiedWebPlatform

The `UnifiedWebPlatform` class provides a unified interface for running inference across different web platforms.

```python
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform

platform = UnifiedWebPlatform(
    model_name="bert-base-uncased",
    model_type="text",
    platform="webgpu"  # or "webnn", "cpu"
)
```

### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_name` | str | Name of the model to load | Required |
| `model_type` | str | Type of model (text, vision, audio, multimodal) | Required |
| `platform` | str | Target platform (webgpu, webnn, cpu) | "auto" |
| `config` | dict | Additional configuration options | {} |

### Methods

#### `run_inference(inputs)`

Runs inference on the provided inputs.

**Parameters:**
- `inputs` (dict): Input data for the model

**Returns:**
- `dict`: Model outputs

**Example:**
```python
results = platform.run_inference({
    "input_text": "Hello, world!"
})
```

#### `run_inference_streaming(inputs, callback)`

Runs inference with streaming output.

**Parameters:**
- `inputs` (dict): Input data for the model
- `callback` (function): Callback function for token handling

**Returns:**
- `dict`: Complete model outputs after streaming

**Example:**
```python
def handle_token(token):
    print(token, end="", flush=True)

results = platform.run_inference_streaming(
    {"input_text": "Hello, world!"}, 
    callback=handle_token
)
```

#### `get_performance_stats()`

Returns performance statistics from the most recent inference run.

**Returns:**
- `dict`: Performance metrics

**Example:**
```python
stats = platform.get_performance_stats()
print(f"Inference time: {stats['inference_time_ms']}ms")
```

#### `optimize_for_browser(browser_name)`

Applies browser-specific optimizations.

**Parameters:**
- `browser_name` (str): Browser name (chrome, firefox, safari, edge)

**Returns:**
- `bool`: Success status

**Example:**
```python
platform.optimize_for_browser("firefox")
```

### Properties

#### `browser_info`

Information about the current browser environment.

**Type:** dict

**Example:**
```python
info = platform.browser_info
print(f"Browser: {info['name']}, Version: {info['version']}")
```

#### `supported_features`

List of supported features for the current platform and browser.

**Type:** dict

**Example:**
```python
features = platform.supported_features
if features.get("shader_precompilation", False):
    print("Shader precompilation is supported")
```

## WebGPUStreamingInference

The `WebGPUStreamingInference` class provides specialized streaming token generation capabilities for WebGPU.

```python
from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference

streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "quantization": "int4",
        "kv_cache_optimization": True
    }
)
```

### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_path` | str | Path to the model | Required |
| `config` | dict | Configuration options | {} |

### Methods

#### `generate(prompt, max_tokens=100)`

Generates text from a prompt.

**Parameters:**
- `prompt` (str): Input text prompt
- `max_tokens` (int): Maximum tokens to generate

**Returns:**
- `str`: Generated text

**Example:**
```python
output = streaming.generate("Once upon a time", max_tokens=50)
```

#### `generate_async(prompt, max_tokens=100)`

Generates text asynchronously.

**Parameters:**
- `prompt` (str): Input text prompt
- `max_tokens` (int): Maximum tokens to generate

**Returns:**
- `Promise`: Resolves to generated text

**Example:**
```python
async def generate():
    output = await streaming.generate_async("Once upon a time", max_tokens=50)
    print(output)
```

#### `stream_websocket(websocket, config=None)`

Streams generation responses over WebSocket.

**Parameters:**
- `websocket` (WebSocket): WebSocket connection
- `config` (dict): Generation configuration

**Returns:**
- None

**Example:**
```python
# Server-side code
@app.websocket("/generate")
async def websocket_generate(websocket):
    await streaming.stream_websocket(websocket)
```

#### `optimize_for_latency()`

Applies latency optimizations for streaming inference.

**Returns:**
- `bool`: Success status

**Example:**
```python
streaming.optimize_for_latency()
```

### Configuration Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `quantization` | str | Quantization method (fp32, fp16, int8, int4) | "fp16" |
| `kv_cache_optimization` | bool | Enable KV cache optimization | True |
| `optimize_compute_transfer` | bool | Enable compute/transfer overlap | True |
| `adaptive_batch_size` | bool | Enable adaptive batch sizing | True |
| `browser_optimizations` | bool | Enable browser-specific optimizations | True |
| `prefetch_size` | int | Number of tokens to prefetch | 3 |

## HardwareDetection

The `HardwareDetection` module provides utilities for detecting and selecting hardware platforms.

```python
from fixed_web_platform.browser_capability_detection import detect_capabilities

capabilities = detect_capabilities()
```

### Functions

#### `detect_capabilities()`

Detects browser capabilities and available hardware.

**Returns:**
- `dict`: Detected capabilities

**Example:**
```python
capabilities = detect_capabilities()
if capabilities.get("webgpu", {}).get("supported", False):
    print("WebGPU is supported")
```

#### `get_optimal_platform(model_type, model_size)`

Determines the optimal platform for a given model.

**Parameters:**
- `model_type` (str): Type of model
- `model_size` (str): Size of model (tiny, small, medium, large)

**Returns:**
- `str`: Recommended platform

**Example:**
```python
platform = get_optimal_platform("text", "small")
```

## ConfigurationManager

The `ConfigurationManager` class handles validation and management of framework configurations.

```python
from fixed_web_platform.unified_framework.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager(user_config={
    "precision": "int4",
    "enable_streaming": True
})
```

### Methods

#### `validate()`

Validates the current configuration.

**Returns:**
- `bool`: Validation result

**Example:**
```python
is_valid = config_manager.validate()
if not is_valid:
    print(f"Validation errors: {config_manager.validation_errors}")
```

#### `get_optimized_config(browser_name)`

Gets browser-optimized configuration.

**Parameters:**
- `browser_name` (str): Target browser

**Returns:**
- `dict`: Optimized configuration

**Example:**
```python
firefox_config = config_manager.get_optimized_config("firefox")
```

#### `apply_template(template_name)`

Applies a predefined configuration template.

**Parameters:**
- `template_name` (str): Name of the template

**Returns:**
- `bool`: Success status

**Example:**
```python
config_manager.apply_template("low_memory")
```

## ErrorHandler

The `ErrorHandler` class provides cross-component error handling and recovery.

```python
from fixed_web_platform.unified_framework.error_handler import ErrorHandler

error_handler = ErrorHandler(mode="graceful")
```

### Methods

#### `handle_error(error, context=None, recoverable=None)`

Handles errors with appropriate recovery strategies.

**Parameters:**
- `error` (Exception): The error that occurred
- `context` (dict): Context information about the error
- `recoverable` (bool): Whether the error is recoverable

**Returns:**
- `dict`: Error handling result

**Example:**
```python
try:
    # Some operation
except Exception as e:
    result = error_handler.handle_error(e, {"component": "inference"})
```

#### `register_error_callback(error_type, callback)`

Registers a callback for a specific error type.

**Parameters:**
- `error_type` (str): Type of error
- `callback` (function): Callback function

**Returns:**
- None

**Example:**
```python
def memory_error_handler(error_info):
    print(f"Memory error: {error_info}")

error_handler.register_error_callback("MemoryError", memory_error_handler)
```

## ModelShardingManager

The `ModelShardingManager` class handles distribution of large models across multiple browser tabs.

```python
from fixed_web_platform.unified_framework.model_sharding import ModelShardingManager

sharding_manager = ModelShardingManager(
    model_name="llama-7b",
    num_shards=4,
    shard_type="layer"
)
```

### Methods

#### `initialize_sharding()`

Initializes the model sharding across browser tabs.

**Returns:**
- `bool`: Success status

**Example:**
```python
success = sharding_manager.initialize_sharding()
```

#### `run_inference_sharded(inputs)`

Runs inference using the sharded model.

**Parameters:**
- `inputs` (dict): Input data

**Returns:**
- `dict`: Model outputs

**Example:**
```python
result = sharding_manager.run_inference_sharded({"input_text": "Hello"})
```

## Examples

### Basic Inference

```python
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform

# Create platform with automatic browser detection
platform = UnifiedWebPlatform(
    model_name="bert-base-uncased",
    model_type="text"
)

# Run inference
result = platform.run_inference({
    "input_text": "Hello, world!"
})

print(result)
```

### Streaming Text Generation

```python
from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference

# Create streaming inference handler
streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "quantization": "int4",
        "kv_cache_optimization": True,
        "low_latency": True
    }
)

# Generate with streaming
def on_token(token):
    print(token, end="", flush=True)

streaming.generate(
    "Write a short story about a robot learning to paint.",
    max_tokens=200,
    callback=on_token
)
```

### Browser-Specific Optimizations

```python
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform
from fixed_web_platform.browser_capability_detection import detect_capabilities

# Detect browser capabilities
capabilities = detect_capabilities()
browser_name = capabilities.get("browser", {}).get("name", "unknown")

# Create platform with browser-specific optimizations
platform = UnifiedWebPlatform(
    model_name="whisper-small",
    model_type="audio",
    platform="webgpu",
    config={
        "optimize_for_browser": browser_name
    }
)

# Apply additional optimizations for Firefox if detected
if browser_name.lower() == "firefox":
    platform.config.update({
        "workgroup_size": (256, 1, 1),  # Firefox-optimized workgroup size
        "enable_compute_shaders": True   # Firefox performs ~20% better with compute shaders
    })
```

### Error Handling

```python
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform
from fixed_web_platform.unified_framework.error_handler import ErrorHandler

# Create custom error handler
error_handler = ErrorHandler(mode="graceful")

# Register custom error callbacks
def memory_error_handler(error_info):
    print("Memory pressure detected, reducing model complexity...")
    # Implement recovery strategy

error_handler.register_error_callback("MemoryError", memory_error_handler)

# Create platform with custom error handler
platform = UnifiedWebPlatform(
    model_name="llava-7b",
    model_type="multimodal",
    platform="webgpu",
    config={
        "error_handler": error_handler
    }
)
```

## Related Documentation

- [Web Platform Integration Guide](../WEB_PLATFORM_INTEGRATION_GUIDE.md)
- [WebGPU Streaming Documentation](../WEBGPU_STREAMING_DOCUMENTATION.md)
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md)
- [Browser-Specific Optimizations](browser_specific_optimizations.md)
- [Hardware Selection Guide](../HARDWARE_SELECTION_GUIDE.md)
- [Model-Specific Optimization Guides](../model_specific_optimizations/)
- [Configuration Validation Guide](../CONFIGURATION_VALIDATION_GUIDE.md)
- [WebSocket Protocol Specification](../websocket_protocol_spec.md)