# WebGPUStreamingInference API Reference

**Version:** 1.0.0  
**Last Updated:** March 6, 2025

## Overview

The `WebGPUStreamingInference` class provides a specialized API for streaming token generation with WebGPU acceleration. It supports low-latency token generation, ultra-low precision (2-bit/3-bit) quantization, browser-specific optimizations, and WebSocket integration.

## Class: WebGPUStreamingInference

```python
from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference

streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "quantization": "int4",
        "optimize_for_latency": True
    }
)
```

### Constructor

```python
WebGPUStreamingInference(model_path, config=None)
```

Creates a new WebGPUStreamingInference instance.

**Parameters:**
- `model_path` (str): Path to the model or model identifier
- `config` (dict, optional): Configuration options for streaming inference

### Configuration Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `quantization` | str | Quantization method (fp32, fp16, int8, int4, int3, int2) | "fp16" |
| `kv_cache_optimization` | bool | Enable KV cache optimization | True |
| `optimize_compute_transfer` | bool | Enable compute/transfer overlap | True |
| `optimize_for_latency` | bool | Enable latency optimizations | True |
| `adaptive_batch_size` | bool | Enable adaptive batch sizing | True |
| `browser_optimizations` | bool | Enable browser-specific optimizations | True |
| `prefetch_size` | int | Number of tokens to prefetch | 3 |
| `workgroup_size` | tuple | WebGPU workgroup size (width, height, depth) | (128, 1, 1) |
| `max_batch_size` | int | Maximum batch size | 4 |

### Methods

#### `generate(prompt, max_tokens=100, callback=None)`

Generates text from a prompt with optional streaming via callback.

**Parameters:**
- `prompt` (str): Input text prompt
- `max_tokens` (int, optional): Maximum tokens to generate
- `callback` (function, optional): Callback function for token streaming

**Returns:**
- `str`: Generated text

**Example:**
```python
# Non-streaming usage
output = streaming.generate("Once upon a time", max_tokens=50)

# Streaming usage
def on_token(token):
    print(token, end="", flush=True)

output = streaming.generate(
    "Once upon a time", 
    max_tokens=50,
    callback=on_token
)
```

#### `generate_async(prompt, max_tokens=100, callback=None)`

Asynchronously generates text from a prompt.

**Parameters:**
- `prompt` (str): Input text prompt
- `max_tokens` (int, optional): Maximum tokens to generate
- `callback` (function, optional): Callback function for token streaming

**Returns:**
- `Promise`: Promise that resolves to the generated text

**Example:**
```python
import asyncio

async def generate():
    output = await streaming.generate_async(
        "Once upon a time", 
        max_tokens=50,
        callback=lambda token: print(token, end="", flush=True)
    )
    print("\nGeneration complete.")
    
asyncio.run(generate())
```

#### `stream_websocket(websocket, config=None)`

Streams generation responses over a WebSocket connection.

**Parameters:**
- `websocket` (WebSocket): WebSocket connection object
- `config` (dict, optional): Generation configuration for this connection

**Returns:**
- None

**Example:**
```python
# Server-side code (with FastAPI)
@app.websocket("/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Get configuration from client
    data = await websocket.receive_json()
    
    # Configure streaming
    streaming_config = {
        "max_tokens": data.get("max_tokens", 100),
        "temperature": data.get("temperature", 0.7)
    }
    
    # Stream responses
    await streaming.stream_websocket(websocket, streaming_config)
```

#### `optimize_for_latency()`

Applies optimizations to minimize latency for streaming token generation.

**Returns:**
- `bool`: Success status

**Example:**
```python
streaming.optimize_for_latency()
```

#### `optimize_for_browser(browser_name)`

Applies browser-specific optimizations.

**Parameters:**
- `browser_name` (str): Browser name (chrome, firefox, safari, edge)

**Returns:**
- `bool`: Success status

**Example:**
```python
streaming.optimize_for_browser("firefox")
```

#### `get_performance_stats()`

Returns performance statistics from the most recent generation.

**Returns:**
- `dict`: Dictionary with performance metrics

**Example:**
```python
stats = streaming.get_performance_stats()
print(f"Average token latency: {stats['avg_token_latency_ms']}ms")
print(f"First token latency: {stats['first_token_latency_ms']}ms")
print(f"Prefetch hits: {stats['prefetch_hit_rate']:.2f}")
```

### Events

#### `on_error`

Set a callback function for error handling.

**Example:**
```python
def handle_error(error):
    print(f"Error during generation: {error}")
    
streaming.on_error = handle_error
```

#### `on_memory_pressure`

Set a callback function for handling memory pressure events.

**Example:**
```python
def handle_memory_pressure(info):
    print(f"Memory pressure detected: {info['message']}")
    # Implement recovery strategy
    
streaming.on_memory_pressure = handle_memory_pressure
```

#### `on_timeout`

Set a callback function for handling timeout events.

**Example:**
```python
def handle_timeout(info):
    print(f"Operation timed out: {info['message']}")
    # Implement recovery strategy
    
streaming.on_timeout = handle_timeout
```

## Browser-Specific Optimizations

The WebGPUStreamingInference class includes several browser-specific optimizations:

### Firefox Optimizations

Firefox shows approximately 20% better performance for audio models using specialized compute shader optimizations:

```python
from fixed_web_platform.browser_capability_detection import detect_capabilities

capabilities = detect_capabilities()
browser = capabilities.get("browser", {}).get("name", "").lower()

streaming = WebGPUStreamingInference(
    model_path="models/whisper-small",
    config={
        "quantization": "int8",
        "optimize_for_browser": browser
    }
)

# Firefox-specific optimizations are automatically applied
# - Uses 256x1x1 workgroups (vs Chrome's 128x2x1)
# - Optimizes shared memory access patterns
# - Uses specialized audio processing shaders
```

### Chrome/Edge Optimizations

Chrome and Edge benefit from compute/transfer overlap optimizations:

```python
streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "quantization": "int4",
        "optimize_compute_transfer": True,
        "optimize_for_browser": "chrome"
    }
)
```

### Safari Optimizations

Safari requires special handling for memory pressure and timeout recovery:

```python
streaming = WebGPUStreamingInference(
    model_path="models/opt-350m",
    config={
        "quantization": "int8",
        "optimize_for_browser": "safari",
        "conservative_memory": True
    }
)

# Register Safari-specific error handlers
streaming.on_memory_pressure = handle_safari_memory_pressure
streaming.on_timeout = handle_safari_timeout
```

## Ultra-Low Precision Support

The WebGPUStreamingInference class supports ultra-low precision inference with 2-bit, 3-bit, and 4-bit quantization:

```python
# 4-bit quantization (best balance of quality and speed)
streaming_4bit = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={"quantization": "int4"}
)

# 3-bit quantization (good for small devices)
streaming_3bit = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={"quantization": "int3"}
)

# 2-bit quantization (fastest but lower quality)
streaming_2bit = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={"quantization": "int2"}
)

# Mixed precision (best for multimodal models)
streaming_mixed = WebGPUStreamingInference(
    model_path="models/llava-7b",
    config={
        "quantization": "mixed",
        "attention_precision": "int4",
        "feedforward_precision": "int3"
    }
)
```

## Advanced Configuration

### KV Cache Optimization

```python
streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "kv_cache_optimization": True,
        "kv_cache_strategy": "dynamic",  # or "static", "progressive"
        "max_kv_cache_mb": 512
    }
)
```

### Adaptive Batch Sizing

```python
streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "adaptive_batch_size": True,
        "initial_batch_size": 4,
        "min_batch_size": 1,
        "max_batch_size": 8,
        "batch_size_strategy": "performance"  # or "memory", "balanced"
    }
)
```

### Advanced Prefetching

```python
streaming = WebGPUStreamingInference(
    model_path="models/llama-7b",
    config={
        "prefetch_enabled": True,
        "prefetch_size": 3,
        "prefetch_strategy": "adaptive",  # or "fixed", "context-aware"
        "token_prediction_enabled": True
    }
)
```

## WebSocket Protocol

When using the `stream_websocket` method, the following message protocol is used:

### Client to Server Messages

Initial Configuration:
```json
{
  "type": "config",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "options": {
    "stream": true
  }
}
```

Control Messages:
```json
{
  "type": "control",
  "action": "stop"
}
```

### Server to Client Messages

Initialization:
```json
{
  "type": "init",
  "model_info": {
    "name": "llama-7b",
    "quantization": "int4"
  }
}
```

Token:
```json
{
  "type": "token",
  "token": " the",
  "token_id": 262,
  "generated_text": " Once upon a time the",
  "finished": false
}
```

Completion:
```json
{
  "type": "completion",
  "generated_text": "Once upon a time there was a robot who wanted to learn to paint...",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  },
  "finish_reason": "stop"
}
```

Error:
```json
{
  "type": "error",
  "error": "memory_pressure",
  "message": "WebGPU memory limit exceeded",
  "recoverable": true
}
```

## Related Documentation

- [Unified Framework API](../unified_framework_api.md)
- [WebGPU Streaming Documentation](../WEBGPU_STREAMING_DOCUMENTATION.md)
- [WebSocket Protocol Specification](../websocket_protocol_spec.md)
- [Model-Specific Optimization Guides](../model_specific_optimizations/)