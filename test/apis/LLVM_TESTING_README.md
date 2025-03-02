# LLVM API Testing Guide

This document outlines how to test the LLVM API backend implementation in the IPFS Accelerate framework.

## Implementation Status ✅ COMPLETE

The LLVM API backend implementation is now complete with full feature parity with other APIs:

- ✅ **Queue system**: Priority-based request queueing
- ✅ **Circuit breaker pattern**: Fault tolerance with automatic recovery
- ✅ **Request batching**: Improved throughput for compatible models
- ✅ **Monitoring system**: Comprehensive performance metrics
- ✅ **Endpoint multiplexing**: Multiple endpoint configurations
- ✅ **Parameterized handlers**: Custom parameters for specific use cases
- ✅ **Custom optimization**: Speed, memory, and balanced optimization profiles

## Overview

LLVM (Low Level Virtual Machine) API provides a high-performance interface for running deep learning models with LLVM optimizations. LLVM is particularly valuable for model execution across different architectures because it:

1. Provides hardware-specific optimizations through its compilation infrastructure
2. Enables just-in-time (JIT) compilation for runtime optimization
3. Supports various precision modes (FP32, FP16, INT8) for performance/accuracy tradeoffs
4. Facilitates efficient batch processing for higher throughput

The test suite validates three main aspects:

1. **Standard API Implementation**: Verifies the core API functionality and interfaces
2. **Performance Benchmarks**: Measures throughput, latency, and optimization effectiveness
3. **Real Connection**: Tests actual connection to a running LLVM server instance

## Test Files

- `test_llvm.py` - Core API implementation tests
- `test_llvm_unified.py` - Unified test runner with performance tests
- `LLVM_TESTING_README.md` - This documentation file

## Requirements

- Python 3.8+
- LLVM server (for real connection tests)
- Python packages: requests, numpy, unittest

## Running Tests

### Standard API Tests

Tests the implementation of the LLVM API client:

```bash
python test_llvm_unified.py --standard
```

This validates:
- Endpoint handler creation
- API request formatting
- Response parsing
- Error handling
- Model metadata retrieval
- Inference methods
- Batch processing capabilities

### Performance Tests

Benchmarks the performance of the API:

```bash
python test_llvm_unified.py --performance
```

This measures:
- Single inference time
- Batch inference time
- Inputs per second throughput
- Batch speedup factor
- Precision mode performance comparison

### Real Connection Tests

Tests connection to an actual LLVM server:

```bash
python test_llvm_unified.py --real
```

This verifies:
- Server availability
- Model availability
- Inference functionality

### All Tests

Run all test suites:

```bash
python test_llvm_unified.py --all
```

## Configuration Options

You can customize the tests with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model to use for testing | default |
| `--api-url` | URL for LLVM API | http://localhost:8090 |
| `--timeout` | Timeout in seconds for API requests | 30 |

Example with custom configuration:
```bash
python test_llvm_unified.py --all --model resnet50 --api-url http://llvm-server:8090 --timeout 60
```

## Environment Variables

You can also set configuration via environment variables:

- `LLVM_API_URL` - URL for the LLVM API
- `LLVM_API_KEY` - API key for authentication
- `LLVM_MODEL` - Model to use for testing
- `LLVM_TIMEOUT` - Timeout in seconds
- `LLVM_MAX_RETRIES` - Maximum retry attempts for backoff
- `LLVM_RETRY_DELAY` - Initial delay for retry attempts
- `LLVM_BACKOFF_FACTOR` - Multiplier for exponential backoff
- `LLVM_MAX_CONCURRENT` - Maximum concurrent requests
- `LLVM_FAILURE_THRESHOLD` - Failures before circuit breaker opens
- `LLVM_RESET_TIMEOUT` - Circuit breaker reset timeout
- `LLVM_BATCH_SIZE` - Maximum batch size for batching
- `LLVM_BATCH_TIMEOUT` - Timeout for batching
- `SKIP_PERFORMANCE_TESTS` - Set to "true" to skip performance tests
- `SKIP_REAL_TESTS` - Set to "true" to skip real connection tests

## Test Results

Test results are saved in the `collected_results/` directory:

- `llvm_test_results.json` - Standard API test results
- `llvm_performance_*.json` - Performance test results
- `llvm_connection_*.json` - Real connection test results
- `llvm_summary_*.json` - Test summary reports

## Advanced API Features

### LlvmClient Class

The LLVM implementation now includes a complete `LlvmClient` class with all advanced features:

```python
from ipfs_accelerate_py.api_backends.llvm import LlvmClient

# Initialize client with configuration
client = LlvmClient(
    api_key="your_api_key",
    base_url="http://localhost:8090",
    max_retries=3,
    max_concurrent_requests=10,
    failure_threshold=5
)

# Run inference with priority levels
result = client.run_inference(
    model_id="resnet50",
    inputs=[1.0, 2.0, 3.0, 4.0],
    parameters={"precision": "fp16"},
    priority=0  # 0=HIGH, 1=NORMAL, 2=LOW
)

# Get metrics
metrics = client.get_metrics()
```

### Priority Queue System

The implementation includes a priority-based queue system with three priority levels:

```python
# High priority request (critical tasks)
result = client.run_inference(model_id="resnet50", inputs=data, priority=0)

# Normal priority request (default)
result = client.run_inference(model_id="resnet50", inputs=data, priority=1)

# Low priority request (background tasks)
result = client.run_inference(model_id="resnet50", inputs=data, priority=2)
```

### Circuit Breaker Pattern

The implementation includes a fault-tolerance circuit breaker with three states:

- **CLOSED**: Normal operation, requests are processed
- **OPEN**: Service is failing, requests are rejected immediately
- **HALF-OPEN**: Testing if service has recovered

The circuit breaker automatically handles failures and recovery:

```python
# Circuit breaker is managed internally
# You can check current state and metrics with:
metrics = client.get_metrics()
print(f"Circuit state: {client.circuit_state}")
print(f"Failure count: {client.failure_count}")
```

### Request Batching

The implementation includes automatic request batching for compatible models:

```python
# Enable batching (on by default)
client.batch_enabled = True
client.max_batch_size = 8
client.batch_timeout = 0.1  # 100ms

# Make requests that will be automatically batched
results = []
for i in range(10):
    # These will be batched together for better throughput
    results.append(client.run_inference("resnet50", input_data[i]))
```

### Monitoring System

The implementation includes comprehensive metrics collection:

```python
# Get current metrics
metrics = client.get_metrics()

# Get and reset metrics
metrics = client.get_metrics(reset=True)

# Metrics include:
# - Request counts
# - Success/failure rates
# - Latency statistics (min, max, avg, percentiles)
# - Retry statistics
# - Batch statistics
# - Per-model statistics
```

### Parameterized Endpoint Handlers

Create handlers with different precision and optimization settings:

```python
# Create a handler with fp16 precision
handler = llvm.create_llvm_endpoint_handler_with_params(
    "http://localhost:8090",
    model="resnet50",
    parameters={"precision": "fp16"}
)

# Create a handler optimized for batch processing
handler = llvm.create_llvm_endpoint_handler_with_params(
    "http://localhost:8090",
    model="resnet50",
    parameters={"batch_size": 8}
)
```

### Request Formatting with Parameters

Control inference parameters for each request:

```python
# Format request with specific parameters
result = llvm.format_request_with_params(
    handler,
    "Input data",
    {"precision": "fp16", "optimize": True}
)

# Format a structured request
result = llvm.format_structured_request(
    handler,
    {
        "text": "Input text",
        "context": "Additional context",
        "options": {"return_details": True}
    }
)
```

### Batch Processing

Methods for efficient batch processing:

```python
# Process a batch of inputs
results = llvm.process_batch(
    "http://localhost:8090",
    ["Input 1", "Input 2", "Input 3"],
    "resnet50"
)

# Process a batch with parameters
results = llvm.process_batch_with_params(
    "http://localhost:8090",
    ["Input 1", "Input 2", "Input 3"],
    "resnet50",
    {"batch_size": 8}
)

# Process a batch and retrieve metrics
results, metrics = llvm.process_batch_with_metrics(
    "http://localhost:8090",
    ["Input 1", "Input 2", "Input 3"],
    "resnet50"
)
```

### Model Information and Statistics

Methods for getting model metadata and performance stats:

```python
# Get model information
info = llvm.get_model_info("http://localhost:8090", "resnet50")

# Get model performance statistics
stats = llvm.get_model_statistics("http://localhost:8090", "resnet50")
```

### Model Optimization

Methods for optimizing model execution:

```python
# Optimize model for speed
result = llvm.optimize_model(
    "http://localhost:8090",
    "resnet50",
    optimization_type="speed"
)

# Optimize model for memory efficiency
result = llvm.optimize_model(
    "http://localhost:8090",
    "resnet50",
    optimization_type="memory"
)

# Balanced optimization
result = llvm.optimize_model(
    "http://localhost:8090",
    "resnet50",
    optimization_type="balanced"
)
```

### Endpoint Multiplexing

Create and manage multiple endpoint configurations:

```python
# Create multiple endpoints with different settings
endpoint1 = client.create_endpoint(
    api_key="key1",
    max_concurrent_requests=5,
    max_retries=3
)

endpoint2 = client.create_endpoint(
    api_key="key2",
    max_concurrent_requests=10,
    max_retries=5
)

# Make request using specific endpoint
result = client.make_request_with_endpoint(
    endpoint_id=endpoint1,
    data=input_data,
    model="resnet50"
)

# Get statistics for specific endpoint
stats = client.get_stats(endpoint_id=endpoint1)

# Get statistics for all endpoints
all_stats = client.get_stats()
```

## Running a Local LLVM Server

For real connection tests, you need an LLVM server. Here's a simple way to run one:

1. Pull and run the LLVM server Docker image:
   ```bash
   docker run -d --name llvm-server \
       -p 8090:8090 \
       -v /path/to/models:/models \
       llvm/api-server:latest
   ```

2. Load your model:
   ```bash
   curl -X POST http://localhost:8090/models/load \
       -H "Content-Type: application/json" \
       -d '{"model_path": "/models/resnet50", "model_name": "resnet50"}'
   ```

3. The server should now be available at http://localhost:8090

## LLVM API Request Format

LLVM generally expects requests in this format:

```json
{
  "input": "Your input data here",
  "parameters": {
    "precision": "fp32",
    "batch_size": 1
  }
}
```

Responses follow this format:

```json
{
  "result": "Inference result",
  "status": "success",
  "metrics": {
    "inference_time": 0.0234,
    "throughput": 42.5
  }
}
```

For batch processing:

```json
{
  "results": ["Result 1", "Result 2"],
  "status": "success",
  "metrics": {
    "batch_size": 2,
    "inference_time": 0.0456
  }
}
```

## Supported Models

The LLVM API backend supports a variety of models:

| Model | Type | Batch Support | Precision Modes |
|-------|------|--------------|-----------------|
| resnet50 | vision | Yes | fp32, fp16, int8 |
| bert-base | nlp | Yes | fp32, fp16, int8 |
| mobilenet | vision | Yes | fp32, fp16, int8 |
| t5-small | nlp | Yes | fp32, fp16 |
| yolov5 | vision | Yes | fp32, fp16, int8 |
| llama-7b | nlp | No | fp32, fp16 |
| whisper-small | audio | Yes | fp32, fp16 |
| wav2vec2 | audio | Yes | fp32, fp16, int8 |
| vit-base | vision | Yes | fp32, fp16 |
| stable-diffusion | generative | No | fp32, fp16 |

## Troubleshooting

### Common Issues

- **Connection errors**: Verify LLVM server is running with `curl http://localhost:8090/status`
- **Model not found**: Ensure the model is properly loaded on the server
- **Timeout errors**: Increase timeout with `--timeout` parameter
- **Memory issues**: For large models, try optimizing with lower precision (fp16/int8)
- **Circuit breaker open**: Check server health or reset with `client.circuit_state = "CLOSED"`
- **Queue full**: Increase queue size or check for bottlenecks

### Debugging Commands

```bash
# Check if LLVM server is running
curl http://localhost:8090/status

# Check available models
curl http://localhost:8090/models

# Get information about a specific model
curl http://localhost:8090/models/resnet50

# Test simple inference
curl -X POST http://localhost:8090/infer \
    -H "Content-Type: application/json" \
    -d '{"input": "test input", "model": "resnet50"}'
```

### Advanced Debugging

Getting detailed metrics for diagnostics:

```python
# Get current metrics
metrics = client.get_metrics()

# Check circuit breaker state
print(f"Circuit state: {client.circuit_state}")
print(f"Failure count: {client.failure_count}")

# Check queue status
print(f"Queue enabled: {client.queue_enabled}")
print(f"Queue length: {len(client.request_queue)}")
print(f"Active requests: {client.current_requests}")

# Access recent request history
print(f"Recent requests: {len(client.recent_requests)}")
```

## Development Notes

When implementing new features in the LLVM backend:

1. Add corresponding tests in `test_llvm.py`
2. Update the `test_llvm_unified.py` for performance testing
3. Document new features in this README

The test suite is designed to gracefully handle missing features, so it will not fail if a feature is not yet implemented.