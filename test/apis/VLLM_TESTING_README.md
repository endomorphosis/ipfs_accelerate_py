# VLLM API Testing Guide

This document outlines how to test the VLLM API backend implementation in the IPFS Accelerate framework.

## Implementation Status ✅ COMPLETE

The VLLM API backend implementation is now complete with full feature parity with other APIs:

- ✅ **Queue system**: Priority-based request queueing
- ✅ **Circuit breaker pattern**: Fault tolerance with automatic recovery
- ✅ **Request batching**: Improved throughput for compatible models
- ✅ **Monitoring system**: Comprehensive performance metrics
- ✅ **Endpoint multiplexing**: Multiple endpoint configurations
- ✅ **Parameterized handlers**: Custom parameters for specific use cases
- ✅ **Custom optimization**: Speed, memory, and balanced optimization profiles

## Overview

VLLM (Variable Length Language Model) API provides a high-performance inference engine for LLMs. VLLM is particularly valuable for serving language models efficiently because it:

1. Implements PagedAttention for efficient memory usage and scaling to longer sequences
2. Provides continuous batching and kernel fusion for high throughput
3. Supports quantization (AWQ, SqueezeLLM, GPTQ) for reduced memory usage
4. Enables tensor parallelism for distributed inference across multiple GPUs

The test suite validates three main aspects:

1. **Standard API Implementation**: Verifies the core API functionality and interfaces
2. **Performance Benchmarks**: Measures throughput, latency, and optimization effectiveness
3. **Real Connection**: Tests actual connection to a running VLLM server instance

## Test Files

- `test_vllm.py` - Core API implementation tests
- `test_vllm_unified.py` - Unified test runner with performance tests
- `VLLM_TESTING_README.md` - This documentation file

## Requirements

- Python 3.8+
- VLLM server (for real connection tests)
- Python packages: requests, numpy, unittest

## Running Tests

### Standard API Tests

Tests the implementation of the VLLM API client:

```bash
python generators/models/test_vllm_unified.py --standard
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
python generators/models/test_vllm_unified.py --performance
```

This measures:
- Single inference time
- Batch inference time
- Inputs per second throughput
- Batch speedup factor
- Precision mode performance comparison

### Real Connection Tests

Tests connection to an actual VLLM server:

```bash
python generators/models/test_vllm_unified.py --real
```

This verifies:
- Server availability
- Model availability
- Inference functionality

### All Tests

Run all test suites:

```bash
python generators/models/test_vllm_unified.py --all
```

## Configuration Options

You can customize the tests with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model to use for testing | default |
| `--api-url` | URL for VLLM API | http://localhost:8000 |
| `--timeout` | Timeout in seconds for API requests | 30 |

Example with custom configuration:
```bash
python generators/models/test_vllm_unified.py --all --model meta-llama/Llama-2-7b-chat-hf --api-url http://vllm-server:8000 --timeout 60
```

## Environment Variables

You can also set configuration via environment variables:

- `VLLM_API_URL` - URL for the VLLM API
- `VLLM_API_KEY` - API key for authentication
- `VLLM_MODEL` - Model to use for testing
- `VLLM_TIMEOUT` - Timeout in seconds
- `VLLM_MAX_RETRIES` - Maximum retry attempts for backoff
- `VLLM_RETRY_DELAY` - Initial delay for retry attempts
- `VLLM_BACKOFF_FACTOR` - Multiplier for exponential backoff
- `VLLM_MAX_CONCURRENT` - Maximum concurrent requests
- `VLLM_FAILURE_THRESHOLD` - Failures before circuit breaker opens
- `VLLM_RESET_TIMEOUT` - Circuit breaker reset timeout
- `VLLM_BATCH_SIZE` - Maximum batch size for batching
- `VLLM_BATCH_TIMEOUT` - Timeout for batching
- `SKIP_PERFORMANCE_TESTS` - Set to "true" to skip performance tests
- `SKIP_REAL_TESTS` - Set to "true" to skip real connection tests

## Test Results

Test results are saved in the `collected_results/` directory:

- `vllm_test_results.json` - Standard API test results
- `vllm_performance_*.json` - Performance test results
- `vllm_connection_*.json` - Real connection test results
- `vllm_summary_*.json` - Test summary reports

## Advanced API Features

### VllmClient Class

The VLLM implementation now includes a complete `VllmClient` class with all advanced features:

```python
from ipfs_accelerate_py.api_backends.vllm import VllmClient

# Initialize client with configuration
client = VllmClient(
    api_key="your_api_key",
    base_url="http://localhost:8000",
    max_retries=3,
    max_concurrent_requests=10,
    failure_threshold=5
)

# Run inference with priority levels
result = client.run_inference(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    inputs="What is machine learning?",
    parameters={"temperature": 0.7, "max_tokens": 100},
    priority=0  # 0=HIGH, 1=NORMAL, 2=LOW
)

# Get metrics
metrics = client.get_metrics()
```

### Priority Queue System

The implementation includes a priority-based queue system with three priority levels:

```python
# High priority request (critical tasks)
result = client.run_inference(model_id="meta-llama/Llama-2-7b-chat-hf", inputs=prompt, priority=0)

# Normal priority request (default)
result = client.run_inference(model_id="meta-llama/Llama-2-7b-chat-hf", inputs=prompt, priority=1)

# Low priority request (background tasks)
result = client.run_inference(model_id="meta-llama/Llama-2-7b-chat-hf", inputs=prompt, priority=2)
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
    results.append(client.run_inference("meta-llama/Llama-2-7b-chat-hf", input_prompts[i]))
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
# Create a handler with temperature setting
handler = vllm.create_vllm_endpoint_handler_with_params(
    "http://localhost:8000",
    model="meta-llama/Llama-2-7b-chat-hf",
    parameters={"temperature": 0.7}
)

# Create a handler optimized for batch processing
handler = vllm.create_vllm_endpoint_handler_with_params(
    "http://localhost:8000",
    model="meta-llama/Llama-2-7b-chat-hf",
    parameters={"batch_size": 8}
)
```

### Request Formatting with Parameters

Control inference parameters for each request:

```python
# Format request with specific parameters
result = vllm.format_request_with_params(
    handler,
    "Tell me about machine learning",
    {"temperature": 0.7, "max_tokens": 100}
)

# Format a structured request
result = vllm.format_structured_request(
    handler,
    {
        "prompt": "Tell me about machine learning",
        "context": "I'm a beginner in AI",
        "options": {"return_logprobs": True}
    }
)
```

### Batch Processing

Methods for efficient batch processing:

```python
# Process a batch of inputs
results = vllm.process_batch(
    "http://localhost:8000",
    ["Tell me about Python", "Explain machine learning", "What is deep learning?"],
    "meta-llama/Llama-2-7b-chat-hf"
)

# Process a batch with parameters
results = vllm.process_batch_with_params(
    "http://localhost:8000",
    ["Tell me about Python", "Explain machine learning", "What is deep learning?"],
    "meta-llama/Llama-2-7b-chat-hf",
    {"temperature": 0.7, "max_tokens": 100}
)

# Process a batch and retrieve metrics
results, metrics = vllm.process_batch_with_metrics(
    "http://localhost:8000",
    ["Tell me about Python", "Explain machine learning", "What is deep learning?"],
    "meta-llama/Llama-2-7b-chat-hf"
)
```

### Model Information and Statistics

Methods for getting model metadata and performance stats:

```python
# Get model information
info = vllm.get_model_info("http://localhost:8000", "meta-llama/Llama-2-7b-chat-hf")

# Get model performance statistics
stats = vllm.get_model_statistics("http://localhost:8000", "meta-llama/Llama-2-7b-chat-hf")
```

### Model Optimization

Methods for optimizing model execution:

```python
# Optimize model for speed
result = vllm.optimize_model(
    "http://localhost:8000",
    "meta-llama/Llama-2-7b-chat-hf",
    optimization_type="speed"
)

# Optimize model for memory efficiency
result = vllm.optimize_model(
    "http://localhost:8000",
    "meta-llama/Llama-2-7b-chat-hf",
    optimization_type="memory"
)

# Balanced optimization
result = vllm.optimize_model(
    "http://localhost:8000",
    "meta-llama/Llama-2-7b-chat-hf",
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
    data=input_prompt,
    model="meta-llama/Llama-2-7b-chat-hf"
)

# Get statistics for specific endpoint
stats = client.get_stats(endpoint_id=endpoint1)

# Get statistics for all endpoints
all_stats = client.get_stats()
```

## Running a Local VLLM Server

For real connection tests, you need a VLLM server. Here's a simple way to run one:

1. Pull and run the VLLM server Docker image:
   ```bash
   docker run -d --name vllm-server \
       -p 8000:8000 \
       --gpus all \
       ghcr.io/vllm-project/vllm-openai:latest \
       --model meta-llama/Llama-2-7b-chat-hf \
       --tensor-parallel-size 1 \
       --host 0.0.0.0
   ```

2. For quantized models, specify the quantization method:
   ```bash
   docker run -d --name vllm-server-quantized \
       -p 8000:8000 \
       --gpus all \
       ghcr.io/vllm-project/vllm-openai:latest \
       --model TheBloke/Llama-2-7B-Chat-AWQ \
       --quantization awq \
       --host 0.0.0.0
   ```

3. The server should now be available at http://localhost:8000 with OpenAI-compatible API endpoints

## VLLM API Request Format

### Native VLLM API

The native VLLM API expects requests in this format:

```json
{
  "prompt": "Your prompt here",
  "temperature": 0.7,
  "max_tokens": 100,
  "top_p": 0.9
}
```

Responses follow this format:

```json
{
  "text": "Generated text response",
  "metadata": {
    "finish_reason": "length",
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 50,
      "total_tokens": 60
    }
  }
}
```

### OpenAI-Compatible API

VLLM also offers an OpenAI-compatible API with the following format:

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about VLLM."}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

With responses in OpenAI format:

```json
{
  "id": "cmpl-abc123",
  "object": "chat.completion",
  "created": 1678888888,
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "VLLM is a high-performance inference engine..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 100,
    "total_tokens": 120
  }
}
```

## Supported Models

The VLLM API backend supports a wide range of modern language models:

| Model | Type | Batch Support | Quantization Support |
|-------|------|--------------|-----------------|
| meta-llama/Llama-2-7b-chat-hf | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| meta-llama/Llama-2-13b-chat-hf | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| meta-llama/Llama-2-70b-chat-hf | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| meta-llama/Llama-3-8b-instruct | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| meta-llama/Llama-3-70b-instruct | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| mistralai/Mistral-7B-Instruct-v0.2 | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| mistralai/Mistral-7B-Instruct-v0.3 | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| Anthropic/claude-3-haiku-20240307 | LLM | Yes | FP16, BF16 |
| microsoft/Phi-3-mini-4k-instruct | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| google/gemma-2-9b-it | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| google/gemma-2-27b-it | LLM | Yes | FP16, BF16, GPTQ, AWQ |
| anthropic/claude-3-opus-20240229 | LLM | Yes | FP16, BF16 |
| anthropic/claude-3-sonnet-20240229 | LLM | Yes | FP16, BF16 |

## Troubleshooting

### Common Issues

- **CUDA Out of Memory**: For large models, use tensor parallelism with `--tensor-parallel-size` or quantization
- **Connection errors**: Verify VLLM server is running with `curl http://localhost:8000/health`
- **Model not found**: Ensure the model is properly loaded on the server
- **Slow first request**: VLLM performs JIT compilation on first inference, subsequent requests will be faster
- **Slow streaming**: Increase the chunk size for streaming or adjust KV cache settings
- **Circuit breaker open**: Check server health or reset with `client.circuit_state = "CLOSED"`
- **Queue full**: Increase queue size or check for bottlenecks

### Debugging Commands

```bash
# Check if VLLM server is running
curl http://localhost:8000/health

# Check server model info
curl http://localhost:8000/model

# Test native VLLM inference
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, world!", "max_tokens": 20}'

# Test OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-2-7b-chat-hf",
      "messages": [{"role": "user", "content": "Hello, world!"}],
      "max_tokens": 20
    }'
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

## Performance Optimization

For VLLM-specific performance optimizations, please refer to the comprehensive [Performance Optimization Plan](../PERFORMANCE_OPTIMIZATION_PLAN.md), which includes detailed information on:

1. Connection pooling optimization
2. Dynamic batch processing
3. Memory management optimization
4. Adaptive concurrency control
5. Enhanced request compression

The performance optimization plan provides concrete implementation examples and a phased approach to implementing these improvements.

## Error Handling

For detailed information on error handling in the VLLM API, please refer to the comprehensive [API Error Documentation](../API_ERROR_DOCUMENTATION.md), which covers:

1. Connection errors
2. Authentication errors
3. Request formatting errors
4. Service availability errors
5. Response processing errors

The error documentation includes code examples and best practices for handling different types of errors.

## Development Notes

When implementing new features in the VLLM backend:

1. Add corresponding tests in `test_vllm.py`
2. Update the `test_vllm_unified.py` for performance testing
3. Document new features in this README
4. Follow the optimization guidelines in the Performance Optimization Plan
5. Implement proper error handling as per the API Error Documentation

The test suite is designed to gracefully handle missing features, so it will not fail if a feature is not yet implemented.