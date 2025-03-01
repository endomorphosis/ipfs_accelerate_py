# VLLM API Testing Guide

This document provides information on testing the VLLM (Very Large Language Model) API integration within the IPFS Accelerate Python framework.

## Overview

VLLM is a high-throughput and memory-efficient inference engine for LLMs. It features:
- Fast inference with PagedAttention
- Continuous batching of incoming requests
- Quantization support (AWQ, GPTQ, SqueezeLLM)
- Tensor parallelism across multiple GPUs
- Streaming generation capabilities
- Support for LoRA adapters
- KV cache optimization
- Prompt caching

The test files in this directory provide comprehensive testing for VLLM API integration, including:
- Basic API functionality
- Performance benchmarking
- Real server connection tests
- Advanced features testing
- API key multiplexing, queueing, and backoff

## Test Files

### 1. `test_vllm.py`

This file contains the core test functionality for the VLLM API. It includes tests for:

- **API Endpoint Handler Creation**
  - Basic endpoint handler creation
  - Parameterized endpoint handler creation

- **Request Formatting**
  - Standard inference request formatting
  - Batch request formatting
  - Chat request formatting
  - Parameter handling

- **Error Handling**
  - Connection errors
  - Server errors (HTTP 500)
  - Invalid JSON responses

- **Batch Processing**
  - Basic batch inference
  - Batch processing with parameters
  - Batch processing with metrics retrieval

- **Streaming Generation**
  - Basic streaming generation
  - Streaming with different parameters

- **Model Information**
  - Model information retrieval
  - Model statistics retrieval

- **LoRA Adapter Management**
  - Listing LoRA adapters
  - Loading LoRA adapters
  - Inference with LoRA adapters

- **Quantization**
  - Setting quantization configuration
  - Different quantization methods (AWQ, GPTQ, SqueezeLLM)
  - Different bit precision (3-bit, 4-bit, 8-bit)

- **API Key Multiplexing**
  - Multiple endpoint creation with different API keys
  - Endpoint-specific configuration
  - Per-endpoint statistics

- **Queuing and Backoff**
  - Request queuing
  - Queue processing
  - Exponential backoff for rate limiting
  - Concurrent request management

- **Request Tracking**
  - Request ID handling
  - Request statistics collection

### 2. `test_vllm_unified.py`

This file provides a unified test runner with additional test methods and support for command-line arguments. It includes:

- **Standard API Tests**
  - Comprehensive testing of all API functionality

- **Performance Tests**
  - Single request latency measurement
  - Batch processing throughput
  - Streaming generation performance
  - Parameter impact on performance

- **Real Connection Tests**
  - Server health checks
  - Model availability verification
  - Actual inference testing
  - Streaming verification
  - Batch processing verification
  - LoRA adapter verification

- **Advanced Feature Tests**
  - LoRA adapter functionality
  - Quantization configuration
  - Tensor parallelism settings
  - KV cache configuration
  - Prompt caching testing
  - API key multiplexing functionality
  - Queue processing and backoff mechanisms

## Running Tests

You can run tests using the `test_vllm_unified.py` script with various command-line arguments:

```bash
# Run all standard tests
python -m test.apis.test_vllm_unified --standard

# Run performance tests
python -m test.apis.test_vllm_unified --performance

# Run real connection tests
python -m test.apis.test_vllm_unified --real

# Run advanced feature tests
python -m test.apis.test_vllm_unified --advanced

# Run all test categories
python -m test.apis.test_vllm_unified --all

# Specify a different model
python -m test.apis.test_vllm_unified --model meta-llama/Llama-2-7b-chat-hf

# Specify a custom API URL
python -m test.apis.test_vllm_unified --api-url http://vllm-server:8000

# Set a custom timeout
python -m test.apis.test_vllm_unified --timeout 60
```

## Environment Variables

The following environment variables control test behavior:

- `VLLM_API_URL`: The URL of the VLLM API server (default: `http://localhost:8000`)
- `VLLM_MODEL`: The model to use for testing (default: `meta-llama/Llama-2-7b-chat-hf`)
- `VLLM_TIMEOUT`: Timeout in seconds for API requests (default: `30`)
- `SKIP_PERFORMANCE_TESTS`: Set to "true" to skip performance tests
- `SKIP_REAL_TESTS`: Set to "true" to skip real connection tests
- `SKIP_ADVANCED_TESTS`: Set to "true" to skip advanced feature tests

## Test Results

Test results are stored in the `collected_results` directory with timestamps:

- Standard test results: `vllm_test_results.json`
- Performance test results: `vllm_performance_YYYYMMDD_HHMMSS.json`
- Real connection test results: `vllm_connection_YYYYMMDD_HHMMSS.json`
- Advanced feature test results: `vllm_advanced_YYYYMMDD_HHMMSS.json`
- Summary report: `vllm_summary_YYYYMMDD_HHMMSS.json`

## API Key Multiplexing Features

The VLLM API implementation supports multiplexing multiple API keys, with each key having its own:
- Request counters
- Request queue
- Concurrency limits
- Backoff settings
- Usage statistics

### Creating Multiple Endpoints

```python
from ipfs_accelerate_py.api_backends import vllm

# Initialize API client with default key
vllm_client = vllm(resources={}, metadata={"vllm_api_key": "default-api-key"})

# Create an endpoint with a specific API key
endpoint_id = vllm_client.create_endpoint(
    api_key="custom-api-key-1",
    max_retries=5,
    initial_retry_delay=1,
    backoff_factor=2,
    max_retry_delay=60,
    queue_enabled=True,
    max_concurrent_requests=5,
    queue_size=100
)

# Create another endpoint with different settings
endpoint_id2 = vllm_client.create_endpoint(
    api_key="custom-api-key-2",
    max_retries=3,
    max_concurrent_requests=10
)
```

### Using Specific Endpoints

```python
# Make a request using a specific endpoint
response = vllm_client(
    prompt="Hello, how are you?",
    endpoint_id=endpoint_id,
    request_id="custom-request-id-123"  # Optional
)

# Make a request with another endpoint
response2 = vllm_client(
    prompt="Hello, tell me a joke",
    endpoint_id=endpoint_id2
    # request_id will be auto-generated if not provided
)
```

### Getting Usage Statistics

```python
# Get statistics for a specific endpoint
stats = vllm_client.get_stats(endpoint_id)
print(f"Total requests: {stats['total_requests']}")
print(f"Successful requests: {stats['successful_requests']}")
print(f"Failed requests: {stats['failed_requests']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Queue size: {stats['queue_size']}")

# Get aggregate statistics across all endpoints
all_stats = vllm_client.get_stats()
print(f"Total endpoints: {all_stats['endpoints_count']}")
print(f"Total requests: {all_stats['total_requests']}")
```

### Load Balancing

```python
# Select an endpoint based on strategy
endpoint_id = vllm_client.select_endpoint(strategy="round-robin")
# Available strategies: "round-robin", "least-loaded", "fastest"

# Make a request with the selected endpoint
response = vllm_client(
    prompt="Hello, world!",
    endpoint_id=endpoint_id
)
```

## Queueing and Backoff

The VLLM implementation includes robust queueing and backoff mechanisms:

### Queue Processing

- Requests are queued when the endpoint reaches its `max_concurrent_requests` limit
- A background thread processes queued requests in FIFO order
- Each request in the queue has its complete context to be processed when dequeued
- Queue size limit prevents memory issues from too many pending requests

### Backoff Mechanism

- Exponential backoff when encountering rate limiting
- Starting with `initial_retry_delay`, each retry increases by `backoff_factor`
- Respects `Retry-After` headers from API responses
- Maximum retry count to prevent infinite retry loops
- Maximum delay cap to prevent excessive waits

### Request Tracking

Each request can have an associated request_id, either:
- Provided by the caller
- Auto-generated in the format: `req_{timestamp}_{hash}`

Request IDs help with:
- Tracking requests through logging
- Associating requests with responses
- Debugging and tracing request flows

## VLLM API Request Format

The VLLM API generally expects requests in this format:

```json
{
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stop": ["\n", "User:"],
  "n": 1
}
```

For batch requests:

```json
{
  "prompts": [
    "Hello, how are you?",
    "Tell me a joke",
    "What is the capital of France?"
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

For streaming:

```json
{
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}
```

## Troubleshooting

### Common Issues

- **Connection errors**: Verify VLLM server is running with `curl http://localhost:8000/health`
- **Model not found**: Verify the model is properly loaded on the server
- **Timeout errors**: Increase timeout with `--timeout` parameter
- **Too many requests**: Check if you've hit rate limits and use backoff
- **Queue full**: Increase queue_size or reduce request frequency

### Debugging Commands

```bash
# Check if VLLM server is running
curl http://localhost:8000/health

# Get information about loaded models
curl http://localhost:8000/model

# Test simple inference
curl -X POST http://localhost:8000/generate \
    -d '{"prompt": "Hello, world!", "max_tokens": 20}' \
    -H "Content-Type: application/json"
```

## Development Notes

When implementing new features in the VLLM backend:

1. Add corresponding tests in `test_vllm.py`
2. Update the `test_vllm_unified.py` for performance and advanced feature testing
3. Document new features in this README

The test suite is designed to gracefully handle missing features by checking for their existence using `hasattr()`, so it will not fail if a feature is not yet implemented.