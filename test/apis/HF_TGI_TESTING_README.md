# HuggingFace Text Generation Inference (TGI) Testing Guide

This guide provides information on testing the HuggingFace Text Generation Inference (TGI) API integration within the IPFS Accelerate Python framework.

## Overview

HuggingFace Text Generation Inference (TGI) is a high-performance solution for deploying large language models for text generation. The TGI API offers:

- High-performance text generation with optimized inference
- Support for a wide range of language models (LLM, PEFT, LoRA)
- Optimized for efficiency with a Rust-based implementation
- Continuous batching for high throughput
- Quantization (8-bit, 4-bit) to reduce memory usage
- Tensor parallelism for distributed inference
- Streaming generation for real-time responses
- Parameter control (temperature, top-p, top-k)
- Self-hosted deployment options with Docker containers

The test files in this directory validate the implementation of the HuggingFace TGI API backend, including:

- Basic API functionality
- Request formatting and parameter handling  
- Error handling and resilience
- Streaming response processing
- Multimodal capabilities
- API key multiplexing
- Request tracking
- Queue processing and backoff mechanisms
- Usage statistics

## Test Files

### 1. `test_hf_tgi.py`

This file contains the core test functionality for the HuggingFace TGI API, including tests for:

- **API Endpoint Handler Creation**
  - Basic endpoint handler creation
  - Parameter validation

- **Request Formatting**
  - Text prompt formatting
  - Chat request formatting
  - Parameter handling (temperature, top_p, max_tokens)
  - Request ID tracking

- **Response Processing**
  - Response format validation
  - Token count extraction
  - Error handling

- **Streaming Generation**
  - Stream initialization
  - Chunk processing
  - Stream termination

- **Generation Parameters**
  - Temperature and sampling settings
  - Sequence length constraints
  - Repetition penalties
  - Stop sequences

- **API Key Multiplexing**
  - Multiple endpoint creation with different API keys
  - Endpoint-specific configuration
  - Per-endpoint usage statistics

- **Queuing and Backoff**
  - Request queuing
  - Queue processing
  - Exponential backoff for rate limiting
  - Concurrent request management

- **Error Handling**
  - Authentication errors (401)
  - Rate limiting (429)
  - Invalid requests (400)
  - Server errors (500)

## Running Tests

You can run the HuggingFace TGI API tests using:

```bash
# Run standard test suite
python -m test.apis.test_hf_tgi

# Run with specific API key
HF_API_KEY="your-api-key-here" python -m test.apis.test_hf_tgi

# Run with a specific model
HF_MODEL_ID="google/t5-efficient-tiny" python -m test.apis.test_hf_tgi

# Run with multiple API keys
HF_API_KEY="key1" HF_API_KEY_1="key2" HF_API_KEY_2="key3" python -m test.apis.test_hf_tgi
```

## Environment Variables

The following environment variables control test behavior:

- `HF_API_KEY`: Your HuggingFace API key (required)
- `HF_MODEL_ID`: The model ID to use for testing (default: "google/t5-efficient-tiny")
- `HF_API_KEY_1`, `HF_API_KEY_2`, etc.: Additional API keys for multiplexing tests
- `HF_MODEL_ID_1`, `HF_MODEL_ID_2`, etc.: Models to use with additional API keys

## API Key Multiplexing Features

The HuggingFace TGI API implementation supports multiplexing multiple API keys, which provides:

1. **Multiple API Keys**: Each key with its own:
   - Request counters and limits
   - Separate request queue
   - Customizable concurrency settings
   - Distinct backoff parameters
   - Usage statistics tracking

2. **Request Tracking**: Each request can have its own request_id for:
   - Debugging and logging
   - Correlation between requests and responses
   - Usage tracking and billing

3. **Queue System**: Automatic handling of concurrent requests:
   - Configurable maximum concurrent requests
   - FIFO queue for pending requests
   - Future-based wait mechanism for queued requests
   - Proper resource cleanup

4. **Backoff Mechanism**: Intelligent handling of rate limits:
   - Exponential backoff with customizable parameters
   - Respects Retry-After headers when provided
   - Configurable maximum retries
   - Per-endpoint backoff state tracking

### Creating Multiple Endpoints

```python
from ipfs_accelerate_py.api_backends import gemini

# Initialize with default API key
gemini_client = gemini(resources={}, metadata={"gemini_api_key": "default-key"})

# Create endpoint with specific API key
endpoint1 = gemini_client.create_endpoint(
    api_key="custom-key-1",
    max_concurrent_requests=5,
    queue_size=20,
    max_retries=3,
    initial_retry_delay=1,
    backoff_factor=2
)

# Create another endpoint with different configuration
endpoint2 = gemini_client.create_endpoint(
    api_key="custom-key-2",
    max_concurrent_requests=10,
    queue_size=50,
    max_retries=5
)
```

### Making Requests with Specific Endpoints

```python
# Use specific endpoint with request_id tracking
response = gemini_client.chat(
    messages=[{"role": "user", "content": "Hello, Gemini!"}],
    endpoint_id=endpoint1,
    request_id="req-123-abc"
)

# Request without explicit request_id (auto-generated)
response2 = gemini_client.chat(
    messages=[{"role": "user", "content": "Another question"}],
    endpoint_id=endpoint2
)
```

### Usage Statistics

```python
# Get statistics for a specific endpoint
stats = gemini_client.get_stats(endpoint1)
print(f"Total requests: {stats['total_requests']}")
print(f"Successful requests: {stats['successful_requests']}")
print(f"Failed requests: {stats['failed_requests']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Input tokens: {stats['input_tokens']}")
print(f"Output tokens: {stats['output_tokens']}")

# Get aggregate statistics across all endpoints
all_stats = gemini_client.get_stats()
print(f"Total endpoints: {all_stats['endpoints_count']}")
print(f"Total requests: {all_stats['total_requests']}")
```

## Test Result Structure

Test results are saved to `collected_results/gemini_test_results.json` and include:

```json
{
  "multiplexing_endpoint_creation": "Success", 
  "usage_statistics": "Success",
  "queue_enabled": "Success",
  "request_queue": "Success",
  "max_concurrent_requests": "Success",
  "current_requests": "Success",
  "max_retries": "Success", 
  "initial_retry_delay": "Success",
  "backoff_factor": "Success",
  "queue_processing": "Success",
  "endpoint_handler": "Success",
  "test_endpoint": "Success",
  "test_endpoint_params": "Success",
  "post_request": "Success",
  "post_request_headers": "Success",
  "request_id_tracking": "Success",
  "chat_method": "Success",
  "streaming": "Success",
  "error_handling_auth": "Success",
  "error_handling_rate_limit": "Success", 
  "error_handling_400": "Success",
  "image_processing": "Success"
}
```

## Troubleshooting

### Common Issues

1. **API Key Authentication Failures**:
   - Check if your API key is valid
   - Verify the environment variable is set correctly
   - Confirm the API key has proper permissions

2. **Rate Limiting**:
   - Reduce concurrent requests
   - Implement proper backoff handling
   - Use multiple API keys with load balancing

3. **Request Format Errors**:
   - Check content format matches Gemini API expectations
   - Verify parameter names and value ranges

4. **Performance Issues**:
   - Adjust max_concurrent_requests to match your quota
   - Configure appropriate queue sizes
   - Use streaming for interactive applications

### Debugging Commands

```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Check connectivity to Gemini API
curl -s -o /dev/null -w "%{http_code}" \
  "https://generativelanguage.googleapis.com/v1/models?key=$GEMINI_API_KEY"

# Test simple request
curl -X POST \
  "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=$GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hello, world!"}]}]}'
```