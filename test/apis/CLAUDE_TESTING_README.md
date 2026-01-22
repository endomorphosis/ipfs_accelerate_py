# Claude (Anthropic) API Testing Guide

This guide provides information on testing the Claude API integration within the IPFS Accelerate Python framework.

## Overview

The Claude API by Anthropic provides access to state-of-the-art large language models. Key features include:

- Text generation with Claude 3 models (Opus, Sonnet, Haiku)
- Multiple input/output formats including text and tools
- Streaming capability for real-time responses
- Optional function/tool calling capabilities
- Detailed token usage statistics
- Fine-grained control over generation parameters

The test files in this directory validate the implementation of the Claude API backend, including:

- Basic API functionality
- Request formatting and parameter handling
- Streaming response processing
- Error handling and resilience
- API key multiplexing
- Request tracking
- Queue processing and backoff mechanisms
- Usage statistics

## Test Files

### 1. `test_claude.py`

This file contains the core test functionality for the Claude API, including tests for:

- **API Endpoint Handler Creation**
  - Basic endpoint handler creation
  - Parameter validation

- **Request Formatting**
  - Message formatting for chat API
  - Parameter handling (max_tokens, temperature)
  - Request ID tracking

- **Response Processing**
  - Response format validation
  - Token usage extraction
  - Error handling

- **Streaming Generation**
  - Stream initialization
  - Chunk processing 
  - Stream termination
  - Content block handling

- **API Key Multiplexing**
  - Multiple endpoint creation with different API keys
  - Endpoint-specific configuration
  - Per-endpoint usage statistics

- **Advanced Queue System (Updated March 2025)**
  - Thread-safe request queue implementation
  - Prioritized queue processing
  - Proper thread management and resource cleanup
  - Robust error handling with retry logic
  - Configurable concurrency controls

- **Comprehensive Backoff System**
  - Exponential backoff for rate limiting
  - Circuit breaker pattern for service outages
  - Retry-After header handling
  - Customizable retry configuration
  - Individual backoff state per endpoint

- **Error Handling**
  - Authentication errors (401)
  - Rate limiting (429)
  - Invalid requests (400)
  - Server errors (500)
  - Service outage detection and recovery

### 2. `test_api_backoff_queue.py`

This test file (found in the parent directory) specifically validates the queue and backoff systems:

- **Queue Testing**: Verifies that concurrent requests are properly handled
- **Backoff Testing**: Confirms exponential backoff for rate limits and errors
- **Concurrency Control**: Tests that max_concurrent_requests is enforced
- **Request Tracking**: Validates proper request ID generation and tracking

## Running Tests

You can run various Claude API tests using:

```bash
# Run standard test suite
python -m test.apis.test_claude

# Run with specific API key
ANTHROPIC_API_KEY="your-api-key-here" python -m test.apis.test_claude

# Run specific queue and backoff tests
python generators/models/test_api_backoff_queue.py --api claude

# Run comprehensive tests with mock support (no API key needed)
MOCK_CLAUDE_TEST=1 python generators/models/test_api_backoff_queue.py --api claude

# Run as part of the complete test suite
python run_queue_backoff_tests.py --apis claude
```

### Advanced Testing Features (March 2025)

The updated test suite includes:

1. **Mock Testing Support**: Tests can now run without real API keys by using the built-in mock response system
2. **Thread-Safety Validation**: Tests verify proper locking and resource management in concurrent scenarios
3. **Queue Prioritization**: Verification that high-priority requests are processed before lower priority ones
4. **Circuit Breaker Testing**: Simulation of service outages to test detection and recovery
5. **Comprehensive Metrics Validation**: Tests that all usage statistics are properly tracked

## Environment Variables

The following environment variables control test behavior:

- `ANTHROPIC_API_KEY`: Your Anthropic Claude API key (required)

## API Key Multiplexing Features

The Claude API implementation supports multiplexing multiple API keys, which provides:

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
from ipfs_accelerate_py.api_backends import claude

# Initialize with default API key
claude_client = claude(resources={}, metadata={"claude_api_key": "default-key"})

# Create endpoint with specific API key
endpoint1 = claude_client.create_endpoint(
    api_key="custom-key-1",
    max_concurrent_requests=5,
    queue_size=20,
    max_retries=3,
    initial_retry_delay=1,
    backoff_factor=2
)

# Create another endpoint with different configuration
endpoint2 = claude_client.create_endpoint(
    api_key="custom-key-2",
    max_concurrent_requests=10,
    queue_size=50,
    max_retries=5
)
```

### Making Requests with Specific Endpoints

```python
# Use specific endpoint with request_id tracking
response = claude_client.chat(
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    endpoint_id=endpoint1,
    request_id="req-123-abc"
)

# Request without explicit request_id (auto-generated)
response2 = claude_client.chat(
    messages=[{"role": "user", "content": "Another question"}],
    endpoint_id=endpoint2
)
```

### Usage Statistics

```python
# Get statistics for a specific endpoint
stats = claude_client.get_stats(endpoint1)
print(f"Total requests: {stats['total_requests']}")
print(f"Successful requests: {stats['successful_requests']}")
print(f"Failed requests: {stats['failed_requests']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Input tokens: {stats['input_tokens']}")
print(f"Output tokens: {stats['output_tokens']}")

# Get aggregate statistics across all endpoints
all_stats = claude_client.get_stats()
print(f"Total endpoints: {all_stats['endpoints_count']}")
print(f"Total requests: {all_stats['total_requests']}")
```

## Test Result Structure

Test results are saved to `collected_results/claude_test_results.json` and include:

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
  "model_compatibility": "Success"
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
   - Verify the message format follows Claude API documentation
   - Ensure all required fields are present
   - Check parameter ranges (temperature, max_tokens)

4. **Performance Issues**:
   - Adjust max_concurrent_requests to match your quota
   - Configure appropriate queue sizes
   - Use streaming for interactive applications

### Debugging Commands

```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY

# Check connectivity to Claude API
curl -s -o /dev/null -w "%{http_code}" \
  -X POST \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  "https://api.anthropic.com/v1/messages" \
  -d '{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":"test"}],"max_tokens":1}'

# Test simple request 
curl -X POST \
  "https://api.anthropic.com/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":"Hello, Claude!"}],"max_tokens":100}'
```