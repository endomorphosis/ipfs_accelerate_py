# Advanced API Features Documentation

This document describes the advanced features implemented in the API backends of the IPFS Accelerate Python framework.

## Overview of Advanced Features

The API backends have been enhanced with several advanced features to improve reliability, performance, and monitoring:

1. **Priority Queue System**
   - Allows critical requests to be processed before less important ones
   - Configurable priority levels (HIGH, NORMAL, LOW)
   - Thread-safe implementation for concurrent environments

2. **Circuit Breaker Pattern**
   - Prevents overwhelming failing services
   - Automatically detects persistent failures
   - Implements self-healing with CLOSED, OPEN, and HALF-OPEN states
   - Configurable failure thresholds and reset timeouts

3. **Enhanced Monitoring and Reporting**
   - Detailed statistics tracking for all requests
   - Error classification and tracking
   - Performance metrics collection
   - Comprehensive reporting capabilities

4. **Request Batching**
   - Optimizes throughput for supported models
   - Automatically combines similar requests
   - Configurable batch size and timeout
   - Model-specific batching strategies

## Configuration

All advanced features can be configured through initialization parameters or by setting attributes after initialization:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create a client with custom settings
client = openai_api(
    resources={},
    metadata={
        "openai_api_key": "your-api-key",
        "max_concurrent_requests": 10,
        "queue_size": 50,
        "max_retries": 3,
        "backoff_factor": 1.5
    }
)

# Or configure after initialization
client.max_concurrent_requests = 10  # Set concurrency limit
client.queue_size = 50              # Set queue capacity
client.max_retries = 3              # Configure retry attempts
client.backoff_factor = 1.5         # Adjust backoff aggressiveness

# Circuit breaker settings
client.failure_threshold = 5        # Failures before opening circuit
client.reset_timeout = 30           # Seconds to wait before half-open

# Priority settings
client.PRIORITY_HIGH = 0            # Highest priority (lowest number)
client.PRIORITY_NORMAL = 1          # Normal priority
client.PRIORITY_LOW = 2             # Low priority

# Batching settings
client.batching_enabled = True      # Enable/disable batching
client.max_batch_size = 10          # Maximum batch size
client.batch_timeout = 0.5          # Maximum wait time (seconds)
```

## Priority Queue Usage

The priority queue system allows you to prioritize important requests:

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Queue with explicit priority (HIGH, NORMAL, or LOW)
future = client.queue_with_priority(
    {
        "endpoint": "chat/completions",
        "data": {...},
        "api_key": client.api_key,
        "request_id": "req-123",
    },
    client.PRIORITY_HIGH  # This request will be processed before lower priority ones
)

# Wait for the result
while not future.get("completed", False):
    time.sleep(0.1)

# Check for errors
if future.get("error"):
    # Handle error
    print(f"Error: {future['error']}")
else:
    # Process result
    result = future["result"]
```

## Circuit Breaker Usage

The circuit breaker pattern is automatically applied to all requests. You can check its status and manually control it if needed:

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Check circuit state
if client.circuit_state == "OPEN":
    print(f"Circuit is OPEN, service appears to be down. Will retry in {client.reset_timeout}s")
elif client.circuit_state == "HALF_OPEN":
    print("Circuit is HALF-OPEN, testing service availability")
else:
    print("Circuit is CLOSED, service operating normally")

# Manually track a result (for external operations)
client.track_request_result(success=True)  # Record successful operation
client.track_request_result(success=False, error_type="ConnectionError")  # Record failure
```

## Monitoring and Reporting

The enhanced monitoring system provides detailed insights into API performance:

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Make some requests
client.chat(
    model_name="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

# Get raw statistics
stats = client.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Successful requests: {stats['successful_requests']}")
print(f"Failed requests: {stats['failed_requests']}")
print(f"Average response time: {stats['average_response_time']:.3f}s")

# Generate a comprehensive report
report = client.generate_report(include_details=True)
print(f"Success rate: {report['summary']['success_rate']:.1f}%")
print(f"Retry rate: {report['summary']['retry_rate']:.1f}%")
print(f"Circuit breaker state: {report['circuit_breaker']['state']}")
print(f"Models used: {report['models']}")
print(f"Error types: {report['errors']}")

# Reset statistics
client.reset_stats()
```

## Request Batching

The request batching system automatically optimizes supported operations:

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Enable and configure batching
client.batching_enabled = True
client.max_batch_size = 16  # Larger batch size
client.batch_timeout = 0.2  # Shorter waiting time

# Define supported models
client.embedding_models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
client.supported_batch_models = client.embedding_models

# Requests to these models will be automatically batched
embeddings1 = client.make_request("embeddings", {
    "model": "text-embedding-3-small",
    "input": "First text to embed"
})

embeddings2 = client.make_request("embeddings", {
    "model": "text-embedding-3-small",
    "input": "Second text to embed"
})

# The requests are batched behind the scenes for better performance
```

## Testing Advanced Features

A comprehensive test suite is available to verify the advanced features:

```bash
# Test one API backend
python test_enhanced_api_features.py --api openai

# Test with your own API key
python test_enhanced_api_features.py --api groq --key your-api-key
```

## Troubleshooting

Common issues and their solutions:

### Queue Full Errors

If you see "Request queue is full" errors:

1. Increase the queue size: `client.queue_size = 200`
2. Reduce the incoming request rate
3. Increase the concurrency limit: `client.max_concurrent_requests = 10`

### Circuit Breaker Tripping

If the circuit breaker opens frequently:

1. Increase the failure threshold: `client.failure_threshold = 10`
2. Increase the backoff factor: `client.backoff_factor = 3`
3. Check for API service issues or rate limits

### Performance Optimization

For optimal performance:

1. Enable batching for supported models
2. Adjust batch size based on model and use case
3. Use the least-loaded strategy for API key multiplexing
4. Monitor the success rate and adjust settings accordingly

## Implementation Details

The advanced features are implemented in a modular way to ensure compatibility across all API backends:

- `queue_with_priority()`: Queues requests with priority levels
- `check_circuit_breaker()`: Implements the circuit breaker state machine
- `track_request_result()`: Tracks success/failure for circuit breaker and monitoring
- `update_stats()`: Updates request statistics in a thread-safe manner
- `get_stats()` and `generate_report()`: Provide monitoring insights
- `add_to_batch()` and `_process_batch()`: Handle request batching