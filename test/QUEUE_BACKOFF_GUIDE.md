# Queue and Backoff Implementation Guide

## Overview

The IPFS Accelerate Python framework implements a standardized queue and backoff system across all API backends to provide:

1. **Request Queueing**: Thread-safe management of concurrent requests
2. **Exponential Backoff**: Automatic retry with increasing delays for transient errors
3. **Circuit Breaker**: Automatic detection and handling of service outages
4. **Request Tracking**: Unique IDs for all requests to enable tracing and diagnostics

This guide explains how these features work and how to configure them for your needs.

## Queue System

The queue system manages concurrent requests to avoid overwhelming API rate limits or causing connection errors.

### Queue Features

- **Thread-safe implementation**: Prevents race conditions in multi-threaded environments
- **Configurable concurrency limits**: Control how many requests execute simultaneously
- **Automatic queue processing**: Background thread handles queued requests
- **Priority levels**: Support for HIGH, NORMAL, and LOW priority requests (when enabled)
- **Status monitoring**: Track queue size, processing time, and success rates

### Queue Configuration

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Basic queue configuration
client.queue_enabled = True               # Enable/disable queueing (default: True)
client.max_concurrent_requests = 5        # Maximum concurrent requests (default: 5)
client.queue_size = 100                   # Maximum queue capacity (default: 100)

# Queue monitoring
current_size = len(client.request_queue)  # Get current queue size
active = client.current_requests          # Get number of active requests

# Queue monitoring with the API (if implemented)
if hasattr(client, "get_queue_info"):
    info = client.get_queue_info()
    print(f"Queue size: {info.get('size', 0)}")
    print(f"Active requests: {info.get('active_requests', 0)}")
    print(f"Queue capacity: {info.get('capacity', 0)}")
```

### How the Queue Works

1. When a request is made, the system checks if the number of current requests exceeds `max_concurrent_requests`
2. If capacity is available, the request is processed immediately
3. If at capacity, the request is added to the queue with a unique request ID
4. A background thread processes queued requests as capacity becomes available
5. Each request is tracked with a future object to enable waiting for results

## Exponential Backoff

The backoff system automatically retries failed requests with increasing delays to handle temporary issues like network problems or rate limits.

### Backoff Features

- **Exponential delay**: Each retry waits longer than the previous one
- **Configurable retry count**: Control maximum number of retry attempts
- **Configurable delay parameters**: Adjust initial delay and backoff factor
- **Maximum delay cap**: Prevent excessively long waits
- **Error categorization**: Different handling for retryable vs. non-retryable errors

### Backoff Configuration

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Configure backoff settings
client.max_retries = 5                    # Maximum retry attempts (default: 5)
client.initial_retry_delay = 1            # Initial delay in seconds (default: 1)
client.backoff_factor = 2                 # Multiply delay by this factor each retry (default: 2)
client.max_retry_delay = 60               # Maximum delay cap in seconds (default: 60)
```

### How Backoff Works

1. When a request fails, the system checks if the error is retryable (e.g., network error, rate limit)
2. If retryable and `retry_count < max_retries`, the system waits for the calculated delay:
   ```
   delay = min(initial_retry_delay * (backoff_factor ^ retry_count), max_retry_delay)
   ```
3. After waiting, the request is retried
4. If the retry succeeds, the result is returned; if it fails, the process repeats
5. If `max_retries` is reached, the last error is propagated to the caller

### Delay Calculation Examples

With default settings (`initial_retry_delay=1`, `backoff_factor=2`, `max_retry_delay=60`):

| Retry | Calculation         | Delay  |
|-------|---------------------|--------|
| 1     | 1 * (2^0)           | 1s     |
| 2     | 1 * (2^1)           | 2s     |
| 3     | 1 * (2^2)           | 4s     |
| 4     | 1 * (2^3)           | 8s     |
| 5     | 1 * (2^4)           | 16s    |

## Circuit Breaker Pattern

The circuit breaker pattern prevents repeated requests to a service that is experiencing an outage.

### Circuit Breaker Features

- **Three-state machine**: CLOSED (normal), OPEN (failing), HALF-OPEN (testing recovery)
- **Automatic failure detection**: Opens the circuit after consecutive failures
- **Self-healing**: Automatically tests recovery after a timeout period
- **Fast-fail for service outages**: Prevents unnecessary requests during outages
- **Configurable thresholds**: Adjust sensitivity to failures

### Circuit Breaker Configuration

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Configure circuit breaker (if implemented)
if hasattr(client, "circuit_failure_threshold"):
    client.circuit_failure_threshold = 5  # Failures before opening circuit (default: 5)
    client.circuit_reset_timeout = 30     # Seconds before trying half-open (default: 30)
    client.circuit_success_threshold = 3  # Success count to close circuit (default: 3)

# Check circuit state (if implemented)
if hasattr(client, "get_circuit_state"):
    state = client.get_circuit_state()
    print(f"Circuit state: {state}")  # CLOSED, OPEN, or HALF_OPEN
```

### How Circuit Breaker Works

1. **CLOSED state** (normal operation):
   - All requests are processed normally
   - Failures are counted
   - When `failure_count >= circuit_failure_threshold`, circuit changes to OPEN

2. **OPEN state** (service outage):
   - All requests immediately fail with "Circuit Breaker Open" error
   - After `circuit_reset_timeout` seconds, circuit changes to HALF-OPEN

3. **HALF-OPEN state** (testing recovery):
   - One test request is allowed through
   - If successful, `success_count` is incremented
   - When `success_count >= circuit_success_threshold`, circuit changes to CLOSED
   - If the test request fails, circuit returns to OPEN and resets the timeout

## Request Tracking

All requests are tracked with unique IDs to enable diagnostics and monitoring.

### Request Tracking Features

- **Automatic ID generation**: Each request gets a unique ID
- **Custom ID support**: Specify your own IDs for integration with existing systems
- **Timestamp tracking**: Record when requests start and complete
- **Status tracking**: Track success/failure and error types
- **Performance metrics**: Measure response times and throughput

### Request Tracking Usage

```python
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Send request with custom ID
response = client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    request_id="my-custom-id-123"  # Custom request ID
)

# If no ID is specified, one is generated automatically
# Format: "req_{timestamp}_{random_hash}"
# Example: "req_1740878332_a1b2c3d4"
```

## Testing Queue and Backoff

The framework includes comprehensive tests for queue and backoff functionality:

```bash
# Test basic queue functionality for a specific API
python test_api_backoff_queue.py --api openai

# Run comprehensive Ollama tests with specific model
python test_ollama_backoff_comprehensive.py --model llama3 --host http://localhost:11434

# Test with different concurrency settings
python test_ollama_backoff_comprehensive.py --queue-size 10 --max-concurrent 2

# Run all queue and backoff tests
python run_queue_backoff_tests.py

# Test specific APIs only
python run_queue_backoff_tests.py --apis openai groq claude

# Skip specific APIs
python run_queue_backoff_tests.py --skip-apis llvm opea ovms
```

## Implementation Details

The queue and backoff implementation follows these patterns:

### Core Components

1. **Request Queue**: List-based queue for pending requests
2. **Queue Lock**: Threading.RLock for thread-safety
3. **Queue Processor**: Background thread that processes queued requests
4. **Backoff Calculator**: Computes exponential delays for retries
5. **Circuit Breaker**: State machine for managing service outages

### Standard Attributes

All API backends implement these standard attributes:

```python
# Queue Settings
self.queue_enabled = True
self.queue_size = 100
self.queue_processing = False
self.current_requests = 0
self.max_concurrent_requests = 5
self.request_queue = []
self.queue_lock = threading.RLock()

# Backoff Settings
self.max_retries = 5
self.initial_retry_delay = 1
self.backoff_factor = 2
self.max_retry_delay = 60

# Circuit Breaker Settings (when implemented)
self.circuit_state = "CLOSED"
self.failure_count = 0
self.success_count = 0
self.circuit_failure_threshold = 5
self.circuit_reset_timeout = 30
self.circuit_success_threshold = 3
self.last_failure_time = 0
```

## Best Practices

1. **Concurrency Tuning**:
   - Set `max_concurrent_requests` based on API rate limits
   - For OpenAI: 5-10 concurrent requests
   - For Claude: 3-5 concurrent requests
   - For Groq: 20-30 concurrent requests
   - For Ollama: 1-3 concurrent requests (depends on hardware)

2. **Backoff Tuning**:
   - Increase `initial_retry_delay` for APIs with strict rate limits
   - Decrease `backoff_factor` for less aggressive backoff
   - Increase `max_retries` for critical operations
   - Decrease `max_retries` for time-sensitive operations

3. **Circuit Breaker Tuning**:
   - Increase `circuit_failure_threshold` for less sensitive triggering
   - Decrease `circuit_reset_timeout` for faster recovery attempts
   - Increase `circuit_success_threshold` for more cautious recovery

4. **Queue Size Considerations**:
   - Larger queues (100+) handle request spikes but may delay responses
   - Smaller queues (10-50) fail faster when overloaded
   - Monitor queue size during operation to adjust as needed

## Troubleshooting

### Common Issues

1. **Queue Overflow**:
   - **Symptom**: "Request queue is full" errors
   - **Solution**: Increase `queue_size` or reduce request rate

2. **Excessive Retries**:
   - **Symptom**: Requests taking too long to complete or fail
   - **Solution**: Reduce `max_retries` or adjust `backoff_factor`

3. **Circuit Breaker Triggering Too Often**:
   - **Symptom**: "Circuit breaker is OPEN" errors during normal operation
   - **Solution**: Increase `circuit_failure_threshold` or improve error handling

4. **Concurrent Request Limits Too Low**:
   - **Symptom**: Requests queuing unnecessarily, slow overall throughput
   - **Solution**: Increase `max_concurrent_requests` based on API limits

### Debugging

For debugging queue and backoff issues:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check queue state
print(f"Queue size: {len(client.request_queue)}")
print(f"Active requests: {client.current_requests}")
print(f"Queue enabled: {client.queue_enabled}")

# Check backoff state
print(f"Max retries: {client.max_retries}")
print(f"Backoff factor: {client.backoff_factor}")

# Check circuit breaker state (if implemented)
if hasattr(client, "circuit_state"):
    print(f"Circuit state: {client.circuit_state}")
    print(f"Failure count: {client.failure_count}")
    print(f"Success count: {client.success_count}")
```

## Performance Impact

The queue and backoff implementation adds minimal overhead:

- **Memory Usage**: ~100KB per 1000 queued requests
- **CPU Usage**: Negligible for typical workloads
- **Latency Impact**: <5ms per request for queue processing
- **Throughput Impact**: Can actually improve throughput by preventing rate limit errors

## Conclusion

The standardized queue and backoff implementation across all API backends ensures robust handling of rate limits, transient errors, and service outages. By configuring these systems appropriately, you can achieve optimal performance and reliability when using the IPFS Accelerate Python framework.