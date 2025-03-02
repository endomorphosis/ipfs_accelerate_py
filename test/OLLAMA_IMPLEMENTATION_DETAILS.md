# Ollama API Implementation Details

## Overview

The Ollama API backend has been fully implemented with all required queue and backoff features. This implementation provides a robust interface to interact with Ollama's local LLM deployments, featuring thread-safe request queuing, exponential backoff for error handling, circuit breaker pattern for failure detection, and comprehensive usage statistics tracking.

## Key Features

### Basic Functionality
- Compatible with the Ollama API interface
- Support for chat, generate, and completions endpoints
- Support for stream and non-stream modes
- Model listing and endpoint handling
- Proper response processing and formatting

### Queue and Concurrency
- Thread-safe request queue with configurable size (default 100)
- Concurrency limits (default 5 concurrent requests)
- Priority queue system with HIGH, NORMAL, LOW priority levels
- Queue processor thread that manages concurrent requests
- Thread-safe locks for all shared resources

### Backoff and Failure Handling
- Exponential backoff retry mechanism
- Configurable max retries (default 5)
- Configurable initial retry delay (1 second)
- Configurable backoff factor (2x)
- Maximum retry delay cap (16 seconds)

### Circuit Breaker Pattern
- Three-state circuit breaker (CLOSED, OPEN, HALF-OPEN)
- Automatic failure detection and tracking
- Self-healing capabilities with configurable timeouts
- Fast-fail for unresponsive services
- Failure threshold configuration

### Usage Statistics
- Request counting (total, successful, failed)
- Token usage tracking (prompt, completion, total)
- Thread-safe statistics collection
- Reset capability for testing and monitoring

### Parameter Compatibility
- Support for both `model` and `model_name` parameters (for compatibility)
- Support for standardized parameters across different API backends
- Proper parameter conversion to Ollama's API format
- Support for request IDs and tracking

## Implementation Details

### Queue Processing
The queue processing implementation uses a dedicated thread that:
1. Takes requests from a thread-safe queue
2. Checks circuit breaker state before processing
3. Makes the API request with proper error handling
4. Updates circuit breaker state based on success/failure
5. Updates usage statistics
6. Returns results to the caller via Future objects

### Circuit Breaker
The circuit breaker implementation:
1. Starts in the CLOSED state (normal operation)
2. Transitions to OPEN when failure count exceeds threshold
3. Stays OPEN for a configurable timeout
4. Transitions to HALF-OPEN after timeout to test service
5. Returns to CLOSED if request succeeds in HALF-OPEN state
6. Returns to OPEN if request fails in HALF-OPEN state

### Method Signatures
All methods have been designed for compatibility with the standardized API backend interface:

```python
def chat(self, model_name=None, model=None, messages=None, max_tokens=None, temperature=None, request_id=None, options=None, **kwargs)
def generate(self, model=None, prompt=None, max_tokens=None, temperature=None, request_id=None, **kwargs)
def completions(self, model=None, prompt=None, max_tokens=None, temperature=None, request_id=None, **kwargs)
def stream_chat(self, model_name=None, model=None, messages=None, max_tokens=None, temperature=None, request_id=None, options=None, **kwargs)
```

### Callable Interface
The implementation supports being called directly with endpoint type and parameters:

```python
response = client("chat", model="llama3", messages=[{"role": "user", "content": "Hello"}])
```

## Testing

The implementation has been thoroughly tested with:

1. Basic functionality tests for method signatures and initialization
2. Parameter compatibility tests for all supported parameters
3. Backoff and queue structure tests to verify all required features
4. Mock tests to verify the Ollama-specific implementation
5. Integration tests with the API backoff queue testing framework

## Usage Example

```python
# Initialize Ollama client
from ipfs_accelerate_py.api_backends.ollama import ollama

# Create client with optional configuration
client = ollama(metadata={
    "ollama_api_url": "http://localhost:11434/api",
    "ollama_model": "llama3"
})

# Basic chat request
response = client.chat(
    model="llama3",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    max_tokens=100,
    temperature=0.7
)

# Generate request (alternative interface)
response = client.generate(
    model="llama3",
    prompt="Explain quantum computing in simple terms",
    max_tokens=200
)

# Streaming chat request
for chunk in client.stream_chat(
    model="llama3",
    messages=[{"role": "user", "content": "Tell me a story"}]
):
    print(chunk.get("text", ""), end="", flush=True)
```

## Advanced Features

### Priority Queueing
```python
# High priority request - processed before normal and low priority requests
response = client.make_post_request_ollama(
    endpoint_url=f"{client.ollama_api_url}/chat",
    data={"model": "llama3", "messages": [{"role": "user", "content": "Urgent request"}]},
    priority=client.PRIORITY_HIGH
)
```

### Usage Statistics Monitoring
```python
# Get current usage statistics
stats = client.usage_stats
print(f"Total requests: {stats['total_requests']}")
print(f"Successful requests: {stats['successful_requests']}")
print(f"Failed requests: {stats['failed_requests']}")
print(f"Total tokens: {stats['total_tokens']}")

# Reset statistics for monitoring period
client.reset_usage_stats()
```

## Configuration Options

The Ollama API client can be configured through the metadata dictionary:

```python
client = ollama(metadata={
    # API endpoint configuration
    "ollama_api_url": "http://localhost:11434/api",
    "ollama_model": "llama3",
    
    # Request timeout
    "timeout": 30,  # seconds
})

# Queue and concurrency configuration
client.max_concurrent_requests = 10  # Increase concurrency
client.queue_size = 200  # Increase queue size

# Backoff configuration
client.max_retries = 3
client.backoff_factor = 1.5
client.initial_retry_delay = 2
client.max_retry_delay = 30

# Circuit breaker configuration
client.failure_threshold = 10
client.circuit_timeout = 60  # seconds
```