# API Key Multiplexing Guide

## Overview

The enhanced API Key Multiplexing feature provides robust management of multiple API keys for OpenAI, Groq, Claude, and Gemini APIs, with substantial improvements:

1. **Per-Endpoint Management**: Each endpoint has its own counters, API key, backoff, and queue
2. **Request Tracking**: Every request can have its own request_id for tracking and debugging
3. **Detailed Statistics**: Track usage statistics per endpoint for better resource monitoring
4. **Advanced Queueing**: Configurable queue size and concurrency limits per endpoint
5. **Smart Backoff**: Exponential backoff with retry logic that respects server response headers

## Key Features

### 1. Per-Endpoint Configuration

Each endpoint can have its own configuration:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_key` | API key for this specific endpoint | (Primary key) |
| `max_concurrent_requests` | Maximum concurrent requests | 5 |
| `queue_size` | Maximum queue size | 100 |
| `initial_retry_delay` | Initial delay for backoff (seconds) | 1 |
| `backoff_factor` | Factor to multiply delay on each retry | 2 |
| `max_retries` | Maximum number of retries for failed requests | 5 |
| `max_retry_delay` | Maximum retry delay (seconds) | 60 |
| `queue_enabled` | Enable/disable queueing for the endpoint | True |

### 2. Request IDs

Each request can have an optional request_id parameter:
- Automatically generated if not provided
- Helps track requests through logging and metrics
- Useful for debugging and tracing request flows

### 3. Detailed Statistics

Each endpoint tracks:
- Total requests
- Successful requests
- Failed requests
- Total tokens
- Input tokens
- Output tokens
- Queue size
- Current active requests

## Implementation

### Creating Endpoints

```python
from ipfs_accelerate_py.api_backends import claude, openai_api, gemini, groq

# Initialize API client with default key
claude_client = claude(resources={}, metadata={"claude_api_key": "default-api-key"})

# Create an endpoint with a specific API key
endpoint_id = claude_client.create_endpoint(
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
endpoint_id2 = claude_client.create_endpoint(
    api_key="custom-api-key-2",
    max_retries=3,
    max_concurrent_requests=10
)
```

### Making Requests with Specific Endpoints

```python
# Make a request using a specific endpoint
response = claude_client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    endpoint_id=endpoint_id,
    request_id="custom-request-id-123"  # Optional
)

# Make a request with another endpoint
response2 = claude_client.chat(
    messages=[{"role": "user", "content": "Hello again"}],
    endpoint_id=endpoint_id2
    # request_id will be auto-generated if not provided
)
```

### Tracking Usage Statistics

```python
# Get statistics for a specific endpoint
stats = claude_client.get_stats(endpoint_id)
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

### Resetting Statistics

```python
# Reset stats for a specific endpoint
claude_client.reset_stats(endpoint_id)

# Reset stats for all endpoints
claude_client.reset_stats()
```

### Updating Endpoint Settings

```python
# Update settings for an existing endpoint
claude_client.update_endpoint(
    endpoint_id,
    max_retries=10,
    queue_enabled=False
)
```

## Advanced Usage Patterns

### Load Balancing Across Multiple API Keys

```python
from ipfs_accelerate_py.api_backends import claude
import random

# Initialize API client
claude_client = claude(resources={}, metadata={})

# Create multiple endpoints with different API keys
api_keys = [
    "api-key-1",
    "api-key-2",
    "api-key-3"
]

endpoints = []
for i, key in enumerate(api_keys):
    endpoint_id = claude_client.create_endpoint(
        api_key=key,
        max_concurrent_requests=5,
        queue_size=20
    )
    endpoints.append(endpoint_id)

# Function to select an endpoint based on current usage
def select_endpoint():
    # Get stats for all endpoints
    endpoint_stats = {endpoint: claude_client.get_stats(endpoint) for endpoint in endpoints}
    
    # First try to find an endpoint that isn't at capacity
    available_endpoints = [
        ep for ep, stats in endpoint_stats.items() 
        if stats["current_requests"] < claude_client.endpoints[ep]["max_concurrent_requests"]
    ]
    
    if available_endpoints:
        # Sort by current usage and pick the least used
        return min(available_endpoints, key=lambda ep: endpoint_stats[ep]["current_requests"])
    else:
        # All endpoints at capacity, pick one with the shortest queue
        return min(endpoints, key=lambda ep: len(claude_client.endpoints[ep]["request_queue"]))

# Make a request with load balancing
def balanced_request(messages):
    endpoint_id = select_endpoint()
    return claude_client.chat(
        messages=messages,
        endpoint_id=endpoint_id,
        request_id=f"req-{random.randint(1000, 9999)}"
    )

# Example usage
response = balanced_request([{"role": "user", "content": "Hello, API!"}])
```

### Routing Requests Based on Priority

```python
# Create tiered endpoints with different concurrency settings
high_priority = claude_client.create_endpoint(
    api_key="high-priority-key",
    max_concurrent_requests=10,  # Higher concurrency for important requests
    queue_size=50,
    max_retries=5
)

medium_priority = claude_client.create_endpoint(
    api_key="medium-priority-key",
    max_concurrent_requests=5,
    queue_size=20,
    max_retries=3
)

low_priority = claude_client.create_endpoint(
    api_key="low-priority-key",
    max_concurrent_requests=2,
    queue_size=10,
    max_retries=2
)

# Function to route requests based on priority
def route_request(messages, priority="medium"):
    if priority == "high":
        endpoint_id = high_priority
    elif priority == "medium":
        endpoint_id = medium_priority
    else:
        endpoint_id = low_priority
        
    return claude_client.chat(
        messages=messages,
        endpoint_id=endpoint_id,
        request_id=f"pri-{priority}-{int(time.time())}"
    )

# Example usage
high_priority_response = route_request(
    [{"role": "user", "content": "Urgent question: How to handle a system outage?"}],
    priority="high"
)

low_priority_response = route_request(
    [{"role": "user", "content": "What's the weather like today?"}],
    priority="low"
)
```

## Technical Implementation Details

### Queue System

1. Each endpoint has its own request queue
2. When an endpoint reaches its `max_concurrent_requests` limit, new requests enter the queue
3. A background thread processes the queue as resources become available
4. Queued requests include all necessary context to complete them when dequeued

### Backoff Mechanism

1. When rate limited, each endpoint applies exponential backoff
2. Starting with `initial_retry_delay`, each retry increases by `backoff_factor`
3. Respects `Retry-After` headers from API responses when available
4. Caps retry delays at `max_retry_delay` to prevent excessive waits

### Request Tracking

1. Request IDs can be provided or auto-generated
2. Auto-generated IDs follow the format: `req_{timestamp}_{hash}`
3. Request IDs are included in API request headers when possible
4. Used for associating requests with their responses and debugging

### Statistics Collection

1. Each request updates the endpoint's counters
2. Token usage is extracted from API responses when available
3. Statistics can be reset per endpoint or globally
4. Stats are maintained separately for each endpoint for precise monitoring

## API-Specific Notes

### Claude (Anthropic)

- API key is passed via `x-api-key` header
- Token usage is tracked from response metadata
- Supports automatic retry on rate limiting (429 responses)

### OpenAI

- Supports all OpenAI API methods (chat, embedding, moderation, etc.)
- API key is passed via `Authorization: Bearer {key}` header
- Handles OpenAI-specific rate limiting responses

### Gemini

- Supports both text and multimodal inputs
- API key is passed via URL parameter and/or header
- Tracks token usage when available in response metadata

### Groq

- Compatible with OpenAI-style API interface
- API key is passed via `Authorization: Bearer {key}` header
- Handles Groq-specific rate limiting patterns

## Best Practices

1. **API Key Management**:
   - Use different API keys for different purposes or priority levels
   - Monitor usage per endpoint to identify issues with specific keys
   - Rotate keys periodically following security best practices

2. **Configuration Optimization**:
   - Adjust `max_concurrent_requests` based on API provider rate limits
   - Set appropriate `queue_size` based on application traffic patterns
   - Configure backoff parameters based on API provider recommendations

3. **Request Tracking**:
   - Use meaningful request ID prefixes for easier debugging
   - Include request IDs in logs and metrics
   - Correlate request IDs with application-level operations

4. **Error Handling**:
   - Monitor failed requests ratio per endpoint
   - Implement circuit breakers for consistently failing endpoints
   - Add alerts for queues approaching capacity

## Conclusion

The enhanced API multiplexing system provides a robust foundation for applications that need to:
- Manage and optimize usage across multiple API keys
- Track detailed usage metrics per endpoint
- Handle rate limiting and backoff gracefully
- Ensure reliable operation with queuing and prioritization

All API backends (Claude, OpenAI, Gemini, and Groq) now implement this consistent interface, making it easy to use multiple providers or switch between them as needed.