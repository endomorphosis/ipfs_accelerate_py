# Advanced API Features Configuration Guide

This document explains how to configure and customize the advanced features in the API backends of the IPFS Accelerate Python framework.

## Priority Queue Configuration

The priority queue system allows critical requests to be processed before less important ones:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create client
client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Priority levels (lower number = higher priority)
client.PRIORITY_HIGH = 0    # Critical requests
client.PRIORITY_NORMAL = 1  # Standard requests
client.PRIORITY_LOW = 2     # Background tasks

# Queue settings
client.queue_size = 100     # Maximum queue capacity
client.max_concurrent_requests = 5  # Concurrency limit
```

## Circuit Breaker Configuration

The circuit breaker pattern prevents overwhelming failing services:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create client
client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Circuit breaker settings
client.failure_threshold = 5    # Failures before opening circuit
client.reset_timeout = 30       # Seconds to wait before trying half-open
```

## Monitoring and Reporting Configuration

The enhanced monitoring system provides detailed insights into API performance:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create client
client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Enable/disable metrics collection
client.collect_metrics = True

# Reset statistics
client.reset_stats()

# Generate a report
report = client.generate_report(include_details=True)
```

## Request Batching Configuration

The request batching system automatically optimizes supported operations:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create client
client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Batching settings
client.batching_enabled = True  # Enable/disable batching
client.max_batch_size = 10      # Maximum items per batch
client.batch_timeout = 0.5      # Maximum wait time (seconds)

# Configure supported models
client.embedding_models = ["text-embedding-ada-002", "text-embedding-3-small"]
client.completion_models = []  # Models supporting batched completions
client.supported_batch_models = client.embedding_models + client.completion_models
```

## API Key Multiplexing Configuration

The API key multiplexing system distributes requests across multiple API keys:

```python
from ipfs_accelerate_py.api_key_multiplexing import ApiKeyMultiplexer

# Create multiplexer
multiplexer = ApiKeyMultiplexer()

# Add multiple API keys
multiplexer.add_openai_key("key1", "sk-api-key-1", max_concurrent=5)
multiplexer.add_openai_key("key2", "sk-api-key-2", max_concurrent=5)
multiplexer.add_openai_key("key3", "sk-api-key-3", max_concurrent=5)

# Get client with different strategies
client_rr = multiplexer.get_openai_client(strategy="round-robin")
client_ll = multiplexer.get_openai_client(strategy="least-loaded")
```

## Retry and Backoff Configuration

The retry and backoff system handles transient errors:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create client
client = openai_api(resources={}, metadata={"openai_api_key": "your-key"})

# Retry settings
client.max_retries = 5        # Maximum retry attempts
client.initial_retry_delay = 1  # Initial delay in seconds
client.backoff_factor = 2     # Multiplicative factor
client.max_retry_delay = 16   # Maximum delay cap in seconds
```

## Advanced Configuration Example

Combining multiple advanced features:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create client with comprehensive configuration
client = openai_api(
    resources={},
    metadata={
        "openai_api_key": "your-api-key",
    }
)

# Priority queue settings
client.PRIORITY_HIGH = 0
client.PRIORITY_NORMAL = 1
client.PRIORITY_LOW = 2
client.queue_size = 200
client.max_concurrent_requests = 10

# Circuit breaker settings
client.circuit_state = "CLOSED"
client.failure_threshold = 8
client.reset_timeout = 60

# Retry and backoff settings
client.max_retries = 5
client.initial_retry_delay = 1
client.backoff_factor = 2
client.max_retry_delay = 32

# Batching settings
client.batching_enabled = True
client.max_batch_size = 16
client.batch_timeout = 0.2
client.embedding_models = ["text-embedding-ada-002", "text-embedding-3-small"]
client.supported_batch_models = client.embedding_models

# Monitoring settings
client.collect_metrics = True
```

## Environment Variables Configuration

You can also configure API credentials through environment variables:

```bash
# OpenAI API keys
export OPENAI_API_KEY="your-primary-key"
export OPENAI_API_KEY_1="your-secondary-key-1"
export OPENAI_API_KEY_2="your-secondary-key-2"

# Claude API keys
export ANTHROPIC_API_KEY="your-primary-key"
export ANTHROPIC_API_KEY_1="your-secondary-key-1"

# Groq API keys
export GROQ_API_KEY="your-primary-key"
export GROQ_API_KEY_1="your-secondary-key-1"

# Gemini API keys
export GOOGLE_API_KEY="your-primary-key"
export GOOGLE_API_KEY_1="your-secondary-key-1"
```

Or use a `.env` file in your project directory:

```
OPENAI_API_KEY=your-primary-key
OPENAI_API_KEY_1=your-secondary-key-1
OPENAI_API_KEY_2=your-secondary-key-2
ANTHROPIC_API_KEY=your-anthropic-key
GROQ_API_KEY=your-groq-key
```

## Configuration Best Practices

1. **Priority Levels**: Use `PRIORITY_HIGH` sparingly for truly critical requests to avoid starving lower priority tasks.

2. **Queue Size**: Set based on your application's memory constraints and expected request volume.

3. **Concurrency Limits**: Match to API provider's rate limits to avoid excessive 429 errors.

4. **Circuit Breaker**: Tune `failure_threshold` based on API reliability; higher for stable APIs, lower for unstable ones.

5. **Batching**: Enable for embedding and vector operations to maximize throughput.

6. **Backoff Strategy**: For APIs with strict rate limits, increase `backoff_factor` to spread retries further apart.

7. **Monitoring**: Always enable `collect_metrics` in production environments for diagnostics.

8. **API Keys**: Use multiple keys with multiplexing for high-volume applications to avoid rate limits.