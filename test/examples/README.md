# API Examples & Semantic Similarity Caching

This directory contains example implementations for working with the IPFS Accelerate API backends, including an advanced semantic similarity-based caching system.

## Advanced API Features (March 2025)

All API backends now implement these advanced features:

- **Thread-Safe Request Queue**: Manages concurrent requests with proper locking
- **Exponential Backoff**: Handles rate limits with progressive retry delays
- **Circuit Breaker Pattern**: Detects service outages and provides fast-fail
- **Request Tracking**: Generates and manages unique request IDs
- **Multiple Endpoints**: Allows different configurations for different use cases

### Example: Using the Queue and Backoff Systems

```python
from ipfs_accelerate_py.api_backends import claude

# Initialize the API with credentials
api = claude(resources={}, metadata={"anthropic_api_key": "your-key-here"})

# Configure the queue and backoff systems
api.max_concurrent_requests = 5    # Maximum concurrent requests
api.queue_size = 100               # Maximum queue size
api.max_retries = 5                # Maximum retry attempts
api.backoff_factor = 2             # Multiplier for retry delays

# Create multiple endpoints with different configurations
endpoint1 = api.create_endpoint(
    api_key="key1",
    max_concurrent_requests=3,     # Lower concurrency
    max_retries=3                  # Fewer retries
)

endpoint2 = api.create_endpoint(
    api_key="key2",
    max_concurrent_requests=10,    # Higher concurrency
    max_retries=7                  # More retries
)

# Use specific endpoint for request
response = api.chat(
    messages=[{"role": "user", "content": "Hello"}],
    endpoint_id=endpoint1,
    request_id="custom-request-id"  # Optional custom tracking ID
)
```

## Semantic Similarity Caching for the Gemini API

This directory contains an implementation of semantic similarity-based caching for the Google Gemini API. Instead of requiring exact string matches for cache hits, this implementation uses embeddings and cosine similarity to find semantically similar cached entries.

## Core Components

### SemanticCache Class

The `SemanticCache` class provides the fundamental caching infrastructure:

- **Embedding Generation**: Converts text queries into embedding vectors
- **Similarity Computation**: Uses cosine similarity to compare embeddings
- **Cache Management**: Handles LRU eviction, entry expiration, and thread safety
- **Configurable Thresholds**: Adjustable similarity thresholds and TTL

### SemanticCacheGeminiClient

A wrapper for the Gemini API client that integrates semantic caching:

- **Transparent Integration**: Wraps the existing Gemini client without API changes
- **Intelligent Caching**: Only caches deterministic responses (temperature=0)
- **Performance Tracking**: Detailed statistics on hit/miss rates and similarity scores
- **Simple Configuration**: Adjustable similarity thresholds and cache sizes

## Key Features

1. **Semantic Matching**: Matches semantically similar queries even when phrased differently
2. **Fallback Mechanism**: Works without an embedding model using hash-based representation
3. **Thread Safety**: Full thread safety with proper locking mechanisms
4. **Automatic Cleanup**: Periodic removal of expired entries
5. **Detailed Statistics**: Comprehensive cache usage statistics and similarity metrics
6. **LRU Eviction**: Least recently used entry eviction when cache is full
7. **Metadata Support**: Store and retrieve additional metadata with cache entries

## Example Use Cases

1. **Repetitive User Queries**: Efficiently handle semantically similar repeated questions
2. **API Cost Reduction**: Reduce API calls for common queries
3. **Latency Improvement**: Faster responses for semantically similar queries
4. **Load Reduction**: Decrease load on the Gemini API service
5. **Offline Fallback**: Provide responses even during API outages

## Implementation Notes

This implementation balances several considerations:

- **Efficiency**: Minimal overhead for cache lookups
- **Accuracy**: Configurable similarity threshold to prevent false cache hits
- **Simplicity**: Easy to integrate with existing code
- **Observability**: Detailed statistics and logging

## Testing

The example demonstrates cache behavior with semantically similar queries:

```python
# Example prompts with semantic similarity
prompts = [
    "What is the capital of France?",
    "Could you tell me the capital city of France?",  # Semantically similar
    "What's the capital of France?",  # Semantically similar
    "What is the population of Paris?",  # Different question
    "What is the capital of Italy?",  # Different country
    "What's France's capital city?",  # Very similar to earlier prompts
    "Paris is the capital of which country?",  # Related but different structure
    "Tell me about the capital of France",  # Request for more information
]
```

With a similarity threshold of 0.85, queries that are semantically similar enough will get cache hits, providing faster responses and reducing API calls.

## Integration

To use this caching mechanism with your existing Gemini client:

```python
from semantic_cache import SemanticCacheGeminiClient
from your_module import GeminiClient

# Create base client
base_client = GeminiClient()

# Create semantic cache wrapper
cached_client = SemanticCacheGeminiClient(
    base_client=base_client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600
)

# Use the cached client exactly like the original
response = await cached_client.generate_content("What is the capital of France?")
```