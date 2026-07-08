# API Examples & Advanced Features

This directory contains example implementations for working with the IPFS Accelerate API backends, including queue and backoff systems, semantic caching, and other advanced features.

## Advanced API Features (March 2025)

All API backends now implement these advanced features:

- **Queue and Backoff System**: Manages concurrent requests with exponential retry
- **Circuit Breaker Pattern**: Detects service outages and provides fast-fail
- **API Key Multiplexing**: Manages multiple API keys with rotation strategies
- **Semantic Caching**: Caches semantically similar requests to reduce API calls
- **Request Batching**: Combines compatible requests for improved throughput

## Example Scripts

This directory contains example scripts demonstrating various features:

- **queue_backoff_example.py**: Demonstrates queue and backoff functionality with multiple APIs
- **semantic_cache.py**: Implements semantic caching based on embedding similarity
- **groq_semantic_cache.py**: Groq-specific implementation of semantic caching
- **openai_semantic_cache.py**: OpenAI-specific implementation of semantic caching
- **claude_semantic_cache.py**: Claude-specific implementation of semantic caching

### Running the Queue and Backoff Example

```bash
# Set API credentials (replace with your actual keys)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-claude-key"
export GROQ_API_KEY="your-groq-key"
export OLLAMA_HOST="http://localhost:11434"

# Run the example script
python queue_backoff_example.py
```

The script demonstrates:
- Queue functionality with varying concurrency limits
- Exponential backoff behavior with rate limits
- Circuit breaker pattern for service outages
- Different queue sizes and their performance impact
- Request tracking with unique IDs

### Example: Using the Queue and Backoff Systems

```python
from ipfs_accelerate_py.api_backends import claude

# Initialize the API with credentials
api = claude(resources={}, metadata={"anthropic_api_key": "your-key-here"})

# Configure the queue and backoff systems
api.max_concurrent_requests = 5    # Maximum concurrent requests
api.queue_size = 100               # Maximum queue size
api.max_retries = 5                # Maximum retry attempts
api.initial_retry_delay = 1        # Starting delay in seconds
api.backoff_factor = 2             # Multiplier for retry delays
api.max_retry_delay = 60           # Maximum delay cap in seconds

# Use with custom request ID for tracking
response = api.chat(
    messages=[{"role": "user", "content": "Hello"}],
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