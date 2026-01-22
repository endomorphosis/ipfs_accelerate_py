# Claude API Semantic Caching Implementation

This directory contains an advanced implementation of semantic similarity-based caching for the Claude API. It extends the base semantic caching system to work with Claude's specific API structure, including the Messages API format.

## Key Components

### ClaudeSemanticCache Class

A specialized cache implementation for Claude's message format:

- **Message Format Support**: Handles Claude's specific message structure
- **Content Block Processing**: Properly extracts text from content blocks
- **Embedding Generation**: Converts messages to embeddings for similarity comparison
- **Adaptive Interface**: Works with various embedding model interfaces

### SemanticCacheClaudeClient

A wrapper for the Claude API client that implements semantic caching:

- **Anthropic Client Compatibility**: Works with the official Anthropic Python SDK
- **Messages API Support**: Full support for the Messages API format
- **Fallback Mechanisms**: Includes fallback embedding generation when API is unavailable
- **Token Savings Tracking**: Estimates token and cost savings from cache hits

## Features Specific to Claude's API

1. **Message Format Handling**: Properly processes Claude's nested message structure
2. **Content Block Extraction**: Extracts text from various content block types
3. **Claude Embedding Support**: Uses Claude's embedding API when available
4. **Adaptive Client Detection**: Works with different Claude client implementations 
5. **Stream Handling**: Proper handling of streaming responses (which bypass the cache)
6. **Token Usage Estimation**: Tracks estimated token savings from cache hits

## Cache Performance Benefits

The implementation is designed to maximize the benefits of Claude's API:

- **Reduced API Costs**: Fewer API calls means lower costs
- **Faster Response Times**: Cache hits provide near-instant responses
- **Token Savings**: Detailed tracking of token savings for cost analysis
- **API Quota Management**: Helps stay within API rate limits

## Usage Guide

```python
import anthropic
from claude_semantic_cache import SemanticCacheClaudeClient

# Create a base client
client = anthropic.Anthropic(api_key="your_api_key")

# Wrap with semantic cache
cached_client = SemanticCacheClaudeClient(
    base_client=client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600  # 1 hour
)

# Use like the regular client
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

# For deterministic responses that can be cached
response = await cached_client.chat(
    messages=messages, 
    temperature=0.0  # Important: zero temperature for caching
)

# Get cache statistics
stats = cached_client.get_cache_stats()
print(f"Cache hit rate: {stats['cache_hits'] / stats['total_requests']:.1%}")
print(f"Estimated token savings: {stats['token_savings']}")
```

## Implementation Notes

1. **Deterministic Generation**: Only caches responses with temperature=0 for consistency
2. **Thread Safety**: Fully thread-safe with proper locking mechanisms
3. **Memory Management**: LRU eviction policy to control memory usage
4. **Fallback System**: Works even when embedding API is unavailable
5. **Compatibility**: Works with both async and sync Claude clients

## Configuration Recommendations

| Use Case | Similarity Threshold | Cache Size | TTL |
|----------|----------------------|------------|-----|
| High Precision | 0.90 - 0.95 | 500 | 1 hour |
| Balanced | 0.85 - 0.90 | 1000 | 6 hours |
| High Recall | 0.80 - 0.85 | 2000 | 24 hours |

## Advanced Usage

### Using Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer

# Load a sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Use with Claude cache
cached_client = SemanticCacheClaudeClient(
    base_client=client,
    embedding_model=embedding_model,
    similarity_threshold=0.85
)
```

### Cache Control

```python
# Clear the cache
cached_client.clear_cache()

# Disable caching temporarily
cached_client.set_cache_enabled(False)

# Re-enable caching
cached_client.set_cache_enabled(True)
```