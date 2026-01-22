# Semantic Cache Integration Guide - All API Backends

This comprehensive guide provides detailed instructions for integrating semantic caching with all supported API backends in the IPFS Accelerate Python Framework.

## Overview

Semantic caching significantly improves performance and reduces API costs by:

1. Caching responses based on semantic similarity rather than exact string matching
2. Automatically detecting semantically equivalent queries
3. Returning cached responses instantly for similar queries
4. Reducing API costs by minimizing redundant requests

## Supported APIs

This framework includes semantic cache implementations for:

| API Provider | File Location | Key Features |
|--------------|---------------|-------------|
| OpenAI | `examples/openai_semantic_cache.py` | All models, streaming, function calls |
| Claude (Anthropic) | `examples/claude_cache/claude_semantic_cache.py` | All models, Messages API, multi-part messages |
| Gemini (Google) | `examples/semantic_cache.py` | All models, content generation, multi-modal |
| Groq | `examples/groq_semantic_cache.py` | All models, OpenAI-compatible interface |

## Integration Process

To integrate semantic caching with an API backend:

1. Import the appropriate semantic cache implementation
2. Create a wrapper around the base client
3. Configure the cache parameters
4. Use the wrapped client as you would the normal client

### Example: OpenAI Integration

```python
import asyncio
from openai import AsyncClient
from examples.openai_semantic_cache import SemanticCacheOpenAIClient

# Create the base client
base_client = AsyncClient(api_key="your-api-key")

# Create the cached client wrapper
cached_client = SemanticCacheOpenAIClient(
    base_client=base_client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600  # Cache entries expire after 1 hour
)

# Use the cached client just like the normal client
async def example():
    response = await cached_client.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.0  # Use 0 temperature for deterministic responses
    )
    print(response)

asyncio.run(example())
```

### Example: Claude Integration

```python
import anthropic
from examples.claude_cache.claude_semantic_cache import SemanticCacheClaudeClient

# Create the base client
base_client = anthropic.Anthropic(api_key="your-api-key")

# Create the cached client wrapper
cached_client = SemanticCacheClaudeClient(
    base_client=base_client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600
)

# Use the cached client as normal
async def example():
    response = await cached_client.chat(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.0
    )
    print(response)

asyncio.run(example())
```

### Example: Gemini Integration

```python
import google.generativeai as genai
from examples.semantic_cache import SemanticCacheGeminiClient

# Configure the base client
genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-pro')

# Create the cached client wrapper
cached_client = SemanticCacheGeminiClient(
    base_client=model,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600
)

# Use the cached client as normal
async def example():
    response = await cached_client.generate_content(
        "What is the capital of France?",
        temperature=0.0
    )
    print(response)

asyncio.run(example())
```

### Example: Groq Integration

```python
import groq
from examples.groq_semantic_cache import SemanticCacheGroqClient

# Create the base client
client = groq.AsyncClient(api_key="your-api-key")

# Create the cached client wrapper
cached_client = SemanticCacheGroqClient(
    base_client=client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600
)

# Use the cached client as normal
async def example():
    response = await cached_client.create_chat_completion(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.0
    )
    print(response)

asyncio.run(example())
```

## Configuration Options

All semantic cache implementations share these common configuration parameters:

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| `similarity_threshold` | Minimum similarity score for cache hit | 0.85 | 0.75 - 0.95 |
| `max_cache_size` | Maximum number of entries in cache | 1000 | 100 - 10000 |
| `ttl` | Time-to-live in seconds | 3600 | 300 - 86400 |
| `embedding_model` | Optional custom embedding model | None | - |
| `cache_enabled` | Whether caching is enabled | True | - |

### Use Case-Specific Configurations

Different use cases may require different cache configurations:

| Use Case | Similarity Threshold | Cache Size | TTL |
|----------|----------------------|------------|-----|
| High Accuracy | 0.90 - 0.95 | 500 | 1 hour |
| Balanced | 0.85 - 0.90 | 1000 | 6 hours |
| High Hit Rate | 0.80 - 0.85 | 2000 | 24 hours |
| Development | 0.75 - 0.80 | 100 | 1 hour |

## Embedding Models

By default, each cache implementation will use:

1. The API provider's embedding API if available (OpenAI, Claude)
2. A fallback hash-based pseudo-embedding if no embedding API is available

You can provide a custom embedding model for better semantic similarity detection:

```python
from sentence_transformers import SentenceTransformer

# Load a sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create client with custom embeddings
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    embedding_model=embedding_model,
    similarity_threshold=0.85
)
```

## Cache Control Methods

All cache implementations provide these methods for controlling the cache:

```python
# Clear the cache
cached_client.clear_cache()

# Enable or disable caching
cached_client.set_cache_enabled(False)
cached_client.set_cache_enabled(True)

# Get cache statistics
stats = cached_client.get_cache_stats()
print(f"Cache hit rate: {stats['cache_hits'] / stats['total_requests']:.1%}")
print(f"Estimated token savings: {stats['token_savings']}")
```

## Performance Optimization

### Memory Usage

Control memory usage with these options:

```python
# Reduce cache size for memory-constrained environments
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    max_cache_size=500,  # Smaller cache
    ttl=1800  # Shorter TTL (30 minutes)
)
```

### When to Bypass Cache

The cache is automatically bypassed in these situations:

1. Non-deterministic generations (temperature > 0)
2. Streaming responses
3. Function calling/tool use (OpenAI)
4. When cache is explicitly disabled

You can also bypass the cache for specific requests:

```python
# Set temperature > 0 to bypass cache
response = await cached_client.create_chat_completion(
    messages=[{"role": "user", "content": "Generate a creative story"}],
    temperature=0.7  # Cache bypassed
)

# Or disable cache globally
cached_client.set_cache_enabled(False)
```

## Monitoring Performance

Get detailed performance metrics:

```python
stats = cached_client.get_cache_stats()

# Cache efficiency
hit_rate = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
print(f"Cache hit rate: {hit_rate:.1%}")
print(f"Average similarity score: {stats['avg_similarity']:.4f}")

# Cost savings (approximate)
token_savings = stats['token_savings']
cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens (adjust for your model)
cost_savings = token_savings * cost_per_1k_tokens / 1000
print(f"Tokens saved: {token_savings}")
print(f"Cost savings: ${cost_savings:.4f}")

# Cache utilization
utilization = stats['active_entries'] / stats['max_size'] * 100
print(f"Cache utilization: {utilization:.1f}% ({stats['active_entries']}/{stats['max_size']})")
```

## Implementing in Production

Recommendations for production use:

1. **Persistent Cache**: Consider implementing a persistent cache database for longer-term storage
2. **Distributed Caching**: For high-traffic applications, implement a shared cache (Redis, Memcached)
3. **Monitoring**: Track cache statistics and adjust parameters based on performance
4. **Regular Maintenance**: Schedule periodic cache cleanups for stale entries
5. **API-specific Tuning**: Adjust similarity thresholds based on each provider's embedding quality

## Integration with Complete API Backends

When integrating with your API backend implementation, follow this pattern:

```python
class MyApiBackend:
    def __init__(self, api_key=None, **kwargs):
        # API client initialization
        self.api_key = api_key or os.environ.get("API_KEY")
        self.base_url = kwargs.get("base_url", "https://api.example.com")
        
        # Cache configuration from kwargs
        self.use_semantic_cache = kwargs.get('use_semantic_cache', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.85)
        self.cache_size = kwargs.get('cache_size', 1000)
        self.cache_ttl = kwargs.get('cache_ttl', 3600)
        
        # Initialize the client
        self.client = self._create_client()
        
        # Initialize semantic cache if enabled
        if self.use_semantic_cache:
            self.cached_client = self._create_cached_client(self.client)
        else:
            self.cached_client = self.client
    
    def _create_client(self):
        # Create the base API client
        return ApiClient(api_key=self.api_key)
    
    def _create_cached_client(self, client):
        # Create the cached client wrapper
        from examples.my_api_semantic_cache import SemanticCacheMyApiClient
        return SemanticCacheMyApiClient(
            base_client=client,
            similarity_threshold=self.similarity_threshold,
            max_cache_size=self.cache_size,
            ttl=self.cache_ttl
        )
    
    async def generate_content(self, prompt, **kwargs):
        # Use the cached client for all requests
        return await self.cached_client.generate_content(prompt, **kwargs)
```

## Conclusion

Semantic caching provides significant performance improvements and cost savings across all API backends. By integrating the appropriate cache implementation for each API provider, you can create a more efficient and cost-effective application while maintaining the same API interface.

For specific details on each implementation, refer to the documentation and examples in the respective files:
- `examples/openai_semantic_cache.py`
- `examples/claude_cache/claude_semantic_cache.py`
- `examples/semantic_cache.py` (Gemini)
- `examples/groq_semantic_cache.py`