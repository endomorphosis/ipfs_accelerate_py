# Semantic Cache Integration Guide for API Backends

This comprehensive guide explains how to integrate semantic caching with any API backend in the IPFS Accelerate Python Framework.

## What is Semantic Caching?

Unlike traditional caching that relies on exact string matching, semantic caching uses vector embeddings to detect when queries are semantically similar but phrased differently. This allows for much higher cache hit rates, especially for natural language queries.

## Benefits of Semantic Caching

1. **Higher Cache Hit Rates**: Cache hits even when queries are rephrased slightly
2. **Reduced API Costs**: Fewer API calls means lower costs
3. **Improved Response Times**: Cache hits return responses instantly
4. **Lower Server Load**: Reduces load on API providers
5. **Better User Experience**: Faster and more consistent responses

## Supported API Backends

This framework currently includes semantic cache implementations for:

- Gemini API
- Claude API (Anthropic)
- OpenAI API
- Groq API (coming soon)

## Integration Process Overview

To integrate semantic caching with any API backend:

1. Create a cache implementation that supports your API's message format
2. Add embedding functionality (or use the fallback hash-based pseudo embeddings)
3. Create a client wrapper that integrates with your existing client
4. Update your API endpoints to use the cached client

## Step 1: Create Basic Cache Implementation

Each API backend requires a specific implementation to handle its unique message format. However, the core cache functionality is shared:

```python
class ApiSemanticCache:
    """Cache implementation customized for your API's format."""
    
    def __init__(
        self,
        embedding_model=None,
        similarity_threshold=0.85,
        max_cache_size=1000,
        ttl=3600,
        use_lru=True
    ):
        # Initialize cache storage and configuration
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache = OrderedDict()  # Main cache storage
        self.lock = threading.RLock()  # Thread safety
        
    def _generate_embedding(self, query):
        """Convert query to embedding vector."""
        # Custom implementation for your API format
        
    def get(self, query, metadata=None):
        """Get a cached response for a semantically similar query."""
        # Implementation for finding similar entries
        
    def put(self, query, response, metadata=None):
        """Add a query-response pair to the cache."""
        # Implementation for storing entries
```

## Step 2: Create API-Specific Client Wrapper

After implementing the cache, create a wrapper for your API client:

```python
class SemanticCacheApiClient:
    """A wrapper adding semantic caching to your API client."""
    
    def __init__(
        self,
        base_client,
        embedding_model=None,
        similarity_threshold=0.85,
        max_cache_size=1000,
        ttl=3600,
        cache_enabled=True
    ):
        self.base_client = base_client
        self.cache_enabled = cache_enabled
        
        # Initialize the cache
        self.cache = ApiSemanticCache(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size,
            ttl=ttl
        )
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_similarity": 0.0,
        }
        self.stats_lock = threading.Lock()
    
    async def generate_content(self, query, **kwargs):
        """Main method that integrates caching."""
        # Update stats
        with self.stats_lock:
            self.stats["total_requests"] += 1
        
        # Skip cache for non-deterministic requests
        if not self.cache_enabled or kwargs.get('temperature', 0) > 0:
            return await self._direct_api_call(query, **kwargs)
        
        # Try cache first
        cached_response, similarity, _ = self.cache.get(query)
        
        if cached_response is not None:
            # Cache hit
            with self.stats_lock:
                self.stats["cache_hits"] += 1
            return cached_response
        
        # Cache miss - call the API directly
        with self.stats_lock:
            self.stats["cache_misses"] += 1
            
        response = await self._direct_api_call(query, **kwargs)
        
        # Store in cache
        self.cache.put(query, response)
        
        return response
    
    async def _direct_api_call(self, query, **kwargs):
        """Direct call to the API without caching."""
        # Implementation specific to your API
```

## Step 3: Integration with API Endpoint

Update your API endpoint to use the cached client:

```python
def create_client(api_key=None, **kwargs):
    """Factory function to create a client with caching."""
    # Create base client
    base_client = YourApiClient(api_key)
    
    # Check if caching is enabled
    use_cache = kwargs.get('use_semantic_cache', True)
    if not use_cache:
        return base_client
    
    # Create cached client wrapper
    return SemanticCacheApiClient(
        base_client=base_client,
        similarity_threshold=kwargs.get('similarity_threshold', 0.85),
        max_cache_size=kwargs.get('cache_size', 1000),
        ttl=kwargs.get('cache_ttl', 3600)
    )
```

## Cache Configuration Best Practices

Tune these parameters based on your specific use case:

| Parameter | Description | Recommended Range |
|-----------|-------------|------------------|
| `similarity_threshold` | Minimum similarity for cache hit | 0.75 - 0.95 |
| `max_cache_size` | Maximum entries in cache | 100 - 10000 |
| `ttl` | Time-to-live in seconds | 300 - 86400 |

### Use Case Specific Settings

| Use Case | Similarity Threshold | Cache Size | TTL |
|----------|----------------------|------------|-----|
| High Accuracy | 0.90 - 0.95 | 500 | 1 hour |
| Balanced | 0.85 - 0.90 | 1000 | 6 hours |
| High Hit Rate | 0.80 - 0.85 | 2000 | 24 hours |
| Development | 0.75 - 0.80 | 100 | 1 hour |

## Performance Considerations

1. **Memory Usage**: Embeddings can consume significant memory with large caches
2. **Embedding Generation**: Computing embeddings adds overhead to requests
3. **Thread Safety**: All implementations are thread-safe for concurrent use
4. **API Costs**: Using embedding APIs incurs additional costs
5. **Fallback Mechanism**: Hash-based pseudo embeddings provide a fallback when embedding APIs are unavailable

## Example Usage

Here's how to use a cached API client:

```python
import asyncio
from your_api_module import create_client

async def main():
    # Create a client with semantic caching
    client = create_client(
        api_key="your-api-key",
        use_semantic_cache=True,
        similarity_threshold=0.85,
        cache_size=1000,
        cache_ttl=3600
    )
    
    # First request - cache miss
    response1 = await client.generate_content("What is the capital of France?")
    
    # Second request - semantic similarity, likely cache hit
    response2 = await client.generate_content("Can you tell me France's capital city?")
    
    # Print cache statistics
    print(client.get_cache_stats())

asyncio.run(main())
```

## Advanced Cache Control

Additional methods available for cache control:

```python
# Clear the cache
client.clear_cache()

# Enable or disable cache
client.set_cache_enabled(False)

# Get cache statistics
stats = client.get_cache_stats()
```

## Implementing Embeddings

For optimal performance, use API-specific embedding endpoints:

1. **OpenAI Embeddings**: `text-embedding-3-small` or `text-embedding-3-large`
2. **Claude Embeddings**: `claude-3-sonnet-20240229-v1:0` 
3. **Gemini Embeddings**: `models/embedding-001`
4. **Custom Embeddings**: SentenceTransformers or other local models

## Complete Implementation Examples

For complete implementation examples, see:

1. `examples/semantic_cache.py` - Gemini implementation
2. `examples/claude_cache/claude_semantic_cache.py` - Claude implementation
3. `examples/openai_semantic_cache.py` - OpenAI implementation

## Monitoring and Metrics

The cache implementation provides comprehensive metrics:

- **Total Requests**: Number of requests processed
- **Cache Hits**: Number of successful cache retrievals
- **Cache Misses**: Number of cache misses requiring API calls
- **Average Similarity**: Average similarity score for cache queries
- **Token Savings**: Estimated number of tokens saved
- **Cost Savings**: Estimated cost savings based on provider pricing
- **Hit Rate**: Percentage of requests served from cache

## Troubleshooting

Common issues and solutions:

1. **Low Cache Hit Rate**: Try lowering the similarity threshold
2. **High Memory Usage**: Reduce max cache size or TTL
3. **Embedding Errors**: Check API key permissions or switch to fallback embeddings
4. **Thread Safety Issues**: Ensure all cache access is protected by the lock
5. **Stale Results**: Reduce TTL or implement cache invalidation

## Next Steps

After implementing semantic caching, consider these enhancements:

1. **Persistent Cache**: Save cache to disk between sessions
2. **Cache Warming**: Pre-populate cache with common queries
3. **Query Optimization**: Preprocess queries for better embedding similarity
4. **Custom Embedding Models**: Train domain-specific embeddings
5. **Distributed Cache**: Share cache across multiple instances