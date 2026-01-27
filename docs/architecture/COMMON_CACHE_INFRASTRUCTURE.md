# Common Cache Infrastructure

This document describes the common caching infrastructure for API calls in ipfs_accelerate_py. The infrastructure provides a unified way to cache responses from various APIs (LLM APIs, HuggingFace Hub, Docker, GitHub, etc.) with support for **content-addressed CID-based lookups**, TTL-based expiration, content validation, disk persistence, and optional P2P cache sharing.

## Overview

The common cache infrastructure enables:

1. **Content-addressed caching** - Cache keys are CIDs (Content Identifiers) computed from the query parameters
2. **Fast lookups** - O(1) lookups by hashing the query to get the CID directly
3. **CID index** - Maintains an in-memory index for prefix-based searches and operation filtering
4. **Unified caching pattern** - All APIs use the same base caching mechanism
5. **Content-addressed validation** - Detect stale cache based on content changes, not just TTL
6. **Flexible TTL policies** - Different operations can have different cache lifetimes
7. **Disk persistence** - Cache survives process restarts
8. **P2P sharing** - Optional cache distribution via libp2p (like GitHub cache)
9. **Thread safety** - Safe for concurrent access
10. **Statistics tracking** - Monitor cache performance

## Content-Addressed Caching with CID

The key innovation in this cache infrastructure is the use of **Content Identifiers (CID)** with multiformats:

### How It Works

1. **Query to CID**: When you make a query, the cache computes a CID from the query parameters
2. **Direct Lookup**: The CID is used as the cache key, enabling O(1) lookup
3. **Multiformats**: Uses the multiformats library to create proper CIDs (base32, version 1)
4. **Fallback**: If multiformats is unavailable, uses SHA256 hashing with a prefix

### Example

```python
# Query parameters
query = {
    "operation": "completion",
    "prompt": "Explain quantum computing",
    "model": "gpt-4",
    "temperature": 0.0
}

# Compute CID from query
cid = cache._compute_cid(json.dumps(query, sort_keys=True))
# Result: "bafkreih4kovkbjv6xjmklyz7d2n6l6ydbqz7qzgk5rqh5c6v7kp2xqnqje"

# Cache lookup is now a simple hash table lookup
cached_result = cache._cache.get(cid)
```

### Benefits

- **Fast lookups**: O(1) instead of string matching
- **Deterministic**: Same query always produces same CID
- **Content-addressed**: The CID represents the query itself
- **P2P friendly**: CIDs can be used for distributed cache lookups
- **Collision-resistant**: SHA256 provides strong collision resistance

## Architecture

### Base Classes

```
BaseAPICache (abstract)
├── Uses CID-based cache keys
├── Maintains CID index for fast lookups
├── LLMAPICache (OpenAI, Claude, Gemini, Groq, etc.)
├── HuggingFaceHubCache (Model/dataset metadata)
├── DockerAPICache (Image/container info)
└── GitHubAPICache (Already exists, can be migrated)
```

### Key Components

1. **`BaseAPICache`** - Abstract base class with common caching logic and CID generation
2. **`CacheEntry`** - Data class representing a cached item with TTL and validation
3. **`CIDCacheIndex`** - Index for fast CID-based lookups, prefix search, and operation filtering
4. **Cache Registry** - Global registry for managing multiple cache instances
5. **Cache Adapters** - Specific implementations for each API type

## CID-Based Operations

### Lookup by CID Prefix

Find cache entries by CID prefix (useful for debugging or exploring related entries):

```python
cache = get_global_llm_cache()

# Find all entries starting with a CID prefix
entries = cache.find_by_cid_prefix("bafkre")
print(f"Found {len(entries)} entries with prefix 'bafkre'")
```

### Lookup by Operation

Find all cached entries for a specific operation:

```python
# Find all cached completions
completions = cache.find_by_operation("completion")
print(f"Found {len(completions)} cached completions")

# Find all cached model info from HuggingFace
hf_cache = get_global_hf_hub_cache()
model_infos = hf_cache.find_by_operation("model_info")
```

### CID Index Statistics

Get statistics about the CID index:

```python
stats = cache.get_stats()
print(f"CID Index Stats:")
print(f"  Total CIDs: {stats['cid_index']['total_cids']}")
print(f"  Operations: {stats['cid_index']['operations']}")
print(f"  Operation counts: {stats['cid_index']['operation_counts']}")
```

## Usage

### LLM API Cache

Cache responses from OpenAI, Claude, Gemini, Groq, and other LLM APIs:

```python
from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache

# Get the global LLM cache instance
cache = get_global_llm_cache()

# Check cache before making API call
prompt = "Explain quantum computing"
model = "gpt-4"
temperature = 0.0

cached_response = cache.get_completion(
    prompt=prompt,
    model=model,
    temperature=temperature
)

if cached_response:
    print("Cache hit! Using cached response")
    response = cached_response
else:
    print("Cache miss, calling API...")
    # Make actual API call
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    # Cache the response
    cache.cache_completion(
        prompt=prompt,
        response=response,
        model=model,
        temperature=temperature
    )
```

### HuggingFace Hub Cache

Cache model metadata and file listings:

```python
from ipfs_accelerate_py.common.hf_hub_cache import get_global_hf_hub_cache

cache = get_global_hf_hub_cache()

# Check cache for model info
model_id = "meta-llama/Llama-2-7b-hf"
cached_info = cache.get("model_info", model=model_id)

if cached_info:
    model_info = cached_info
else:
    # Fetch from HuggingFace Hub API
    from huggingface_hub import hf_api
    api = hf_api.HfApi()
    model_info = api.model_info(model_id)
    
    # Cache for 1 hour
    cache.put("model_info", model_info, model=model_id)

# Check model files
cached_files = cache.get("model_files", model=model_id)

if not cached_files:
    files = api.list_repo_files(model_id)
    cache.put("model_files", files, model=model_id)
```

### Docker API Cache

Cache Docker image and container information:

```python
from ipfs_accelerate_py.common.docker_cache import get_global_docker_cache
import docker

cache = get_global_docker_cache()
client = docker.from_env()

# Check cache for image info
image_name = "ubuntu:22.04"
cached_image = cache.get("image_inspect", image=image_name)

if cached_image:
    image_data = cached_image
else:
    image = client.images.get(image_name)
    image_data = image.attrs
    
    # Cache for 30 minutes
    cache.put("image_inspect", image_data, image=image_name)

# Container status (shorter TTL)
container_id = "abc123"
cached_status = cache.get("container_inspect", container=container_id)

if not cached_status:
    container = client.containers.get(container_id)
    status_data = container.attrs
    cache.put("container_inspect", status_data, container=container_id, ttl=30)
```

### Unified Cache Management

Manage all caches together:

```python
from ipfs_accelerate_py.common.base_cache import (
    get_all_caches,
    shutdown_all_caches
)

# Get stats from all caches
caches = get_all_caches()
for cache_name, cache in caches.items():
    stats = cache.get_stats()
    print(f"{cache_name}:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  API calls saved: {stats['api_calls_saved']}")
    print(f"  Cache size: {stats['cache_size']}/{stats['max_cache_size']}")

# Gracefully shutdown all caches on exit
import atexit
atexit.register(shutdown_all_caches)
```

## Configuration

### Per-Cache Configuration

Each cache can be configured individually:

```python
from ipfs_accelerate_py.common.llm_cache import configure_llm_cache

cache = configure_llm_cache(
    cache_dir="/custom/cache/dir",
    default_ttl=1800,  # 30 minutes default
    max_cache_size=2000,
    enable_persistence=True,
    enable_p2p=False  # Disable P2P for now
)
```

### Environment Variables

Configure cache behavior via environment variables:

```bash
# Global cache directory
export IPFS_ACCELERATE_CACHE_DIR=/var/cache/ipfs_accelerate

# P2P configuration (if enabled)
export CACHE_DISCOVERY_REFRESH_INTERVAL=120
export CACHE_MIN_CONNECTED_PEERS=1
```

## Default TTL Values

Each cache adapter defines sensible defaults for different operations:

### LLM API Cache
- `completion`: 3600s (1 hour) - deterministic responses
- `chat_completion`: 1800s (30 min) - conversations
- `embedding`: 86400s (24 hours) - embeddings are deterministic
- `model_list`: 3600s (1 hour)

### HuggingFace Hub Cache
- `model_info`: 3600s (1 hour)
- `model_files`: 1800s (30 min)
- `dataset_info`: 3600s (1 hour)
- `search_models`: 600s (10 min)
- `repo_commits`: 300s (5 min)
- `download_url`: 86400s (24 hours)

### Docker API Cache
- `image_inspect`: 1800s (30 min)
- `image_history`: 3600s (1 hour)
- `container_inspect`: 30s - state changes frequently
- `container_list`: 30s
- `volume_inspect`: 300s (5 min)
- `registry_tags`: 1800s (30 min)

## Content Validation

The cache uses content-addressed validation to detect stale entries:

1. **Validation fields** - Extract key fields from API response (e.g., `lastModified`, `sha`, `State`)
2. **Content hash** - Compute SHA256 hash (or CID if multiformats available) of validation fields
3. **Staleness check** - Compare stored hash with current hash to detect changes

Example for HuggingFace model:

```python
# Validation fields extracted from response
validation_fields = {
    "sha": "abc123...",  # Git commit hash
    "lastModified": "2025-01-15T10:00:00Z",
    "downloads": 150000,
    "likes": 500
}

# Content hash computed and stored with cache entry
content_hash = compute_hash(validation_fields)

# On subsequent access, if validation fields change, cache is invalidated
# even if TTL hasn't expired
```

## Cache Statistics

Monitor cache performance:

```python
cache = get_global_llm_cache()
stats = cache.get_stats()

print(f"Cache name: {stats['cache_name']}")
print(f"Total requests: {stats['total_requests']}")
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"API calls saved: {stats['api_calls_saved']}")
print(f"API calls made: {stats['api_calls_made']}")
print(f"Expirations: {stats['expirations']}")
print(f"Evictions: {stats['evictions']}")
```

## Cache Invalidation

### Manual Invalidation

Invalidate specific entries or patterns:

```python
# Invalidate a specific entry
cache.invalidate("model_info", model="meta-llama/Llama-2-7b-hf")

# Invalidate all entries matching a pattern
count = cache.invalidate_pattern("model_info")
print(f"Invalidated {count} entries")

# Clear entire cache
cache.clear()
```

### Automatic Invalidation

Cache entries are automatically invalidated when:
- TTL expires
- Content validation fails (for operations that support it)
- Cache size limit is reached (LRU eviction)

## Best Practices

### 1. Use Appropriate TTLs

Match TTLs to data volatility:
- **Static data** (embeddings, image history): Long TTL (hours/days)
- **Semi-static data** (model metadata, image info): Medium TTL (minutes/hour)
- **Dynamic data** (container status, search results): Short TTL (seconds/minutes)

### 2. Enable Content Validation

For APIs that support it, content validation prevents serving stale data:

```python
class MyAPICache(BaseAPICache):
    def extract_validation_fields(self, operation, data):
        if operation == "get_resource":
            return {
                "version": data.get("version"),
                "updated_at": data.get("updated_at")
            }
        return None
```

### 3. Monitor Cache Performance

Regularly check cache statistics to ensure good hit rates:

```python
stats = cache.get_stats()
if stats['hit_rate'] < 0.3:  # Less than 30%
    logger.warning("Low cache hit rate, consider adjusting TTLs")
```

### 4. Handle Cache Misses Gracefully

Always handle the case where cache returns None:

```python
response = cache.get("operation", param="value")
if response is None:
    response = make_api_call()
    cache.put("operation", response, param="value")
```

### 5. Clean Up on Exit

Ensure caches are properly saved on shutdown:

```python
import atexit
from ipfs_accelerate_py.common.base_cache import shutdown_all_caches

atexit.register(shutdown_all_caches)
```

## Advanced Features

### P2P Cache Sharing

Enable P2P sharing to distribute cache across multiple nodes:

```python
cache = configure_llm_cache(
    enable_p2p=True,
    p2p_listen_port=9100,
    github_repo="owner/repo"  # For peer discovery
)
```

**Note**: P2P features require:
- `libp2p` installed
- Network connectivity between peers
- Optional: GitHub repository for peer registry

### Custom Cache Adapters

Create custom cache adapters for new APIs:

```python
from ipfs_accelerate_py.common.base_cache import BaseAPICache

class MyAPICache(BaseAPICache):
    DEFAULT_TTLS = {
        "get_data": 600,
        "list_items": 300
    }
    
    def get_cache_namespace(self):
        return "my_api"
    
    def extract_validation_fields(self, operation, data):
        if operation == "get_data":
            return {
                "version": data.get("version"),
                "checksum": data.get("checksum")
            }
        return None
    
    def get_default_ttl_for_operation(self, operation):
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)

# Use your custom cache
cache = MyAPICache()
from ipfs_accelerate_py.common.base_cache import register_cache
register_cache("my_api", cache)
```

## Migration from Existing Caches

If you have existing cache implementations, migrate them to use the common infrastructure:

### Before (Custom Implementation)

```python
class MyCache:
    def __init__(self):
        self._cache = {}
    
    def get(self, key):
        return self._cache.get(key)
    
    def put(self, key, value):
        self._cache[key] = value
```

### After (Using BaseAPICache)

```python
from ipfs_accelerate_py.common.base_cache import BaseAPICache

class MyCache(BaseAPICache):
    def get_cache_namespace(self):
        return "my_cache"
    
    def extract_validation_fields(self, operation, data):
        return None  # Or implement validation
```

Benefits:
- ✅ TTL-based expiration
- ✅ Disk persistence
- ✅ Thread safety
- ✅ Statistics tracking
- ✅ P2P sharing ready
- ✅ Content validation

## Troubleshooting

### Cache Not Persisting

Check that the cache directory is writable:

```python
import os
cache_dir = os.path.expanduser("~/.cache/llm_api")
assert os.access(cache_dir, os.W_OK), "Cache directory not writable"
```

### Low Hit Rate

1. Check if operations are using consistent parameters
2. Verify TTLs aren't too short
3. Monitor cache evictions (may need larger `max_cache_size`)

### High Memory Usage

Reduce cache size:

```python
cache = configure_llm_cache(max_cache_size=500)  # Default is 1000
```

Or disable persistence to avoid disk usage:

```python
cache = configure_llm_cache(enable_persistence=False)
```

## Performance Impact

Expected improvements with caching:

| API Type | Without Cache | With Cache | Speedup |
|----------|---------------|------------|---------|
| LLM Completion | 1-5s | <0.01s | **100-500x** |
| HF Model Info | 0.2-1s | <0.01s | **20-100x** |
| Docker Inspect | 0.05-0.2s | <0.01s | **5-20x** |
| GitHub API | 0.5-2s | <0.01s | **50-200x** |

**API Cost Savings**: With a 70% hit rate, caching reduces API calls by 70%, saving:
- LLM APIs: $0.002-0.10 per 1K tokens × 70% = significant cost reduction
- HuggingFace Hub: Rate limit relief (1000 req/hour → effectively 3000+)
- GitHub API: Rate limit relief (5000 req/hour → effectively 15000+)

## Related Documentation

- [GitHub API Cache](../GITHUB_API_CACHE.md) - GitHub-specific caching
- [CLI Retry and Cache](../CLI_RETRY_AND_CACHE.md) - CLI wrapper caching
- [Distributed Cache](../DISTRIBUTED_CACHE.md) - P2P cache sharing
- [Caching Infrastructure Alignment](../CACHING_INFRASTRUCTURE_ALIGNMENT.md) - Cross-repo caching

## Future Enhancements

Planned improvements:

1. **Semantic caching** - Cache similar prompts with vector similarity
2. **Cache warming** - Pre-fetch commonly accessed entries
3. **Smart invalidation** - Auto-invalidate related entries
4. **Compression** - Compress large cached responses
5. **Redis backend** - Optional Redis/Memcached support for distributed deployments
6. **Cache analytics** - Detailed performance metrics and dashboards
