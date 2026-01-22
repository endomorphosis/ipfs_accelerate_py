# GitHub API Caching

This document explains the GitHub API caching feature that reduces the number of API requests and improves performance.

## Overview

The GitHub CLI wrapper now includes intelligent caching to minimize API calls to GitHub. This helps:

- **Avoid rate limiting**: GitHub has rate limits (5000 requests/hour for authenticated users)
- **Improve performance**: Cached responses are returned instantly
- **Reduce network usage**: Fewer API calls means less bandwidth
- **Persist across sessions**: Cache is saved to disk and reloaded on restart

## Features

### Automatic Caching

The cache automatically stores responses for common operations:

- `list_repos()` - Repository listings (5 min TTL)
- `get_repo_info()` - Repository details (5 min TTL)
- `list_workflow_runs()` - Workflow runs (60 sec TTL)
- `get_workflow_run()` - Workflow details (60 sec TTL)
- `list_runners()` - Runner status (30 sec TTL)

Different operations have different TTLs (Time-To-Live) based on how frequently the data changes.

### Thread-Safe

All cache operations are thread-safe using locks, making it suitable for concurrent access.

### Disk Persistence

Cache entries are automatically saved to disk (`~/.cache/github_cli/`) and restored on startup, so your cache survives restarts.

## Usage

### Basic Usage

```python
from ipfs_accelerate_py.github_cli import GitHubCLI

# Caching is enabled by default
gh = GitHubCLI()

# First call hits the API
repos = gh.list_repos(owner="endomorphosis", limit=10)  # ~1-2 seconds

# Second call uses cache
repos = gh.list_repos(owner="endomorphosis", limit=10)  # ~0.001 seconds (1000x faster!)
```

### Disable Caching

```python
# Disable caching entirely
gh = GitHubCLI(enable_cache=False)

# Or disable for specific calls
repos = gh.list_repos(owner="endomorphosis", use_cache=False)
```

### Custom Cache Configuration

```python
from ipfs_accelerate_py.github_cli import GitHubCLI, configure_cache

# Configure global cache
cache = configure_cache(
    cache_dir="/custom/cache/dir",
    default_ttl=600,  # 10 minutes
    max_cache_size=2000,  # Store up to 2000 entries
    enable_persistence=True
)

# Use custom cache instance
gh = GitHubCLI(cache=cache)
```

### Cache Statistics

```python
from ipfs_accelerate_py.github_cli import get_global_cache

cache = get_global_cache()
stats = cache.get_stats()

print(f"Total requests: {stats['total_requests']}")
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['cache_size']}/{stats['max_cache_size']}")
```

### Manual Cache Management

```python
from ipfs_accelerate_py.github_cli import get_global_cache

cache = get_global_cache()

# Invalidate specific entry
cache.invalidate("list_repos", owner="endomorphosis", limit=10)

# Invalidate all entries matching a pattern
cache.invalidate_pattern("list_repos")  # Clear all repo listings

# Clear entire cache
cache.clear()
```

## Cache Keys

Cache keys are generated from the operation name and parameters:

```
list_repos:owner=endomorphosis:limit=10
get_workflow_run:repo=owner/repo:run_id=12345
```

This ensures different parameter combinations get separate cache entries.

## Performance Impact

### Without Cache
- **Repository listing**: ~1-2 seconds per call
- **Workflow runs**: ~0.5-1 second per call
- **API calls**: Every request hits GitHub

### With Cache (after initial load)
- **Repository listing**: ~0.001 seconds (cached)
- **Workflow runs**: ~0.001 seconds (cached)
- **API calls**: Only when cache expires or data not cached

**Example savings**: If your autoscaler checks 10 repos every 60 seconds:
- Without cache: ~600 API calls/hour
- With cache: ~120 API calls/hour (80% reduction!)

## Configuration Options

### GitHubCLI Options

```python
gh = GitHubCLI(
    gh_path="gh",              # Path to gh executable
    enable_cache=True,         # Enable caching
    cache=None,                # Custom cache instance (or None for global)
    cache_ttl=300              # Default TTL in seconds
)
```

### Cache Configuration

```python
from ipfs_accelerate_py.github_cli import configure_cache

cache = configure_cache(
    cache_dir=None,            # Cache directory (default: ~/.cache/github_cli)
    default_ttl=300,           # Default TTL in seconds (5 minutes)
    max_cache_size=1000,       # Max entries in memory
    enable_persistence=True    # Save to disk
)
```

## Cache TTLs by Operation

| Operation | Default TTL | Rationale |
|-----------|-------------|-----------|
| `list_repos` | 5 minutes | Repository list doesn't change often |
| `get_repo_info` | 5 minutes | Repo metadata is relatively stable |
| `list_workflow_runs` | 60 seconds | Workflow status changes frequently |
| `get_workflow_run` | 60 seconds | Run details can change |
| `list_runners` | 30 seconds | Runner status changes very frequently |

## Testing

Run the test script to see cache performance:

```bash
python test_github_cache.py
```

Example output:
```
GitHub API Cache Performance Test
============================================================

1. First call to list_repos (cache miss):
   Time: 1.234s
   Repos found: 10

2. Second call to list_repos (cache hit):
   Time: 0.001s
   Repos found: 10
   Speed improvement: 1234.0x faster

3. Cache Statistics:
   Total requests: 2
   Cache hits: 1
   Cache misses: 1
   Hit rate: 50.0%
   Cache size: 1/500

Cache Benefits:
============================================================
✓ Cached requests are 1234.0x faster
✓ Reduced API calls by 50.0%
✓ Cache persists across sessions
✓ Automatic expiration prevents stale data
```

## Best Practices

1. **Use defaults**: The default TTLs are well-tuned for most use cases
2. **Monitor cache stats**: Check hit rates to ensure cache is effective
3. **Invalidate when needed**: If you make changes (e.g., create a runner), invalidate related cache
4. **Adjust TTL for your use case**: Longer TTL = fewer API calls but potentially stale data
5. **Consider disk space**: Persistent cache uses disk space; clear periodically if needed

## Integration with Autoscaler

The autoscaler automatically benefits from caching:

```python
from ipfs_accelerate_py.github_autoscaler import GitHubRunnerAutoscaler

# Autoscaler uses cached GitHub CLI by default
autoscaler = GitHubRunnerAutoscaler(
    owner="endomorphosis",
    interval=60,  # Check every 60 seconds
    # GitHub API calls are automatically cached!
)

autoscaler.start()
```

This means:
- Repository lists are cached for 5 minutes
- Workflow runs are cached for 60 seconds
- Runner status is cached for 30 seconds
- **80%+ reduction in API calls** for typical autoscaler usage

## Troubleshooting

### Cache Not Working

Check that caching is enabled:
```python
gh = GitHubCLI(enable_cache=True)  # Ensure this is True
```

### Stale Data

If you're getting stale data, reduce the TTL:
```python
cache = configure_cache(default_ttl=60)  # 1 minute instead of 5
```

Or invalidate specific entries:
```python
cache.invalidate_pattern("list_workflow_runs")
```

### High Memory Usage

Reduce the cache size:
```python
cache = configure_cache(max_cache_size=100)  # Store fewer entries
```

Or disable persistence:
```python
cache = configure_cache(enable_persistence=False)
```

### Permission Errors

If you get permission errors for the cache directory, specify a custom location:
```python
cache = configure_cache(cache_dir="/tmp/github_cache")
```

## Implementation Details

The cache uses:
- **LRU eviction**: Least-recently-used entries are evicted when cache is full
- **MD5 hashing**: Not used (keys are human-readable strings)
- **JSON serialization**: For disk persistence
- **Threading locks**: For thread safety
- **Timestamp-based expiration**: Each entry has a creation timestamp and TTL

## Future Enhancements

Potential improvements:
- Cache warming (pre-fetch commonly accessed data)
- Smart invalidation (invalidate related entries automatically)
- Compression for disk storage
- Memory-mapped cache for very large datasets
- Redis/Memcached backend option for distributed caching
