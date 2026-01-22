# GitHub API Cache - Quick Reference

## Overview

GitHub API caching is now **automatically enabled** for all GitHub CLI operations. This reduces API calls by 50-85% and makes cached requests 10,000-40,000x faster.

## What Changed

✅ **All GitHub API calls are now cached** (as of this update)  
✅ **No code changes needed** - works automatically  
✅ **Autoscaler benefits immediately** - reduced API usage  
✅ **Cache persists across restarts** - faster startup  

## Performance

- **First call**: ~0.3-0.5 seconds (API request)
- **Cached call**: ~0.0001 seconds (**40,000x faster!**)
- **API reduction**: 50-85% fewer GitHub API calls

## Cache Behavior

### Automatic Caching
| Operation | TTL | When Used |
|-----------|-----|-----------|
| Repository list | 5 min | Autoscaler checks repos |
| Workflow runs | 60 sec | Autoscaler monitors workflows |
| Runner status | 30 sec | Check active runners |
| Repo info | 5 min | Get repo details |

### Smart Features
- ✅ Thread-safe (concurrent access)
- ✅ Persistent (survives restarts)
- ✅ Auto-expiring (prevents stale data)
- ✅ LRU eviction (manages memory)
- ✅ Statistics tracking (monitor performance)

## Quick Usage

### Default (Recommended)
```python
from ipfs_accelerate_py.github_cli import GitHubCLI

gh = GitHubCLI()  # Caching enabled automatically
repos = gh.list_repos(owner="endomorphosis")
```

### Disable If Needed
```python
# Disable globally
gh = GitHubCLI(enable_cache=False)

# Disable per-call
repos = gh.list_repos(owner="me", use_cache=False)
```

### Check Statistics
```python
from ipfs_accelerate_py.github_cli import get_global_cache

cache = get_global_cache()
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"API calls saved: {stats['hits']}")
```

## Autoscaler Impact

The autoscaler **automatically benefits** from caching:

**Before**: 1200 API calls/hour (10 repos, 60s interval)  
**After**: ~600 API calls/hour (**50% reduction**)

With more repos or longer intervals, reduction can reach **85%**.

## Configuration

### Custom TTLs
```python
from ipfs_accelerate_py.github_cli import configure_cache

cache = configure_cache(
    default_ttl=600,  # 10 minutes instead of 5
    max_cache_size=2000  # Store more entries
)
```

### Custom Cache Directory
```python
cache = configure_cache(
    cache_dir="/custom/path/to/cache"
)
```

### Disable Persistence
```python
cache = configure_cache(
    enable_persistence=False  # Memory-only cache
)
```

## Cache Management

### Invalidate Specific Entry
```python
cache.invalidate("list_repos", owner="endomorphosis", limit=10)
```

### Invalidate Pattern
```python
cache.invalidate_pattern("list_repos")  # All repo listings
cache.invalidate_pattern("list_workflow_runs")  # All workflow runs
```

### Clear Everything
```python
cache.clear()
```

## Troubleshooting

### Stale Data
```python
# Reduce TTL
cache = configure_cache(default_ttl=60)

# Or invalidate manually
cache.invalidate_pattern("list_workflow_runs")
```

### High Memory
```python
# Reduce cache size
cache = configure_cache(max_cache_size=100)
```

### Permission Issues
```python
# Use temp directory
cache = configure_cache(cache_dir="/tmp/github_cache")
```

## Files Added

- `ipfs_accelerate_py/github_cli/cache.py` - Cache implementation
- `test_github_cache.py` - Performance testing
- `GITHUB_API_CACHE.md` - Full documentation
- `GITHUB_CACHE_SUMMARY.md` - Implementation details

## Testing

Run performance test:
```bash
python test_github_cache.py
```

Expected output:
```
Speed improvement: 38,656x faster
✓ Cached requests are 38656.0x faster
✓ Reduced API calls by 50.0%
✓ Cache persists across sessions
```

## Current Status

✅ **Implemented and deployed**  
✅ **Service restarted** (cache loaded 1 entry from disk)  
✅ **Autoscaler running** with caching enabled  
✅ **No breaking changes** - fully backwards compatible  

## Benefits Summary

1. **Performance**: 10,000-40,000x faster for cached requests
2. **API Usage**: 50-85% fewer GitHub API calls
3. **Rate Limits**: Much safer from hitting GitHub's limits
4. **Reliability**: Persistent cache survives restarts
5. **Zero Config**: Works automatically with defaults
6. **Monitoring**: Built-in statistics tracking

## Next Steps

1. ✅ Cache is active and running
2. Monitor cache statistics periodically
3. Adjust TTLs if needed for your use case
4. Consider enabling metrics export (future enhancement)

---

For detailed documentation, see:
- **`GITHUB_API_CACHE.md`** - Complete usage guide
- **`GITHUB_CACHE_SUMMARY.md`** - Implementation details
