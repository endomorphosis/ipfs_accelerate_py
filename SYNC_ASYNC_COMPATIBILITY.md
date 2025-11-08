# Synchronous and Asynchronous Compatibility Guide

## Overview

The P2P cache system is designed to work seamlessly in **both synchronous and asynchronous contexts**. This document explains how it works and best practices for using it.

## ✅ Supported Usage Patterns

### 1. Synchronous Usage (GitHub Autoscaler, CLI)

The cache works perfectly in synchronous code like the GitHub autoscaler:

```python
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

# Create cache in synchronous context
cache = GitHubAPICache(enable_p2p=True)

# Synchronous operations
cache.put("repos/owner/name", repo_data, ttl=300)
data = cache.get("repos/owner/name")
stats = cache.get_stats()
```

**Result**: ✅ All operations complete immediately without blocking

### 2. Asynchronous Usage

The cache also works in async functions:

```python
async def fetch_data():
    cache = GitHubAPICache(enable_p2p=True)
    
    # Synchronous methods called from async context
    cache.put("key", data, ttl=300)
    result = cache.get("key")
    
    # Mix with async operations
    await some_async_operation()
    more_data = cache.get("other_key")
```

**Result**: ✅ Synchronous cache methods work correctly in async context

### 3. Multi-Threading

The cache is thread-safe:

```python
import threading

cache = GitHubAPICache(enable_p2p=True)

def worker(thread_id):
    cache.put(f"thread/{thread_id}", data)
    result = cache.get(f"thread/{thread_id}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
```

**Result**: ✅ Thread-safe operations with proper locking

### 4. Mixed Sync/Async

Combining synchronous and asynchronous usage:

```python
# Sync
cache = GitHubAPICache(enable_p2p=True)
cache.put("key1", data1)

# Async
async def process():
    cache.put("key2", data2)
    await asyncio.sleep(1)
    return cache.get("key1")

result = asyncio.run(process())
```

**Result**: ✅ Works correctly

## 🔧 Technical Implementation

### Background Event Loop for P2P

The P2P functionality uses a **background thread with a dedicated event loop**:

```python
def _init_p2p(self) -> None:
    # Create new event loop for background thread
    self._event_loop = asyncio.new_event_loop()
    
    def run_event_loop():
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()
    
    # Start in daemon thread
    self._p2p_thread = threading.Thread(
        target=run_event_loop,
        daemon=True,
        name="p2p-event-loop"
    )
    self._p2p_thread.start()
    
    # Schedule initialization (non-blocking)
    future = asyncio.run_coroutine_threadsafe(
        self._start_p2p_host(),
        self._event_loop
    )
```

### Non-Blocking P2P Operations

Runtime P2P operations are **always non-blocking**:

```python
def _broadcast_in_background(self, cache_key: str, entry: CacheEntry):
    """Broadcast cache entry without blocking."""
    if not self.enable_p2p or not self._p2p_host:
        return
    
    # Schedule async operation in background event loop
    asyncio.run_coroutine_threadsafe(
        self._broadcast_cache_entry(cache_key, entry),
        self._event_loop
    )
```

### Key Design Decisions

1. **Separate event loop**: P2P has its own event loop in a background thread
2. **Non-blocking initialization**: P2P starts initializing without blocking cache creation
3. **Thread-safe**: All cache operations use proper locking
4. **Graceful degradation**: If P2P fails, cache still works in local-only mode

## 📊 Test Results

```
TEST SUMMARY
======================================================================
✅ PASS     | Synchronous Usage
✅ PASS     | Asynchronous Operations
✅ PASS     | Multi-Threading
✅ PASS     | Mixed Sync/Async
✅ PASS     | P2P Initialization
======================================================================
Total: 5/5 tests passed (100%)
```

## 🚀 Production Deployment

### Current Status

The GitHub autoscaler is running in production with P2P enabled:

```bash
$ systemctl --user status github-autoscaler.service
● github-autoscaler.service - GitHub Actions Runner Autoscaler with P2P Cache
     Active: active (running)
     Memory: 58.1M (limit: 512.0M)
```

### Configuration

```ini
[Service]
Environment=CACHE_ENABLE_P2P=true
Environment=P2P_LISTEN_PORT=9000
Environment=CACHE_DEFAULT_TTL=300
ExecStart=/home/barberb/ipfs_accelerate_py/.venv/bin/python3 github_autoscaler.py --interval 60
```

## 🎯 Best Practices

### 1. Cache Creation

```python
# Good: Create once, reuse
cache = GitHubAPICache(enable_p2p=True)

# Bad: Creating new cache for each operation
def get_data():
    cache = GitHubAPICache()  # Don't do this repeatedly
    return cache.get("key")
```

### 2. Shutdown Handling

```python
# Automatic shutdown on program exit
cache = GitHubAPICache(enable_p2p=True)
# ... use cache ...
# __del__() handles cleanup automatically

# Or explicit shutdown
cache.shutdown()
```

### 3. Error Handling

```python
try:
    cache = GitHubAPICache(enable_p2p=True)
    # P2P may not initialize immediately
    # Cache still works in local-only mode
except Exception as e:
    logger.warning(f"Cache initialization issue: {e}")
    # Handle gracefully
```

## ⚠️ Known Limitations

1. **P2P initialization delay**: P2P may take a few seconds to fully initialize
2. **Network requirements**: P2P requires open ports and network connectivity
3. **Bootstrap peers**: Need at least one peer to form P2P network
4. **Event loop lifecycle**: Event loop runs until process exits

## 🔍 Debugging

### Check if P2P is working

```python
cache = GitHubAPICache(enable_p2p=True)
stats = cache.get_stats()

print(f"P2P enabled: {stats['p2p_enabled']}")
print(f"Connected peers: {stats['connected_peers']}")
print(f"Peer hits: {stats['peer_hits']}")
```

### Monitor P2P status

```bash
python3 monitor_p2p_cache.py --interval 5
```

### Test sync/async compatibility

```bash
python3 test_sync_async_usage.py
```

## 📝 Summary

**Question**: Does this work for both synchronous and asynchronous uses?

**Answer**: ✅ **YES!**

- ✅ Synchronous code (GitHub autoscaler): **Works perfectly**
- ✅ Asynchronous code: **Works perfectly**
- ✅ Multi-threading: **Thread-safe**
- ✅ Mixed sync/async: **Works correctly**
- ✅ P2P background operations: **Non-blocking**

The cache is designed to be **transparent to the caller** - you don't need to worry about whether your code is sync or async. Just create the cache and use it!

## 📚 Related Documentation

- [PRODUCTION_DEPLOYMENT_SUMMARY.md](./PRODUCTION_DEPLOYMENT_SUMMARY.md) - Production deployment details
- [P2P_CACHE_SPECIFICATION.md](./P2P_CACHE_SPECIFICATION.md) - P2P architecture
- [test_sync_async_usage.py](./test_sync_async_usage.py) - Comprehensive tests
- [monitor_p2p_cache.py](./monitor_p2p_cache.py) - Monitoring tool

---

**Last Updated**: 2025-11-08  
**Status**: ✅ Production Ready  
**Version**: 1.0.0
