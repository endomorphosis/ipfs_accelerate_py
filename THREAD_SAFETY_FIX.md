# P2P Cache Thread-Safety Fix

## Problem Statement

The P2P cache singleton in `ipfs_accelerate_py/github_cli/cache.py` was not thread-safe, causing critical issues in multi-threaded environments:

### The Issue
- Flask with `threaded=True` creates multiple threads
- Multiple tool calls happen simultaneously
- Each accesses `get_global_cache()` at the same time
- The singleton check `if _global_cache is None` passes for multiple threads
- Multiple P2P hosts try to bind to port 9100 simultaneously
- Everything deadlocks

### Root Cause
```python
# BEFORE (NOT THREAD-SAFE):
def get_global_cache(**kwargs):
    global _global_cache
    
    if _global_cache is None:  # ← Race condition here!
        _global_cache = GitHubAPICache(**kwargs)  # ← Multiple threads create instances
    
    return _global_cache
```

Multiple threads could pass the `if _global_cache is None` check before any of them created the instance, resulting in multiple `GitHubAPICache` instances being created. Each instance would try to initialize a P2P host that binds to port 9100, causing port conflicts and deadlocks.

## Solution

Implemented **double-checked locking** pattern with a threading.Lock:

```python
# AFTER (THREAD-SAFE):
_global_cache_lock = Lock()

def get_global_cache(**kwargs):
    global _global_cache
    
    # First check (without lock, for performance)
    if _global_cache is None:
        with _global_cache_lock:  # ← Acquire lock
            # Second check (with lock, ensures only one thread creates instance)
            if _global_cache is None:
                _global_cache = GitHubAPICache(**kwargs)
    
    return _global_cache
```

### How It Works

1. **First Check (Fast Path)**: Check if `_global_cache is None` without acquiring the lock. If the instance already exists, return it immediately without any locking overhead.

2. **Acquire Lock**: If the instance doesn't exist, acquire `_global_cache_lock` to ensure only one thread proceeds.

3. **Second Check (Safe Path)**: Check again if `_global_cache is None` while holding the lock. This handles the case where multiple threads passed the first check simultaneously - only the first one will find it still `None` and create the instance.

4. **Create Instance**: Create the `GitHubAPICache` instance (only one thread does this).

### Benefits

✅ **Thread-Safe**: Only one thread creates the singleton instance, even with concurrent access  
✅ **No Port Conflicts**: Single P2P host binds to port 9100, no deadlocks  
✅ **Performance**: Lock only acquired when instance is null (first access only)  
✅ **Backward Compatible**: No changes to the public API

## Testing

### Unit Tests
Run the thread-safety tests:
```bash
python test_cache_thread_safety.py
```

Tests verify:
- 10 concurrent threads all get the same singleton instance
- `configure_cache()` is thread-safe with 5 concurrent threads
- P2P port binding works without conflicts (when libp2p available)

### Demonstration
Run the demonstration script to see the fix in action:
```bash
python demo_thread_safety_fix.py
```

Simulates Flask with `threaded=True`:
- 20 concurrent requests
- Each accesses `get_global_cache()`
- All get the same instance (no port conflicts)

## Files Changed

| File | Changes |
|------|---------|
| `ipfs_accelerate_py/github_cli/cache.py` | Added `_global_cache_lock`, implemented double-checked locking in `get_global_cache()` and `configure_cache()` |
| `test_cache_thread_safety.py` | Comprehensive thread-safety tests |
| `demo_thread_safety_fix.py` | Demonstration of fix in Flask-like scenario |

## Impact

### Before Fix
```
Thread 1: get_global_cache() → Creates instance A → P2P binds to port 9100 ✓
Thread 2: get_global_cache() → Creates instance B → P2P tries port 9100 ✗ DEADLOCK
Thread 3: get_global_cache() → Creates instance C → P2P tries port 9100 ✗ DEADLOCK
```

### After Fix
```
Thread 1: get_global_cache() → Acquires lock → Creates instance A → P2P binds to port 9100 ✓
Thread 2: get_global_cache() → Waits for lock → Gets instance A (already created) ✓
Thread 3: get_global_cache() → Gets instance A immediately (no lock needed) ✓
```

## Additional Notes

### Python Memory Model
The double-checked locking pattern is safe in Python due to the Global Interpreter Lock (GIL), which ensures that bytecode operations are atomic. However, relying on the GIL alone is not recommended for portable code. Using an explicit lock makes the intent clear and works correctly regardless of Python implementation.

### Alternative Approaches Considered
1. **Single Lock Check**: Simpler but requires lock acquisition on every call
2. **Module-Level Initialization**: Would work but less flexible for configuration
3. **Thread-Local Storage**: Overkill and would create multiple instances

The double-checked locking pattern provides the best balance of safety and performance.

## References

- [Python threading.Lock documentation](https://docs.python.org/3/library/threading.html#lock-objects)
- [Double-checked locking pattern](https://en.wikipedia.org/wiki/Double-checked_locking)
- Original issue: "P2P cache singleton not thread-safe"
