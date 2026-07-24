# Synchronous And Asynchronous Cache Compatibility

`GitHubAPICache` exposes synchronous cache operations and can be used by
synchronous or asynchronous application code. Optional P2P work runs behind
its own background runtime when enabled; local cache operations should still be
bounded and treated as ordinary blocking Python calls.

## Synchronous use

```python
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

cache = GitHubAPICache(enable_p2p=False)
cache.put("key", {"value": 1}, ttl=60)
assert cache.get("key")["value"] == 1
```

## Async or threaded callers

Use the same cache object only when the application understands that `put`,
`get`, invalidation, and statistics calls are synchronous. For high-volume
async services, isolate blocking work in an executor and bound concurrent
access. The cache uses internal locking, but it does not turn a blocking cache
operation into an async one.

```python
import asyncio
from functools import partial
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

cache = GitHubAPICache(enable_p2p=False)

async def read_cache():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(cache.get, "key"))
```

## P2P boundary

Enable P2P explicitly with `CACHE_ENABLE_P2P=true` and the `mcp-p2p` or
`libp2p` extra. P2P initialization can fail independently of local caching;
the expected degraded mode is local-only operation with a clear capability or
health signal. Configure distinct ports, reachable peers, encryption, and
shutdown behavior before using it across hosts.

## Validation

```bash
python test/test_sync_async_usage.py
python -m pytest test/test_github_actions_p2p_cache.py -q
```

The script-style test reports its own environment-dependent result. Do not
copy a historical pass count or production status into a deployment decision.

See [GitHub API cache](../github/GITHUB_API_CACHE.md), [P2P guide](../p2p/README.md),
and [testing](../../development/testing.md).
