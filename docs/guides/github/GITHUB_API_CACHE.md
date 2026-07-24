# GitHub API Cache

`ipfs_accelerate_py.github_cli.cache` provides a local, persistent cache for
GitHub API wrapper operations. It can reduce repeated API calls, but cached
data is still subject to TTL, invalidation, permissions, and GitHub freshness.
P2P sharing is a separate optional capability and is disabled by default.

## Basic usage

```python
from ipfs_accelerate_py.github_cli import GitHubCLI, configure_cache, get_global_cache

cache = configure_cache(
    default_ttl=300,
    max_cache_size=500,
    enable_persistence=True,
)
gh = GitHubCLI(enable_cache=True, cache=cache)

repos = gh.list_repos(owner="owner", limit=10)
stats = get_global_cache().get_stats()
print(repos)
print(stats)
```

The wrapper caches operations such as repository metadata, workflow runs, and
runner status. The operation and request parameters form the cache identity.
Use a shorter TTL or bypass the cache when freshness is more important than
latency:

```python
fresh_repo = gh.get_repo_info("owner/repo", use_cache=False)
cache.invalidate_pattern("list_workflow_runs")
cache.clear()
```

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `IPFS_ACCELERATE_CACHE_DIR` | `~/.cache/github_cli` | Local persistent cache directory. |
| `CACHE_DEFAULT_TTL` | `300` | Default TTL used by the global cache factory. |
| `CACHE_ENABLE_P2P` | `false` | Enable optional P2P cache sharing. |
| `CACHE_LISTEN_PORT` | `9100` | P2P listen port when enabled. |
| `CACHE_BOOTSTRAP_PEERS` | empty | Comma-separated peer multiaddrs. |
| `CACHE_P2P_SHARED_SECRET` | empty | Recommended shared encryption secret for peers. |
| `GH_TOKEN` / `GITHUB_TOKEN` | unset | GitHub authentication source. |

The Python constructor and `configure_cache()` parameters override environment
defaults. Keep cache directories private and do not put tokens in cache keys,
logs, or checked-in configuration.

## P2P mode

P2P cache sharing requires the `mcp-p2p` or `libp2p` extra, encryption support,
peer identity/discovery, and a network policy that permits the selected ports:

```bash
python -m pip install -e ".[mcp-p2p]"
export CACHE_ENABLE_P2P=true
export CACHE_P2P_SHARED_SECRET="use-a-random-shared-secret"
```

P2P is best treated as a cache hint. A peer hit is not an authoritative source
of current GitHub state, and a successful local cache initialization does not
prove that remote peers are reachable. Use the dedicated P2P setup and
security guides for an isolated multi-peer experiment.

## Validation

The repository contains script-style integration checks and unit/contract
tests. Start with the local cache path:

```bash
python -m pytest test/test_github_actions_p2p_cache.py -q
python scripts/validation/verify_p2p_cache.py
```

The verification script reports missing optional dependencies or credentials;
that is a capability result, not a failed core-package import. A live
multi-machine test must additionally verify peer identity, encryption,
firewall/NAT behavior, and TTL/invalidation semantics.

## Related documentation

- [GitHub integration index](README.md)
- [P2P guide](../p2p/README.md)
- [P2P cache quick reference](GITHUB_CACHE_QUICK_REF.md)
- [GitHub cache overview](../../features/github-cache/overview.md)
- [Testing guide](../../development/testing.md)
