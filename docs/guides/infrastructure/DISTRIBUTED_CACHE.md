# Distributed GitHub Cache

The GitHub cache has two distinct modes:

1. local persistent caching through `GitHubAPICache`; and
2. optional P2P sharing through the libp2p integration.

P2P is disabled by default. A local cache hit does not prove peer reachability,
freshness, or authorization.

## Local cache

```python
from ipfs_accelerate_py.github_cli import GitHubCLI, configure_cache

cache = configure_cache(default_ttl=300, max_cache_size=1000)
gh = GitHubCLI(enable_cache=True, cache=cache)
print(gh.list_repos(owner="owner", limit=10))
print(cache.get_stats())
```

Configure invalidation and persistence for the workload rather than assuming a
fixed hit rate:

```python
cache.invalidate_pattern("list_workflow_runs")
cache.clear()
```

## Enable P2P explicitly

```bash
python -m pip install -e ".[mcp-p2p]"
export CACHE_ENABLE_P2P=true
export CACHE_LISTEN_PORT=9100
export CACHE_P2P_SHARED_SECRET="use-a-random-secret"
```

For remote peers, set a reachable `CACHE_BOOTSTRAP_PEERS` multiaddr and apply
firewall, identity, encryption, and rate-limit policy. A peer's response is a
cache hint; the application must still apply TTL and invalidation rules.

## Current validation

```bash
python -m pytest test/test_github_actions_p2p_cache.py -q
python scripts/validation/verify_p2p_cache.py
```

For a TaskQueue P2P RPC smoke, use
`scripts/validation/p2p_taskqueue_cache_smoke.py`. It is not a substitute for
an application-level GitHub cache test.

## Related documentation

- [GitHub API cache](../github/GITHUB_API_CACHE.md)
- [GitHub cache overview](../../features/github-cache/overview.md)
- [P2P setup](../p2p/README.md)
- [Deployment](../deployment/README.md)
