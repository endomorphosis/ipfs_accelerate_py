# GitHub API Cache Quick Reference

Use the GitHub wrapper with explicit cache configuration when a workload can
tolerate bounded staleness:

```python
from ipfs_accelerate_py.github_cli import GitHubCLI, configure_cache

cache = configure_cache(default_ttl=300, max_cache_size=1000)
gh = GitHubCLI(enable_cache=True, cache=cache)
repos = gh.list_repos(owner="owner", limit=10)
print(cache.get_stats())
```

Useful operations:

```python
cache.invalidate("list_repos", owner="owner", limit=10)
cache.invalidate_pattern("list_workflow_runs")
cache.clear()
```

P2P sharing is optional and defaults to disabled:

```bash
python -m pip install -e ".[mcp-p2p]"
export CACHE_ENABLE_P2P=true
export CACHE_LISTEN_PORT=9100
export CACHE_P2P_SHARED_SECRET="use-a-random-shared-secret"
```

Do not copy historical benchmark ratios into capacity plans. Measure hit rate,
TTL, API latency, object sizes, and provider rate limits in the target workload.

Run the current checks from the repository root:

```bash
python -m pytest test/test_github_actions_p2p_cache.py -q
python scripts/validation/verify_p2p_cache.py
```

See the [full cache guide](GITHUB_API_CACHE.md), [P2P guide](../p2p/README.md),
and [GitHub cache overview](../../features/github-cache/overview.md).
