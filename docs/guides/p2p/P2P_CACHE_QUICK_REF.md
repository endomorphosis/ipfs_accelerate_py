# P2P Cache Quick Reference

P2P cache sharing is an optional extension of the local GitHub API cache. The
current default is local-only operation; enable P2P deliberately after the
local cache contract is working.

```bash
python -m pip install -e ".[mcp-p2p]"
export CACHE_ENABLE_P2P=true
export CACHE_LISTEN_PORT=9100
export CACHE_P2P_SHARED_SECRET="use-a-random-secret"
```

For a remote peer, configure a reachable `CACHE_BOOTSTRAP_PEERS` multiaddr and
use distinct ports. Do not advertise `127.0.0.1` to another machine.

```python
from ipfs_accelerate_py.github_cli import GitHubCLI, get_global_cache

gh = GitHubCLI(enable_cache=True)
print(gh.list_repos(owner="owner", limit=10))
print(get_global_cache().get_stats())
```

P2P entries are cache hints, not authoritative GitHub state. Validate shared
secret handling, peer identity, TTL, invalidation, firewall/NAT behavior, and
degraded local-only operation.

Run the current local checks:

```bash
python -m pytest test/test_github_actions_p2p_cache.py -q
python scripts/validation/verify_p2p_cache.py
```

For a TaskQueue P2P RPC smoke, use
`scripts/validation/p2p_taskqueue_cache_smoke.py`. The older
`tools/github_p2p_cache_smoke.py` path is not present in this checkout.

See [P2P setup](README.md), [GitHub API cache](../github/GITHUB_API_CACHE.md),
and [distributed cache](../infrastructure/DISTRIBUTED_CACHE.md).
