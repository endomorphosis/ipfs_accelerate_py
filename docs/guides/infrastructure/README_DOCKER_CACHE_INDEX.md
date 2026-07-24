# Docker And Cache Documentation Index

Docker runner cache notes in this directory describe optional deployment
experiments. They are not part of the core package defaults and should not be
treated as a production recipe without checking the current compose files,
workflow files, image provenance, network policy, and secrets.

## Current starting points

- [Docker guide](../docker/README.md) for the repository's current container
  entry points and security boundary.
- [GitHub API cache](../github/GITHUB_API_CACHE.md) for the local cache API and
  current P2P defaults.
- [Distributed cache](DISTRIBUTED_CACHE.md) for optional cache sharing.
- [P2P setup](../p2p/README.md) for optional libp2p/MCP-P2P installation.
- [Deployment guide](../deployment/README.md) for process, network, and resource
  controls.

## Current checks

Run from the repository root:

```bash
python -m pytest test/test_docker_executor.py -q
python -m pytest test/test_github_actions_p2p_cache.py -q
python scripts/validation/verify_p2p_cache.py
```

For a live TaskQueue cache RPC smoke, use
`scripts/validation/p2p_taskqueue_cache_smoke.py` with an explicitly configured
peer. A local test pass does not prove Docker-to-host connectivity, peer
discovery, or GitHub API freshness.

## Historical records

The older Docker runner cache plan and implementation summaries are retained as
design history. They may mention files, workflow examples, fixed performance
targets, or deployment scripts that are not present in this checkout. Use them
to understand prior decisions, then verify every command against the current
source before reuse.
