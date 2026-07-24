# P2P GitHub Cache Validation

This page describes the boundary for a multi-host GitHub cache experiment. It
does not claim that a two-laptop network is available or secure by default.
P2P sharing is opt-in, requires the optional `mcp-p2p`/`libp2p` dependencies,
and must be protected by peer identity, encryption, firewall rules, and a
shared secret.

## Prepare both hosts

```bash
python -m pip install -e ".[mcp-p2p]"
export CACHE_ENABLE_P2P=true
export CACHE_P2P_SHARED_SECRET="generate-a-random-secret-and-share-it-out-of-band"
export CACHE_DEFAULT_TTL=300
```

Give each process a distinct listen port and configure
`CACHE_BOOTSTRAP_PEERS` with reachable multiaddrs. Never use `127.0.0.1` in a
multi-host bootstrap address; it points back to the current host.

## Validate the local contract first

Run from the repository root before opening a firewall:

```bash
python -m pytest test/test_github_actions_p2p_cache.py -q
python scripts/validation/verify_p2p_cache.py
```

The first check uses controlled API/cache doubles to verify cache-hit/miss
ordering and configuration handling. The verification script reports whether
libp2p, cryptography, multiformats, credentials, and local cache operations
are available.

## Validate a live peer path

There is no maintained `tools/github_p2p_cache_smoke.py` in the current
checkout. Do not use that command from older copies of this runbook. For a
TaskQueue P2P cache RPC smoke, use the current script instead:

```bash
IPFS_ACCELERATE_PY_TASK_P2P_TOKEN="..." \
  python scripts/validation/p2p_taskqueue_cache_smoke.py \
  --multiaddr /ip4/HOST/tcp/PORT/p2p/PEER_ID
```

That smoke validates the TaskQueue cache RPC, not GitHub API semantics. A
GitHub cache deployment still needs an application-level test that confirms
cache identity, encryption, TTL, invalidation, and stale-data policy.

## Operational checks

- Confirm both peers advertise routable addresses and have distinct ports.
- Confirm the shared secret is identical without placing it in source control.
- Confirm `connected_peers` and cache statistics from each process.
- Test a cache miss followed by a hit, then expiry/invalidation.
- Verify that a peer outage falls back to local cache or GitHub API behavior.
- Keep API responses and authentication material out of durable logs.

See the [GitHub API cache guide](GITHUB_API_CACHE.md), [P2P guide](../p2p/README.md),
and [deployment guidance](../deployment/README.md).
