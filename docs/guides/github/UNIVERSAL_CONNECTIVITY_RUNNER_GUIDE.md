# Universal Connectivity Runner Guide (ipfs_accelerate_py)

This repo’s P2P cache sharing uses **py-libp2p** to exchange cached GitHub API responses between self-hosted runners, reducing repeated calls to GitHub.

It borrows the *idea* of “universal connectivity” (multiple discovery/transport/NAT traversal techniques) from:
- https://github.com/libp2p/universal-connectivity

## 1) What is (and isn’t) cached

**Cached + P2P-shared:**
- GitHub API calls made through this project’s Python layer (`ipfs_accelerate_py.github_cli.*`), including the autoscaler/dashboard operations.
- Copilot CLI responses when invoked through this repo’s Python wrapper (`ipfs_accelerate_py.copilot_cli.wrapper.CopilotCLI`) which uses the same cache.

**Not transparently cached:**
- The official `gh` binary (and `gh copilot`) cannot be “made to use libp2p” without modifying GitHub CLI itself or routing its requests through a compatible caching proxy (not implemented here).

If you want caching benefits, make calls through the repo’s Python wrappers/CLI, not directly via `gh`.

## 2) Environment variables to enable/shape caching

These are read by `ipfs_accelerate_py.github_cli.cache.get_global_cache()`:

- `CACHE_ENABLE_P2P=true` (default)
- `CACHE_LISTEN_PORT=9100` (default)
- `CACHE_BOOTSTRAP_PEERS=/ip4/…/tcp/9100/p2p/…` (comma-separated multiaddrs)
- `CACHE_DEFAULT_TTL=300` (increase to reduce API calls; e.g. `3600`)

Optional: public bootstrap nodes

- By default, the service does **not** auto-add public libp2p bootstrap nodes (like `bootstrap.libp2p.io`) because they’re often not reachable with the enabled transports and can slow startup with repeated failed connect attempts.
- To enable adding those public bootstrap nodes when no other peers are configured, set `IPFS_ACCELERATE_ENABLE_PUBLIC_BOOTSTRAP=1`.

Persistence (important for good hit rate across restarts):
- `CACHE_DIR=/tmp/ipfs_accelerate_github_cli_cache` (recommended under systemd hardening)
- Or `IPFS_ACCELERATE_CACHE_DIR=/tmp/ipfs_accelerate_github_cli_cache` (same purpose)

Security/encryption (recommended):
- `GITHUB_TOKEN=…` (used as shared secret for P2P message encryption; peers must match to decrypt)

## 3) systemd hardening gotcha (very common)

If the service uses `ProtectHome=read-only`, then `~/.cache/...` may exist but be **non-writable**.
That prevents persistence and makes the cache hit rate look “low” after restarts.

Fix: point the cache dir to a writable path (e.g. `/tmp/...`) via `CACHE_DIR` or `IPFS_ACCELERATE_CACHE_DIR`.

## 4) Network requirements for “non-zero peers”

At minimum, for TCP multiaddr connectivity:

- Allow **inbound + outbound TCP** on `CACHE_LISTEN_PORT` (default `9100`) between runners.
- Ensure NAT/firewall rules allow those connections.

Optional discovery / “universal connectivity” style features:

- mDNS discovery (LAN only): allow UDP multicast `224.0.0.251:5353` (and local LAN multicast in general).
- DHT / relay / hole punching: these require more advanced libp2p protocol support and public relay/bootstrap infrastructure.
  This repo currently focuses on TCP + explicit bootstrap multiaddrs and GitHub-backed peer registry.

## 5) Troubleshooting checklist

1. Check dashboard endpoints:
   - `/api/mcp/cache/stats` for `hit_rate`, `cache_size`, `cache_dir`, `api_calls_made`, `bootstrap_peers_configured`
   - `/api/mcp/peers` for `peer_count` (connected) and `connectivity.discovered_peers`
   - If you’re testing peer exchange, look for `known_peer_multiaddrs` and `peer_exchange_last_iso`

2. Confirm the process is actually connecting:
   - You should see logs indicating connections and `peer_count > 0`.
   - Look for "Refreshing bootstrap peer connections..." logs every 5 minutes (automatic reconnect).

3. Confirm all peers share the same `GITHUB_TOKEN` (or disable encryption) if you expect cross-peer cache decryption.

4. Ensure the cache dir is writable in the service context.

5. Verify TCP connectivity between runners:
   - `telnet <other-runner-ip> 9100` should succeed
   - Check firewall/NAT rules allow bidirectional TCP on the configured port

## 6) Connectivity Patterns Explained (from universal-connectivity)

**What we implement (py-libp2p capabilities):**

| Feature | Status | Notes |
|---------|--------|-------|
| TCP transport | ✅ Supported | Primary transport; requires direct reachability |
| Identify protocol | ✅ Enabled | Peers exchange supported protocols/addresses |
| Periodic bootstrap refresh | ✅ Implemented | Re-connects to bootstrap peers every 5 minutes |
| Peer registry (file-based) | ✅ Implemented | Local discovery for same-host/LAN runners |
| Connection keepalive | ✅ Basic | Periodic reconnect attempts maintain peer list |

**What py-libp2p *has* but we don't currently use:**

| Feature | Status | Next Steps |
|---------|--------|------------|
| mDNS (LAN discovery) | ⚠️  Available | Could enable for automatic LAN peer discovery |
| Kademlia DHT | ⚠️  Partial | py-libp2p has DHT but not fully integrated here |
| Circuit Relay v1 | ⚠️  Available | Could route connections through relay peers |
| AutoNAT v1 | ⚠️  Available | Could detect external address automatically |

**What universal-connectivity uses that py-libp2p lacks:**

| Feature | Status | Impact |
|---------|--------|--------|
| QUIC / WebRTC / WebTransport | ❌ Not available | Limits browser + mobile peer support |
| Circuit Relay v2 | ❌ Not available | More efficient relay protocol |
| DCUtR (hole-punching) | ❌ Not available | NAT traversal without relay |
| AutoNAT v2 | ❌ Not available | Better NAT detection |

**Best practices for maintaining peer connectivity:**

1. **Static bootstrap peers**: Always configure at least 2-3 known-reachable multiaddrs via `CACHE_BOOTSTRAP_PEERS`.
2. **External address**: If behind NAT, set `EXTERNAL_IP` or configure port forwarding so other peers can reach you.
3. **Symmetric connectivity**: If peer A can reach peer B, ensure B can also reach A (firewall rules, NAT hairpinning).
4. **Monitor connection churn**: Watch logs for "Refreshing bootstrap..." and successful reconnects.
5. **Shared encryption key**: All peers must use the same `GITHUB_TOKEN` to decrypt P2P messages.

