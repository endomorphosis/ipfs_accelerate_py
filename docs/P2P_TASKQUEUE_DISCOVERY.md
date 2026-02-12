# TaskQueue P2P Discovery (LAN + WAN)

This project’s TaskQueue transport supports discovery so a client can connect *without pre-sharing the remote multiaddr*.

## What “discovery” can and cannot do

- Discovery helps a client **find a peer ID / address to dial**.
- Discovery does **not** guarantee connectivity through NAT/firewalls. If both peers are behind NAT with no inbound connectivity, you’ll typically need one of:
  - Port forwarding / open inbound TCP to the service
  - A VPN/overlay network (WireGuard, Tailscale, Zerotier)
  - A relay + hole punching setup (optional; see env flags below)

## Discovery mechanisms (order)

When `RemoteQueue.multiaddr` is empty, the client tries:

1. **announce-file**: reads a local JSON hint containing `{peer_id, multiaddr}`.
2. **bootstrap peers (direct dial)**: only used when you explicitly set `IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS` to TaskQueue endpoints.
3. **rendezvous**: discovers peers advertising in a namespace.
4. **DHT**: finds providers for the namespace (or finds a specific peer ID).
5. **mDNS**: LAN-only fallback.

You can see which one succeeded using:

- `./scripts/p2p_rpc.py discover`

## Recommended WAN setup (practical)

For internet use, you generally want a **stable, reachable “coordination” node** (public IP or port-forwarded) that is always online:

- Run the TaskQueue service on that node.
- Configure clients to discover/dial it via one of:
  - **Bootstrap peers**: set `IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS` to the coordination node’s multiaddr.
  - **Rendezvous**: enable rendezvous on the coordination node and keep a shared namespace.
  - **DHT**: keep DHT enabled on both sides (still benefits from reachable bootstrap/routing peers).

If your *service* is behind NAT and you need the public box to dial **inbound to it**, you will likely need a VPN/overlay network or port forwarding.

## Useful environment variables

- `IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS`
  - Comma-separated multiaddrs.
  - If set, the client will also attempt to dial these directly as TaskQueue endpoints.
- `IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS=1|0`
- `IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS=...`
- `IPFS_ACCELERATE_PY_TASK_P2P_DHT=1|0`
- `IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE=/path/to/task_p2p_announce.json` (or `0` to disable)

### Optional NAT traversal (best-effort)

These are disabled by default.

- `IPFS_ACCELERATE_PY_TASK_P2P_RELAY=1|0`
  - Enables Circuit Relay v2 protocol handlers.
- `IPFS_ACCELERATE_PY_TASK_P2P_RELAY_HOP=1|0`
  - When enabled alongside `*_P2P_RELAY=1`, allows this node to act as a relay (requires it to be reachable by others).
- `IPFS_ACCELERATE_PY_TASK_P2P_HOLEPUNCH=1|0`
  - Enables DCUtR hole punching support.

## CLI quickstart

- Discovery trace:
  - `./scripts/p2p_rpc.py discover --pretty`

- Once it can reach a peer, normal operations:
  - `./scripts/p2p_rpc.py status --pretty`
  - `./scripts/p2p_rpc.py call-tool --tool get_server_status --pretty`
  - `./scripts/p2p_rpc.py cache-set --key demo --value '"hello"' --pretty`
  - `./scripts/p2p_rpc.py cache-get --key demo --pretty`
