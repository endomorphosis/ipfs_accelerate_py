# Systemd Services for IPFS Accelerate

This directory contains optional systemd unit files to run the MCP server(s) and the Performance Dashboard on boot.

## Install

Quick install helper:

```bash
sudo deployments/systemd/install.sh
```

Install both MCP instances (so two can run on one machine):

```bash
sudo deployments/systemd/install.sh --both
```

Install a dedicated libp2p relay (Circuit Relay v2 HOP) on a VPS:

```bash
sudo deployments/systemd/install.sh --unit ipfs-accelerate-relay.service
```

Manual install:

```bash
# Copy units
sudo cp deployments/systemd/ipfs-accelerate.service /etc/systemd/system/
sudo cp deployments/systemd/ipfs-accelerate-mcp.service /etc/systemd/system/
sudo cp deployments/systemd/ipfs-accelerate-dashboard.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable ipfs-accelerate.service
sudo systemctl enable ipfs-accelerate-mcp.service
sudo systemctl enable ipfs-accelerate-dashboard.service

# Start them now
sudo systemctl start ipfs-accelerate.service
sudo systemctl start ipfs-accelerate-mcp.service
sudo systemctl start ipfs-accelerate-dashboard.service

# Check status
systemctl status ipfs-accelerate.service
systemctl status ipfs-accelerate-mcp.service
systemctl status ipfs-accelerate-dashboard.service
```

## Configuration
- Units default to:
  - MCP API on port 9000, binds `0.0.0.0`.
  - Additional MCP instance (`ipfs-accelerate-mcp.service`) on port 9001.
  - Performance Dashboard on port 8080, binds `0.0.0.0`.
  - P2P cache sharing (libp2p) on port 9100.
  - TaskQueue P2P RPC (MCP-over-P2P) on port 9100 (and 9101 for the additional MCP instance).
- Adjust environment or ports by editing the unit files.

### Public libp2p bootstraps (/dnsaddr)

If you rely on the default libp2p bootstrap peers (`/dnsaddr/bootstrap.libp2p.io/...`), enable dnsaddr TXT expansion so the node dials concrete `/ip4|/ip6/.../tcp/.../p2p/...` multiaddrs:

- `IPFS_ACCELERATE_PY_TASK_P2P_DNSADDR_RESOLVE=1`
- (optional) `IPFS_DATASETS_PY_TASK_P2P_DNSADDR_RESOLVE=1`

### Multi-node P2P requirements

For peer discovery and cache sharing across machines, ensure every node has:

- `CACHE_ENABLE_P2P=true`
- `CACHE_LISTEN_PORT=9100` (or a consistent, reachable alternative)
- `IPFS_ACCELERATE_GITHUB_REPO=owner/repo` (all nodes must match)
- GitHub auth available to `gh` (prefer `GH_TOKEN`; optionally also set `GITHUB_TOKEN`)

Recommended approach: create an environment file and reference it from the units.

Example:

```bash
sudo mkdir -p /etc/ipfs-accelerate
sudo tee /etc/ipfs-accelerate/secrets.env >/dev/null <<'EOF'
IPFS_ACCELERATE_GITHUB_REPO=endomorphosis/ipfs_accelerate_py
GH_TOKEN=...your_token...
EOF
sudo chmod 600 /etc/ipfs-accelerate/secrets.env
```

### Fast two-node bring-up (recommended)

If you have exactly two boxes and want them to find each other immediately (without waiting on DHT/rendezvous propagation), set each node to explicitly dial the other as a TaskQueue bootstrap endpoint:

- On box A: get box B's multiaddr from its journal line `multiaddr=...` (must include `/p2p/<peer_id>`)
- On box B: set `IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS` to that multiaddr (comma-separated list supported)
- Repeat the same in the other direction (A dials B, B dials A)

This only affects TaskQueue service discovery/dialing (used by MCP-over-P2P tool calls and cache RPC).

## Logs
- All logs go to journald by default:
  - `sudo journalctl -u ipfs-accelerate -f`
  - `sudo journalctl -u ipfs-accelerate-mcp -f`
  - `sudo journalctl -u ipfs-accelerate-dashboard -f`

## Firewall
```bash
sudo ufw allow 9000/tcp
sudo ufw allow 9001/tcp
sudo ufw allow 8080/tcp
sudo ufw allow 9100/tcp
sudo ufw allow 9101/tcp
# Relay VPS unit default port
sudo ufw allow 9102/tcp
# mDNS peer discovery (LAN)
sudo ufw allow 5353/udp
```

## VPS Relay Notes

- The relay unit is [deployments/systemd/ipfs-accelerate-relay.service](deployments/systemd/ipfs-accelerate-relay.service).
- It runs the libp2p TaskQueue service primarily as a libp2p host with Circuit Relay v2 enabled in **HOP** mode.
- Remote tools/cache RPC are disabled by default on the relay (`IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=0`, `IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE=0`).
- Ensure TCP `9102` is reachable from the internet (security group / firewall).
- If the VPS has an unusual networking setup, set `IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP` in `/etc/ipfs-accelerate/secrets.env` to the relay's public IP.

## Troubleshooting

### "Changing to the requested working directory failed: Permission denied"

If you see a CHDIR permission error on boot, it usually means system services can't reliably `chdir` into a home-directory checkout (common with encrypted/automounted homes).

Use the provided `ipfs-accelerate-mcp.service` which sets `WorkingDirectory=/` and relies on `IPFS_ACCELERATE_REPO_DIR` instead.