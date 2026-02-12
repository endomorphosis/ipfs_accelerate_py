# Systemd Services for IPFS Accelerate

This directory contains optional systemd unit files to run the MCP API and the Performance Dashboard on boot.

## Install

```bash
# Copy units
sudo cp deployments/systemd/ipfs-accelerate.service /etc/systemd/system/
sudo cp deployments/systemd/ipfs-accelerate-dashboard.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable ipfs-accelerate.service
sudo systemctl enable ipfs-accelerate-dashboard.service

# Start them now
sudo systemctl start ipfs-accelerate.service
sudo systemctl start ipfs-accelerate-dashboard.service

# Check status
systemctl status ipfs-accelerate.service
systemctl status ipfs-accelerate-dashboard.service
```

## Configuration
- Units default to:
  - MCP API on port 9000, binds `0.0.0.0`.
  - Performance Dashboard on port 8080, binds `0.0.0.0`.
  - P2P cache sharing (libp2p) on port 9100.
  - TaskQueue P2P RPC (MCP-over-P2P) on port 9100.
- Adjust environment or ports by editing the unit files.

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

## Logs
- All logs go to journald by default:
  - `sudo journalctl -u ipfs-accelerate -f`
  - `sudo journalctl -u ipfs-accelerate-dashboard -f`

## Firewall
```bash
sudo ufw allow 9000/tcp
sudo ufw allow 8080/tcp
sudo ufw allow 9100/tcp
# mDNS peer discovery (LAN)
sudo ufw allow 5353/udp
```