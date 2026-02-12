# MCP Server P2P Connection Setup Guide

## Problem

GitHub Actions runners were not connecting to the MCP server via libp2p for P2P cache sharing, causing excessive API calls. The runners need to know the MCP server's address to establish P2P connections.

## Solution

Configure the MCP server's libp2p multiaddr as a GitHub Secret so workflows can connect to it.

## Setup Steps

Before starting, ensure every node uses the same rendezvous repo:

```bash
export IPFS_ACCELERATE_GITHUB_REPO=endomorphosis/ipfs_accelerate_py
```

### 1. Get Your MCP Server's P2P Address

On the machine running your MCP server, get the peer ID and multiaddr:

```python
from ipfs_accelerate_py.github_cli.cache import get_global_cache

cache = get_global_cache()
stats = cache.get_stats()

peer_id = stats.get('peer_id')
print(f"Peer ID: {peer_id}")

# Get your public IP
import urllib.request
public_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
print(f"Public IP: {public_ip}")

# P2P cache port (default 9100)
p2p_port = 9100

# Construct multiaddr
multiaddr = f"/ip4/{public_ip}/tcp/{p2p_port}/p2p/{peer_id}"
print(f"\nBootstrap Multiaddr:\n{multiaddr}")
```

Example output:
```
Peer ID: QmYourPeerID123456789abcdef...
Public IP: 203.0.113.42
Bootstrap Multiaddr:
/ip4/203.0.113.42/tcp/9100/p2p/QmYourPeerID123456789abcdef...
```

### 2. Configure GitHub Secret

1. Go to your GitHub repository: `https://github.com/endomorphosis/ipfs_accelerate_py`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `MCP_P2P_BOOTSTRAP_PEERS`
5. Value: Your multiaddr from step 1 (the full `/ip4/...` string)
6. Click **Add secret**

### 3. Verify Configuration

After setting the secret, the next workflow run will show:

```
📦 Installing P2P cache dependencies...
🚀 Initializing P2P cache for runner communication...
✓ P2P bootstrap peers configured: /ip4/203.0.113.42/tcp/9100/p2p/QmYour...
✓ P2P cache initialized
```

If the secret is not set, you'll see:
```
⚠️  No bootstrap peers configured (set MCP_P2P_BOOTSTRAP_PEERS secret)
   Runners will not connect to MCP server for P2P cache sharing
```

## Network Requirements

### Firewall Configuration

Your MCP server must allow incoming connections on the P2P cache port (default 9100):

```bash
# For UFW (Ubuntu)
sudo ufw allow 9100/tcp

# For firewalld (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=9100/tcp
sudo firewall-cmd --reload

# For iptables
sudo iptables -A INPUT -p tcp --dport 9100 -j ACCEPT
sudo iptables-save
```

### NAT/Router Configuration

If your MCP server is behind NAT, configure port forwarding:
- External port: 9100 (TCP)
- Internal IP: Your MCP server's local IP
- Internal port: 9100 (TCP)

### Cloud Provider Security Groups

For cloud-hosted MCP servers (AWS, GCP, Azure):
- Add inbound rule for TCP port 9100
- Source: `0.0.0.0/0` (GitHub Actions IPs)

## Verifying P2P Connections

### On MCP Server

Check connected peers:

```python
from ipfs_accelerate_py.github_cli.cache import get_global_cache

cache = get_global_cache()
stats = cache.get_stats()

print(f"Connected peers: {stats.get('connected_peers', 0)}")
print(f"P2P enabled: {stats.get('p2p_enabled', False)}")
```

Or check the MCP server logs:
```bash
# If running as systemd service
sudo journalctl -u ipfs-accelerate-mcp -f | grep -i "peer\|p2p\|connect"

# If running directly
tail -f /path/to/mcp-server.log | grep -i "peer\|p2p\|connect"
```

You should see messages like:
```
✓ P2P host started, listening on port 9100
Peer ID: QmYour...
✓ Connected to peer: QmRunner...
```

### In GitHub Actions Logs

Check workflow logs for P2P initialization:
```
✓ P2P bootstrap peers configured: /ip4/203.0.113.42/tcp/9100/p2p/QmYour...
P2P host started, listening on port 9001
Connecting to 1 bootstrap peer(s)...
✓ Connected to bootstrap peer
```

## Troubleshooting

### Issue: "No bootstrap peers configured"

**Cause:** GitHub Secret not set or misconfigured

**Solution:** Follow step 2 above to set the `MCP_P2P_BOOTSTRAP_PEERS` secret

### Issue: "Timeout connecting to bootstrap peer"

**Causes:**
- Firewall blocking port 9100
- MCP server not running
- Wrong IP address in multiaddr
- NAT/port forwarding not configured

**Solutions:**
1. Verify firewall rules (see Network Requirements above)
2. Check MCP server is running: `systemctl status ipfs-accelerate-mcp`
3. Verify public IP: `curl https://api.ipify.org`
4. Test port accessibility: `nc -zv <public-ip> 9100` (from external network)

### Issue: "libp2p not available"

**Cause:** P2P dependencies not installed on runner

**Solution:** The workflow now automatically installs dependencies. If it still fails, check the workflow logs for installation errors.

### Issue: Connections established but no cache hits

**Causes:**
- Cache TTL expired
- Different cache keys between runner and MCP server
- Network latency issues

**Solutions:**
1. Check cache stats on MCP server
2. Increase cache TTL if needed
3. Monitor network latency between runners and MCP server

## Multiple Workflows

If you have multiple workflows (amd64, arm64, multiarch), they all use the same secret but different P2P ports:

- amd64-ci.yml: Port 9000 (connects to MCP server on 9100)
- arm64-ci.yml: Port 9001 (connects to MCP server on 9100)
- multiarch-ci.yml: Port 9002 (connects to MCP server on 9100)

All workflows connect to the same MCP server bootstrap peer.

## Security Notes

### Secret Protection

The `MCP_P2P_BOOTSTRAP_PEERS` secret is only visible to:
- Repository administrators
- Workflows with appropriate permissions

GitHub Actions masks secret values in logs automatically.

### P2P Encryption

All P2P cache messages are encrypted using:
- Algorithm: AES-256
- Key derivation: PBKDF2 from GitHub token
- Only runners with the same GitHub token can decrypt messages

Notes:
- Prefer setting `GH_TOKEN` for GitHub CLI auth.
- Some components also accept `GITHUB_TOKEN`.

### Network Security

- P2P connections use authenticated libp2p protocols
- Peer identity verified via cryptographic peer IDs
- Connections require matching GitHub access tokens

## Monitoring

### Cache Performance

Monitor API call reduction:

```python
from ipfs_accelerate_py.github_cli.cache import get_global_cache

stats = get_global_cache().get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"API calls saved: {stats['api_calls_saved']}")
print(f"Peer hits: {stats.get('peer_hits', 0)}")
```

### Expected Results

Without P2P: 500 API calls (5 runners × 100 calls)
With P2P: ~100 API calls (80% reduction)

## Getting Help

If you continue to see issues with P2P connections:

1. Check this guide's troubleshooting section
2. Verify all network requirements are met
3. Check MCP server logs for connection attempts
4. Review GitHub Actions workflow logs for error messages
5. Open an issue with logs and configuration details

## References

- [GitHub Actions P2P Setup](./GITHUB_ACTIONS_P2P_SETUP.md)
- [P2P Cache Quick Reference](./P2P_CACHE_QUICK_REF.md)
- [libp2p Connection Fix Summary](./LIBP2P_FIX_SUMMARY.md)
