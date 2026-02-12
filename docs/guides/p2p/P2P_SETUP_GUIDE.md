# P2P Peer System Setup Guide

This guide explains how to set up and verify the P2P peer system for distributed cache sharing in IPFS Accelerate.

## Overview

The P2P peer system uses [py-libp2p](https://github.com/libp2p/py-libp2p) to enable distributed cache sharing across multiple instances. This allows:

- **Reduced API calls**: Share cached GitHub API responses across peers
- **Faster responses**: Get data from nearby peers instead of API servers
- **Better resilience**: Continue working even if one peer goes down

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Linux, macOS, or Windows with WSL

## Installation

### 1. Install py-libp2p and Dependencies

The latest version of py-libp2p is actively maintained on GitHub:
https://github.com/libp2p/py-libp2p

Install using pip:

```bash
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main" pymultihash>=0.8.2
```

### 2. Install System Dependencies

Some of libp2p's dependencies require system libraries:

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    libgmp-dev \
    build-essential
```

#### macOS:
```bash
brew install gmp
```

#### Windows (WSL):
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    libgmp-dev \
    build-essential
```

### 3. Install IPFS Accelerate with P2P Support

```bash
pip install -r requirements.txt
```

The requirements.txt includes:
- `libp2p @ git+https://github.com/libp2p/py-libp2p@main` - P2P networking (tracks upstream main)
- `pymultihash>=0.8.2` - Required by libp2p for peer IDs
- `multiformats>=0.3.0` - Content addressing

## Verification

### 1. Check libp2p Installation

```bash
python3 -c "from libp2p import new_host; print('✅ libp2p installed successfully')"
```

### 2. Check pymultihash Installation

```bash
python3 -c "import pymultihash; print('✅ pymultihash installed successfully')"
```

### 3. Test P2P System via Python

```python
from ipfs_accelerate_py.github_cli.cache import get_global_cache

# Get cache instance with P2P enabled
cache = get_global_cache()

# Check P2P status
stats = cache.get_stats()
print(f"P2P Enabled: {stats.get('p2p_enabled', False)}")
print(f"P2P Peers: {stats.get('p2p_peers', 0)}")
```

### 4. Check via MCP Dashboard

1. Start the MCP server:
   ```bash
   python3 -m ipfs_accelerate_py.mcp_dashboard
   ```

2. Open browser to: http://localhost:8899

3. Navigate to the "Overview" tab

4. Look for the "🌐 P2P Peer System" section:
   - **Status**: Should show "✓ Active" when P2P is working
   - **Active Peers**: Shows number of connected peers
   - **P2P Enabled**: Should show "✓ Enabled"

## Configuration

### Environment Variables

Configure P2P behavior using environment variables:

```bash
# Enable P2P (default: true)
export CACHE_ENABLE_P2P=true

# Set listen port (default: 9100)
export CACHE_LISTEN_PORT=9100

# Ensure all nodes rendezvous in the same repo (format: owner/repo)
export IPFS_ACCELERATE_GITHUB_REPO=owner/repo

# Provide GitHub auth for `gh` so the Issue-backed peer registry works
# (preferred) for GitHub CLI:
export GH_TOKEN=...
# (optional) some codepaths still read GITHUB_TOKEN:
export GITHUB_TOKEN=...

# Set bootstrap peers (optional)
export CACHE_BOOTSTRAP_PEERS="/ip4/192.168.1.100/tcp/9100/p2p/QmPeerID1,/ip4/192.168.1.101/tcp/9100/p2p/QmPeerID2"
```

### Programmatic Configuration

```python
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

# Create cache with P2P enabled
cache = GitHubAPICache(
    enable_p2p=True,
  p2p_listen_port=9100,
    bootstrap_peers=[
    "/ip4/192.168.1.100/tcp/9100/p2p/QmPeerID1"
    ]
)
```

## Troubleshooting

### P2P Shows "Disabled" in Dashboard

**Cause**: libp2p or pymultihash not installed

**Solution**:
```bash
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main" pymultihash>=0.8.2
```

### ImportError: No module named 'pymultihash'

**Cause**: pymultihash package not installed

**Solution**:
```bash
pip install pymultihash>=0.8.2
```

### Build Errors During Installation

**Cause**: Missing system dependencies for cryptography libraries

**Solution** (Ubuntu/Debian):
```bash
sudo apt-get install -y python3-dev libgmp-dev build-essential
```

### P2P Enabled but No Peers Connect

**Possible causes**:
1. **Firewall blocking**: Ensure port 9100 (or configured port) is open
2. **No bootstrap peers**: Set `CACHE_BOOTSTRAP_PEERS` environment variable
3. **Network isolation**: Peers must be able to reach each other's IP addresses

**Multi-node checklist (must match on every node)**:
1. `CACHE_ENABLE_P2P=true`
2. `IPFS_ACCELERATE_GITHUB_REPO=owner/repo`
3. GitHub auth available to `gh` (set `GH_TOKEN` or run `gh auth login`)
4. TCP reachability to the listen port (default `9100`) between nodes

**Check connectivity**:
```bash
# On peer 1, get the multiaddr
python3 -c "from ipfs_accelerate_py.github_cli.cache import get_global_cache; cache = get_global_cache(); print(cache._p2p_listen_addr)"

# On peer 2, try to connect using the multiaddr from peer 1
export CACHE_BOOTSTRAP_PEERS="/ip4/<peer1-ip>/tcp/9000/p2p/<peer1-id>"
```

### Check Logs for P2P Issues

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Start MCP server and check logs
python3 -m ipfs_accelerate_py.mcp_dashboard
```

Look for messages like:
- `✓ P2P cache sharing enabled`
- `P2P host started on port XXXX`
- `Connected to X peer(s)`

## Architecture

### How P2P Cache Works

1. **Initialization**: Each instance starts a libp2p host on the configured port
2. **Discovery**: Peers discover each other via bootstrap peers or mDNS
3. **Connection**: Peers establish encrypted connections using libp2p
4. **Sharing**: When data is cached, it's broadcast to all connected peers
5. **Validation**: Content-addressed hashing ensures data integrity

### Security

- **Encryption**: All peer-to-peer communication is encrypted
- **Authentication**: Peers are authenticated using libp2p peer IDs
- **Content Validation**: Multihash ensures data hasn't been tampered with

## API Reference

### MCP Server API Endpoints

#### Get P2P Peer Status
```
GET /api/mcp/peers
```

Response:
```json
{
  "enabled": true,
  "active": true,
  "peer_count": 3,
  "peers": [
    {
      "peer_id": "QmPeerID1",
      "runner_name": "runner-1",
      "listen_port": 9000,
      "last_seen": "2025-12-13T03:00:00Z"
    }
  ]
}
```

#### Get Cache Stats (includes P2P info)
```
GET /api/mcp/cache/stats
```

Response:
```json
{
  "available": true,
  "total_entries": 150,
  "total_size_mb": 2.5,
  "hit_rate": 0.85,
  "p2p_enabled": true,
  "p2p_peers": 3
}
```

### Python API

```python
from ipfs_accelerate_py.mcp.tools.dashboard_data import (
    get_peer_status,
    get_cache_stats
)

# Get P2P peer status
peer_status = get_peer_status()
print(f"P2P enabled: {peer_status['enabled']}")
print(f"Active peers: {peer_status['peer_count']}")

# Get cache stats
cache_stats = get_cache_stats()
print(f"P2P peers: {cache_stats['p2p_peers']}")
```

## Related Documentation

- [py-libp2p GitHub Repository](https://github.com/libp2p/py-libp2p)
- [libp2p Specifications](https://github.com/libp2p/specs)
- [Universal Connectivity Pattern](https://github.com/libp2p/universal-connectivity)

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review the logs with `LOG_LEVEL=DEBUG`
3. Open an issue on GitHub with:
   - Python version (`python --version`)
   - OS and version
   - Complete error messages
   - Output from verification commands
