# libp2p Universal Connectivity Setup Guide

This guide explains how to test libp2p connectivity for IPFS Accelerate following the [universal-connectivity](https://github.com/libp2p/universal-connectivity) pattern.

## Overview

IPFS Accelerate now includes libp2p support for peer-to-peer connectivity, enabling distributed cache sharing across:
- Multiple local instances
- GitHub Actions runners
- Remote machines
- Browser environments (future WebRTC support)

## Prerequisites

The libp2p Python library is automatically installed with IPFS Accelerate:

```bash
pip install ipfs_accelerate_py
```

Or install manually:
```bash
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main"
```

## Quick Start

### 1. Start MCP Server with P2P

Start the MCP server which automatically enables P2P:

```bash
ipfs-accelerate mcp start
```

The server will:
- Create a libp2p host on port 9100 (configurable)
- Generate a unique Peer ID
- Register with local peer discovery
- Attempt to connect to IPFS bootstrap nodes
- Display peer information in the dashboard

### 2. Check P2P Status

Visit the dashboard at `http://localhost:9000/dashboard` to see:
- P2P enabled status
- Number of connected peers
- Peer information

Or check via API:
```bash
curl http://localhost:9000/api/mcp/peers
```

### 3. Test Connectivity

Run the universal connectivity test:

```bash
# Automated test
python test_universal_connectivity.py --automated

# Interactive mode (allows connecting to other peers)
python test_universal_connectivity.py
```

## Connecting Multiple Instances

### Local Testing

1. **Start first instance:**
   ```bash
   ipfs-accelerate mcp start --port 9000
   ```
   Note the Peer ID displayed in logs.

2. **Start second instance:**
   ```bash
   # Use different port for second instance
   ipfs-accelerate mcp start --port 9001
   ```

3. **Connect instances:**
   The instances will automatically discover each other via the local peer registry at:
   ```
   ~/.cache/p2p_peers/
   ```

### Manual Connection

To manually connect to a specific peer:

```python
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

# Create cache with P2P enabled
cache = GitHubAPICache(
    enable_p2p=True,
    p2p_listen_port=9100,
    p2p_bootstrap_peers=[
        "/ip4/127.0.0.1/tcp/9200/p2p/<PEER_ID_HERE>"
    ]
)
```

### GitHub Actions

When running in GitHub Actions, peers automatically register and discover each other using the peer registry system.

Environment variables for configuration:
```yaml
env:
  CACHE_ENABLE_P2P: "true"
  P2P_LISTEN_PORT: "9100"
  GITHUB_REPOSITORY: "${{ github.repository }}"
```

## Architecture

### libp2p Compatibility Layer

IPFS Accelerate includes a compatibility layer (`libp2p_compat.py`) that fixes issues between libp2p and newer dependency versions:

1. **multihash.Func** - Creates enum-like class for hash algorithm codes
2. **multihash.digest()** - Implements function with `.encode()` method support
3. **Multiaddr conversion** - Automatic string to Multiaddr object conversion

### Peer Discovery

Three methods are used for peer discovery:

1. **Local Registry** - File-based peer registry in `~/.cache/p2p_peers/`
2. **Environment Variables** - `P2P_BOOTSTRAP_PEERS` for static peers
3. **Bootstrap Nodes** - Standard libp2p bootstrap nodes

### Supported Transports

Currently supported:
- ✅ TCP
- ✅ Local (Unix sockets)

Planned:
- 🔄 QUIC
- 🔄 WebRTC (for browser connectivity)
- 🔄 WebTransport
- 🔄 WebSockets

## Configuration

### Environment Variables

```bash
# Enable P2P
export CACHE_ENABLE_P2P=true

# Set listen port
export P2P_LISTEN_PORT=9100

# Add bootstrap peers (comma-separated)
export P2P_BOOTSTRAP_PEERS="/ip4/192.168.1.100/tcp/9100/p2p/QmPeerID1,/ip4/192.168.1.101/tcp/9100/p2p/QmPeerID2"

# Set GitHub repository for runner discovery
export GITHUB_REPOSITORY="owner/repo"
```

### Programmatic Configuration

```python
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

cache = GitHubAPICache(
    enable_p2p=True,
    p2p_listen_port=9100,
    p2p_bootstrap_peers=[
        "/ip4/192.168.1.100/tcp/9100/p2p/QmPeerID"
    ],
    github_repo="owner/repo",
    enable_peer_discovery=True
)
```

## Testing Connectivity

### Test Script

```bash
# Run all connectivity tests
python test_universal_connectivity.py --automated

# Interactive mode
python test_universal_connectivity.py
```

### Manual Testing

```python
import anyio
from libp2p import new_host
from multiaddr import Multiaddr

async def test_connection():
    # Create host
    addr = Multiaddr('/ip4/0.0.0.0/tcp/9200')
    host = new_host(listen_addrs=[addr])
    
    print(f"Peer ID: {host.get_id().pretty()}")
    print(f"Listening on: {addr}")
    
    # Keep running
    await anyio.Event().wait()

anyio.run(test_connection)
```

## Troubleshooting

### libp2p not found

```bash
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main"
```

### Port already in use

Change the port:
```bash
ipfs-accelerate mcp start --port 9001
```

Or set environment variable:
```bash
export P2P_LISTEN_PORT=9101
```

### No peers connecting

1. Check firewall settings
2. Verify peer multiaddrs are correct
3. Check peer registry:
   ```bash
   ls -la ~/.cache/p2p_peers/
   cat ~/.cache/p2p_peers/peer_*.json
   ```

### Bootstrap peer connection failures

This is normal and expected. The bootstrap peers use DNS addresses that may not resolve immediately. Local peer discovery will still work.

## Security

### Encryption

P2P messages are encrypted using the GitHub token as a shared secret. This ensures only runners with the same GitHub access can decrypt messages.

To enable encryption:
```bash
export GITHUB_TOKEN=ghp_your_token_here
# Or use gh CLI:
gh auth login
```

Without encryption, P2P will still work but messages won't be encrypted.

## Performance

### Metrics

Check P2P performance via the dashboard:
- Peer count
- Cache hit rate
- P2P cache hits
- Network traffic

### Tuning

Optimize for your use case:

```python
cache = GitHubAPICache(
    enable_p2p=True,
    p2p_listen_port=9100,
    default_ttl=300,  # Cache TTL in seconds
    max_cache_size=1000  # Max cache entries
)
```

## Examples

See the `examples/` directory for:
- Multi-peer connectivity test
- Cache sharing demonstration
- GitHub Actions integration

## References

- [libp2p Specifications](https://github.com/libp2p/specs)
- [Universal Connectivity](https://github.com/libp2p/universal-connectivity)
- [libp2p Python](https://github.com/libp2p/py-libp2p)
- [Multiaddr](https://github.com/multiformats/multiaddr)

## Contributing

Contributions welcome! Areas for improvement:
- WebRTC support for browser connectivity
- WebTransport implementation
- NAT traversal improvements
- DHT integration
- Performance optimizations
