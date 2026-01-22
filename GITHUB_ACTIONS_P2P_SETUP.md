# GitHub Actions P2P Cache Setup

This document explains how the P2P cache system works in GitHub Actions to reduce API rate limiting by sharing cached responses between runners.

## Overview

The P2P cache system allows GitHub Actions runners to share cached API responses, dramatically reducing the number of API calls made. This is especially important when running multiple workflows simultaneously.

## How It Works

### 1. Peer Discovery

When a runner starts, it:
1. Initializes the P2P cache with a unique peer ID
2. Registers itself in a shared peer registry (file-based or environment variables)
3. Discovers other active runners
4. Connects to discovered peers via libp2p

### 2. Cache Sharing

When a runner makes a GitHub API call:
1. First checks local cache
2. If not found, checks connected peers' caches
3. If still not found, makes the actual API call
4. Broadcasts the new cache entry to all connected peers

### 3. Automatic Cleanup

- Peer entries expire after 30 minutes of inactivity
- Stale peers are automatically cleaned up
- Runners reconnect if disconnected

## Configuration

### Workflow Configuration

Add these environment variables to your workflow:

```yaml
env:
  # Enable P2P cache sharing between runners
  CACHE_ENABLE_P2P: 'true'
  # Port for P2P communication (use different ports for different workflows)
  CACHE_LISTEN_PORT: '9000'
```

### Workflow Steps

Add a P2P initialization step at the beginning of your job:

```yaml
jobs:
  your-job:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Initialize P2P Cache
      id: p2p-init
      run: |
        echo "🚀 Initializing P2P cache for runner communication..."
        chmod +x .github/scripts/p2p_peer_bootstrap.sh
        .github/scripts/p2p_peer_bootstrap.sh init
        echo "CACHE_ENABLE_P2P=true" >> $GITHUB_ENV
        echo "CACHE_LISTEN_PORT=9000" >> $GITHUB_ENV
    
    # Your other steps...
```

## Bootstrap Peers

### Automatic Fallback to libp2p Bootstrap Nodes

If no custom bootstrap peers are configured and no other runners are discovered, the system automatically falls back to standard libp2p bootstrap nodes:

- `/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN`
- `/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa`
- `/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb`
- `/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt`

This ensures runners can connect to the libp2p network even without explicit configuration.

### Static Bootstrap Peers (Optional)

For faster peer discovery within your organization, you can configure static bootstrap peers:

```yaml
env:
  CACHE_ENABLE_P2P: 'true'
  CACHE_LISTEN_PORT: '9000'
  # Comma-separated list of peer multiaddrs
  CACHE_BOOTSTRAP_PEERS: '/ip4/10.0.0.1/tcp/9000/p2p/QmPeer123...'
```

**Note:** When custom bootstrap peers are provided via `CACHE_BOOTSTRAP_PEERS`, the libp2p fallback nodes are not used. This allows you to keep traffic within your private network if desired.

## Benefits

### Reduced API Calls

Without P2P cache:
- Each runner makes its own API calls
- 5 runners × 100 calls = 500 API calls total
- Higher chance of hitting rate limits

With P2P cache:
- First runner makes API calls and shares results
- Other runners use cached data
- 5 runners × ~20 calls = ~100 API calls total (80% reduction)

### Faster Workflow Execution

- Runners get data from local peers instead of GitHub API
- Lower latency (local network vs internet)
- Faster workflow completion

### Better Resource Utilization

- Fewer network requests
- Lower bandwidth usage
- Better API quota management

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GitHub Actions                         │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Runner 1 │◄───┤ Runner 2 │◄───┤ Runner 3 │          │
│  │ (AMD64)  │    │ (ARM64)  │    │ (AMD64)  │          │
│  │ Port:9000│    │ Port:9001│    │ Port:9000│          │
│  └─────┬────┘    └─────┬────┘    └─────┬────┘          │
│        │               │               │                │
│        └───────────────┼───────────────┘                │
│                        │                                │
│                   P2P Cache Network                      │
│                (libp2p connections)                      │
│                                                           │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Shared Cache:                                           │
│  • Repository lists                                      │
│  • Workflow runs                                         │
│  • Job details                                           │
│  • Artifact information                                  │
│  • Rate limit status                                     │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Port Assignment

To avoid conflicts, use different ports for different architectures or workflows:

- AMD64 workflows: Port 9000
- ARM64 workflows: Port 9001
- Multi-arch workflows: Port 9002

Example:

```yaml
# amd64-ci.yml
env:
  CACHE_LISTEN_PORT: '9000'

# arm64-ci.yml
env:
  CACHE_LISTEN_PORT: '9001'
```

## Monitoring

### View Cache Statistics

Check cache performance during workflow execution:

```bash
python3 -c "
from ipfs_accelerate_py.github_cli.cache import get_global_cache
stats = get_global_cache().get_stats()
print(f'Hit rate: {stats[\"hit_rate\"]:.1%}')
print(f'API calls saved: {stats[\"api_calls_saved\"]}')
print(f'Connected peers: {stats.get(\"connected_peers\", 0)}')
"
```

### Debug Peer Connections

List discovered peers:

```bash
.github/scripts/p2p_peer_bootstrap.sh discover
```

## Troubleshooting

### No Peers Discovered

**Problem:** Runners can't find each other

**Solutions:**
1. Check that P2P is enabled (`CACHE_ENABLE_P2P=true`)
2. Verify the system is using libp2p bootstrap nodes (check logs for "Using standard libp2p bootstrap node(s)")
3. If using custom `CACHE_BOOTSTRAP_PEERS`, verify the multiaddr format is correct
4. Check network connectivity - libp2p bootstrap nodes require internet access
5. Ensure libp2p dependencies are installed: `pip install libp2p cryptography py-multiformats-cid`

### Port Conflicts

**Problem:** P2P fails to start due to port already in use

**Solutions:**
1. Use different ports for different workflows
2. Check for other services using the port
3. Use dynamic port assignment

### High API Usage Despite P2P

**Problem:** API calls aren't being reduced

**Solutions:**
1. Verify P2P is enabled and peers are connected
2. Check cache TTL settings (may be too short)
3. Ensure all runners are using the same cache implementation
4. Monitor peer connection status

## Security

### Encryption

All P2P messages are encrypted using AES-256 with the GitHub token as the shared secret. Only runners with the same GitHub token can decrypt messages.

### Network Security

- P2P connections are authenticated
- Only runners in the same repository/workflow can connect
- Peer identity is verified using libp2p peer IDs

### Data Privacy

- Cached data never leaves the runner infrastructure
- No data is sent to external services
- Peer discovery is local to the workflow run

## Requirements

### Dependencies

The following Python packages are required for P2P cache:

```bash
pip install libp2p cryptography py-multiformats-cid
```

These are optional dependencies - the cache works without them but falls back to local-only caching.

### GitHub Actions Runners

- Self-hosted runners: Full P2P support
- GitHub-hosted runners: Limited P2P (runners may not be on same network)

## Implementation Details

### Files

- `.github/scripts/p2p_peer_bootstrap.sh` - Bootstrap script for peer setup
- `ipfs_accelerate_py/github_cli/p2p_bootstrap_helper.py` - Simplified peer discovery
- `ipfs_accelerate_py/github_cli/cache.py` - Main cache implementation with P2P

### Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CACHE_ENABLE_P2P` | `true` | Enable P2P cache sharing |
| `CACHE_LISTEN_PORT` | `9100` | Port for libp2p listener |
| `CACHE_BOOTSTRAP_PEERS` | (none) | Comma-separated peer multiaddrs |
| `CACHE_DEFAULT_TTL` | `300` | Cache TTL in seconds |
| `CACHE_DIR` | `~/.cache/github_cli` | Cache storage directory |

## Future Improvements

1. **Dynamic peer discovery via GitHub API**
   - Use GitHub Actions artifacts for peer registry
   - Automatic peer discovery without configuration

2. **Intelligent cache distribution**
   - Distribute cache load across peers
   - Prioritize peers by response time

3. **Enhanced monitoring**
   - Real-time peer connection status
   - Cache hit rate visualization
   - API call reduction metrics

4. **Cross-workflow caching**
   - Share cache between different workflows
   - Repository-wide cache coordination

## Examples

See the following workflows for complete examples:

- `.github/workflows/amd64-ci.yml` - AMD64 with P2P cache
- `.github/workflows/arm64-ci.yml` - ARM64 with P2P cache

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review workflow logs for P2P initialization messages
3. Run verification: `python3 verify_p2p_cache.py`
4. Open an issue on GitHub
