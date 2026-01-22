# Docker Runner Cache - Quick Start Guide

## TL;DR - Fast Setup

```bash
# 1. Install dependencies
pip install libp2p>=0.4.0 pymultihash>=0.8.2 py-multiformats-cid cryptography

# 2. Run diagnostic
python test_docker_runner_cache_connectivity.py

# 3. Configure environment (for Docker runners)
export CACHE_ENABLE_P2P=true
export CACHE_LISTEN_PORT=9000
export CACHE_BOOTSTRAP_PEERS=/ip4/172.17.0.1/tcp/9100/p2p/YOUR_MCP_PEER_ID

# 4. Run with host network
docker run --network host \
  -e CACHE_ENABLE_P2P \
  -e CACHE_LISTEN_PORT \
  -e CACHE_BOOTSTRAP_PEERS \
  your-image
```

## Problem: Cache Not Working in Docker

**Symptom:** GitHub Actions runners in Docker can't connect to P2P cache

**Root Cause:** Network isolation prevents container → host connections

## Quick Fixes

### Fix 1: Use Host Network (Easiest) ⭐

```yaml
# .github/workflows/your-workflow.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests in Docker
        run: |
          docker run --network host \
            -e CACHE_ENABLE_P2P=true \
            -e CACHE_LISTEN_PORT=9000 \
            your-image
```

**Pros:** Simple, no configuration needed  
**Cons:** Less secure (shares host network)

### Fix 2: Configure Host IP (More Secure)

```bash
# Get host IP from container perspective
HOST_IP=$(docker network inspect bridge | jq -r '.[0].IPAM.Config[0].Gateway')

# Or use this for GitHub Actions
HOST_IP=172.17.0.1  # Default Docker bridge gateway

# Configure bootstrap peers with host IP
export CACHE_BOOTSTRAP_PEERS="/ip4/${HOST_IP}/tcp/9100/p2p/YOUR_PEER_ID"

# Run container with bridge network
docker run --rm \
  -p 9000:9000 \
  -e CACHE_ENABLE_P2P=true \
  -e CACHE_LISTEN_PORT=9000 \
  -e CACHE_BOOTSTRAP_PEERS \
  your-image
```

**Pros:** Maintains isolation, more secure  
**Cons:** Requires knowing host IP

### Fix 3: Use Docker Compose (Best for Multiple Services)

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    command: python ipfs_mcp/mcp_server.py
    ports:
      - "9100:9100"
    environment:
      CACHE_ENABLE_P2P: "true"
      CACHE_LISTEN_PORT: "9100"
    networks:
      - cache-net

  runner:
    build: .
    command: python your_script.py
    environment:
      CACHE_ENABLE_P2P: "true"
      CACHE_LISTEN_PORT: "9000"
      # Use DNS name instead of IP
      CACHE_BOOTSTRAP_PEERS: "/dns4/mcp-server/tcp/9100/p2p/${PEER_ID}"
    networks:
      - cache-net
    depends_on:
      - mcp-server

networks:
  cache-net:
    driver: bridge
```

**Pros:** Clean, scalable, service discovery  
**Cons:** Requires docker-compose

## Alternative: Use IPFS Instead

If P2P connectivity is too complex, consider using IPFS/Kubo:

### Option A: Local IPFS Daemon

```bash
# 1. Install IPFS
wget https://dist.ipfs.io/kubo/latest/kubo_linux-amd64.tar.gz
tar -xvzf kubo_linux-amd64.tar.gz
cd kubo
sudo bash install.sh

# 2. Initialize and start IPFS
ipfs init
ipfs daemon &

# 3. Use IPFS cache backend
export CACHE_BACKEND=ipfs
export IPFS_API=/ip4/127.0.0.1/tcp/5001
```

### Option B: Web3.Storage (Storacha)

```bash
# 1. Get API token from https://web3.storage
export WEB3_STORAGE_TOKEN=your_token_here

# 2. Use Storacha cache backend
export CACHE_BACKEND=storacha
```

### Option C: S3-Compatible Storage

```bash
# Works with AWS S3, MinIO, Backblaze B2, etc.
export CACHE_BACKEND=s3
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_S3_BUCKET=github-api-cache
export AWS_S3_REGION=us-east-1
```

## Troubleshooting

### Issue: "libp2p not installed"

```bash
# Install P2P dependencies
pip install libp2p>=0.4.0 pymultihash>=0.8.2 py-multiformats-cid cryptography
```

### Issue: "Cannot connect to bootstrap peer"

```bash
# Check if MCP server is running
netstat -tulpn | grep 9100

# Test connectivity from container
docker run --rm --network host nicolaka/netshoot nc -zv localhost 9100

# Check firewall
sudo ufw status
sudo ufw allow 9100/tcp
```

### Issue: "P2P enabled but not connecting"

1. Check logs for errors:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   from ipfs_accelerate_py.github_cli import GitHubCLI
   gh = GitHubCLI(enable_cache=True)
   ```

2. Verify bootstrap peer format:
   ```bash
   # Correct format:
   /ip4/192.168.1.100/tcp/9100/p2p/QmYourPeerID...
   
   # Wrong:
   192.168.1.100:9100  # ❌ Not a multiaddr
   /ip4/127.0.0.1/tcp/9100  # ❌ Missing peer ID
   ```

3. Check network mode:
   ```bash
   # If using default bridge network, use host IP
   export CACHE_BOOTSTRAP_PEERS=/ip4/172.17.0.1/tcp/9100/p2p/QmPeer...
   
   # If using host network, use localhost
   export CACHE_BOOTSTRAP_PEERS=/ip4/127.0.0.1/tcp/9100/p2p/QmPeer...
   ```

### Issue: "Cache hits = 0"

1. Check if cache is enabled:
   ```python
   from ipfs_accelerate_py.github_cli.cache import get_global_cache
   cache = get_global_cache()
   print(f"P2P enabled: {cache.enable_p2p}")
   print(f"Connected peers: {len(cache._p2p_connected_peers)}")
   ```

2. Verify same operations:
   ```python
   # First call
   repos1 = gh.list_repos(owner="test", limit=10)
   
   # Second call (should hit cache)
   repos2 = gh.list_repos(owner="test", limit=10)
   
   # Check stats
   stats = cache.get_stats()
   print(f"Hit rate: {stats['hit_rate']:.1%}")
   ```

## Verification

After setup, verify everything works:

```bash
# Run full diagnostic
python test_docker_runner_cache_connectivity.py

# Expected output:
# ✅ P2P Dependencies Check
# ✅ Cache Module Import
# ✅ Cache Initialization
# ✅ Network Connectivity
# ✅ Cache Operations
# ✅ Encryption Setup
# ✅ Environment Variables
# ✅ Docker Network Mode
# 
# 🎉 All tests passed!
```

## Performance Expectations

With P2P cache working:

- **Cache hit rate:** 60-80% (depends on workload)
- **API call reduction:** 50-80%
- **Response time:** <10ms for cached data vs 100-500ms for API calls
- **Connected peers:** 1-10 (depends on concurrent runners)

Example stats:
```
Total requests: 1000
Cache hits: 750 (75%)
Cache misses: 250 (25%)
API calls saved: 750
API calls made: 250
Connected peers: 3
```

## GitHub Actions Example

Complete workflow with P2P cache:

```yaml
name: Test with P2P Cache

on: [push]

env:
  CACHE_ENABLE_P2P: 'true'
  CACHE_LISTEN_PORT: '9000'
  CACHE_BOOTSTRAP_PEERS: ${{ secrets.MCP_P2P_BOOTSTRAP_PEERS }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install P2P dependencies
      run: |
        pip install libp2p>=0.4.0 pymultihash>=0.8.2 \
                    py-multiformats-cid cryptography
    
    - name: Run diagnostic
      run: python test_docker_runner_cache_connectivity.py
    
    - name: Run tests with cache
      run: |
        python -c "
        from ipfs_accelerate_py.github_cli import GitHubCLI
        gh = GitHubCLI(enable_cache=True)
        repos = gh.list_repos(owner='test', limit=10)
        print(f'Found {len(repos)} repos')
        "
    
    - name: Show cache stats
      run: |
        python -c "
        from ipfs_accelerate_py.github_cli.cache import get_global_cache
        stats = get_global_cache().get_stats()
        print(f'Hit rate: {stats[\"hit_rate\"]:.1%}')
        print(f'API calls saved: {stats[\"api_calls_saved\"]}')
        print(f'Connected peers: {len(stats.get(\"connected_peers\", []))}')
        "
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Run diagnostic test
3. ✅ Choose fix (host network, bridge + host IP, or docker-compose)
4. ✅ Update workflows
5. ✅ Verify cache connectivity
6. ⏳ Monitor performance
7. ⏳ Consider alternatives (IPFS, Storacha, S3) if needed

## Support

- **Full Plan:** See [DOCKER_RUNNER_CACHE_PLAN.md](./DOCKER_RUNNER_CACHE_PLAN.md)
- **Previous Work:** See [GITHUB_API_CACHE.md](./GITHUB_API_CACHE.md)
- **P2P Setup:** See [GITHUB_ACTIONS_P2P_SETUP.md](./GITHUB_ACTIONS_P2P_SETUP.md)
- **Run Tests:** `python test_docker_runner_cache_connectivity.py`
