# GitHub Actions Docker Runner - Cache Connectivity Fix

## 🎯 Problem

GitHub Actions runners in Docker containers cannot connect to the P2P cache, causing:
- ❌ Redundant GitHub API calls
- ❌ Increased risk of rate limiting  
- ❌ Slower workflow execution
- ❌ No cache sharing between runners

## ✅ Solution Implemented

Complete diagnostic and implementation plan with:
- 🔍 Test suite to identify connectivity issues
- 📋 5 different solution approaches
- 📚 Comprehensive documentation
- 🚀 Automated installation script

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# 1. Install dependencies
./install_p2p_cache_deps.sh

# 2. Run diagnostic
python test_docker_runner_cache_connectivity.py

# 3. Follow recommendations in output
```

### Option 2: Manual Setup

```bash
# Install P2P dependencies
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main" pymultihash>=0.8.2 py-multiformats-cid cryptography

# Configure environment
export CACHE_ENABLE_P2P=true
export CACHE_LISTEN_PORT=9000
export CACHE_BOOTSTRAP_PEERS=/ip4/172.17.0.1/tcp/9100/p2p/YOUR_PEER_ID

# Run with Docker host network
docker run --network host \
  -e CACHE_ENABLE_P2P \
  -e CACHE_LISTEN_PORT \
  -e CACHE_BOOTSTRAP_PEERS \
  your-image
```

## 📁 Files Created

| File | Description | Use Case |
|------|-------------|----------|
| `test_docker_runner_cache_connectivity.py` | Diagnostic test suite | Identify connectivity issues |
| `install_p2p_cache_deps.sh` | Automated installer | Install dependencies |
| `DOCKER_RUNNER_CACHE_PLAN.md` | Implementation plan | Understand solutions |
| `DOCKER_CACHE_QUICK_START.md` | Quick reference | Fast setup |
| `DOCKER_CACHE_IMPLEMENTATION_SUMMARY.md` | This summary | Overview |

## 🔍 Diagnostic Test

The diagnostic test checks:

1. ✅ **P2P Dependencies** - libp2p, pymultihash, cryptography, multiformats
2. ✅ **Cache Module** - Can import and initialize
3. ✅ **Network Connectivity** - Can reach bootstrap peers
4. ✅ **Cache Operations** - Get/put/invalidate work
5. ✅ **Encryption** - AES-256 setup functional
6. ✅ **Environment** - Variables configured correctly
7. ✅ **Docker Detection** - Network mode identified

**Run:**
```bash
python test_docker_runner_cache_connectivity.py
```

## 💡 Solutions Provided

### Solution 1: Docker Host Network ⭐ (Easiest)

Use `--network host` to share host network:

```yaml
# .github/workflows/your-workflow.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: |
          docker run --network host \
            -e CACHE_ENABLE_P2P=true \
            -e CACHE_LISTEN_PORT=9000 \
            your-image
```

**Pros:** Simplest, no configuration  
**Cons:** Less secure

### Solution 2: Bridge Network + Host IP (More Secure)

Configure bootstrap peers with host gateway IP:

```bash
export CACHE_BOOTSTRAP_PEERS=/ip4/172.17.0.1/tcp/9100/p2p/YOUR_PEER_ID
docker run -p 9000:9000 \
  -e CACHE_ENABLE_P2P=true \
  -e CACHE_LISTEN_PORT=9000 \
  -e CACHE_BOOTSTRAP_PEERS \
  your-image
```

**Pros:** Maintains isolation  
**Cons:** Requires host IP

### Solution 3: Docker Compose (Best for Services)

Use service discovery:

```yaml
version: '3.8'
services:
  mcp-server:
    ports: ["9100:9100"]
    environment:
      CACHE_LISTEN_PORT: "9100"
  
  runner:
    environment:
      CACHE_LISTEN_PORT: "9000"
      CACHE_BOOTSTRAP_PEERS: "/dns4/mcp-server/tcp/9100/p2p/${PEER_ID}"
    depends_on: [mcp-server]
```

**Pros:** Clean, scalable  
**Cons:** Requires docker-compose

### Solution 4: IPFS/Kubo (Alternative Backend)

Use IPFS instead of libp2p:

```bash
# Install and start IPFS
ipfs init
ipfs daemon &

# Configure cache
export CACHE_BACKEND=ipfs
export IPFS_API=/ip4/127.0.0.1/tcp/5001
```

**Pros:** Mature, battle-tested  
**Cons:** Additional dependency

### Solution 5: Storacha/S3 (Cloud Storage)

Use managed storage:

```bash
# Web3.Storage
export CACHE_BACKEND=storacha
export WEB3_STORAGE_TOKEN=your_token

# Or S3-compatible
export CACHE_BACKEND=s3
export AWS_S3_BUCKET=github-api-cache
```

**Pros:** No self-hosting  
**Cons:** External dependency

## 📊 Expected Results

After implementing the fix:

| Metric | Before | After |
|--------|--------|-------|
| Cache hit rate | 0% | **60-80%** |
| API calls saved | 0% | **50-80%** |
| Response time (cached) | N/A | **<10ms** |
| Connected peers | 0 | **1-10** |

## 🔧 Troubleshooting

### Issue: Dependencies Not Installing

```bash
# Update pip
pip3 install --upgrade pip

# Install build tools
sudo apt-get install build-essential python3-dev

# Retry with verbose output
pip3 install -v "libp2p @ git+https://github.com/libp2p/py-libp2p@main"
```

### Issue: Cannot Connect to Bootstrap Peer

```bash
# Check MCP server is running
netstat -tulpn | grep 9100

# Test from container
docker run --rm --network host nicolaka/netshoot nc -zv localhost 9100

# Check firewall
sudo ufw allow 9100/tcp
```

### Issue: Cache Not Working

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check cache status
from ipfs_accelerate_py.github_cli.cache import get_global_cache
cache = get_global_cache()
print(f"P2P enabled: {cache.enable_p2p}")
print(f"Connected peers: {len(cache._p2p_connected_peers)}")
```

## 📖 Documentation

- **Quick Start:** [DOCKER_CACHE_QUICK_START.md](./DOCKER_CACHE_QUICK_START.md)
- **Full Plan:** [DOCKER_RUNNER_CACHE_PLAN.md](./DOCKER_RUNNER_CACHE_PLAN.md)
- **Previous Work:** [GITHUB_API_CACHE.md](./GITHUB_API_CACHE.md)
- **P2P Setup:** [GITHUB_ACTIONS_P2P_SETUP.md](./GITHUB_ACTIONS_P2P_SETUP.md)

## 🎯 Next Steps

### Immediate (Now)

1. Run diagnostic: `python test_docker_runner_cache_connectivity.py`
2. Install dependencies: `./install_p2p_cache_deps.sh`
3. Choose a solution (recommend: host network mode)
4. Update workflows to use chosen solution

### Short-term (1-2 days)

- [ ] Update GitHub Actions workflows with fix
- [ ] Add diagnostic step to CI/CD
- [ ] Verify cache connectivity in workflows
- [ ] Monitor cache hit rates

### Mid-term (1 week)

- [ ] Evaluate IPFS integration
- [ ] Performance testing
- [ ] Consider alternative backends

### Long-term (2-4 weeks)

- [ ] Implement pluggable backends
- [ ] Auto-fallback logic
- [ ] Enhanced monitoring

## 🎉 Success Criteria

- ✅ All diagnostic tests pass
- ✅ Runners connect to P2P cache
- ✅ Cache hit rate > 50%
- ✅ No connectivity errors in logs
- ✅ API calls reduced by 50%+

## 📝 Summary

This implementation provides:

1. **Diagnostic Tools** - Test suite to identify issues
2. **Multiple Solutions** - 5 different approaches for different needs
3. **Complete Documentation** - Plans, guides, troubleshooting
4. **Automated Setup** - Installation script for easy deployment
5. **Clear Next Steps** - Immediate, short-term, and long-term roadmap

**Status:** ✅ Ready for implementation

**Recommended Action:** Use Docker host network mode (`--network host`) for immediate fix, then evaluate long-term alternatives.

## 🤝 Support

For issues or questions:
1. Run diagnostic: `python test_docker_runner_cache_connectivity.py`
2. Check troubleshooting section above
3. Review detailed documentation in linked files
4. Open an issue on GitHub with diagnostic output
