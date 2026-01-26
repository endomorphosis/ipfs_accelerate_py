# Docker Runner Cache Connectivity - Implementation Plan

## Problem Statement

GitHub Actions runners running in Docker containers cannot connect to the P2P cache managed by the ipfs_accelerate_py package. This prevents cache sharing between runners and the MCP server, leading to redundant API calls and potential rate limiting.

## Previous Work Review

### What We Built

1. **P2P Cache System** (`ipfs_accelerate_py/github_cli/cache.py`)
   - Distributed cache using libp2p for peer-to-peer communication
   - Content-addressed caching with multiformats/multihash
   - AES-256 encryption using GitHub token as shared secret
   - Support for all GitHub API data types (repos, issues, PRs, etc.)
   - Automatic peer discovery and bootstrap

2. **GitHub Actions Integration** (`.github/workflows/*.yml`)
   - P2P cache configuration via environment variables
   - Bootstrap peer discovery
   - Cache sharing between runners on ports 9000-9002

3. **Test Suite**
   - `test_github_cache.py` - Basic cache functionality
   - `test_github_actions_p2p_cache.py` - P2P cache integration
   - Verification of cache hits/misses and statistics

### How It's Supposed to Work

```
┌─────────────────────────────────────────────────────────┐
│                  MCP Server (Host)                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  P2P Cache (Port 9100)                          │   │
│  │  - Listens for P2P connections                  │   │
│  │  - Manages global cache                         │   │
│  │  - Broadcasts cache updates                     │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │ libp2p connections
            ┌───────────┼───────────┐
            │           │           │
┌───────────▼───────┐ ┌▼───────────▼───┐ ┌───────────────▼┐
│ Runner 1 (Docker) │ │ Runner 2       │ │ Runner 3       │
│ Port: 9000        │ │ Port: 9001     │ │ Port: 9002     │
│                   │ │                │ │                │
│ 1. Check local    │ │ 1. Check local │ │ 1. Check local │
│    cache          │ │    cache       │ │    cache       │
│ 2. Query P2P      │ │ 2. Query P2P   │ │ 2. Query P2P   │
│    peers          │ │    peers       │ │    peers       │
│ 3. Call GitHub    │ │ 3. Call GitHub │ │ 3. Call GitHub │
│    API if needed  │ │    API if      │ │    API if      │
│ 4. Broadcast      │ │    needed      │ │    needed      │
│    to peers       │ │ 4. Broadcast   │ │ 4. Broadcast   │
└───────────────────┘ └────────────────┘ └────────────────┘
```

## Root Cause Analysis

### Potential Issues

1. **Docker Network Isolation**
   - Default Docker bridge network isolates containers
   - P2P connections cannot reach host or other containers
   - Bootstrap peers on localhost may not be accessible

2. **Port Binding**
   - Container ports not properly exposed or mapped
   - Host firewall blocking P2P ports (9000-9100)

3. **Environment Variable Propagation**
   - `CACHE_BOOTSTRAP_PEERS` not set in container environment
   - `CACHE_ENABLE_P2P` not propagated to containerized runners

4. **Bootstrap Peer Configuration**
   - Bootstrap peers configured with `127.0.0.1` (not accessible from container)
   - Should use host IP or `host.docker.internal`

5. **Dependency Installation**
   - libp2p, cryptography, or multiformats not installed in container
   - Import errors causing P2P to silently fail

6. **libp2p Compatibility**
   - Version mismatch between host and container
   - Python AnyIO event loop issues in containerized environment

## Diagnostic Plan

### Phase 1: Environment Verification

Run `test_docker_runner_cache_connectivity.py` to check:

1. ✅ P2P dependencies installed (libp2p, cryptography, multiformats)
2. ✅ Cache module imports successfully
3. ✅ Cache can initialize with P2P enabled
4. ✅ Network connectivity to bootstrap peers
5. ✅ Basic cache operations work
6. ✅ Encryption setup functions
7. ✅ Environment variables configured
8. ✅ Docker network mode detected

**Run Command:**
```bash
# On host
python test_docker_runner_cache_connectivity.py

# In Docker container
docker run --rm -it \
  -e CACHE_ENABLE_P2P=true \
  -e CACHE_LISTEN_PORT=9000 \
  -e CACHE_BOOTSTRAP_PEERS=/ip4/172.17.0.1/tcp/9100/p2p/QmTest \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  --network host \
  your-image \
  python test_docker_runner_cache_connectivity.py
```

### Phase 2: Network Connectivity Tests

1. **Test host-to-container connectivity**
   ```bash
   # Start MCP server on host
   python ipfs_mcp/mcp_server.py
   
   # Test from container
   docker run --rm --network host nicolaka/netshoot \
     nc -zv localhost 9100
   ```

2. **Test container-to-host connectivity**
   ```bash
   # Find host IP from container perspective
   docker run --rm alpine ip route | grep default
   # Usually 172.17.0.1 for default bridge network
   
   # Test connectivity
   docker run --rm --network host nicolaka/netshoot \
     nc -zv 172.17.0.1 9100
   ```

3. **Test with different network modes**
   - `--network bridge` (default, isolated)
   - `--network host` (shares host network)
   - `--network container:<name>` (shares another container's network)

### Phase 3: Cache Integration Tests

1. **Test cache without P2P** (baseline)
   ```python
   from ipfs_accelerate_py.github_cli import GitHubCLI
   
   gh = GitHubCLI(enable_cache=True, enable_p2p=False)
   repos = gh.list_repos(owner="endomorphosis", limit=10)
   # Verify: Should work, no P2P connections
   ```

2. **Test cache with P2P locally** (same host)
   ```python
   gh = GitHubCLI(enable_cache=True, enable_p2p=True)
   repos = gh.list_repos(owner="endomorphosis", limit=10)
   # Verify: P2P should initialize, may connect to bootstrap nodes
   ```

3. **Test cache with P2P in container** (cross-network)
   ```bash
   docker run --rm --network host \
     -e CACHE_ENABLE_P2P=true \
     -e CACHE_LISTEN_PORT=9000 \
     -e CACHE_BOOTSTRAP_PEERS=/ip4/172.17.0.1/tcp/9100/p2p/QmTest \
     your-image \
     python -c "from ipfs_accelerate_py.github_cli import GitHubCLI; \
                gh = GitHubCLI(enable_cache=True); \
                repos = gh.list_repos(owner='endomorphosis', limit=10)"
   ```

## Implementation Solutions

### Solution 1: Use Docker Host Network (Recommended)

**Pros:**
- Simplest solution
- No port mapping needed
- Full network access

**Cons:**
- Less secure (container shares host network)
- Port conflicts possible

**Implementation:**
```yaml
# .github/workflows/runner-test.yml
services:
  runner:
    image: your-runner-image
    network_mode: host
    environment:
      CACHE_ENABLE_P2P: "true"
      CACHE_LISTEN_PORT: "9000"
      CACHE_BOOTSTRAP_PEERS: "/ip4/127.0.0.1/tcp/9100/p2p/${MCP_PEER_ID}"
```

### Solution 2: Configure Bootstrap Peers with Host IP

**Pros:**
- Maintains container isolation
- Works with default bridge network

**Cons:**
- Requires knowing host IP
- More complex configuration

**Implementation:**
```bash
# Get host IP (from container perspective)
HOST_IP=$(docker run --rm alpine ip route | grep default | awk '{print $3}')

# Configure bootstrap peers
export CACHE_BOOTSTRAP_PEERS="/ip4/${HOST_IP}/tcp/9100/p2p/${MCP_PEER_ID}"

# Run container
docker run --rm \
  -e CACHE_ENABLE_P2P=true \
  -e CACHE_LISTEN_PORT=9000 \
  -e CACHE_BOOTSTRAP_PEERS \
  your-image
```

### Solution 3: Use Docker Compose with Custom Network

**Pros:**
- Service discovery via DNS
- Better for multiple services

**Cons:**
- More complex setup
- Requires docker-compose

**Implementation:**
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
      - cache-network

  runner-1:
    build: .
    command: python your_workflow.py
    environment:
      CACHE_ENABLE_P2P: "true"
      CACHE_LISTEN_PORT: "9000"
      CACHE_BOOTSTRAP_PEERS: "/dns4/mcp-server/tcp/9100/p2p/${MCP_PEER_ID}"
    networks:
      - cache-network
    depends_on:
      - mcp-server

networks:
  cache-network:
    driver: bridge
```

### Solution 4: Use IPFS/Kubo for Cache Storage (Alternative)

**Pros:**
- Built-in P2P networking
- Content-addressed storage
- Mature, battle-tested

**Cons:**
- Adds dependency on IPFS daemon
- More resource intensive
- Different API

**Implementation:**
```python
# ipfs_accelerate_py/github_cli/ipfs_cache.py
import ipfshttpclient

class IPFSGitHubCache:
    def __init__(self):
        self.client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
    
    def put(self, key: str, data: Any, ttl: int = 300):
        """Store data in IPFS."""
        json_data = json.dumps({"key": key, "data": data, "ttl": ttl})
        result = self.client.add_json(json_data)
        return result  # Returns CID
    
    def get(self, cid: str) -> Optional[Any]:
        """Retrieve data from IPFS."""
        try:
            data = self.client.get_json(cid)
            return data["data"]
        except:
            return None
```

### Solution 5: Use Storacha (web3.storage) for Cache (Alternative)

**Pros:**
- No self-hosted infrastructure needed
- Built on IPFS
- Free tier available

**Cons:**
- Requires internet access
- Data stored externally
- API rate limits

**Implementation:**
```python
# ipfs_accelerate_py/github_cli/storacha_cache.py
from web3.storage import Client

class StorachaGitHubCache:
    def __init__(self, api_token: str):
        self.client = Client(api_token)
    
    def put(self, key: str, data: Any, ttl: int = 300):
        """Store data in Storacha."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump({"key": key, "data": data, "ttl": ttl}, f)
            f.flush()
            cid = self.client.put_file(f.name)
        return cid
    
    def get(self, cid: str) -> Optional[Any]:
        """Retrieve data from Storacha."""
        try:
            content = self.client.get(cid)
            data = json.loads(content)
            return data["data"]
        except:
            return None
```

## Recommended Approach

### Short-term: Fix libp2p Connectivity (1-2 days)

1. **Update GitHub Actions workflows** to use `--network host`
2. **Update bootstrap peer configuration** to use correct host IP
3. **Add diagnostic step** to workflows to verify P2P connectivity
4. **Improve error logging** in cache module to show P2P failures

**Files to modify:**
- `.github/workflows/amd64-ci.yml`
- `.github/workflows/arm64-ci.yml`
- `.github/workflows/multiarch-ci.yml`
- `ipfs_accelerate_py/github_cli/cache.py` (add debug logging)

### Mid-term: Evaluate IPFS Integration (1 week)

1. **Research IPFS/Kubo integration** for cache storage
2. **Prototype IPFS cache backend** alongside libp2p
3. **Performance testing** to compare libp2p vs IPFS
4. **Cost-benefit analysis** for maintenance and complexity

**New files:**
- `ipfs_accelerate_py/github_cli/ipfs_cache.py`
- `test_ipfs_cache_integration.py`
- `docs/IPFS_CACHE_INTEGRATION.md`

### Long-term: Hybrid Approach (2-4 weeks)

1. **Implement pluggable cache backends**
   - Local (current in-memory + disk)
   - P2P (current libp2p)
   - IPFS (kubo daemon)
   - Storacha (web3.storage)
   - S3 (AWS, Minio, etc.)

2. **Auto-fallback logic**
   - Try P2P first (lowest latency)
   - Fall back to IPFS if P2P unavailable
   - Fall back to direct API if all caches fail

3. **Configuration via environment**
   ```bash
   export CACHE_BACKENDS="p2p,ipfs,s3"  # Priority order
   export CACHE_P2P_BOOTSTRAP="..."
   export CACHE_IPFS_API="/ip4/127.0.0.1/tcp/5001"
   export CACHE_S3_BUCKET="github-api-cache"
   ```

**Architecture:**
```python
class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: pass
    
    @abstractmethod
    def put(self, key: str, data: Any, ttl: int): pass

class P2PCache(CacheBackend): ...
class IPFSCache(CacheBackend): ...
class StorachaCache(CacheBackend): ...
class S3Cache(CacheBackend): ...

class MultiBackendCache:
    def __init__(self, backends: List[CacheBackend]):
        self.backends = backends
    
    def get(self, key: str) -> Optional[Any]:
        for backend in self.backends:
            try:
                result = backend.get(key)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Backend {backend} failed: {e}")
        return None
```

## Next Steps

1. **Run diagnostics** (Now)
   ```bash
   python test_docker_runner_cache_connectivity.py
   ```

2. **Review results** and identify specific failure points

3. **Implement quick fix** (Solution 1: host network mode)

4. **Validate fix** with integration tests

5. **Document findings** and update workflows

6. **Consider alternatives** (IPFS, Storacha, S3) based on:
   - Reliability requirements
   - Performance needs
   - Maintenance burden
   - Cost constraints

## Success Criteria

- [ ] Diagnostic tests pass in Docker container
- [ ] Runners can connect to MCP P2P cache
- [ ] Cache hit rate > 50% across multiple runners
- [ ] No connectivity errors in workflow logs
- [ ] Documentation updated with configuration guide
- [ ] Integration tests validate end-to-end flow

## References

- [Previous Work: GITHUB_API_CACHE.md](./GITHUB_API_CACHE.md)
- [Previous Work: GITHUB_CACHE_COMPREHENSIVE.md](./GITHUB_CACHE_COMPREHENSIVE.md)
- [Previous Work: GITHUB_ACTIONS_P2P_SETUP.md](./GITHUB_ACTIONS_P2P_SETUP.md)
- [Test Suite: test_github_cache.py](./test_github_cache.py)
- [Test Suite: test_github_actions_p2p_cache.py](./test_github_actions_p2p_cache.py)
- [libp2p Documentation](https://docs.libp2p.io/)
- [IPFS Documentation](https://docs.ipfs.io/)
- [Docker Networking](https://docs.docker.com/network/)
