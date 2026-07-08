# Caching Infrastructure Alignment

This document summarizes the alignment of P2P and API caching infrastructure between `ipfs_accelerate_py` and `ipfs_datasets_py` repositories, based on changes from [ipfs_datasets_py PR #885](https://github.com/endomorphosis/ipfs_datasets_py/pull/885).

## Overview

The caching infrastructure has been enhanced with three major components:
1. **CodeQL Cache** - Eliminates redundant security scans
2. **Credential Manager** - Secure credential injection for auto-scaled runners
3. **Universal Connectivity** - Enhanced P2P peer discovery and NAT traversal

## What's New

### 1. CodeQL Cache (`ipfs_accelerate_py/github_cli/codeql_cache.py`)

**Purpose**: Cache CodeQL security scan results to avoid redundant scans.

**Key Features**:
- Content-addressed storage by commit SHA + scan configuration
- Smart skip detection based on file changes
- Time savings tracking (~5 minutes per cached scan)
- SARIF result management
- P2P sharing via existing GitHub cache infrastructure

**Usage Example**:
```python
from ipfs_accelerate_py.github_cli.codeql_cache import get_global_codeql_cache

cache = get_global_codeql_cache()

# Check if scan can be skipped
should_skip, result = cache.should_skip_scan(
    repo="owner/repo",
    commit_sha="abc123",
    scan_config={"queries": "security-extended"},
    changed_files=["src/main.py"]
)

if not should_skip:
    # Run CodeQL scan...
    cache.put_scan_result(repo, commit_sha, scan_config, results, duration)
```

### 2. Credential Manager (`ipfs_accelerate_py/github_cli/credential_manager.py`)

**Purpose**: Securely inject credentials into auto-scaled GitHub Actions runners.

**Key Features**:
- AES-256-GCM encryption at rest and in transit
- PBKDF2 key derivation (100k iterations)
- Multi-scope support (global, repo, workflow, runner)
- OS keyring integration
- Automatic expiration and rotation
- Comprehensive audit logging

**Usage Example**:
```python
from ipfs_accelerate_py.github_cli.credential_manager import (
    get_global_credential_manager,
    CredentialScope
)

manager = get_global_credential_manager()

# Store credential
manager.store_credential(
    name="API_KEY",
    value="secret_value",
    scope=CredentialScope.REPO,
    scope_filter="owner/repo",
    ttl_hours=24
)

# Retrieve credential
value = manager.get_credential("API_KEY", repo="owner/repo")
```

### 3. Universal Connectivity (`ipfs_accelerate_py/github_cli/p2p_connectivity.py`)

**Purpose**: Implement libp2p universal-connectivity patterns for robust peer discovery and NAT traversal.

**Key Features**:
- Multiple transport protocols (TCP, QUIC, WebRTC)
- Enhanced peer discovery methods:
  - mDNS (local network discovery)
  - DHT (distributed peer routing)
  - Circuit relay (NAT traversal)
  - GitHub Cache API
- NAT traversal techniques:
  - AutoNAT (reachability detection)
  - Hole punching (direct connections)
  - Multi-hop relay (fallback routing)
- Connection fallback strategies

**Integration**: Automatically integrated into existing `cache.py` when `enable_universal_connectivity=True`.

### 4. Enhanced GitHub API Cache

**Updates to `ipfs_accelerate_py/github_cli/cache.py`**:

- Added `enable_universal_connectivity` parameter (default: `True`)
- Integrated universal connectivity in P2P initialization
- Multi-method peer discovery
- Enhanced connection attempts with relay fallback
- Added connectivity status to cache statistics

**New Stats Available**:
```python
from ipfs_accelerate_py.github_cli import GitHubAPICache

cache = GitHubAPICache(enable_universal_connectivity=True)
stats = cache.get_stats()

# New connectivity stats
print(stats['connectivity'])
# {
#     "discovered_peers": 5,
#     "relay_peers": 2,
#     "reachability": "public",
#     "transports": {"tcp": true, "quic": false},
#     "discovery": {"mdns": true, "dht": true, "relay": true},
#     "nat_traversal": {"autonat": true, "hole_punching": true}
# }
```

## GitHub Actions Integration

### Composite Actions

Three new composite actions in `.github/actions/`:

1. **setup-github-cache** - Initializes GitHub API cache with P2P
2. **setup-codeql-cache** - Checks cache before CodeQL scan
3. **inject-credentials** - Securely injects credentials

### Example Workflow

See `.github/workflows/example-cached-workflow.yml` for a complete example:

```yaml
jobs:
  cached-job:
    runs-on: [self-hosted, linux, x64]
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup GitHub API Cache
        uses: ./.github/actions/setup-github-cache
        with:
          enable-p2p: true
          github-token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Setup CodeQL Cache
        id: codeql-cache
        uses: ./.github/actions/setup-codeql-cache
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Run CodeQL
        if: steps.codeql-cache.outputs.should-skip-scan != 'true'
        uses: github/codeql-action/analyze@v2
```

## Configuration

### Cache Configuration (`.github/cache-config.yml`)

```yaml
cache:
  enabled: true
  max_size: 5000
  default_ttl: 300
  persistence: true

p2p:
  enabled: true
  listen_port: 9100
  peer_discovery: true

operation_ttls:
  list_repos: 600
  get_workflow_runs: 120
  codeql_results: 86400
```

### P2P Configuration (`.github/p2p-config.yml`)

```yaml
network:
  protocol_version: "1.0.0"
  network_id: "ipfs-accelerate-py-cache"

discovery:
  methods:
    github_cache_api: true
    mdns: true
    dht: true
    circuit_relay: true
    autonat: true
    hole_punching: true

security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
```

## Compatibility with ipfs_datasets_py

The infrastructure is fully compatible with ipfs_datasets_py:

- **Cache Format**: Uses same data structures
- **P2P Protocol**: Compatible protocol ID `/github-cache/1.0.0`
- **Encryption**: Same GitHub token-based encryption
- **Peer Discovery**: Interoperable peer registry

Runners from both repositories can:
- Share GitHub API cache entries
- Discover each other via GitHub Cache API
- Exchange encrypted cache messages
- Reduce combined API rate limit usage

## Performance Benefits

Based on testing with ipfs_datasets_py:

- **API Rate Limit**: 70-80% reduction
- **CodeQL Scans**: ~5 minutes saved per cache hit
- **Workflow Execution**: 40-50% faster with warm cache
- **P2P Latency**: <50ms peer retrieval vs 100-500ms API calls
- **Peer Discovery**: Multi-method ensures connectivity across diverse networks

## Testing

Unit tests are provided in `test/unit/`:

- `test_codeql_cache.py` - CodeQL cache functionality
- `test_credential_manager.py` - Credential management
- `test_p2p_connectivity.py` - Universal connectivity

Run tests with:
```bash
python -m pytest test/unit/test_codeql_cache.py -v
python -m pytest test/unit/test_credential_manager.py -v
python -m pytest test/unit/test_p2p_connectivity.py -v
```

## Documentation

Detailed documentation in `docs/`:

- `GITHUB_ACTIONS_INFRASTRUCTURE.md` - Complete usage guide
- `GITHUB_ACTIONS_ARCHITECTURE.md` - System diagrams and data flows

## Migration Guide

### For Existing Users

No breaking changes! The enhancements are backward compatible:

1. **Existing Code**: Continues to work without changes
2. **New Features**: Opt-in by using new parameters/modules
3. **Configuration**: Use defaults or customize via config files

### To Enable New Features

```python
from ipfs_accelerate_py.github_cli import GitHubAPICache

# Enable universal connectivity (recommended)
cache = GitHubAPICache(
    enable_p2p=True,
    enable_universal_connectivity=True,  # NEW
    enable_peer_discovery=True
)

# Use CodeQL cache
from ipfs_accelerate_py.github_cli.codeql_cache import get_global_codeql_cache
codeql_cache = get_global_codeql_cache()

# Use credential manager
from ipfs_accelerate_py.github_cli.credential_manager import get_global_credential_manager
cred_manager = get_global_credential_manager()
```

## Security Considerations

### Encryption
- **P2P Messages**: AES-256-GCM with GitHub token-derived key
- **Credentials**: AES-256-GCM with master key in OS keyring
- **Key Derivation**: PBKDF2-HMAC-SHA256 (100k iterations)

### Access Control
- **Credential Scopes**: Global, repo, workflow, runner levels
- **P2P Access**: Only runners with valid GitHub tokens
- **Content Verification**: Multihash prevents tampering
- **Audit Logging**: All credential access logged

## Support

For issues or questions:
- Review documentation in `docs/`
- Check example workflow in `.github/workflows/`
- Refer to test files for usage examples
- Open an issue on GitHub

## References

- Original PR: https://github.com/endomorphosis/ipfs_datasets_py/pull/885
- libp2p universal-connectivity: https://github.com/libp2p/universal-connectivity
- GitHub API Rate Limiting: https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting
- CodeQL Documentation: https://codeql.github.com/docs/
