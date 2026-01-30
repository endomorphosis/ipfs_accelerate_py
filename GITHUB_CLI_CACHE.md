# GitHub CLI P2P Caching Guide

## Overview

This repository uses an advanced GitHub CLI caching system with P2P (peer-to-peer) sharing capabilities powered by libp2p and IPFS. This system dramatically reduces GitHub API calls, prevents rate limiting, and speeds up development workflows by sharing cached responses across developers and CI runners.

## Features

### 🚀 Performance
- **Content-Addressed Caching**: Uses multiformats for intelligent cache invalidation
- **5-Minute Default TTL**: Configurable cache lifetime
- **Stale Cache Fallback**: Automatically uses stale cache when rate-limited
- **Parallel P2P Sharing**: Multiple developers share cache via libp2p

### 🔒 Security
- **Encrypted Cache**: Uses GitHub token as shared secret (PBKDF2+Fernet)
- **Token-Based Access**: Only users with same GitHub access can decrypt cache
- **No Token Storage**: Tokens never stored, only used for encryption key derivation

### 🌐 Distribution
- **P2P Network**: libp2p-based peer discovery and sharing
- **IPFS Integration**: Optional IPFS backend for persistence
- **Local Disk Cache**: Fast local cache with optional P2P sync
- **Multi-Runner Support**: CI runners automatically share cache

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Developer  │────▶│ gh_api_cached │────▶│ Local Cache │
│   Terminal  │     │     .py       │     │  (Disk)     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │ GitHub API   │     │   libp2p    │
                    │  (fallback)  │     │   P2P Net   │
                    └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                        ┌─────────────┐
                                        │    IPFS     │
                                        │  (optional) │
                                        └─────────────┘
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_ENABLE_P2P` | `true` | Enable P2P cache sharing |
| `CACHE_LISTEN_PORT` | `9100` | P2P listen port |
| `CACHE_DEFAULT_TTL` | `300` | Cache TTL in seconds (5 minutes) |
| `CACHE_DIR` | `~/.cache/github_cli` | Cache directory path |
| `CACHE_BOOTSTRAP_PEERS` | `""` | Comma-separated peer multiaddrs |
| `GH_TOKEN` or `GITHUB_TOKEN` | Required | GitHub authentication token |

### VSCode Configuration

Cache variables are automatically set in VSCode terminals via `.vscode/settings.json`:

```json
{
  "terminal.integrated.env.linux": {
    "CACHE_ENABLE_P2P": "true",
    "CACHE_LISTEN_PORT": "9100",
    "CACHE_DEFAULT_TTL": "300",
    "CACHE_DIR": "${workspaceFolder}/.cache/github_cli"
  }
}
```

### CI/CD Configuration

Workflows set cache variables in the `env` section:

```yaml
env:
  CACHE_ENABLE_P2P: 'true'
  CACHE_LISTEN_PORT: '9100'
  CACHE_DEFAULT_TTL: '300'
  CACHE_DIR: '/tmp/github_cli_cache'
```

## Usage

### Command Line

Use `tools/gh_api_cached.py` as a drop-in replacement for `gh api`:

```bash
# Standard GitHub API call (cached)
python tools/gh_api_cached.py user --jq '.login'

# Get repository info
python tools/gh_api_cached.py repos/owner/repo --jq '.name'

# Search issues
python tools/gh_api_cached.py "search/issues?q=repo:owner/repo+is:issue"

# Override cache TTL
python tools/gh_api_cached.py user --ttl 600  # 10-minute TTL

# Pass-through (no caching, includes headers)
python tools/gh_api_cached.py user --include
```

### In Python Scripts

```python
from ipfs_accelerate_py.github_cli.cache import get_global_cache

# Get the global cache instance (auto-configured)
cache = get_global_cache()

# Cache a GitHub API response
endpoint = "repos/owner/repo"
data = {"name": "repo", "owner": {"login": "owner"}}
cache.put("gh_api", data, ttl=300, endpoint, tuple())

# Retrieve from cache
cached_data = cache.get("gh_api", endpoint, tuple())

# Get stale cache (fallback)
stale_data = cache.get_stale("gh_api", endpoint, tuple())
```

### In Workflows

Replace direct `gh` calls with cached versions:

```yaml
# Before (direct gh call)
- run: gh issue view "$ISSUE_NUMBER" --json title,body

# After (cached gh call)
- run: python tools/gh_api_cached.py "repos/${{ github.repository }}/issues/$ISSUE_NUMBER" --jq '.title'
```

## Dependencies

### Required
- `cryptography` - For encrypted P2P cache
- `multiformats-cid` - For content-addressed caching

### Optional
- `libp2p` - For P2P cache sharing (highly recommended)
- `anyio` - For async operations

Install all dependencies:
```bash
pip install cryptography multiformats-cid libp2p-stubs
```

## Cache Behavior

### Cache Hit Flow
1. Check local disk cache
2. If expired, check P2P peers
3. If unavailable, fetch from GitHub API
4. Store in local cache and share with P2P peers

### Cache Miss Flow
1. Fetch from GitHub API
2. Store in local disk cache
3. Share with P2P network (if enabled)
4. Return to caller

### Rate Limit Handling
1. Detect GitHub API rate limit error
2. Attempt to use stale cache
3. If stale cache available, return it
4. Otherwise, return rate limit error

## P2P Network

### How It Works

1. **Peer Discovery**: Nodes discover each other via bootstrap peers or mDNS
2. **Content Sharing**: Cached responses are shared via libp2p pubsub
3. **Encryption**: All shared data is encrypted with GitHub token-derived key
4. **Decryption**: Only peers with same GitHub access can decrypt

### Bootstrap Peers

Configure bootstrap peers for faster discovery:

```bash
export CACHE_BOOTSTRAP_PEERS="/ip4/192.168.1.10/tcp/9100/p2p/QmHash1,/ip4/192.168.1.11/tcp/9100/p2p/QmHash2"
```

### Port Configuration

Default port is 9100. Change if needed:

```bash
export CACHE_LISTEN_PORT=9200
```

## Monitoring

### Check Cache Status

```bash
# View cache statistics
python -c "
from ipfs_accelerate_py.github_cli.cache import get_global_cache
cache = get_global_cache()
print(f'Cache hits: {cache.hits}')
print(f'Cache misses: {cache.misses}')
print(f'Hit rate: {cache.hits / (cache.hits + cache.misses) * 100:.1f}%')
"
```

### Monitor P2P Network

```bash
# Check P2P connectivity
python scripts/validation/monitor_p2p_cache.py
```

### Verify Cache Setup

```bash
# Run verification script
python scripts/validation/verify_p2p_cache.py
```

## Troubleshooting

### Cache Not Working

**Problem**: Cache always misses

**Solutions**:
1. Check environment variables: `env | grep CACHE_`
2. Verify cache directory exists: `ls -la $CACHE_DIR`
3. Check permissions: `ls -la ~/.cache/github_cli`
4. Verify GitHub token: `gh auth status`

### P2P Not Connecting

**Problem**: Cache not shared between peers

**Solutions**:
1. Check port is not blocked: `netstat -an | grep 9100`
2. Verify libp2p installed: `pip show libp2p`
3. Check firewall rules
4. Verify bootstrap peers: `echo $CACHE_BOOTSTRAP_PEERS`

### Rate Limiting Issues

**Problem**: Still hitting GitHub rate limits

**Solutions**:
1. Increase cache TTL: `export CACHE_DEFAULT_TTL=600`
2. Check cache hit rate (see Monitoring section)
3. Verify stale cache fallback is working
4. Ensure scripts use `gh_api_cached.py`

### Encryption Errors

**Problem**: Cannot decrypt P2P cache

**Solutions**:
1. Verify same GitHub token: `gh auth token`
2. Check cryptography library: `pip show cryptography`
3. Regenerate cache: `rm -rf $CACHE_DIR/*`

## Performance Tips

### Optimal Configuration

```bash
# For development (aggressive caching)
export CACHE_DEFAULT_TTL=600        # 10 minutes
export CACHE_ENABLE_P2P=true

# For CI (moderate caching)
export CACHE_DEFAULT_TTL=300        # 5 minutes
export CACHE_ENABLE_P2P=true

# For production (conservative)
export CACHE_DEFAULT_TTL=60         # 1 minute
export CACHE_ENABLE_P2P=false       # Disable P2P
```

### Cache Warming

Pre-populate cache before heavy operations:

```bash
# Warm up repository cache
python tools/gh_api_cached.py "repos/$REPO"
python tools/gh_api_cached.py "repos/$REPO/issues"
python tools/gh_api_cached.py "repos/$REPO/pulls"
```

### Selective Caching

Some endpoints shouldn't be cached:

```bash
# Use --include for non-cacheable requests
python tools/gh_api_cached.py "user/repos" --include
```

## Security Considerations

### Token Security
- Tokens are never stored in cache
- Tokens only used for key derivation
- Encryption key derived using PBKDF2 with 100,000 iterations
- Fernet (AES-128-CBC + HMAC-SHA256) used for encryption

### Network Security
- All P2P traffic is encrypted
- Only users with matching GitHub access can decrypt
- libp2p provides secure peer-to-peer channels
- Optional: Use VPN or private network for P2P

### Cache Poisoning
- Content-addressed caching prevents tampering
- Multiformats CIDs verify data integrity
- Encrypted payloads prevent unauthorized modification

## Advanced Features

### Custom Cache Backend

Implement custom cache storage:

```python
from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

cache = GitHubAPICache(
    cache_dir="/custom/path",
    enable_p2p=True,
    p2p_listen_port=9200,
    default_ttl=600
)
```

### Cache Inspection

```python
from pathlib import Path
import json

cache_dir = Path.home() / ".cache" / "github_cli"
for cache_file in cache_dir.glob("gh_api_*.json"):
    data = json.loads(cache_file.read_text())
    print(f"Stored: {data['stored_at']}")
    print(f"TTL: {data['ttl']}")
    print(f"Data: {data['data']}")
```

### IPFS Integration

Enable IPFS backend for persistent cache:

```bash
# Start IPFS daemon
ipfs daemon &

# Cache will automatically use IPFS if available
export CACHE_ENABLE_P2P=true
python tools/gh_api_cached.py user
```

## Best Practices

1. **Always use cached API in automation**: Replace `gh` with `gh_api_cached.py`
2. **Set appropriate TTLs**: Balance freshness vs. cache hits
3. **Enable P2P in teams**: Share cache across developers
4. **Monitor cache performance**: Track hit rates and adjust
5. **Use stale cache fallback**: Handle rate limits gracefully
6. **Warm cache before heavy ops**: Pre-populate frequently used data
7. **Secure your tokens**: Never commit tokens to git
8. **Review cache logs**: Understand caching patterns

## References

- [GitHub CLI Cache Implementation](ipfs_accelerate_py/github_cli/cache.py)
- [Cache Wrapper Script](tools/gh_api_cached.py)
- [VSCode Cache Wrapper](scripts/gh_cached_vscode.py)
- [P2P Verification Script](scripts/validation/verify_p2p_cache.py)
- [Cache Monitoring Script](scripts/validation/monitor_p2p_cache.py)

## Support

For issues or questions:
1. Check this documentation
2. Review [cache.py source code](ipfs_accelerate_py/github_cli/cache.py)
3. Run verification: `python scripts/validation/verify_p2p_cache.py`
4. Create an issue with the `github-cli-cache` label

---

**Last Updated**: January 30, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
