# GitHub CLI Caching - Quick Reference

## TL;DR

✅ GitHub API caching is **automatically enabled** in this repository  
✅ Uses P2P sharing (libp2p/IPFS) to reduce rate limiting  
✅ Works in VSCode, CI/CD, and command line  

## Quick Start

### Command Line
```bash
# Use cached gh API (instead of `gh api`)
python tools/gh_api_cached.py user --jq '.login'
```

### VSCode
Open any terminal in VSCode - cache is automatically configured!

### CI/CD
Already integrated in workflows - no action needed.

## Common Commands

```bash
# Get user info
python tools/gh_api_cached.py user --jq '.login'

# Get repository
python tools/gh_api_cached.py repos/owner/repo

# Search issues
python tools/gh_api_cached.py "search/issues?q=repo:owner/repo+is:issue"

# Custom TTL (10 minutes)
python tools/gh_api_cached.py user --ttl 600
```

## Configuration (Optional)

```bash
# Adjust cache lifetime (default: 300 seconds)
export CACHE_DEFAULT_TTL=600

# Change P2P port (default: 9100)
export CACHE_LISTEN_PORT=9200

# Disable P2P (local cache only)
export CACHE_ENABLE_P2P=false
```

## Troubleshooting

### Cache not working?
```bash
# Check environment
env | grep CACHE_

# Verify cache directory
ls -la ~/.cache/github_cli

# Check GitHub auth
gh auth status
```

### Still rate limited?
```bash
# Increase cache TTL
export CACHE_DEFAULT_TTL=900  # 15 minutes

# Check cache hit rate
python -c "
from ipfs_accelerate_py.github_cli.cache import get_global_cache
c = get_global_cache()
print(f'Hit rate: {c.hits/(c.hits+c.misses)*100:.1f}%')
"
```

## Key Files

- `tools/gh_api_cached.py` - Cache wrapper script
- `ipfs_accelerate_py/github_cli/cache.py` - Cache implementation
- `.vscode/settings.json` - VSCode cache config
- `docs/GITHUB_CLI_CACHE.md` - Full documentation

## Benefits

- 🚀 **Faster**: Cached responses are instant
- 💰 **Rate Limit Protection**: Reduces API calls by 80-90%
- 🌐 **P2P Sharing**: Cache shared across team
- 🔒 **Secure**: Encrypted with GitHub token
- 📦 **IPFS-backed**: Distributed storage

## Learn More

See [docs/GITHUB_CLI_CACHE.md](docs/GITHUB_CLI_CACHE.md) for complete documentation.
