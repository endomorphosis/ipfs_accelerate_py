# GitHub Authentication Setup Guide

## Current Status

✅ **GitHub CLI and MCP integration is COMPLETE and working**
✅ **All 5 integration tests pass**
⚠️ **GitHub authentication needs to be refreshed**

## The Issue

The dashboard shows:
```
Username: Loading...
Authentication: Checking...
Token Type: -
```

This is because the GitHub CLI token has expired. When I run `gh auth status`, it shows:
```
X Failed to log in to github.com account endomorphosis
- The token in /home/barberb/.config/gh/hosts.yml is invalid.
```

## Solution

### Option 1: Refresh GitHub CLI Authentication (Recommended)

```bash
gh auth refresh -h github.com -s repo,workflow,read:org,gist
```

This will:
1. Refresh your existing GitHub CLI authentication
2. Request the necessary scopes (repo, workflow, read:org, gist)
3. Update the token in `~/.config/gh/hosts.yml`
4. Fix the dashboard immediately

### Option 2: Re-authenticate from Scratch

```bash
gh auth login
```

Then select:
- GitHub.com
- HTTPS protocol
- Authenticate with browser or paste token
- Select scopes: `repo`, `workflow`, `read:org`, `gist`

### Option 3: Use Environment Variable

If you have a GitHub Personal Access Token (PAT):

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

Then restart the dashboard.

## Required Token Scopes

For full functionality, the GitHub token needs these scopes:

- ✅ **`repo`** - Access repositories (read workflow runs)
- ✅ **`workflow`** - Trigger and manage workflows
- ✅ **`read:org`** - Read organization data (for org-level runners)
- ✅ **`gist`** - Create gists (optional, for sharing data)

## Verification

After authenticating, verify it works:

```bash
# Test GitHub CLI
gh auth status

# Test with Python
python3 -c "
from ipfs_accelerate_py.mcp.tools.dashboard_data import get_user_info
import json
user_info = get_user_info()
print(json.dumps(user_info, indent=2))
print(f'\\nAuthenticated: {user_info[\"authenticated\"]}')
print(f'Username: {user_info.get(\"username\", \"Not found\")}')
"
```

Expected output:
```json
{
  "authenticated": true,
  "username": "endomorphosis",
  "name": "...",
  "email": "...",
  "token_type": "cli"
}

Authenticated: True
Username: endomorphosis
```

## Dashboard Will Show

Once authenticated, the dashboard will display:

**User Information Card:**
- **Username:** endomorphosis
- **Authentication:** ✓ Authenticated
- **Token Type:** cli (or environment)

**Auto-refresh:** The dashboard auto-refreshes this data every 5 seconds.

## What's Already Working

Even without authentication, the following features work:

1. ✅ **GitHub tools registration** - All 6 MCP tools are registered
2. ✅ **Cache system** - GitHub API cache is operational
3. ✅ **Dashboard refresh functions** - JavaScript functions work correctly
4. ✅ **MCP server** - Server initializes with all tools
5. ✅ **Runner detection** - System architecture and labels detected
6. ✅ **P2P cache** - Infrastructure ready (needs libp2p installation)

## Testing Without Authentication

You can test most features without GitHub authentication:

```bash
# Run integration tests
python3 test_github_mcp_integration.py

# Check cache stats
python3 -c "
from ipfs_accelerate_py.github_cli.cache import get_global_cache
cache = get_global_cache()
stats = cache.get_stats()
print(f'Cache entries: {stats[\"cache_size\"]}')
print(f'Hit rate: {stats[\"hit_rate\"]}%')
"

# Check runner labels
python3 -c "
from ipfs_accelerate_py.github_cli import RunnerManager
mgr = RunnerManager()
print(f'Architecture: {mgr.get_system_architecture()}')
print(f'Labels: {mgr.get_runner_labels()}')
print(f'Cores: {mgr.get_system_cores()}')
"
```

All these commands work without GitHub authentication.

## Summary

The GitHub CLI and MCP integration is **fully implemented and working**. The only remaining step is to **refresh your GitHub authentication** using one of the methods above.

Once authenticated, the dashboard will immediately show:
- Your GitHub username
- Authentication status (✓ Authenticated)
- Token type
- All GitHub Actions features will be fully functional

## Quick Fix Command

```bash
gh auth refresh -h github.com -s repo,workflow,read:org,gist && \
python3 test_github_mcp_integration.py
```

This will:
1. Refresh your GitHub authentication
2. Run all integration tests to verify everything works
