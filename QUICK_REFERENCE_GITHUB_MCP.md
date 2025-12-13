# Quick Reference: GitHub CLI and MCP Integration

## TL;DR

✅ **Implementation Complete** - All GitHub CLI tools integrated with MCP dashboard  
⚠️ **Action Required** - Refresh GitHub authentication: `gh auth refresh`

## Quick Start

### 1. Authenticate GitHub CLI
```bash
gh auth refresh -h github.com -s repo,workflow,read:org,gist
```

### 2. Test Integration
```bash
python3 test_github_mcp_integration.py
```

Expected: "✓ ALL TESTS PASSED!"

### 3. Start Dashboard
```bash
# If Flask is installed:
python3 -m ipfs_accelerate_py.mcp_dashboard --port 8899

# Then open: http://localhost:8899
```

## MCP Tools Available

| Tool Name | Description | Example Call |
|-----------|-------------|--------------|
| `gh_list_runners` | List self-hosted runners | `mcp.call('gh_list_runners', {repo: 'owner/repo'})` |
| `gh_create_workflow_queues` | Create workflow queues | `mcp.call('gh_create_workflow_queues', {since_days: 1})` |
| `gh_get_cache_stats` | Get cache statistics | `mcp.call('gh_get_cache_stats', {})` |
| `gh_get_auth_status` | Get auth status | `mcp.call('gh_get_auth_status', {})` |
| `gh_list_workflow_runs` | List workflow runs | `mcp.call('gh_list_workflow_runs', {repo: 'owner/repo'})` |
| `gh_get_runner_labels` | Get runner labels | `mcp.call('gh_get_runner_labels', {})` |

## Dashboard Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mcp/user` | GET | GitHub user info |
| `/api/mcp/cache/stats` | GET | Cache statistics |
| `/api/mcp/peers` | GET | P2P peer status |
| `/api/mcp/metrics` | GET | System metrics |

## Python Quick Tests

### Test User Info
```python
from ipfs_accelerate_py.mcp.tools.dashboard_data import get_user_info
print(get_user_info())
# Expected: {'authenticated': True, 'username': 'endomorphosis', ...}
```

### Test Cache Stats
```python
from ipfs_accelerate_py.github_cli.cache import get_global_cache
print(get_global_cache().get_stats())
# Shows: total entries, hit rate, P2P status
```

### Test Runner Info
```python
from ipfs_accelerate_py.github_cli import RunnerManager
mgr = RunnerManager()
print(f"Architecture: {mgr.get_system_architecture()}")
print(f"Labels: {mgr.get_runner_labels()}")
# Shows: x64, labels like 'self-hosted,linux,x64,docker,cuda,gpu'
```

## Files Reference

### New Files
- `ipfs_accelerate_py/mcp/tools/github_tools.py` - 6 MCP tools
- `test_github_mcp_integration.py` - Integration tests
- `GITHUB_CLI_MCP_INTEGRATION.md` - Full documentation
- `GITHUB_AUTH_SETUP.md` - Authentication guide

### Modified Files
- `ipfs_accelerate_py/mcp/tools/__init__.py` - Tool registration
- `ipfs_accelerate_py/static/js/dashboard.js` - Refresh functions

### Existing (Used)
- `ipfs_accelerate_py/github_cli/wrapper.py` - GitHub CLI wrapper
- `ipfs_accelerate_py/github_cli/cache.py` - Cache with P2P
- `ipfs_accelerate_py/mcp/tools/dashboard_data.py` - Dashboard data

## Common Commands

### Check Authentication
```bash
gh auth status
```

### Refresh Authentication
```bash
gh auth refresh -h github.com -s repo,workflow,read:org,gist
```

### Run Tests
```bash
python3 test_github_mcp_integration.py
```

### Check Cache Stats
```bash
python3 -c "from ipfs_accelerate_py.github_cli.cache import get_global_cache; print(get_global_cache().get_stats())"
```

### Check System Info
```bash
python3 -c "from ipfs_accelerate_py.github_cli import RunnerManager; m=RunnerManager(); print(f'Arch: {m.get_system_architecture()}, Labels: {m.get_runner_labels()}')"
```

## Troubleshooting

### Dashboard shows "Loading..."
**Problem:** GitHub authentication expired  
**Solution:** `gh auth refresh -h github.com -s repo,workflow,read:org,gist`

### "Rate limit exceeded"
**Problem:** Too many GitHub API calls  
**Solution:** Cache automatically handles this with stale data fallback

### P2P not working
**Problem:** libp2p not installed  
**Solution:** `pip install py-libp2p` (optional)

### Tests fail
**Problem:** Missing dependencies or authentication  
**Solution:** Check error message, install deps or refresh auth

## Cache Behavior

| Scenario | Cache Behavior | API Call |
|----------|---------------|----------|
| First request | Cache miss | ✓ Makes API call |
| Second request (< 5 min) | Cache hit | ✗ No API call |
| Request from peer | P2P cache hit | ✗ No API call |
| Rate limit error | Stale cache | ✗ No API call |
| Cache expired | Cache miss | ✓ Makes API call |

## Token Scopes Required

- ✅ `repo` - Access repositories
- ✅ `workflow` - Manage workflows
- ✅ `read:org` - Read organization data
- ✅ `gist` - Create gists (optional)

## Expected Dashboard Display

### User Information (After Auth)
```
👤 User Information
Username: endomorphosis
Authentication: ✓ Authenticated
Token Type: cli
```

### Cache Statistics
```
💾 Cache Statistics
Total Entries: 15
Cache Size: 2.3 MB
Hit Rate: 87%
```

### P2P Peer System
```
🌐 P2P Peer System
Status: Active / Disabled
Active Peers: 3 / 0
P2P Enabled: Yes / No
```

## Auto-Refresh

- **Interval:** 5 seconds
- **When:** On overview tab only
- **What:** User info, cache stats, peer status
- **Initial:** Loads immediately on page open

## Success Indicators

✅ **All 5 tests pass**  
✅ **Dashboard shows username**  
✅ **Cache hit rate increases**  
✅ **P2P peers connect** (if libp2p installed)  
✅ **API calls minimized**

## Getting Help

1. **Read full docs:** `GITHUB_CLI_MCP_INTEGRATION.md`
2. **Check auth:** `GITHUB_AUTH_SETUP.md`
3. **Run tests:** `python3 test_github_mcp_integration.py`
4. **Check logs:** Dashboard shows errors in browser console

## One-Line Test

```bash
gh auth refresh -h github.com -s repo,workflow,read:org,gist && python3 test_github_mcp_integration.py
```

Expected output: "✓ ALL TESTS PASSED!"
