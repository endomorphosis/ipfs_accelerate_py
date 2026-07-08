# Self-Hosted Runners Dashboard Fix

## Problem Statement

The IPFS Accelerate MCP dashboard was showing "No self-hosted runners configured" even when self-hosted runners were registered and active in GitHub repositories. This was because the `gh_list_runners` MCP tool was falling back to listing local Docker containers when called without repo/org parameters, instead of querying the GitHub API.

## Root Cause

In `shared/operations.py`, the `list_runners()` method had this logic:

```python
def list_runners(self, repo=None, org=None):
    # If no repo/org specified, list active containerized runners instead
    if not repo and not org:
        return self.list_active_runners()  # Only checks local Docker containers
```

When the dashboard JavaScript (`static/js/github-workflows.js`) called `gh_list_runners` without parameters:

```javascript
const result = await this.mcp.request('tools/call', {
    name: 'gh_list_runners',
    arguments: {}  // No repo or org specified
});
```

It would only return local Docker containers, not the actual self-hosted runners registered via GitHub API.

## Solution

Modified `list_runners()` to aggregate runners from all accessible repositories when no repo/org is specified:

### Key Changes

1. **Auto-detect Organization**: Use the GitHub authentication token to automatically determine the authenticated user/organization
2. **Filter by Activity**: Only check repositories that have been updated in the past 24 hours for better performance
3. **Aggregate Results**: Query runners from multiple repositories and combine them into a single list
4. **Add Context**: Include repository information with each runner for better visibility
5. **Maintain Caching**: Keep efficient caching with 30-second TTL

### Implementation

Added new method `_list_all_runners()` in `shared/operations.py`:

```python
def _list_all_runners(self) -> Dict[str, Any]:
    """
    List all self-hosted runners from recently active repositories.
    
    1. Uses GitHub token to determine authenticated user/org
    2. Gets repositories updated in the past day
    3. Queries runners for each active repository
    4. Aggregates all runners and caches the result
    """
```

### Benefits

1. **Performance**: Only queries repos updated in past 24 hours (typically 10-20 repos vs 100+)
2. **Relevance**: Shows runners from actively-used repositories
3. **Context**: Each runner includes repository information
4. **Caching**: Results cached for 30 seconds to minimize API calls
5. **Robustness**: Handles errors gracefully with partial results

## Testing

### Prerequisites

- GitHub CLI (`gh`) installed and authenticated, OR
- `GITHUB_TOKEN` environment variable set with appropriate permissions

### Manual Testing Steps

1. **Start the MCP Server**:
   ```bash
   cd /home/devel/ipfs_accelerate_py
   source ipfs_env/bin/activate
   python mcp_jsonrpc_server.py --host 0.0.0.0 --port 9000
   ```

2. **Open Dashboard**: Navigate to `http://localhost:9000` in your browser

3. **Check GitHub Workflows Tab**:
   - Click on "GitHub Workflows" tab
   - Expand the "Self-Hosted Runners" section
   - Click "🔄 Refresh" button

4. **Expected Results**:
   - Should see a list of self-hosted runners from your GitHub repositories
   - Each runner shows:
     - Runner name
     - Repository (owner/repo)
     - Status (online/offline)
     - OS type
     - Labels

### Programmatic Testing

```python
from shared import GitHubOperations, SharedCore

# Initialize
shared_core = SharedCore()
github_ops = GitHubOperations(shared_core)

# List all runners
result = github_ops.list_runners()

print(f"Found {result['count']} runners from {result['repos_checked']} repositories")
for runner in result['runners']:
    print(f"  - {runner['name']} ({runner['repository']}) - {runner['status']}")
```

### API Testing

```bash
# Using curl to test the MCP endpoint
curl -X POST http://localhost:9000/api/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gh_list_runners",
    "arguments": {}
  }'
```

## Expected Response Format

```json
{
  "runners": [
    {
      "id": 123,
      "name": "runner-ubuntu-latest",
      "status": "online",
      "busy": false,
      "os": "Linux",
      "labels": [
        {"name": "self-hosted"},
        {"name": "ubuntu"},
        {"name": "x64"}
      ],
      "repository": "endomorphosis/ipfs_accelerate_py",
      "owner": "endomorphosis"
    }
  ],
  "count": 1,
  "repos_checked": 15,
  "repos_with_runners": 1,
  "authenticated_user": "endomorphosis",
  "operation": "list_all_runners",
  "timestamp": 1699724675.123,
  "success": true,
  "cached": false
}
```

## Troubleshooting

### "No runners found"

If the dashboard still shows no runners:

1. **Check Authentication**:
   ```bash
   gh auth status
   # or
   echo $GITHUB_TOKEN
   ```

2. **Check Repository Activity**: Ensure at least one repo has been updated in the past 24 hours
   
3. **Check Permissions**: GitHub token needs `repo` and `admin:org` scopes for runner access

4. **Check Logs**: Look for errors in the MCP server console:
   ```bash
   # Look for lines containing "list_all_runners" or "Error fetching runners"
   tail -f /var/log/ipfs-accelerate-mcp.log
   ```

### "Partial errors" in response

Some repositories may return errors if:
- You don't have admin access to the repository
- The repository doesn't have Actions enabled
- API rate limits are hit

These are logged but don't prevent other runners from being listed.

## Performance Considerations

- **Caching**: Results cached for 30 seconds to minimize API calls
- **Time-based filtering**: Only queries repos updated in past 24 hours
- **API calls**: Approximately 1 call per active repository + 1 for repo list
- **Typical load**: 15-20 API calls per refresh (under rate limit)

## Files Modified

- `shared/operations.py`: Added `_list_all_runners()` method and modified `list_runners()`
- No changes required to JavaScript or HTML - existing code now works correctly

## Compatibility

- Backward compatible: Existing code calling `list_runners(repo="owner/repo")` still works
- New feature: Calling `list_runners()` without parameters now returns aggregated results
- Cache-friendly: P2P cache sharing still works across services

## Related Issues

- Fixes dashboard showing "No self-hosted runners configured"
- Addresses GitHub API integration for runner management
- Improves visibility into self-hosted runner infrastructure
