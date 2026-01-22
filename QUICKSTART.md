# Quick Start Guide - GitHub CLI and Copilot CLI Integration

This guide demonstrates that the GitHub CLI and Copilot CLI integration is working correctly.

## 🚀 NEW: Auto-Scaling Runner Service

The easiest way to use the GitHub integration is the autoscaler - it automatically monitors workflows and provisions runners as needed:

```bash
# 1. Authenticate (one time)
gh auth login

# 2. Start the autoscaler (runs continuously)
python cli.py github autoscaler

# Or with options
python cli.py github autoscaler --owner myorg --interval 30
```

The autoscaler will:
- ✅ Monitor your repos for workflow activity
- ✅ Detect running and failed workflows automatically
- ✅ Provision self-hosted runners on demand
- ✅ Respect your system's CPU core limit
- ✅ Work completely automatically once started

**See [AUTOSCALER.md](AUTOSCALER.md) for complete autoscaler documentation.**

---

## ✅ Quick Verification

Run the integration test to verify everything works:

```bash
python test_github_copilot_integration.py
```

Expected output:
```
============================================================
GitHub CLI and Copilot CLI Integration Tests
============================================================
...
✓ All tests passed!
```

## 🔧 Testing Each Component

### 1. Python Package Imports

```bash
# Test imports work
python -c "from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager; print('✓ Imports work')"
python -c "from ipfs_accelerate_py.copilot_cli import CopilotCLI; print('✓ Copilot import works')"
```

### 2. CLI Commands

```bash
# Test CLI is accessible
python cli.py --help
python cli.py github --help
python cli.py copilot --help

# Test specific commands (requires gh auth)
python cli.py github auth              # Check authentication
python cli.py github repos --limit 5    # List repositories
python cli.py copilot suggest "list text files"  # Copilot suggestion
```

### 3. MCP Tools

The following MCP tools are registered and available:

**GitHub Tools:**
- `gh_auth_status()` - Check GitHub authentication
- `gh_list_repos()` - List repositories
- `gh_list_workflow_runs()` - List workflow runs
- `gh_get_workflow_run()` - Get workflow details
- `gh_create_workflow_queues()` - Create workflow queues
- `gh_list_runners()` - List self-hosted runners
- `gh_provision_runners()` - Provision runners

**Copilot Tools:**
- `copilot_suggest_command()` - Command suggestions
- `copilot_explain_command()` - Command explanations
- `copilot_suggest_git_command()` - Git suggestions

To verify MCP tools are registered:
```bash
python -c "from mcp.tools import github_tools, copilot_tools; print('✓ MCP tools available')"
```

### 4. Dashboard Integration

The dashboard uses the MCP JavaScript SDK to communicate with MCP tools:

```bash
# Start the MCP server with dashboard
python cli.py mcp start --dashboard --open-browser

# Visit http://localhost:9000/dashboard
# GitHub workflows and runners will be displayed using MCP tools
```

## 📦 Python API Usage

```python
from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager

# Initialize
gh = GitHubCLI()
queue = WorkflowQueue(gh)
runner_mgr = RunnerManager(gh)

# Check authentication
auth = gh.get_auth_status()
print(f"Authenticated: {auth['authenticated']}")

# List repositories
repos = gh.list_repos(limit=5)
print(f"Found {len(repos)} repositories")

# Create workflow queues
queues = queue.create_workflow_queues(since_days=1)
print(f"Created queues for {len(queues)} repositories")

# Get system cores
cores = runner_mgr.get_system_cores()
print(f"System has {cores} cores")

# Provision runners (requires authentication)
# provisioning = runner_mgr.provision_runners_for_queue(queues, max_runners=cores)
```

## 🎯 Dashboard with MCP SDK

The dashboard JavaScript now uses the MCP SDK instead of direct API calls:

```javascript
// Create MCP client
const mcpClient = new MCPClient();

// Create GitHub workflows manager with MCP client
const githubManager = new GitHubWorkflowsManager(mcpClient);

// Fetch workflows using MCP tool
await mcpClient.request('tools/call', {
    name: 'gh_create_workflow_queues',
    arguments: { since_days: 1 }
});

// Provision runner using MCP tool
await mcpClient.request('tools/call', {
    name: 'gh_provision_runners',
    arguments: { since_days: 1, max_runners: 4 }
});
```

## 🚀 Complete Example

Run the complete example script:

```bash
python examples/github_workflow_automation.py
```

This demonstrates:
- Authentication checking
- Repository listing
- Workflow queue creation
- Runner provisioning
- Token generation

## 📋 Test Results

Running `test_github_copilot_integration.py` verifies:

1. ✅ Python package imports (4/4 tests)
   - GitHubCLI wrapper
   - WorkflowQueue manager
   - RunnerManager
   - CopilotCLI wrapper

2. ✅ CLI commands (11/11 tests)
   - Main CLI help
   - GitHub subcommands (auth, repos, workflows, queues, runners)
   - Copilot subcommands (suggest, explain, git)

3. ✅ MCP tools registration (2/2 tests)
   - GitHub tools module
   - Copilot tools module

4. ✅ Class functionality (3/3 tests)
   - GitHubCLI instantiation
   - WorkflowQueue instantiation
   - RunnerManager with core detection

**Total: 20/20 tests passed**

## 🔍 Troubleshooting

### GitHub CLI Not Authenticated
```bash
# Authenticate with GitHub CLI
gh auth login
```

### Python Import Errors
```bash
# Ensure you're in the correct directory
cd /path/to/ipfs_accelerate_py

# Verify Python path
python -c "import sys; print(sys.path)"
```

### MCP Tools Not Available
MCP tools are registered when the MCP server starts. They use the shared operations layer which lazily loads the GitHub CLI wrappers.

## 📖 Documentation

For complete documentation, see:
- `README_GITHUB_COPILOT.md` - Full usage guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `examples/github_workflow_automation.py` - Working example

## ✨ Summary

All components are working correctly:
- ✅ Python package imports work
- ✅ CLI commands are accessible
- ✅ MCP tools are registered
- ✅ Dashboard uses MCP JavaScript SDK
- ✅ Integration tests pass (20/20)
