# GitHub CLI and Copilot CLI Integration

This document describes the GitHub CLI and Copilot CLI integration features in the IPFS Accelerate package.

## Overview

IPFS Accelerate now integrates with GitHub CLI (`gh`) and GitHub Copilot CLI to provide:

1. **Python Package Integration**: Use GitHub CLI and Copilot features directly in your Python code
2. **MCP Tools**: Access GitHub and Copilot features through the MCP server
3. **CLI Subcommands**: Use GitHub and Copilot features through the `ipfs-accelerate` CLI
4. **Automated Workflow Management**: Automatically manage GitHub Actions workflows and self-hosted runners

## Prerequisites

### GitHub CLI

Install GitHub CLI:

```bash
# macOS
brew install gh

# Ubuntu/Debian
sudo apt install gh

# Windows (via winget)
winget install --id GitHub.cli
```

Authenticate:

```bash
gh auth login
```

### GitHub Copilot CLI (Optional)

Install GitHub Copilot CLI:

```bash
npm install -g @githubnext/github-copilot-cli
```

## Usage

### 1. Python Package

Import and use the wrappers directly in your Python code:

```python
from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager
from ipfs_accelerate_py.copilot_cli import CopilotCLI

# GitHub CLI
gh = GitHubCLI()
repos = gh.list_repos(owner="myorg", limit=10)
print(f"Found {len(repos)} repositories")

# Workflow Queue Management
queue = WorkflowQueue(gh)
queues = queue.create_workflow_queues(owner="myorg", since_days=1)
print(f"Created queues for {len(queues)} repositories")

# Runner Management
runner_mgr = RunnerManager(gh)
cores = runner_mgr.get_system_cores()
provisioning = runner_mgr.provision_runners_for_queue(queues, max_runners=cores)
print(f"Provisioned {len(provisioning)} runner tokens")

# Copilot CLI
copilot = CopilotCLI()
result = copilot.suggest_command("list all text files")
print(f"Suggested: {result['suggestion']}")
```

### 2. MCP Tools

The GitHub and Copilot features are available as MCP tools:

#### GitHub Tools

- `gh_auth_status()` - Check GitHub authentication status
- `gh_list_repos(owner, limit)` - List repositories
- `gh_list_workflow_runs(repo, status, limit)` - List workflow runs
- `gh_get_workflow_run(repo, run_id)` - Get workflow run details
- `gh_create_workflow_queues(owner, since_days)` - Create workflow queues
- `gh_list_runners(repo, org)` - List self-hosted runners
- `gh_provision_runners(owner, since_days, max_runners)` - Provision runners

#### Copilot Tools

- `copilot_suggest_command(prompt, shell)` - Get command suggestions
- `copilot_explain_command(command)` - Explain a command
- `copilot_suggest_git_command(prompt)` - Get Git command suggestions

### 3. CLI Subcommands

Use the integrated CLI for all GitHub and Copilot operations:

#### GitHub Commands

```bash
# Check authentication
ipfs-accelerate github auth

# List repositories
ipfs-accelerate github repos --owner myorg --limit 20

# List workflow runs
ipfs-accelerate github workflows myorg/myrepo --status in_progress

# Create workflow queues for repos updated in last day
ipfs-accelerate github queues --owner myorg --since-days 1

# List self-hosted runners
ipfs-accelerate github runners list --org myorg

# Provision runners based on workflow queues
ipfs-accelerate github runners provision --owner myorg --since-days 1 --max-runners 4
```

#### Copilot Commands

```bash
# Get command suggestion
ipfs-accelerate copilot suggest "list all text files" --shell bash

# Explain a command
ipfs-accelerate copilot explain "ls -la | grep txt"

# Get Git command suggestion
ipfs-accelerate copilot git "commit all changes with message"
```

## Automated Workflow Queue Management

The package can automatically manage GitHub Actions workflows and self-hosted runners:

### Workflow Queue Creation

Automatically creates queues for repositories with recent activity:

```python
from ipfs_accelerate_py.github_cli import WorkflowQueue

queue = WorkflowQueue()
queues = queue.create_workflow_queues(
    owner="myorg",
    since_days=1  # Repositories updated in last day
)

# Returns dict mapping repo names to workflow lists
# Each workflow includes running and failed workflows
```

### Automatic Runner Provisioning

Provisions self-hosted runners based on workflow load:

```python
from ipfs_accelerate_py.github_cli import RunnerManager

runner_mgr = RunnerManager()
provisioning = runner_mgr.provision_runners_for_queue(
    queues,
    max_runners=None  # Defaults to system CPU cores
)

# Returns dict with registration tokens for each repository
# Tokens are derived from gh CLI and can be used to attach runners
```

### CLI Example

Complete workflow for automating runner provisioning:

```bash
# 1. Check authentication
ipfs-accelerate github auth

# 2. Create workflow queues for repos with recent activity
ipfs-accelerate github queues --owner myorg --since-days 1 --output-json > queues.json

# 3. Provision runners based on system capacity
ipfs-accelerate github runners provision --owner myorg --since-days 1 --output-json > tokens.json

# 4. View provisioning results
cat tokens.json | jq '.provisioning'
```

## Architecture

### Component Structure

```
ipfs_accelerate_py/
├── github_cli/
│   ├── __init__.py
│   └── wrapper.py          # GitHubCLI, WorkflowQueue, RunnerManager
├── copilot_cli/
│   ├── __init__.py
│   └── wrapper.py          # CopilotCLI
├── shared/
│   └── operations.py       # GitHubOperations, CopilotOperations
├── mcp/
│   └── tools/
│       ├── github_tools.py # GitHub MCP tools
│       └── copilot_tools.py # Copilot MCP tools
└── cli.py                  # CLI subcommands
```

### Integration Layers

1. **Wrapper Layer** (`github_cli/`, `copilot_cli/`): Direct Python wrappers for CLI commands
2. **Operations Layer** (`shared/operations.py`): Business logic and shared functionality
3. **MCP Layer** (`mcp/tools/`): MCP server tool registration
4. **CLI Layer** (`cli.py`): Command-line interface

## Features

### Workflow Queue Management

- Automatically detects repositories with activity in the last N days
- Creates queues of running and failed workflows
- Filters workflows by status and time window
- Provides detailed workflow metadata

### Runner Provisioning

- Automatically provisions runners based on workflow load
- Respects system capacity (defaults to CPU core count)
- Derives tokens from authenticated gh CLI
- Prioritizes repositories with the most workflows
- Returns registration tokens for attaching runners

### Dashboard Integration

The GitHub workflows and runner status can be monitored through the integrated dashboard:

```bash
ipfs-accelerate mcp start --dashboard --open-browser
```

Then visit `http://localhost:9000/dashboard` to view:
- Workflow queue status
- Runner provisioning status
- Repository activity
- Workflow success/failure rates

## Error Handling

The integration includes comprehensive error handling:

- Authentication checks before operations
- Timeout handling for long-running commands
- Graceful fallback when tools are unavailable
- Detailed error messages and logging

## Security Considerations

- Tokens are derived from authenticated gh CLI session
- No tokens are stored in code or logs
- Runner provisioning respects organization/repository permissions
- All operations use official GitHub CLI for security

## Troubleshooting

### GitHub CLI Not Found

```
Error: gh CLI not found at gh
```

**Solution**: Install GitHub CLI and ensure it's in your PATH.

### Not Authenticated

```
Error: GitHub CLI is not authenticated
```

**Solution**: Run `gh auth login` and follow the prompts.

### Copilot CLI Not Found

```
Warning: Copilot CLI not found
```

**Solution**: Install Copilot CLI with `npm install -g @githubnext/github-copilot-cli` or the operations will gracefully skip Copilot features.

### Rate Limiting

GitHub API has rate limits. The integration automatically handles rate limit errors and will retry with exponential backoff.

## Examples

### Example 1: Monitor Failing Workflows

```python
from ipfs_accelerate_py.github_cli import WorkflowQueue

queue = WorkflowQueue()
failed = queue.list_failed_runs("myorg/myrepo", since_days=7)

for run in failed:
    print(f"Run #{run['databaseId']}: {run['workflowName']}")
    print(f"  Failed at: {run['updatedAt']}")
    print(f"  Conclusion: {run['conclusion']}")
```

### Example 2: Auto-Provision Runners for Busy Repos

```python
from ipfs_accelerate_py.github_cli import WorkflowQueue, RunnerManager

# Create queues
queue = WorkflowQueue()
queues = queue.create_workflow_queues(owner="myorg", since_days=1)

# Find repos with most workflows
busy_repos = sorted(
    queues.items(),
    key=lambda x: len(x[1]),
    reverse=True
)[:5]  # Top 5

# Provision runners for busy repos
runner_mgr = RunnerManager()
busy_queues = dict(busy_repos)
tokens = runner_mgr.provision_runners_for_queue(busy_queues, max_runners=5)

for repo, token_info in tokens.items():
    print(f"{repo}: {token_info['status']}")
```

### Example 3: CLI Pipeline

```bash
#!/bin/bash
# pipeline.sh - Automated workflow monitoring and runner provisioning

# Check authentication
if ! ipfs-accelerate github auth > /dev/null 2>&1; then
    echo "Error: Not authenticated. Run 'gh auth login'"
    exit 1
fi

# Create workflow queues
echo "Creating workflow queues..."
ipfs-accelerate github queues --owner myorg --since-days 1 --output-json > /tmp/queues.json

# Check for failures
FAILED=$(cat /tmp/queues.json | jq -r '.queues | to_entries[] | select(.value[].conclusion == "failure") | .key' | wc -l)
echo "Found $FAILED repositories with failed workflows"

# Provision runners if needed
if [ $FAILED -gt 0 ]; then
    echo "Provisioning runners..."
    ipfs-accelerate github runners provision --owner myorg --since-days 1 --max-runners 4
fi
```

## Future Enhancements

Planned features for future releases:

- [ ] Dashboard widgets for workflow status
- [ ] Real-time workflow notifications
- [ ] Automatic runner scaling based on queue depth
- [ ] Integration with runner auto-scaling services
- [ ] Workflow performance analytics
- [ ] Custom runner labels and targeting
- [ ] Multi-organization support
- [ ] Webhook integration for event-driven provisioning

## Contributing

Contributions are welcome! Please see the main project README for contribution guidelines.

## License

This integration is part of the IPFS Accelerate project and is licensed under the GNU Affero General Public License v3 or later (AGPLv3+).
