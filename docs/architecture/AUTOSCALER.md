# GitHub Actions Runner Autoscaler

## Overview

The GitHub Actions Runner Autoscaler automatically monitors your GitHub workflows and provisions self-hosted runners as needed. When you're authenticated with GitHub CLI (`gh auth login`), it works out of the box with zero configuration.

## Features

✅ **Zero Configuration** - Works immediately after `gh auth login`
✅ **Automatic Monitoring** - Continuously checks workflow queues
✅ **Smart Provisioning** - Provisions runners based on workflow demand
✅ **System-Aware** - Respects CPU core limits automatically
✅ **Priority-Based** - Provisions runners for busiest repos first
✅ **Easy to Use** - Simple CLI command to start

## Quick Start

```bash
# 1. Authenticate with GitHub CLI (one time)
gh auth login

# 2. Start the autoscaler (runs continuously)
python cli.py github autoscaler

# That's it! The autoscaler will now:
# - Monitor your repositories for workflow activity
# - Detect running and failed workflows
# - Automatically provision self-hosted runners as needed
# - Respect your system's CPU core limit
```

## Usage

### Basic Usage

```bash
# Monitor all accessible repositories
python cli.py github autoscaler

# Monitor specific organization
python cli.py github autoscaler --owner myorg

# Custom poll interval (check every 30 seconds)
python cli.py github autoscaler --interval 30

# Limit maximum runners (override system cores)
python cli.py github autoscaler --max-runners 4

# Monitor repos updated in last 2 days
python cli.py github autoscaler --since-days 2
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--owner` | GitHub owner (user/org) to monitor | All accessible repos |
| `--interval` | Poll interval in seconds | 60 |
| `--since-days` | Monitor repos updated in last N days | 1 |
| `--max-runners` | Maximum runners to provision | System CPU cores |

### Using as Python Script

```bash
# Direct script execution
python github_autoscaler.py

# With options
python github_autoscaler.py --owner myorg --interval 30
```

## How It Works

1. **Authentication Check** - Verifies you're logged into GitHub CLI
2. **Continuous Monitoring** - Polls GitHub every N seconds (default: 60s)
3. **Workflow Detection** - Finds repos with recent activity (default: 1 day)
4. **Queue Analysis** - Identifies running and failed workflows
5. **Smart Provisioning** - Generates runner tokens for repos that need them
6. **Priority Sorting** - Provisions runners for busiest repos first
7. **Capacity Respect** - Never exceeds system CPU core limit

## Example Output

```
============================================================
GitHub Actions Runner Autoscaler Started
============================================================

Monitoring for workflow queues and auto-provisioning runners...
Press Ctrl+C to stop

--- Check #1 at 2025-11-02 14:30:00 ---
Checking workflow queues...
Found 3 repos with 7 workflows
  Running: 2, Failed: 1
Provisioning runners...
✓ Generated 2 runner token(s)
  myorg/repo1: 3 workflows
  myorg/repo2: 2 workflows
Sleeping for 60s...

--- Check #2 at 2025-11-02 14:31:00 ---
Checking workflow queues...
No workflows need runner provisioning
Sleeping for 60s...
```

## Integration with Dashboard

The autoscaler works alongside the MCP dashboard:

```bash
# Terminal 1: Start autoscaler
python cli.py github autoscaler --owner myorg

# Terminal 2: Start dashboard (monitors the same data)
python cli.py mcp start --dashboard --open-browser
```

The dashboard will show:
- Current workflow queues
- Runner provisioning status
- Real-time statistics

## Prerequisites

1. **GitHub CLI** - Must be installed
   ```bash
   # macOS
   brew install gh
   
   # Ubuntu/Debian
   sudo apt install gh
   
   # Windows
   winget install --id GitHub.cli
   ```

2. **Authentication** - Must be logged in
   ```bash
   gh auth login
   ```

3. **Permissions** - Need permissions to:
   - Read workflow runs
   - Create runner registration tokens
   - Access repository/organization settings

## Configuration

### Environment Variables

The autoscaler respects standard GitHub CLI configuration:

```bash
# Use specific GitHub instance
export GH_HOST=github.company.com

# Use specific token
export GH_TOKEN=ghp_yourtoken
```

### Running as Service

For production deployments, run as a system service:

**systemd service example** (`/etc/systemd/system/github-autoscaler.service`):

```ini
[Unit]
Description=GitHub Actions Runner Autoscaler
After=network.target

[Service]
Type=simple
User=runner
WorkingDirectory=/path/to/ipfs_accelerate_py
ExecStart=/usr/bin/python3 /path/to/ipfs_accelerate_py/cli.py github autoscaler --owner myorg
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable github-autoscaler
sudo systemctl start github-autoscaler
sudo systemctl status github-autoscaler
```

## Monitoring

### Logs

The autoscaler logs all activity to stdout. Redirect to file if needed:

```bash
python cli.py github autoscaler --owner myorg 2>&1 | tee autoscaler.log
```

### Health Check

The autoscaler runs continuously. If it stops:
- Check authentication: `gh auth status`
- Check logs for errors
- Verify GitHub CLI is installed: `gh --version`

## Troubleshooting

### "GitHub CLI not authenticated"

**Problem**: Autoscaler fails to start

**Solution**: 
```bash
gh auth login
```

### "No repositories with active workflows"

**Problem**: No workflows detected

**Causes**:
- No repos updated in last day (increase `--since-days`)
- No workflows running or failed
- Permissions issue (can't access repos)

**Solution**:
```bash
# Try monitoring longer period
python cli.py github autoscaler --since-days 7
```

### "Reached max runners limit"

**Problem**: Can't provision more runners

**Cause**: Hit system CPU core limit

**Solution**:
```bash
# Increase limit (use with caution)
python cli.py github autoscaler --max-runners 8
```

### Rate Limiting

**Problem**: GitHub API rate limits

**Solution**: The autoscaler respects rate limits automatically. Increase `--interval` if needed:
```bash
python cli.py github autoscaler --interval 120  # Check every 2 minutes
```

## Best Practices

1. **Start Small** - Begin with short intervals and few repos
2. **Monitor Logs** - Watch first few cycles to verify behavior
3. **Set Limits** - Use `--max-runners` to prevent over-provisioning
4. **Use Specific Owner** - Target specific org/user with `--owner`
5. **Adjust Interval** - Balance responsiveness vs. API usage
6. **Run as Service** - Use systemd for production deployments

## Security Considerations

- Tokens are derived from your authenticated GitHub CLI session
- No tokens are stored in code or logs (only first 20 chars shown)
- Respects GitHub CLI's authentication and permissions
- Runner registration tokens expire (typically 1 hour)
- Requires appropriate repository/org permissions

## API Usage

The autoscaler makes these GitHub API calls per check:
- 1 call to list repositories (if no `--owner`)
- N calls to list workflow runs (N = number of repos with activity)
- M calls to generate tokens (M = runners to provision)

**Example**: With 10 active repos and 60s interval:
- ~10-20 API calls per minute
- Well within GitHub's rate limits (5000/hour authenticated)

## Advanced Usage

### Custom Workflow Detection

Modify `github_autoscaler.py` to customize behavior:

```python
# Only provision for specific workflow states
def check_and_scale(self):
    queues = self.queue_mgr.create_workflow_queues(...)
    
    # Custom filtering
    filtered_queues = {
        repo: [w for w in workflows if w.get('event') == 'push']
        for repo, workflows in queues.items()
    }
    
    self.runner_mgr.provision_runners_for_queue(filtered_queues)
```

### Multiple Organizations

Run multiple instances for different orgs:

```bash
# Terminal 1
python cli.py github autoscaler --owner org1

# Terminal 2
python cli.py github autoscaler --owner org2
```

### Integration with CI/CD

Use in CI/CD pipelines to auto-provision runners:

```yaml
# .github/workflows/autoscale.yml
name: Autoscale Runners
on:
  schedule:
    - cron: '*/5 * * * *'  # Every 5 minutes
jobs:
  autoscale:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Autoscaler
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          pip install -e .
          python cli.py github autoscaler --owner ${{ github.repository_owner }}
```

## FAQ

**Q: Does this provision actual runner instances?**
A: No, it generates registration tokens. You need to use those tokens to start actual runner processes/containers.

**Q: Can I use this with GitHub Enterprise?**
A: Yes, if GitHub CLI is configured for your enterprise instance.

**Q: Will it provision runners for private repos?**
A: Yes, if you have appropriate permissions.

**Q: How do I stop it?**
A: Press Ctrl+C or send SIGTERM to the process.

**Q: Can I run multiple autoscalers?**
A: Yes, each can monitor different orgs or with different settings.

## See Also

- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [Self-Hosted Runners Guide](https://docs.github.com/en/actions/hosting-your-own-runners)
- [README_GITHUB_COPILOT.md](README_GITHUB_COPILOT.md) - Full integration guide
- [QUICKSTART.md](QUICKSTART.md) - Quick verification guide
