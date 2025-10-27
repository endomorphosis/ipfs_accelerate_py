# GitHub Actions Runner Setup Guide

This guide will help you set up a GitHub Actions self-hosted runner on this machine as a backup to your existing runner.

## Quick Setup

### 1. Create a GitHub Personal Access Token

First, you need to create a GitHub Personal Access Token (PAT):

1. Go to [GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Give it a descriptive name like "GitHub Actions Runner - Backup Machine"
4. Set expiration (recommended: 90 days)
5. Select the following scopes:
   - `repo` (Full control of private repositories)
   - `admin:repo_hook` (Admin access to repository hooks)
6. Click "Generate token"
7. Copy the token immediately (you won't see it again)

### 2. Run the Setup Script

```bash
# Set your GitHub token
export GITHUB_TOKEN='your_token_here'

# Run the setup script
./setup_github_actions_runner.sh
```

Or in one command:
```bash
GITHUB_TOKEN='your_token_here' ./setup_github_actions_runner.sh
```

## What the Script Does

The setup script will:

1. **Check system requirements** - Verify Linux OS, architecture, and required tools
2. **Create a dedicated user** - Creates `actions-runner` user for security
3. **Download GitHub Actions runner** - Gets the latest stable runner version (2.319.1)
4. **Configure the runner** - Registers with GitHub using your token
5. **Install as a service** - Sets up systemd service for automatic startup
6. **Setup project dependencies** - Clones repo and installs Python dependencies
7. **Verify installation** - Checks that everything is working

## Runner Configuration

The runner will be configured with:
- **Name**: `$(hostname)-backup-runner` (e.g., `fent-reactor-backup-runner`)
- **Labels**: `linux,x64,self-hosted,backup-runner`
- **Repository**: `endomorphosis/ipfs_accelerate_py`
- **User**: `actions-runner`
- **Service**: Automatic startup on boot

## Managing the Runner

### Check Status
```bash
# Check service status
sudo systemctl status actions.runner.endomorphosis.ipfs_accelerate_py.$(hostname)-backup-runner.service

# Or use the script
./setup_github_actions_runner.sh --status
```

### View Logs
```bash
sudo journalctl -u actions.runner.endomorphosis.ipfs_accelerate_py.$(hostname)-backup-runner.service -f
```

### Stop/Start Runner
```bash
# Stop
sudo systemctl stop actions.runner.endomorphosis.ipfs_accelerate_py.$(hostname)-backup-runner.service

# Start  
sudo systemctl start actions.runner.endomorphosis.ipfs_accelerate_py.$(hostname)-backup-runner.service

# Restart
sudo systemctl restart actions.runner.endomorphosis.ipfs_accelerate_py.$(hostname)-backup-runner.service
```

### Uninstall Runner
```bash
GITHUB_TOKEN='your_token_here' ./setup_github_actions_runner.sh --uninstall
```

## File Locations

- **Runner Directory**: `/home/actions-runner/actions-runner/`
- **Project Directory**: `/home/actions-runner/ipfs_accelerate_py/`
- **Config File**: `/home/actions-runner/runner-config.json`
- **Service Name**: `actions.runner.endomorphosis.ipfs_accelerate_py.$(hostname)-backup-runner.service`

## Using the Backup Runner

Once installed, your workflows can target this specific runner using labels:

```yaml
jobs:
  backup-test:
    runs-on: [self-hosted, linux, backup-runner]
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: python -m pytest
```

## Troubleshooting

### Runner Not Appearing in GitHub

1. Check service status: `./setup_github_actions_runner.sh --status`
2. Check logs: `sudo journalctl -u actions.runner.* -f`
3. Verify token permissions
4. Try restarting: `sudo systemctl restart actions.runner.*`

### Service Won't Start

1. Check disk space: `df -h`
2. Check permissions: `ls -la /home/actions-runner/`
3. Check logs: `sudo journalctl -u actions.runner.* --no-pager`

### Dependencies Issues

1. Update pip: `sudo -u actions-runner python3 -m pip install --upgrade pip`
2. Reinstall requirements: `cd /home/actions-runner/ipfs_accelerate_py && sudo -u actions-runner python3 -m pip install -r requirements.txt`

### GitHub Token Expired

1. Generate new token with same permissions
2. Re-run the setup script with new token

## Security Notes

- The runner runs as a dedicated `actions-runner` user
- The user has sudo access (required for some CI operations)
- Runner directory permissions are restricted to the actions-runner user
- Consider using a dedicated machine or VM for production runners

## Advanced Configuration

### Custom Labels

To add custom labels, edit the script and modify the `RUNNER_LABELS` variable:
```bash
RUNNER_LABELS="linux,x64,self-hosted,backup-runner,gpu,cuda"
```

### Different Repository

To set up for a different repository:
```bash
export GITHUB_REPOSITORY="owner/repo"
GITHUB_TOKEN='your_token_here' ./setup_github_actions_runner.sh
```

### Multiple Runners

To run multiple runners on the same machine, you can modify the `RUNNER_NAME` in the script before running it multiple times.

## GitHub Actions Workflow Examples

Here are some example workflows that can use your backup runner:

### Basic Test Workflow
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test-backup:
    runs-on: [self-hosted, linux, backup-runner]
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest
```

### Benchmark Workflow
```yaml
name: Benchmarks
on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    runs-on: [self-hosted, linux, backup-runner]
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmarks
        run: python benchmarks/benchmark_all_key_models.py
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_results.json
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the GitHub Actions runner documentation: https://docs.github.com/en/actions/hosting-your-own-runners
3. Check the repository issues: https://github.com/endomorphosis/ipfs_accelerate_py/issues

---

**Note**: This backup runner setup is designed to complement your existing GitHub Actions runner infrastructure. Make sure to coordinate with your team about runner usage and resource allocation.