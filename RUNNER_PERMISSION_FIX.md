# GitHub Actions Self-Hosted Runner Permission Fix

## Problem
Self-hosted runners may encounter permission errors during checkout:
```
Error: fatal: Unable to create '/path/to/.git/index.lock': Permission denied
Error: EACCES: permission denied, unlink '/path/to/.git/FETCH_HEAD'
```

## Root Cause
- Stale git lock files from interrupted workflows
- Permission mismatches in workspace
- Git processes not properly terminated

## Immediate Fix

### 1. Manual Cleanup (Run on runner host)
```bash
# Fix all runners
/home/barberb/fix-runner-permissions.sh

# Or fix specific runner
cd /home/barberb/actions-runner-datasets
./cleanup.sh
```

### 2. Add Cleanup Step to Workflows
Add this as the FIRST step in your workflows (before checkout):

```yaml
jobs:
  your-job:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - name: Pre-job cleanup
        run: |
          # Remove stale git locks
          find "${GITHUB_WORKSPACE}" -name "*.lock" -delete 2>/dev/null || true
          # Fix git directory permissions  
          find "${GITHUB_WORKSPACE}/.git" -type d -exec chmod u+rwx {} \; 2>/dev/null || true
          find "${GITHUB_WORKSPACE}/.git" -type f -exec chmod u+rw {} \; 2>/dev/null || true
        continue-on-error: true
        
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          clean: true
          fetch-depth: 1
```

### 3. Use Clean Checkout
Always use `clean: true` in checkout action:

```yaml
- uses: actions/checkout@v4
  with:
    clean: true  # Forces clean checkout
    fetch-depth: 1  # Shallow clone for speed
```

## Automated Solutions

### Option A: Add to .github/workflows/cleanup.yml
Create a reusable cleanup workflow:

```yaml
name: Cleanup Runner Workspace

on:
  workflow_call:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  cleanup:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - name: Clean workspace
        run: |
          cd "$RUNNER_WORKSPACE"
          find . -name "*.lock" -type f -delete
          find . -name ".git" -type d -exec chmod -R u+rwX {} \; 2>/dev/null || true
          # Remove old build artifacts
          find . -name "*.pyc" -delete
          find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

### Option B: Runner Service Configuration
Add cleanup to runner service (systemd):

```bash
# Edit: /etc/systemd/user/github-runner-arm64.service
[Service]
ExecStartPre=/home/barberb/actions-runner-datasets/cleanup.sh
ExecStart=/home/barberb/actions-runner-datasets/run.sh
```

## Prevention

### 1. Ensure Proper Runner Shutdown
Always stop runners gracefully:
```bash
cd /home/barberb/actions-runner-datasets
./svc.sh stop  # Not kill -9
```

### 2. Monitor Disk Space
Runners need space for git operations:
```bash
df -h /home/barberb/actions-runner-datasets/_work
```

### 3. Regular Workspace Cleanup
Add cron job:
```bash
# /etc/cron.daily/cleanup-runners
#!/bin/bash
/home/barberb/fix-runner-permissions.sh
```

## Testing

After applying fixes, test with:
```bash
cd /home/barberb/actions-runner-datasets
./cleanup.sh
# Then trigger a workflow
```

## Monitoring

Check runner health:
```bash
# View runner logs
tail -f /home/barberb/actions-runner-datasets/_diag/Runner*.log

# Check for permission errors
grep -i "permission\|eacces" /home/barberb/actions-runner-datasets/_diag/Worker*.log
```

## Related Issues
- GitHub Actions checkout action documentation
- Self-hosted runner troubleshooting guide
- Git lock file issues
