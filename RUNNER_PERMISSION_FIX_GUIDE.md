# GitHub Actions Runner Permission Fix Guide

## Problem Overview

When using self-hosted GitHub Actions runners, you may encounter permission errors when trying to remove or modify files in the workspace:

```
Error: EACCES: permission denied, unlink '/home/actions-runner/_work/ipfs_datasets_py/ipfs_datasets_py/.github/GITHUB_ACTIONS_FIX_GUIDE.md'
Error: fatal: Unable to create '/path/to/.git/index.lock': Permission denied
```

## Root Causes

1. **Stale git lock files** from interrupted workflows
2. **Permission mismatches** between runner user and workspace files
3. **Git processes** not properly terminated
4. **Insufficient permissions** on `.git` and `.github` directories

## Quick Fix (Immediate)

Run this command on the runner host machine:

```bash
# Navigate to your ipfs_accelerate_py directory
cd /home/barberb/ipfs_accelerate_py

# Run the fix script
./.github/scripts/fix_runner_permissions.sh
```

Or for a specific runner:

```bash
./.github/scripts/fix_runner_permissions.sh /home/actions-runner
```

## Solutions

### Solution 1: Automated Cleanup Workflow (Recommended)

The repository includes an automated cleanup workflow that runs every 6 hours.

**File:** `.github/workflows/runner-cleanup.yml`

**Features:**
- Automatically cleans workspace every 6 hours
- Removes stale git lock files
- Fixes permissions on `.git` and `.github` directories
- Cleans Python cache files
- Reports disk usage
- Can be triggered manually

**Manual trigger:**
```bash
gh workflow run runner-cleanup.yml
```

### Solution 2: Pre-Job Cleanup (Best Practice)

Add the cleanup action as the FIRST step in your workflow jobs:

```yaml
jobs:
  your-job:
    runs-on: [self-hosted, linux, arm64]
    steps:
      # IMPORTANT: This must be the first step!
      - name: Pre-job cleanup
        uses: ./.github/actions/cleanup-workspace
      
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          clean: true  # Always use clean checkout
          fetch-depth: 1
      
      # ... rest of your workflow steps
```

**What this does:**
1. Removes stale git lock files
2. Fixes .git directory permissions
3. Fixes .github directory permissions  
4. Cleans Python cache
5. Reports disk usage

### Solution 3: Manual Cleanup Script

Use the provided script for manual cleanup:

**Location:** `.github/scripts/fix_runner_permissions.sh`

**Usage:**
```bash
# Fix all runners
./.github/scripts/fix_runner_permissions.sh

# Fix specific runner
./.github/scripts/fix_runner_permissions.sh /home/barberb/actions-runner-datasets

# Show help
./.github/scripts/fix_runner_permissions.sh --help
```

**What it does:**
- Removes all git lock files
- Fixes git directory permissions recursively
- Fixes .github directory permissions
- Cleans Python cache files
- Fixes ownership to current user
- Reports summary of actions taken

## Workflow Updates

Update your existing workflows to include pre-job cleanup. Here's an example:

### Before:
```yaml
jobs:
  test:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest
```

### After:
```yaml
jobs:
  test:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - name: Pre-job cleanup
        uses: ./.github/actions/cleanup-workspace
      
      - uses: actions/checkout@v4
        with:
          clean: true
          fetch-depth: 1
      
      - name: Run tests
        run: pytest
```

## Prevention Best Practices

### 1. Always Use Clean Checkout

```yaml
- uses: actions/checkout@v4
  with:
    clean: true  # Forces clean checkout
    fetch-depth: 1  # Shallow clone for speed
```

### 2. Add Pre-Job Cleanup to All Self-Hosted Workflows

Every job on self-hosted runners should start with:

```yaml
- name: Pre-job cleanup
  uses: ./.github/actions/cleanup-workspace
```

### 3. Monitor Runner Health

Set up monitoring:

```bash
# View runner logs
tail -f /home/actions-runner/_diag/Runner*.log

# Check for permission errors
grep -i "permission\|eacces" /home/actions-runner/_diag/Worker*.log
```

### 4. Regular Maintenance

The automated cleanup workflow runs every 6 hours, but you can also:

- Add a cron job on the runner host:
  ```bash
  # /etc/cron.daily/cleanup-runners
  #!/bin/bash
  /home/barberb/ipfs_accelerate_py/.github/scripts/fix_runner_permissions.sh
  ```

- Manually trigger cleanup before important workflows:
  ```bash
  gh workflow run runner-cleanup.yml
  ```

## Troubleshooting

### Issue: Permission denied on specific file

**Error:**
```
Error: EACCES: permission denied, unlink '/path/to/file'
```

**Solution:**
```bash
# On the runner host, fix permissions for that specific file
chmod u+rw /path/to/file

# Or fix the entire workspace
./.github/scripts/fix_runner_permissions.sh
```

### Issue: Git lock file persists

**Error:**
```
fatal: Unable to create '/path/.git/index.lock': File exists
```

**Solution:**
```bash
# On the runner host
rm /path/to/workspace/.git/index.lock

# Or use the cleanup script
./.github/scripts/fix_runner_permissions.sh
```

### Issue: Disk space full

**Error:**
```
No space left on device
```

**Solution:**
```bash
# Check disk usage
df -h /home/actions-runner/_work

# Clean old workspaces
cd /home/actions-runner/_work
find . -type d -name "_temp" -mtime +7 -exec rm -rf {} +
```

## Testing Your Fix

After applying any fix, test with these steps:

1. **Trigger a workflow manually:**
   ```bash
   gh workflow run <your-workflow.yml>
   ```

2. **Check the workflow logs** for permission errors

3. **Verify cleanup worked:**
   ```bash
   # On runner host
   cd /home/actions-runner/_work
   find . -name "*.lock"  # Should return nothing
   ```

## Monitoring

### View Cleanup Workflow Results

```bash
gh run list --workflow=runner-cleanup.yml
gh run view <run-id>
```

### Check Runner Status

```bash
cd /home/actions-runner
./run.sh --once  # Test runner
```

### Monitor Disk Usage

```bash
# Add to crontab for daily reports
0 9 * * * df -h /home/actions-runner/_work | mail -s "Runner Disk Usage" admin@example.com
```

## Files Created

This fix includes the following files:

1. **`.github/scripts/fix_runner_permissions.sh`** - Manual cleanup script
2. **`.github/workflows/runner-cleanup.yml`** - Automated cleanup workflow  
3. **`.github/actions/cleanup-workspace/action.yml`** - Reusable cleanup action
4. **`.github/actions/cleanup-workspace/README.md`** - Action documentation
5. **`RUNNER_PERMISSION_FIX_GUIDE.md`** - This guide (you're reading it!)

## Summary

### For Immediate Fix:
```bash
./.github/scripts/fix_runner_permissions.sh
```

### For Long-term Prevention:
1. Add pre-job cleanup to all workflows
2. Use clean checkout with `clean: true`
3. Let automated cleanup workflow run every 6 hours
4. Monitor runner health regularly

### Quick Checklist:
- [ ] Run the manual fix script once
- [ ] Add cleanup action to workflow files
- [ ] Enable automated cleanup workflow
- [ ] Use `clean: true` in checkout actions
- [ ] Set up monitoring/alerts
- [ ] Test workflows after changes

## Related Documentation

- [RUNNER_PERMISSION_FIX.md](RUNNER_PERMISSION_FIX.md) - Original issue documentation
- [.github/workflows/README_DOCUMENTATION_MAINTENANCE.md](.github/workflows/README_DOCUMENTATION_MAINTENANCE.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Self-hosted Runner Troubleshooting](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners)

## Support

If you continue to experience issues:

1. Check the workflow logs: `gh run view <run-id> --log`
2. Review runner diagnostic logs: `/home/actions-runner/_diag/`
3. Create an issue with the `runner-support` label
4. Include error messages and runner logs
