# Self-Hosted Runner Workflow Setup Guide

## Problem

Self-hosted GitHub Actions runners encounter permission errors when:
1. **Docker containerized workflows** create files as root
2. **Git operations** fail with "Permission denied" on `.git/FETCH_HEAD`, `.git/index`, etc.
3. **Checkout fails** with "Unable to prepare the existing repository"

## Solution

### For ipfs_accelerate_py Repository ✅

Already configured with:
- Automated permission fix script
- Scheduled cleanup workflow (every 6 hours)
- Passwordless sudo for workspace ownership fixes

### For OTHER Repositories (ipfs_datasets_py, ipfs_kit_py, etc.)

Add the pre-checkout cleanup action as the **FIRST step** in workflows that run on self-hosted runners.

## Quick Setup

### Step 1: Copy the Pre-Checkout Action

Copy this directory to your repository:
```bash
# From ipfs_accelerate_py to your repo
cp -r .github/actions/pre-checkout-cleanup /path/to/your-repo/.github/actions/
```

### Step 2: Add to Workflow (FIRST STEP)

In any workflow that uses `runs-on: [self-hosted]`, add this as the **first step**:

```yaml
jobs:
  your-job:
    runs-on: [self-hosted, linux]
    
    steps:
      # ⭐ ADD THIS AS FIRST STEP ⭐
      - name: Pre-checkout cleanup
        uses: ./.github/actions/pre-checkout-cleanup
      
      # Now checkout will work without permission errors
      - name: Checkout code
        uses: actions/checkout@v4
      
      # ... rest of your workflow
```

### Step 3: Example Full Workflow

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: [self-hosted, linux]
    
    steps:
      # IMPORTANT: This must be the first step
      - name: Pre-checkout cleanup
        uses: ./.github/actions/pre-checkout-cleanup
      
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Run tests
        run: |
          python -m pytest tests/
```

## What the Pre-Checkout Action Does

1. **Fixes ownership** of root-owned files from previous containerized workflows
2. **Removes stale git locks** (index.lock, *.lock files)
3. **Configures git safe directory** to prevent permission warnings
4. **Fixes .git permissions** if the directory exists
5. **Uses passwordless sudo** (configured via sudoers in ipfs_accelerate_py)

## Alternative: Manual Checkout with Cleanup

If you can't use the pre-checkout action (workflow uses a different structure), use this pattern:

```yaml
- name: Manual workspace cleanup
  run: |
    # Fix ownership of existing workspace
    current_user=$(whoami)
    if [ -d "${GITHUB_WORKSPACE}" ]; then
      sudo -n chown -R "$current_user:$current_user" "${GITHUB_WORKSPACE}" 2>/dev/null || true
      find "${GITHUB_WORKSPACE}" -name "*.lock" -delete 2>/dev/null || true
    fi
    git config --global --add safe.directory "*"
  continue-on-error: true

- name: Checkout code
  uses: actions/checkout@v4
```

## Why This Happens

### Root Cause
When workflows run Docker containers (especially with volume mounts), files created inside containers are owned by **root** on the host system. The GitHub Actions runner user (e.g., `barberb`) cannot delete or modify these files, causing:

```
Error: EACCES: permission denied, unlink '.git/FETCH_HEAD'
Error: Unable to create '.git/index.lock': Permission denied
```

### The Fix
The sudoers configuration in ipfs_accelerate_py allows passwordless `chown` commands for runner workspaces, so the pre-checkout action can fix ownership before `actions/checkout` runs.

## Verification

After adding the pre-checkout action, you should see in workflow logs:

```
✅ Pre-checkout cleanup completed
   The checkout action will now run safely
```

And no more errors like:
- ❌ "EACCES: permission denied, unlink"
- ❌ "Unable to create '.git/index.lock'"
- ❌ "Unable to prepare the existing repository"

## Troubleshooting

### If You Still Get Permission Errors

1. **Check sudoers configuration**:
   ```bash
   sudo visudo -c
   ls -la /etc/sudoers.d/runner-fix-permissions
   ```

2. **Verify passwordless sudo works**:
   ```bash
   sudo -n chown -R barberb:barberb /home/barberb/actions-runner*/_work/
   ```

3. **Run manual fix**:
   ```bash
   bash /home/barberb/ipfs_accelerate_py/.github/scripts/fix_runner_permissions.sh
   ```

4. **Check runner logs**:
   ```bash
   ps aux | grep Runner.Listener
   journalctl -u actions-runner* -f
   ```

## Automated Maintenance

The ipfs_accelerate_py repository runs automatic cleanup every 6 hours via:
- `.github/workflows/runner-cleanup.yml`
- `.github/scripts/fix_runner_permissions.sh`

This ensures all runners stay healthy even if individual workflows forget to use the cleanup action.

## Repository Coverage

Apply this to all repositories using self-hosted runners:
- ✅ ipfs_accelerate_py (already configured)
- ⏳ ipfs_datasets_py (add pre-checkout action)
- ⏳ ipfs_kit_py (add pre-checkout action)
- ⏳ swissknife (add pre-checkout action)
- ⏳ navichat (add pre-checkout action)

---

**Questions?** See `RUNNER_PERMISSION_FIX_COMPLETE.md` for full details.
