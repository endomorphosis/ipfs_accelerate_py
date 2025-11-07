# Runner Permission Fix - Implementation Summary

**Date:** November 7, 2025  
**Issue:** EACCES permission denied errors when removing files in GitHub Actions runner workspace  
**Status:** ✅ Complete

## Problem Statement

GitHub Actions self-hosted runners encountered permission errors when trying to remove files:

```
Error: EACCES: permission denied, unlink '/home/actions-runner/_work/ipfs_datasets_py/ipfs_datasets_py/.github/GITHUB_ACTIONS_FIX_GUIDE.md'
```

This error occurred because:
- Stale git lock files from interrupted workflows
- Permission mismatches in workspace directories
- Insufficient permissions on `.git` and `.github` directories

## Solution Overview

A comprehensive multi-layered approach was implemented:

1. **Manual Fix Script** - Immediate resolution tool
2. **Automated Cleanup Workflow** - Preventive maintenance
3. **Reusable Cleanup Action** - Pre-job cleanup for workflows
4. **Documentation** - Comprehensive guides and examples

## Files Created

### 1. Manual Cleanup Script
**File:** `.github/scripts/fix_runner_permissions.sh`  
**Purpose:** Fix permissions immediately on runner host  
**Usage:**
```bash
./.github/scripts/fix_runner_permissions.sh
```

**Features:**
- Removes all git lock files
- Fixes git directory permissions
- Fixes .github directory permissions
- Cleans Python cache files
- Fixes file ownership
- Colorized output and progress reporting

### 2. Automated Cleanup Workflow
**File:** `.github/workflows/runner-cleanup.yml`  
**Purpose:** Run regular automated cleanup every 6 hours  
**Schedule:** Every 6 hours (cron: `0 */6 * * *`)

**Features:**
- Automatic scheduled cleanup
- Manual trigger support
- Callable by other workflows
- Disk usage reporting
- Cleanup status summary

### 3. Reusable Cleanup Action
**File:** `.github/actions/cleanup-workspace/action.yml`  
**Purpose:** Composite action for pre-job cleanup in workflows

**Features:**
- Remove stale git locks
- Fix directory permissions
- Clean Python cache
- Report disk usage
- Can be used with `continue-on-error: true`

### 4. Action Documentation
**File:** `.github/actions/cleanup-workspace/README.md`  
**Purpose:** Usage guide for the cleanup action

### 5. Comprehensive Fix Guide
**File:** `RUNNER_PERMISSION_FIX_GUIDE.md`  
**Purpose:** Complete documentation for fixing and preventing permission issues

**Sections:**
- Problem overview and root causes
- Quick fix instructions
- Three solution approaches
- Prevention best practices
- Troubleshooting guide
- Testing procedures
- Monitoring recommendations

### 6. Workflow Update Examples
**File:** `.github/workflows/WORKFLOW_UPDATE_EXAMPLES.md`  
**Purpose:** Examples showing how to update workflows

**Examples:**
- Simple workflow update
- Multi-architecture workflow
- Inline cleanup alternative
- Conditional cleanup
- Testing and troubleshooting

## Implementation Details

### Manual Fix Script Features

```bash
# Run on all runners
./.github/scripts/fix_runner_permissions.sh

# Run on specific runner
./.github/scripts/fix_runner_permissions.sh /home/barberb/actions-runner

# Show help
./.github/scripts/fix_runner_permissions.sh --help
```

**What it does:**
1. Locates runner workspace directories
2. Removes all `.lock` files
3. Recursively fixes `.git` permissions
4. Fixes `.github` directory permissions
5. Cleans Python `__pycache__` directories
6. Fixes ownership to current user
7. Reports summary of actions

### Automated Cleanup Workflow

**Trigger Methods:**
1. **Scheduled:** Runs every 6 hours automatically
2. **Manual:** `gh workflow run runner-cleanup.yml`
3. **Workflow Call:** Can be called by other workflows

**Cleanup Steps:**
1. Remove git lock files
2. Fix .git directory permissions
3. Fix .github directory permissions
4. Clean Python cache
5. Clean build artifacts (egg-info, dist, build)
6. Report disk usage

### Reusable Cleanup Action

**Usage in workflows:**
```yaml
steps:
  - name: Pre-job cleanup
    uses: ./.github/actions/cleanup-workspace
  
  - uses: actions/checkout@v4
    with:
      clean: true
```

**Steps performed:**
1. Remove stale git locks
2. Fix git directory permissions (dirs + files)
3. Fix .github directory permissions
4. Clean Python cache
5. Report disk usage

All steps use `continue-on-error: true` to not block workflows.

## Usage Instructions

### For Immediate Fix (One-time):

```bash
cd /home/barberb/ipfs_accelerate_py
./.github/scripts/fix_runner_permissions.sh
```

### For Long-term Prevention:

1. **Enable automated cleanup** (already in place):
   - Workflow runs every 6 hours automatically
   - No configuration needed

2. **Update workflows** to include pre-job cleanup:
   ```yaml
   jobs:
     your-job:
       runs-on: [self-hosted, linux, arm64]
       steps:
         - name: Pre-job cleanup
           uses: ./.github/actions/cleanup-workspace
         
         - uses: actions/checkout@v4
           with:
             clean: true
   ```

3. **Monitor** runner health:
   ```bash
   # View cleanup workflow runs
   gh run list --workflow=runner-cleanup.yml
   
   # Check runner logs
   tail -f /home/actions-runner/_diag/Runner*.log
   ```

## Workflows to Update

### High Priority (Self-hosted runners):
- [ ] `.github/workflows/amd64-ci.yml`
- [ ] `.github/workflows/arm64-ci.yml`
- [ ] `.github/workflows/multiarch-ci.yml`
- [ ] `.github/workflows/auto-heal-failures.yml`

### Medium Priority:
- [ ] `.github/workflows/test-auto-heal.yml`
- [ ] `.github/workflows/package-test.yml`
- [ ] `.github/workflows/documentation-maintenance.yml`

## Testing

### Test the manual script:
```bash
./.github/scripts/fix_runner_permissions.sh
```

Expected output:
```
[INFO] GitHub Actions Runner Permission Fix Script
[INFO] ===========================================
[INFO] Fixing permissions for runner: actions-runner
[INFO] Cleaning workspace: /home/actions-runner/_work
[INFO] Removing stale git lock files...
[INFO] Fixing git directory permissions...
[INFO] Fixing .github directory permissions...
[INFO] Cleaning Python cache files...
[INFO] Fixing ownership to barberb...
[INFO] ✓ Completed fixing permissions for actions-runner
[INFO] ===========================================
[INFO] Summary:
[INFO]   Runners fixed: 1
[INFO] ===========================================
```

### Test the automated workflow:
```bash
# Trigger manually
gh workflow run runner-cleanup.yml

# Check status
gh run list --workflow=runner-cleanup.yml

# View logs
gh run view <run-id> --log
```

### Test the cleanup action:
Create a test workflow and verify logs show:
- ✅ Lock files removed
- ✅ .git permissions fixed
- ✅ .github permissions fixed
- ✅ Python cache cleaned

## Monitoring

### Automated Cleanup Runs:
```bash
gh run list --workflow=runner-cleanup.yml --limit 5
```

### Runner Diagnostic Logs:
```bash
# View runner logs
tail -f /home/actions-runner/_diag/Runner*.log

# Check for permission errors
grep -i "permission\|eacces" /home/actions-runner/_diag/Worker*.log
```

### Disk Usage:
```bash
df -h /home/actions-runner/_work
du -sh /home/actions-runner/_work/*
```

## Prevention Best Practices

1. ✅ **Always use clean checkout** on self-hosted runners
   ```yaml
   - uses: actions/checkout@v4
     with:
       clean: true
   ```

2. ✅ **Add pre-job cleanup** to all self-hosted workflows
   ```yaml
   - name: Pre-job cleanup
     uses: ./.github/actions/cleanup-workspace
   ```

3. ✅ **Let automated cleanup run** (every 6 hours)
   - No action needed, it's already configured

4. ✅ **Monitor runner health** regularly
   - Check workflow logs
   - Review diagnostic logs
   - Monitor disk space

5. ✅ **Update workflows** as you touch them
   - Add cleanup when modifying existing workflows
   - Include cleanup in new workflows from the start

## Benefits

### Immediate Benefits:
- ✅ Resolves permission denied errors
- ✅ Removes stale git lock files
- ✅ Fixes .git and .github permissions
- ✅ Cleans up disk space

### Long-term Benefits:
- ✅ Prevents future permission issues
- ✅ Automated maintenance every 6 hours
- ✅ Easier to add to new workflows
- ✅ Better runner performance
- ✅ Reduced manual intervention

## Related Documentation

1. **RUNNER_PERMISSION_FIX_GUIDE.md** - Comprehensive fix guide
2. **WORKFLOW_UPDATE_EXAMPLES.md** - Workflow update examples
3. **.github/actions/cleanup-workspace/README.md** - Action usage
4. **RUNNER_PERMISSION_FIX.md** - Original issue documentation

## Support

If you encounter issues:

1. **Check the guides:**
   - `RUNNER_PERMISSION_FIX_GUIDE.md` - Main guide
   - `WORKFLOW_UPDATE_EXAMPLES.md` - Examples

2. **Run the fix script:**
   ```bash
   ./.github/scripts/fix_runner_permissions.sh
   ```

3. **Check workflow logs:**
   ```bash
   gh run view <run-id> --log
   ```

4. **Review runner diagnostics:**
   ```bash
   tail -f /home/actions-runner/_diag/Runner*.log
   ```

5. **Create an issue** with:
   - Error message
   - Workflow file
   - Runner logs
   - Steps to reproduce

## Summary

### What was implemented:
1. ✅ Manual fix script (`.github/scripts/fix_runner_permissions.sh`)
2. ✅ Automated cleanup workflow (`.github/workflows/runner-cleanup.yml`)
3. ✅ Reusable cleanup action (`.github/actions/cleanup-workspace/`)
4. ✅ Comprehensive documentation (4 guide files)

### What to do next:
1. Run the manual fix script once: `./.github/scripts/fix_runner_permissions.sh`
2. Update workflows to include pre-job cleanup
3. Monitor the automated cleanup runs
4. Add cleanup to new workflows as you create them

### Quick Reference:
```bash
# Fix now
./.github/scripts/fix_runner_permissions.sh

# Check cleanup status
gh run list --workflow=runner-cleanup.yml

# View runner logs
tail -f /home/actions-runner/_diag/Runner*.log
```

---

**Status:** Implementation complete ✅  
**Next Steps:** Update existing workflows with pre-job cleanup  
**Documentation:** Complete and ready for use
