# GitHub CLI Authentication Fix

## Problem
Workflow step "assign-to-copilot" failing with:
```
Run # Authenticate gh CLI with the token
The value of the GH_TOKEN environment variable is being used for authentication.
To have GitHub CLI store credentials instead, first clear the value from the environment.
Error: Process completed with exit code 1.
```

## Root Cause
The `gh` CLI tool requires either:
1. `GH_TOKEN` environment variable to be set
2. `GITHUB_TOKEN` environment variable to be set
3. Credentials stored via `gh auth login`

When `GH_TOKEN` is set but invalid/empty, gh CLI fails with exit code 1.

## Solution

### Option 1: Set GH_TOKEN from GITHUB_TOKEN
If a workflow step uses `gh` CLI, ensure the environment variable is set:

```yaml
- name: Some step using gh CLI
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    gh pr list
    # other gh commands
```

### Option 2: Use GITHUB_TOKEN directly
Most `gh` commands also recognize `GITHUB_TOKEN`:

```yaml
- name: Some step using gh CLI
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    gh pr list
```

### Option 3: Authenticate explicitly
```yaml
- name: Authenticate GitHub CLI
  run: |
    echo "${{ secrets.GITHUB_TOKEN }}" | gh auth login --with-token

- name: Use gh CLI
  run: |
    gh pr list
```

## Likely Culprits in This Repo

Based on the workflows in this repository, the issue is likely in one of:

1. **auto-heal-failures.yml** - Uses GitHub API but might have a step trying to use gh CLI
2. **cleanup-auto-heal-branches.yml** - Uses actions/github-script but no explicit gh CLI
3. **documentation-maintenance.yml** - Might use gh CLI for operations

## Debugging Steps

1. Check the actual workflow run logs on GitHub Actions tab
2. Look for which step is named "assign-to-copilot" or contains "# Authenticate gh CLI"
3. Add `GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}` to that step's `env:` section

## Quick Fix Template

If you find the failing step, update it like this:

```yaml
# BEFORE
- name: Some step
  run: |
    gh some-command

# AFTER  
- name: Some step
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    gh some-command
```

## Verification

After applying the fix:
1. Re-run the failed workflow
2. Check that the authentication step passes
3. Verify subsequent gh CLI commands work

---
**Note**: This error appeared recently (6 minutes ago based on user report), suggesting either:
- A workflow was recently added/modified
- GitHub changed something about token handling
- A transient issue with GitHub's authentication service
