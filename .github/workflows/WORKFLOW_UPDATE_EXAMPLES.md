# Workflow Update Examples

This file shows examples of how to update existing workflows to include the pre-job cleanup action.

## Example 1: Simple Workflow Update

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
      # Add this as the FIRST step
      - name: Pre-job cleanup
        uses: ./.github/actions/cleanup-workspace
      
      - uses: actions/checkout@v4
        with:
          clean: true  # Always use clean checkout
          fetch-depth: 1
      
      - name: Run tests
        run: pytest
```

## Example 2: Multi-Architecture Workflow

### Before:
```yaml
jobs:
  build-amd64:
    runs-on: [self-hosted, linux, amd64]
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: docker build .
  
  build-arm64:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: docker build .
```

### After:
```yaml
jobs:
  build-amd64:
    runs-on: [self-hosted, linux, amd64]
    steps:
      - name: Pre-job cleanup
        uses: ./.github/actions/cleanup-workspace
      
      - uses: actions/checkout@v4
        with:
          clean: true
          fetch-depth: 1
      
      - name: Build
        run: docker build .
  
  build-arm64:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - name: Pre-job cleanup
        uses: ./.github/actions/cleanup-workspace
      
      - uses: actions/checkout@v4
        with:
          clean: true
          fetch-depth: 1
      
      - name: Build
        run: docker build .
```

## Example 3: Using Inline Cleanup (Alternative)

If you can't use the composite action, use inline cleanup:

```yaml
jobs:
  test:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - name: Pre-job cleanup
        run: |
          # Remove stale git locks
          find "${GITHUB_WORKSPACE}" -name "*.lock" -delete 2>/dev/null || true
          # Fix git directory permissions
          if [ -d "${GITHUB_WORKSPACE}/.git" ]; then
            find "${GITHUB_WORKSPACE}/.git" -type d -exec chmod u+rwx {} \; 2>/dev/null || true
            find "${GITHUB_WORKSPACE}/.git" -type f -exec chmod u+rw {} \; 2>/dev/null || true
          fi
          # Fix .github directory permissions
          if [ -d "${GITHUB_WORKSPACE}/.github" ]; then
            find "${GITHUB_WORKSPACE}/.github" -type d -exec chmod u+rwx {} \; 2>/dev/null || true
            find "${GITHUB_WORKSPACE}/.github" -type f -exec chmod u+rw {} \; 2>/dev/null || true
          fi
        continue-on-error: true
      
      - uses: actions/checkout@v4
        with:
          clean: true
          fetch-depth: 1
      
      - name: Run tests
        run: pytest
```

## Example 4: Conditional Cleanup (Only for Self-Hosted)

```yaml
jobs:
  test:
    runs-on: ${{ matrix.runner }}
    strategy:
      matrix:
        runner:
          - ubuntu-latest
          - [self-hosted, linux, arm64]
    
    steps:
      # Only run cleanup on self-hosted runners
      - name: Pre-job cleanup (self-hosted only)
        if: runner.environment == 'self-hosted'
        uses: ./.github/actions/cleanup-workspace
      
      - uses: actions/checkout@v4
        with:
          clean: ${{ runner.environment == 'self-hosted' }}
          fetch-depth: 1
      
      - name: Run tests
        run: pytest
```

## Files to Update

Update these workflow files to include pre-job cleanup:

### High Priority (Self-Hosted Runners):
- `.github/workflows/amd64-ci.yml`
- `.github/workflows/arm64-ci.yml`
- `.github/workflows/multiarch-ci.yml`
- `.github/workflows/auto-heal-failures.yml`

### Medium Priority:
- `.github/workflows/test-auto-heal.yml`
- `.github/workflows/package-test.yml`
- `.github/workflows/documentation-maintenance.yml`

### Low Priority (Mostly GitHub-hosted):
- Other workflow files as needed

## Automated Update Script

You can use this script to automatically add pre-job cleanup to all self-hosted workflows:

```bash
#!/bin/bash

# Find all workflow files
for workflow in .github/workflows/*.yml; do
  # Check if workflow uses self-hosted runners
  if grep -q "runs-on:.*self-hosted" "$workflow"; then
    echo "Workflow uses self-hosted runners: $workflow"
    echo "  → Manual review recommended"
  fi
done
```

## Testing Your Changes

After updating a workflow:

1. **Commit changes:**
   ```bash
   git add .github/workflows/your-workflow.yml
   git commit -m "Add pre-job cleanup to prevent permission errors"
   ```

2. **Test manually:**
   ```bash
   gh workflow run your-workflow.yml
   ```

3. **Check logs:**
   ```bash
   gh run list --workflow=your-workflow.yml
   gh run view <run-id> --log
   ```

4. **Verify cleanup worked:**
   Look for these log entries:
   - "🔒 Removing stale git lock files..."
   - "✅ Lock files removed"
   - "📁 Fixing .git directory permissions..."
   - "✅ .git permissions fixed"

## Common Patterns

### Pattern 1: Always Clean on Self-Hosted
```yaml
- uses: actions/checkout@v4
  with:
    clean: true  # ← Always true for self-hosted runners
```

### Pattern 2: Shallow Clone for Speed
```yaml
- uses: actions/checkout@v4
  with:
    clean: true
    fetch-depth: 1  # ← Only fetch latest commit
```

### Pattern 3: Minimal Permissions
```yaml
permissions:
  contents: read  # ← Only what's needed
```

## Troubleshooting Updated Workflows

### Issue: Cleanup action not found

**Error:**
```
Error: Unable to resolve action ./.github/actions/cleanup-workspace
```

**Solution:**
Ensure the action exists and checkout happens AFTER cleanup (for non-local actions):

```yaml
steps:
  # If using inline cleanup:
  - name: Pre-job cleanup
    run: |
      find "${GITHUB_WORKSPACE}" -name "*.lock" -delete 2>/dev/null || true
    continue-on-error: true
  
  - uses: actions/checkout@v4
    with:
      clean: true
```

### Issue: Cleanup fails

**Error:**
```
Error: Process completed with exit code 1
```

**Solution:**
Always use `continue-on-error: true` for cleanup steps:

```yaml
- name: Pre-job cleanup
  uses: ./.github/actions/cleanup-workspace
  continue-on-error: true  # ← Don't fail job if cleanup fails
```

## Best Practices Summary

1. ✅ Add cleanup as the FIRST step
2. ✅ Use `clean: true` in checkout
3. ✅ Use `continue-on-error: true` for cleanup
4. ✅ Use `fetch-depth: 1` for speed
5. ✅ Test workflows after updates
6. ✅ Monitor for permission errors
7. ✅ Keep cleanup action updated

## Next Steps

1. Review your workflows
2. Identify self-hosted runner jobs
3. Add cleanup action to each job
4. Test workflows
5. Monitor for errors
6. Document any custom patterns
