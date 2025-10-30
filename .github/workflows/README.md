# GitHub Actions Workflows

## Auto-Heal System

### Auto-Heal Workflow Failures (`auto-heal-failures.yml`)

**Purpose**: Automatically detects, analyzes, and attempts to fix failing GitHub Actions workflows.

**Triggers**:
- `workflow_run` - Automatically runs when monitored workflows fail
- `workflow_dispatch` - Manual trigger for testing (optional run_id parameter)

**Monitored Workflows**:
- AMD64 CI/CD Pipeline
- ARM64 CI/CD Pipeline
- Multi-Architecture CI/CD Pipeline
- Package Installation Test
- Test Auto-Heal System

**What It Does**:
1. Detects when a monitored workflow fails
2. Analyzes failure logs using GitHub API
3. Categorizes the failure (dependency, resource, syntax, etc.)
4. Creates a tracking issue with full analysis
5. Creates an auto-heal branch for fixes
6. Creates a PR (draft or ready for Copilot)
7. Posts instructions for GitHub Copilot Workspace

**Manual Testing**:
```bash
# Get a failed workflow run ID
gh run list --workflow="AMD64 CI/CD Pipeline" --status=failure --limit=1 --json databaseId --jq '.[0].databaseId'

# Trigger auto-heal manually
gh workflow run auto-heal-failures.yml -f run_id=`<run_id>`
```

**Outputs**:
- GitHub Issue with `auto-heal` label
- Auto-heal branch: `auto-heal/workflow-<run_id>-<timestamp>`
- Draft Pull Request (if automated fixes applied) or placeholder PR for Copilot
- Workflow artifacts with detailed analysis

### Test Auto-Heal System (`test-auto-heal.yml`)

**Purpose**: Simulates various types of workflow failures to test the auto-heal system.

**Trigger**: Manual via `workflow_dispatch`

**Failure Types Available**:
1. **dependency_error** - Python module not found
2. **syntax_error** - Python syntax error
3. **timeout_error** - Job timeout (1 minute)
4. **resource_error** - Disk space error simulation
5. **docker_error** - Docker build failure
6. **test_failure** - Test assertion failure

**How to Use**:
1. Go to Actions tab
2. Select "Test Auto-Heal System"
3. Click "Run workflow"
4. Choose a failure type
5. Watch it fail, then check for auto-heal workflow to trigger

**Expected Behavior**:
- Workflow fails with chosen error type
- Auto-heal workflow triggers within 30 seconds
- Issue is created with failure analysis
- PR is created for Copilot to work on

### Cleanup Old Auto-Heal Branches (`cleanup-auto-heal-branches.yml`)

**Purpose**: Removes stale auto-heal branches that are no longer needed.

**Triggers**:
- `schedule` - Weekly on Sunday at 2 AM UTC
- `workflow_dispatch` - Manual trigger with options

**What It Does**:
1. Lists all auto-heal/* branches
2. Identifies branches older than N days (default: 7)
3. Excludes branches with open PRs
4. Deletes stale branches (or shows them in dry-run mode)

**Parameters**:
- `days_old` - Delete branches older than this many days (default: 7)
- `dry_run` - Preview deletions without actually deleting (default: false)

**Manual Run**:
```bash
# Dry run to see what would be deleted
gh workflow run cleanup-auto-heal-branches.yml -f dry_run=true -f days_old=7

# Actually delete old branches
gh workflow run cleanup-auto-heal-branches.yml -f dry_run=false -f days_old=14
```

## Main CI/CD Workflows

### AMD64 CI/CD Pipeline (`amd64-ci.yml`)

**Purpose**: Tests and builds on AMD64 architecture

**Triggers**: push, pull_request

**Status**: ⚠️ Failing with disk space issues during Docker builds (as of 2025-10-30)

### ARM64 CI/CD Pipeline (`arm64-ci.yml`)

**Purpose**: Tests and builds on ARM64 architecture

**Triggers**: push, pull_request

### Multi-Architecture CI/CD Pipeline (`multiarch-ci.yml`)

**Purpose**: Tests and builds for multiple architectures

**Triggers**: push, pull_request

### Package Installation Test (`package-test.yml`)

**Purpose**: Tests package installation in clean environments

**Triggers**: push, pull_request

## Documentation Maintenance

### Weekly Documentation Maintenance (`documentation-maintenance.yml`)

**Purpose**: Automated documentation updates and maintenance

**Triggers**: schedule (weekly), workflow_dispatch

**Excluded from Auto-Heal**: Yes (to avoid interference)

## Troubleshooting

### Auto-Heal Not Triggering

1. Check that the failed workflow is in the monitored list
2. Verify the workflow concluded with `failure` (not `cancelled`)
3. Check workflow run logs for auto-heal workflow
4. Try manual trigger with workflow_dispatch

### Issue or PR Not Created

1. Check permissions (contents: write, issues: write, pull-requests: write)
2. Review auto-heal workflow logs
3. Check rate limits
4. Verify GitHub token has proper scopes

### Copilot Not Responding

1. Verify Copilot subscription is active for the repo
2. Check that PR was created with proper labels
3. Manually mention @github-copilot in the PR
4. Review Copilot status page

## Quick Reference

```bash
# List all auto-heal issues
gh issue list --label auto-heal

# List all auto-heal PRs
gh pr list --label auto-heal

# View recent workflow runs
gh run list --workflow=auto-heal-failures.yml --limit=5

# View workflow logs
gh run view <run_id> --log

# Trigger test failure
gh workflow run test-auto-heal.yml -f failure_type=dependency_error

# Clean up old branches (dry run)
gh workflow run cleanup-auto-heal-branches.yml -f dry_run=true
```

## Configuration Files

- `.github/workflows/auto-heal-failures.yml` - Main auto-heal workflow
- `.github/workflows/test-auto-heal.yml` - Testing workflow
- `.github/workflows/cleanup-auto-heal-branches.yml` - Cleanup workflow
- `.github/scripts/workflow_failure_analyzer.py` - Failure analysis script
- `.github/scripts/auto_fix_common_issues.py` - Automated fix script
- `.github/scripts/cleanup_old_branches.py` - Branch cleanup script

## Documentation

- `AUTO_HEAL_IMPLEMENTATION_SUMMARY.md` - Complete system overview
- `AUTO_HEAL_FIXES_2025-10-30.md` - Recent fixes and improvements
- `.github/workflows/README.md` - This file

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review workflow logs in Actions tab
3. Create an issue with `auto-heal-support` label
4. Check documentation files

---

**Last Updated**: 2025-10-30  
**Maintained By**: Development Team
