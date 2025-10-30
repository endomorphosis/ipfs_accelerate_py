# Auto-Heal System Quick Reference

## Overview

The Auto-Heal system automatically detects workflow failures, analyzes them, and either applies automated fixes or invokes GitHub Copilot to implement fixes.

## Quick Start

### Automatic Mode (Production)

The auto-heal workflow runs automatically when any monitored workflow fails:
- AMD64 CI/CD Pipeline
- ARM64 CI/CD Pipeline  
- Multi-Architecture CI/CD Pipeline
- Package Installation Test
- Test Auto-Heal System

**No action needed** - the system activates automatically when failures occur.

### Manual Testing Mode

To test the auto-heal system with simulated failures:

```bash
# Simulate a dependency error
gh workflow run "Test Auto-Heal System" --field failure_type=dependency_error

# Other failure types available:
# - syntax_error
# - timeout_error
# - resource_error
# - docker_error
# - test_failure
```

### Manual Dispatch for Specific Failures

To run auto-heal for a specific workflow run:

```bash
# Get the run ID of a failed workflow
gh run list --workflow "AMD64 CI/CD Pipeline" --limit 5 --json databaseId,status,conclusion

# Manually trigger auto-heal for that run
gh workflow run "Auto-Heal Workflow Failures" --field run_id=<RUN_ID>
```

## What Happens When a Workflow Fails

1. **Detection** (< 1 minute)
   - Auto-heal workflow is triggered automatically
   - Failure logs are fetched and analyzed

2. **Issue Creation** (< 1 minute)
   - A tracking issue is created with label `auto-heal`
   - Issue includes complete failure logs and analysis
   - Issue is linked to the failed workflow run

3. **Branch Creation** (< 1 minute)
   - New branch created: `auto-heal/workflow-{run_id}-{timestamp}`
   - Branch starts from the failed commit

4. **Automated Fix Attempt** (1-2 minutes)
   - System tries to fix common issues:
     - Missing dependencies → Added to requirements.txt
     - Timeout issues → Timeout values increased
     - Permission issues → Permission blocks added
     - Syntax issues → Basic YAML/syntax fixes
   
5. **Two Possible Outcomes**:

   **A) Automated Fix Applied**
   - Fixes are committed to the auto-heal branch
   - Regular PR is created (not draft)
   - PR linked to tracking issue
   - Ready for human review and merge

   **B) GitHub Copilot Invoked**
   - Draft PR is created
   - @copilot mentioned in PR body with instructions
   - @copilot /fix command added in comment
   - Copilot analyzes and implements fixes
   - PR updated by Copilot with fixes
   - Ready for human review

## Monitoring Auto-Heal Activity

### Find Auto-Heal Issues

```bash
# List all auto-heal issues
gh issue list --label auto-heal

# List open auto-heal issues
gh issue list --label auto-heal --state open
```

### Find Auto-Heal PRs

```bash
# List all auto-heal PRs
gh pr list --label auto-heal

# List draft PRs waiting for Copilot
gh pr list --label auto-heal --state open --json isDraft,number,title | \
  jq '.[] | select(.isDraft == true)'
```

### Check Auto-Heal Workflow Runs

```bash
# List recent auto-heal workflow runs
gh run list --workflow "Auto-Heal Workflow Failures" --limit 10

# View details of a specific run
gh run view <RUN_ID>

# View logs of a specific run
gh run view <RUN_ID> --log
```

## Understanding Auto-Heal Labels

- `auto-heal` - Created by the auto-heal system
- `automated` - Automated actions taken (may include automated fixes)
- `workflow-failure` - Triggered by a workflow failure
- `automated-fix` - Automated fixes were applied (regular PR)
- `needs-review` - Ready for human review

## Reviewing and Merging Auto-Heal Fixes

### For Automated Fixes (Regular PRs)

1. **Review the Changes**
   ```bash
   gh pr view <PR_NUMBER>
   gh pr diff <PR_NUMBER>
   ```

2. **Check the PR runs the affected workflow**
   - The PR should trigger the same CI workflows
   - Verify the workflow passes with the fixes

3. **Merge if tests pass**
   ```bash
   gh pr merge <PR_NUMBER> --squash
   ```

4. **Close the tracking issue**
   ```bash
   gh issue close <ISSUE_NUMBER>
   ```

### For Copilot-Generated Fixes (Draft PRs)

1. **Wait for Copilot to update the PR**
   - Copilot should add commits to the PR
   - Check PR comments for Copilot's analysis

2. **Review Copilot's changes**
   ```bash
   gh pr view <PR_NUMBER>
   gh pr diff <PR_NUMBER>
   ```

3. **Mark as ready for review if changes look good**
   ```bash
   gh pr ready <PR_NUMBER>
   ```

4. **Run tests and merge**
   - Same process as automated fixes above

## Troubleshooting

### Auto-Heal Didn't Trigger

**Check**:
- Is the failed workflow in the monitored list?
- Did the workflow actually fail (not cancelled/skipped)?
- Check auto-heal workflow permissions
- View auto-heal workflow run logs for errors

### No Automated Fixes Applied

This is normal! Most fixes require Copilot intervention:
- Check for the draft PR with @copilot mention
- Wait for Copilot to analyze and implement fixes
- Review Copilot's changes when ready

### Copilot Didn't Respond

**Try**:
1. Check that Copilot subscription is active
2. Manually @mention Copilot in a new comment
3. Add another `/fix` command comment
4. Check PR for any error messages from Copilot

### Wrong Branch Used

The auto-heal branch is created from the failed commit:
- This ensures fixes are based on the exact state that failed
- You may need to merge/rebase with main before merging
- Or recreate the PR against main if preferred

## Configuration

### Add Workflows to Monitor

Edit `.github/workflows/auto-heal-failures.yml`:

```yaml
workflows:
  - "AMD64 CI/CD Pipeline"
  - "ARM64 CI/CD Pipeline"
  - "Your New Workflow Name"  # Add here
```

### Adjust Permissions

Auto-heal requires these permissions:
```yaml
permissions:
  contents: write      # Create branches, commit fixes
  pull-requests: write # Create PRs
  issues: write        # Create tracking issues
  actions: read        # Read workflow runs and logs
```

### Modify Automated Fix Patterns

Edit `.github/scripts/auto_fix_common_issues.py` to add new automated fix patterns.

Edit `.github/scripts/workflow_failure_analyzer.py` to add new failure detection patterns.

## Best Practices

✅ **Do**:
- Review all auto-heal PRs before merging
- Test fixes locally if uncertain
- Close tracking issues after merging fixes
- Monitor auto-heal effectiveness over time
- Add new fix patterns as common issues emerge

❌ **Don't**:
- Merge auto-heal PRs without review
- Delete auto-heal branches with open PRs
- Disable auto-heal without communication
- Ignore auto-heal issues accumulating
- Bypass the PR process

## Maintenance

### Cleanup Old Branches

Auto-heal branches are automatically cleaned up weekly:
```yaml
# Runs every Sunday at 2 AM UTC
# .github/workflows/cleanup-auto-heal-branches.yml
```

Manual cleanup:
```bash
# Dry run to see what would be deleted
gh workflow run "Cleanup Old Auto-Heal Branches" --field dry_run=true

# Actually delete old branches (>7 days)
gh workflow run "Cleanup Old Auto-Heal Branches"

# Delete branches older than 14 days
gh workflow run "Cleanup Old Auto-Heal Branches" --field days_old=14
```

## Support and Feedback

- **Issues**: Report problems as GitHub issues
- **Improvements**: Suggest enhancements via issues or PRs
- **Questions**: Check documentation or ask in discussions

## Related Documentation

- [AUTO_HEAL_COPILOT_INTEGRATION.md](AUTO_HEAL_COPILOT_INTEGRATION.md) - Detailed integration docs
- [AUTO_HEAL_WORKFLOW_FIXES_2025-10-30.md](AUTO_HEAL_WORKFLOW_FIXES_2025-10-30.md) - Recent fixes
- [Workflow Files](.github/workflows/) - Workflow source code
- [Scripts](.github/scripts/) - Analysis and fix scripts

---

**Last Updated**: October 30, 2025
**Version**: 2.0
