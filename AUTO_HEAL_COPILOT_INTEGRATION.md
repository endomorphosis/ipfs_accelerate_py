# Auto-Heal GitHub Copilot Integration

## Overview

This document describes how the Auto-Heal workflow system integrates with GitHub Copilot agents to automatically fix workflow failures.

## How It Works

### 1. Workflow Failure Detection

When a GitHub Actions workflow fails, the auto-heal system is triggered via the `workflow_run` event:

```yaml
on:
  workflow_run:
    workflows:
      - "AMD64 CI/CD Pipeline"
      - "ARM64 CI/CD Pipeline"
      - "Multi-Architecture CI/CD Pipeline"
      - "Package Installation Test"
      - "Test Auto-Heal System"
    types:
      - completed
```

### 2. Failure Analysis

The system analyzes the failure logs to:
- Identify failed jobs and steps
- Extract error messages
- Categorize the failure type
- Determine confidence level
- Suggest potential fixes

### 3. Issue Creation

A tracking issue is created with:
- Detailed failure logs
- Root cause analysis
- Suggested fixes
- Links to the failed run

### 4. Branch Creation

A new branch is created from the failed commit:
```
auto-heal/workflow-<run-id>-<timestamp>
```

### 5. Automated Fix Attempt

The system tries to apply automated fixes for common issues:
- **Dependency issues**: Missing packages added to requirements.txt
- **Timeout issues**: Timeout values increased
- **Permission issues**: Permission blocks added to workflows
- **Syntax issues**: YAML formatting fixed

### 6. Draft PR with GitHub Copilot

If automated fixes aren't applicable, the system:

1. **Creates a draft PR** with detailed context:
   - Workflow name and run ID
   - Failure logs and error messages
   - Root cause analysis
   - Suggested fixes

2. **Mentions @copilot** in the PR description:
   ```markdown
   @copilot Please analyze the workflow failure described above and implement fixes to resolve the issue:
   
   1. Review the failure analysis and error logs
   2. Identify the root cause
   3. Implement the minimal necessary fixes
   4. Ensure the fix doesn't break other functionality
   5. Test that the workflow passes after your changes
   ```

3. **Adds @copilot /fix comment** to explicitly invoke the agent:
   ```
   @copilot /fix
   
   Please analyze the workflow failure in the PR description and implement fixes 
   to resolve the issue. The failure details and error logs are provided above.
   
   Focus on making minimal changes to fix the specific issue identified in the 
   failure analysis.
   ```

4. **Links the PR to the tracking issue**

### 7. GitHub Copilot Works

The GitHub Copilot agent:
- Analyzes the failure context
- Reviews the error logs
- Implements fixes
- Commits changes to the PR branch
- Updates the PR description

### 8. Human Review

A human reviewer:
- Reviews the Copilot-proposed fixes
- Tests the changes
- Marks the PR as ready for review
- Merges if the tests pass
- Closes the tracking issue

## Key Improvements

The latest version of the auto-heal system includes these improvements:

1. **Simplified Copilot Invocation**: Direct @copilot mention in PR description and comment, following the pattern used by VS Code and GitHub's own tools.

2. **Removed Redundant Steps**: Eliminated the separate issue comment step since Copilot is now invoked directly in the PR.

3. **Better Context**: All failure information is provided in the PR body where Copilot can easily access it.

4. **Dual Invocation**: Both PR description mention and explicit /fix command ensure Copilot picks up the task.

## Example Flow

```mermaid
graph LR
    A[Workflow Fails] --> B[Auto-Heal Triggered]
    B --> C[Analyze Failure]
    C --> D[Create Issue]
    D --> E[Create Branch]
    E --> F{Auto-Fix Available?}
    F -->|Yes| G[Apply Fix & Create PR]
    F -->|No| H[Create Draft PR]
    H --> I[@copilot Mentioned]
    I --> J[Copilot Analyzes]
    J --> K[Copilot Implements Fix]
    K --> L[Human Reviews]
    L --> M[Merge PR]
    G --> L
```

## Testing

To test the auto-heal system with Copilot integration:

1. **Trigger a test failure**:
   ```bash
   gh workflow run "Test Auto-Heal System" --field failure_type=dependency_error
   ```

2. **Watch for the auto-heal workflow** to start

3. **Check the created issue** for failure details

4. **Review the draft PR** that mentions @copilot

5. **Watch for Copilot** to analyze and propose fixes

## Monitoring

Track auto-heal activity:

- **Issues**: Filter by `auto-heal` label
- **Pull Requests**: Filter by `auto-heal` and `automated` labels
- **Workflow Runs**: Check "Auto-Heal Workflow Failures" in Actions tab

## Configuration

The auto-heal workflow can be customized by editing:
```
.github/workflows/auto-heal-failures.yml
```

Key configuration points:
- Monitored workflows (lines 6-11)
- Permissions (lines 21-25)
- Trigger conditions (line 31)

## Troubleshooting

### Copilot Not Responding

If GitHub Copilot doesn't pick up the PR:

1. Verify Copilot subscription is active
2. Check that @copilot is mentioned in PR description
3. Verify the /fix command is in a PR comment
4. Manually mention @copilot again if needed

### Auto-Heal Not Triggering

If auto-heal doesn't start:

1. Check the workflow failed (not cancelled or skipped)
2. Verify the failed workflow is in the monitored list
3. Check auto-heal workflow permissions
4. Review auto-heal workflow run logs

## Benefits

✅ **Automatic Issue Creation**: Every failure is tracked
✅ **Intelligent Analysis**: Root cause identification  
✅ **Automated Fixes**: Common issues fixed without human intervention
✅ **Copilot Integration**: Complex issues handled by AI agent
✅ **Audit Trail**: Complete history of failures and fixes
✅ **Time Savings**: Reduces manual debugging time

## Security Considerations

- All PRs are created as drafts requiring review
- Automated fixes are limited to safe operations
- Human review required before merge
- Complete audit trail maintained
- Branch isolation prevents direct changes to main

## Next Steps

Future enhancements planned:
- [ ] Support for more failure patterns
- [ ] Integration with more CI/CD systems
- [ ] Better Copilot prompt engineering
- [ ] Automatic rollback on fix failures
- [ ] Success rate tracking and analytics

---

**Last Updated**: 2025-10-30
**Version**: 2.0
