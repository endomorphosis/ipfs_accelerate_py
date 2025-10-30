# 🔧 GitHub Actions Auto-Healing System

## Overview

The Auto-Healing System is an automated workflow failure detection and remediation system that uses **GitHub Copilot Workspace** to automatically diagnose and fix failing GitHub Actions workflows.

## 🎯 Features

- **Automatic Failure Detection**: Monitors all GitHub Actions workflows for failures
- **Intelligent Analysis**: Analyzes failure logs to identify root causes
- **AI-Powered Fixes**: Uses GitHub Copilot to generate fixes automatically
- **Issue Tracking**: Creates tracking issues for every failure
- **Pull Request Generation**: Automatically creates PRs with proposed fixes
- **Comprehensive Logging**: Maintains detailed logs and reports for audit
- **Configurable Behavior**: Extensive configuration options for customization

## 🚀 How It Works

```mermaid
graph TD
    A[Workflow Fails] --> B[Auto-Heal Triggered]
    B --> C[Analyze Failure Logs]
    C --> D[Create Tracking Issue]
    D --> E[Create Auto-Heal Branch]
    E --> F[Trigger GitHub Copilot]
    F --> G[Copilot Analyzes & Fixes]
    G --> H[Create Pull Request]
    H --> I[Run Tests on Fix]
    I --> J{Tests Pass?}
    J -->|Yes| K[Ready for Review]
    J -->|No| L[Update Issue]
    K --> M[Manual Review]
    M --> N[Merge PR]
    N --> O[Original Workflow Passes]
```

## 📋 Workflow Lifecycle

### 1. Failure Detection

When any GitHub Actions workflow fails:
- The `workflow_run` event triggers the auto-heal workflow
- Only workflows with `conclusion: failure` are processed

### 2. Failure Analysis

The system:
- Fetches workflow run details via GitHub API
- Identifies all failed jobs and steps
- Extracts relevant error logs
- Creates a structured analysis JSON file

### 3. Issue Creation

A tracking issue is automatically created with:
- Workflow name and run ID
- Failed job details
- Error logs and context
- Links to the failed run
- Auto-heal label for tracking

### 4. Copilot Activation

GitHub Copilot Workspace is triggered via issue comment:
- Receives the complete failure analysis
- Gets context about the repository
- Receives instructions on what to fix
- Creates fixes on a dedicated branch

### 5. Pull Request Creation

Copilot creates a PR that:
- Contains the proposed fixes
- Links to the tracking issue
- Includes detailed explanation of changes
- Runs CI/CD tests to validate the fix

### 6. Review and Merge

The PR is:
- Created as a draft (by default)
- Reviewed by human developers
- Tested automatically via CI/CD
- Merged if tests pass and changes are approved

## 🔧 Setup Instructions

### Prerequisites

1. **GitHub Repository Settings**
   - Enable GitHub Actions
   - Enable GitHub Copilot (requires subscription)
   - Set up branch protection rules (recommended)

2. **Required Permissions**
   - The workflow needs these permissions:
     ```yaml
     permissions:
       contents: write
       pull-requests: write
       issues: write
       actions: read
     ```

### Installation

1. **Add the Auto-Heal Workflow**
   
   The auto-heal workflow is located at:
   ```
   .github/workflows/auto-heal-failures.yml
   ```

2. **Configure Settings**
   
   Edit `.github/auto-heal-config.yml` to customize:
   - Which workflows to monitor
   - Notification preferences
   - Copilot settings
   - Branch naming conventions
   - PR templates

3. **Set Up GitHub Secrets** (Optional)
   
   If using custom GitHub tokens:
   ```bash
   # Add to repository secrets
   GITHUB_TOKEN: <your-token>  # Default token usually works
   ```

4. **Test the System**
   
   Create a test workflow that intentionally fails:
   ```yaml
   name: Test Auto-Heal
   on: [workflow_dispatch]
   jobs:
     test-fail:
       runs-on: ubuntu-latest
       steps:
         - run: exit 1  # Intentional failure
   ```
   
   Run it manually and watch the auto-heal system activate!

## ⚙️ Configuration

### Basic Configuration

Edit `.github/auto-heal-config.yml`:

```yaml
# Enable/disable globally
enabled: true

# Monitor specific workflows
monitored_workflows:
  - "*"  # All workflows

# Exclude specific workflows
excluded_workflows:
  - "Auto-Heal Workflow Failures"
  
# Maximum healing attempts per day
max_heal_attempts_per_day: 3
```

### Advanced Configuration

#### Copilot Settings

```yaml
copilot:
  enabled: true
  model: "gpt-4"
  temperature: 0.2  # Conservative fixes
  include_context:
    - "README.md"
    - "requirements.txt"
```

#### Failure Pattern Recognition

```yaml
failure_patterns:
  - pattern: "no space left on device"
    suggestion: "Add disk cleanup step"
  
  - pattern: "timeout"
    suggestion: "Increase timeout value"
```

#### Security Settings

```yaml
security:
  trusted_branches:
    - "main"
    - "develop"
  require_code_review: true
  run_security_scans: true
  max_file_changes: 20
```

## 📊 Monitoring and Analytics

### View Auto-Heal Activity

1. **Issues Tab**
   - Filter by label: `auto-heal`
   - See all detected failures

2. **Pull Requests**
   - Filter by label: `automated-fix`
   - Review proposed fixes

3. **Actions Tab**
   - View "Auto-Heal Workflow Failures" runs
   - Check analysis artifacts

### Artifacts

Each auto-heal run creates artifacts containing:
- `failure_analysis.json` - Structured failure data
- `failure_report.md` - Human-readable report
- `healing_context.json` - Workflow context
- `copilot_healing_prompt.md` - Instructions for Copilot

Download artifacts from the Actions tab.

## 🎓 Usage Examples

### Example 1: Dependency Issue

**Failure**: Missing Python package

**Auto-Heal Action**:
1. Detects "ModuleNotFoundError: No module named 'requests'"
2. Creates issue #42
3. Copilot adds `requests` to requirements.txt
4. Creates PR #43 with fix
5. Tests pass, ready for review

### Example 2: Timeout Issue

**Failure**: Workflow step timeout

**Auto-Heal Action**:
1. Detects "The job running on runner Hosted Agent has exceeded the maximum execution time"
2. Creates issue #44
3. Copilot increases timeout from 30m to 60m
4. Creates PR #45 with fix
5. Tests pass, ready for review

### Example 3: Docker Build Failure

**Failure**: Docker build fails due to syntax error in Dockerfile

**Auto-Heal Action**:
1. Detects "ERROR: failed to solve: process '/bin/sh -c' failed"
2. Creates issue #46
3. Copilot fixes syntax in Dockerfile
4. Creates PR #47 with fix
5. Tests pass, ready for review

## 🔒 Security Considerations

### What Gets Auto-Healed

✅ **Safe to Auto-Heal**:
- Configuration file errors
- Syntax errors in workflows
- Missing dependencies
- Timeout issues
- Resource constraints

❌ **Not Safe to Auto-Heal** (requires manual review):
- Security vulnerabilities
- Authentication failures
- Credential issues
- Major architectural changes

### Security Best Practices

1. **Always review auto-heal PRs** before merging
2. **Set up branch protection** to require reviews
3. **Enable security scans** in configuration
4. **Limit to trusted branches** only
5. **Set reasonable file change limits**
6. **Monitor auto-heal activity** regularly

## 🐛 Troubleshooting

### Auto-Heal Not Triggering

**Check**:
1. Workflow failed (not cancelled or skipped)
2. Auto-heal workflow is enabled
3. Workflow not in excluded list
4. Permissions are correct

**Debug**:
```bash
# Check workflow runs
gh run list --workflow="auto-heal-failures.yml"

# View specific run
gh run view <run-id>
```

### Copilot Not Responding

**Check**:
1. Copilot subscription is active
2. Issue comment was created
3. Copilot has repository access
4. Instructions are clear in issue

**Manual Trigger**:
Comment on the issue:
```
@github-copilot workspace

Please analyze the failure in the issue description and create a fix.
```

### PR Not Created

**Check**:
1. Branch was created successfully
2. Copilot has push permissions
3. No conflicts with existing branches
4. File changes within limits

## 📈 Best Practices

### For Repository Maintainers

1. **Review Configuration**: Regularly review `.github/auto-heal-config.yml`
2. **Monitor Success Rate**: Track how many auto-heals succeed
3. **Update Patterns**: Add new failure patterns as you discover them
4. **Train Team**: Ensure team knows how to work with auto-heal PRs
5. **Set Limits**: Use `max_heal_attempts_per_day` to prevent loops

### For Developers

1. **Review Auto-Heal PRs Promptly**: Don't let them pile up
2. **Provide Feedback**: Comment on PRs to improve future fixes
3. **Update Documentation**: If auto-heal reveals gaps, update docs
4. **Test Thoroughly**: Don't trust blindly, test the fixes
5. **Close Tracking Issues**: Close issues when PRs are merged

## 🔄 Workflow Integration

### Integrating with Existing Workflows

The auto-heal system works with any workflow. No changes needed to existing workflows!

### Custom Integration

To add custom auto-heal behavior:

```yaml
# In your workflow
on:
  workflow_run:
    workflows: ["My Custom Workflow"]
    types: [completed]

jobs:
  custom-heal:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    # Custom healing logic
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [GitHub API Reference](https://docs.github.com/en/rest)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

## 🤝 Contributing

To improve the auto-healing system:

1. **Report Issues**: Create issues for bugs or enhancement requests
2. **Submit PRs**: Contribute improvements to the auto-heal workflow
3. **Share Patterns**: Add common failure patterns to the config
4. **Improve Documentation**: Help others understand the system

## 📝 License

This auto-healing system is part of the ipfs_accelerate_py project and follows the same license.

## 🎉 Success Stories

Track your auto-heal successes by tagging issues with `auto-heal-success`:

- **Issue #42**: Automatically fixed missing dependency ✅
- **Issue #44**: Resolved timeout issue ✅
- **Issue #46**: Fixed Docker build error ✅

---

## Quick Start Checklist

- [ ] Auto-heal workflow added to `.github/workflows/`
- [ ] Configuration file created at `.github/auto-heal-config.yml`
- [ ] GitHub Copilot subscription active
- [ ] Workflow permissions configured
- [ ] Team trained on reviewing auto-heal PRs
- [ ] Test workflow created and run
- [ ] Monitoring dashboard set up
- [ ] Security settings reviewed

**Need Help?** Create an issue with the `auto-heal-support` label!

---

*Last Updated: $(date +%Y-%m-%d)*
*Auto-Healing System Version: 1.0*
