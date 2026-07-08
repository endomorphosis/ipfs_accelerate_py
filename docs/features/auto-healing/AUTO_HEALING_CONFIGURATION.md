# Auto-Healing Configuration for IPFS Accelerate

This document describes the auto-healing error handling features in IPFS Accelerate.

## Overview

The IPFS Accelerate CLI now includes automatic error handling capabilities that can:

1. **Capture errors** with full stack traces and log context
2. **Create GitHub issues** automatically when errors occur
3. **Generate draft PRs** to fix the issues
4. **Invoke GitHub Copilot** to suggest fixes and auto-heal errors

## Configuration

Auto-healing features are controlled via environment variables:

### Environment Variables

| Variable | Description | Default | Values |
|----------|-------------|---------|--------|
| `IPFS_AUTO_ISSUE` | Automatically create GitHub issues for errors | `false` | `true`, `false`, `1`, `0`, `yes`, `no` |
| `IPFS_AUTO_PR` | Automatically create draft PRs from issues | `false` | `true`, `false`, `1`, `0`, `yes`, `no` |
| `IPFS_AUTO_HEAL` | Invoke GitHub Copilot to suggest fixes | `false` | `true`, `false`, `1`, `0`, `yes`, `no` |
| `IPFS_REPO` | GitHub repository for issue creation | `endomorphosis/ipfs_accelerate_py` | `owner/repo` format |

### Enabling Auto-Healing

To enable all auto-healing features:

```bash
export IPFS_AUTO_ISSUE=true
export IPFS_AUTO_PR=true
export IPFS_AUTO_HEAL=true
export IPFS_REPO=your-org/your-repo  # Optional
```

Or for a single command:

```bash
IPFS_AUTO_ISSUE=true ipfs-accelerate <command>
```

## Features

### 1. Error Capture

When an error occurs in the CLI, the error handler automatically:

- Captures the full stack trace
- Records the command that was run
- Captures the last 50 lines of logs preceding the error
- Collects system context (Python version, working directory, etc.)

### 2. GitHub Issue Creation

When `IPFS_AUTO_ISSUE=true`, errors automatically create GitHub issues containing:

- Error type and message
- Full stack trace
- Preceding log context
- System information
- Auto-generated labels (`auto-generated`, `bug`, optionally `priority`)

Example issue:
```
Title: [Auto-Generated Error] ValueError: Invalid model configuration

Body:
# Auto-Generated Error Report

**Error Type:** `ValueError`
**Command:** `ipfs-accelerate inference generate --model bert-base`
**Timestamp:** 2024-01-31T12:34:56.789Z

## Error Message
```
Invalid model configuration: missing 'model_path' parameter
```

## Stack Trace
```python
Traceback (most recent call last):
  File "cli.py", line 123, in main
    ...
ValueError: Invalid model configuration
```

## Preceding Logs
Last 50 log lines before error:
```
[12:34:50] INFO: Loading model configuration...
[12:34:51] WARNING: Model path not found in config
[12:34:55] ERROR: Configuration validation failed
```

---
*This issue was automatically created by the IPFS Accelerate error handler.*
```

### 3. Draft PR Creation

When `IPFS_AUTO_PR=true`, the system will:

1. Create a GitHub issue for the error
2. Create a new branch named `auto-fix/issue-{number}-{error-type}`
3. Generate a draft PR that:
   - References the issue
   - Includes error details
   - Provides guidance for fixing
   - Automatically closes the issue when merged

### 4. GitHub Copilot Auto-Healing

When `IPFS_AUTO_HEAL=true`, the system will:

1. Invoke GitHub Copilot SDK to analyze the error
2. Generate fix suggestions including:
   - Root cause analysis
   - Recommended code changes
   - Files that need modification
   - Test cases to prevent regression
3. Add suggestions to the draft PR

**Note:** This requires the GitHub Copilot SDK to be installed:
```bash
pip install github-copilot-sdk
```

## Error Aggregation

The system uses P2P error aggregation to:

- Share errors across distributed instances
- Deduplicate similar errors
- Bundle multiple occurrences before creating issues
- Prevent duplicate issues

### Configuration

Error aggregation settings:

- **Bundle interval:** 15 minutes (configurable)
- **Minimum occurrences:** 3 errors before creating issue (configurable)
- **Deduplication:** Based on error signature (type + normalized message)

## Usage Examples

### Enable Auto-Issue for Critical Commands

```bash
# Monitor production deployments
IPFS_AUTO_ISSUE=true ipfs-accelerate mcp start --dashboard

# Track inference errors
IPFS_AUTO_ISSUE=true ipfs-accelerate inference generate --model gpt-4
```

### Full Auto-Healing Pipeline

```bash
# Enable all features
export IPFS_AUTO_ISSUE=true
export IPFS_AUTO_PR=true
export IPFS_AUTO_HEAL=true

# Run CLI - errors will automatically:
# 1. Create issues
# 2. Generate draft PRs
# 3. Invoke Copilot for fixes
ipfs-accelerate <any-command>
```

### Custom Repository

```bash
# Send issues to a different repository
export IPFS_REPO=my-org/my-fork
export IPFS_AUTO_ISSUE=true

ipfs-accelerate <command>
```

## Programmatic Usage

You can also use the error handler programmatically:

```python
from ipfs_accelerate_py.error_handler import CLIErrorHandler

# Initialize error handler
handler = CLIErrorHandler(
    repo='my-org/my-repo',
    enable_auto_issue=True,
    enable_auto_pr=True,
    enable_auto_heal=True,
    log_context_lines=50
)

# Wrap your CLI function
@handler.wrap_cli_main
def my_cli_main():
    # Your CLI logic here
    pass

# Or capture errors manually
try:
    risky_operation()
except Exception as e:
    handler.capture_error(e, context={'operation': 'risky'})
    handler.create_issue_from_error(e)
```

## Security Considerations

1. **GitHub Authentication:** Requires `gh` CLI to be authenticated
2. **Permissions:** Needs write access to the repository for issue/PR creation
3. **Sensitive Data:** Be cautious about exposing sensitive information in stack traces
4. **Rate Limits:** Issue creation is rate-limited to prevent API abuse

## Best Practices

1. **Development:** Keep auto-features disabled during development
   ```bash
   # Development
   ipfs-accelerate <command>
   ```

2. **CI/CD:** Enable auto-issue for production deployments
   ```bash
   # CI/CD pipeline
   IPFS_AUTO_ISSUE=true ipfs-accelerate mcp start
   ```

3. **Testing:** Use a test repository for auto-healing
   ```bash
   # Testing
   export IPFS_REPO=my-org/test-repo
   export IPFS_AUTO_ISSUE=true
   ```

4. **Monitoring:** Review auto-generated issues regularly
   - Check for patterns in errors
   - Merge helpful auto-fix PRs
   - Close false positives

## Troubleshooting

### Issues not being created

1. Check GitHub CLI authentication:
   ```bash
   gh auth status
   ```

2. Verify repository access:
   ```bash
   gh repo view owner/repo
   ```

3. Enable debug logging:
   ```bash
   IPFS_AUTO_ISSUE=true ipfs-accelerate --debug <command>
   ```

### Copilot not working

1. Install Copilot SDK:
   ```bash
   pip install github-copilot-sdk
   ```

2. Verify Copilot access:
   ```bash
   # Check if you have Copilot access in your GitHub account
   ```

## Limitations

1. **PR Creation:** Draft PR creation requires an actual branch with commits
2. **Auto-Healing:** Fully automated fixes require additional infrastructure
3. **P2P Features:** Error aggregation requires P2P connectivity
4. **Rate Limits:** GitHub API has rate limits for issue creation

## Future Enhancements

Planned improvements:

- [ ] Full automated PR creation with actual code changes
- [ ] Integration with GitHub Actions for CI testing
- [ ] Machine learning-based error pattern detection
- [ ] Automatic rollback on critical errors
- [ ] Integration with monitoring services (DataDog, Sentry, etc.)

## Support

For issues or questions:

1. Check existing GitHub issues
2. Review auto-generated issues for similar problems
3. Create a manual issue if auto-healing didn't catch it
4. Contact maintainers

## See Also

- [ErrorAggregator Documentation](../ipfs_accelerate_py/github_cli/error_aggregator.py)
- [CLI Documentation](../ipfs_accelerate_py/cli.py)
- [GitHub CLI Integration](../ipfs_accelerate_py/github_cli/)
- [Copilot SDK Integration](../ipfs_accelerate_py/copilot_sdk/)
