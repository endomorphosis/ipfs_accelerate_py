# Auto-Healing Error Handling System

This implementation adds automatic error handling and GitHub integration to the `ipfs-accelerate` CLI tool, enabling automatic issue creation, draft PR generation, and GitHub Copilot-based auto-healing.

## Overview

The auto-healing system consists of three main components:

1. **Error Handler (`error_handler.py`)**: Captures CLI errors with full context
2. **Error Aggregator (`github_cli/error_aggregator.py`)**: Aggregates errors across P2P peers and manages GitHub integration
3. **CLI Integration (`cli.py`)**: Integrates error handling into the main CLI

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Execution                           │
│                   (ipfs-accelerate)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Error Occurs       │
            └──────────┬───────────┘
                       │
                       ▼
       ┌───────────────────────────────────┐
       │     CLIErrorHandler               │
       │  - Captures stack trace            │
       │  - Captures log context            │
       │  - Determines severity             │
       └────────┬──────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────┐
    │    ErrorAggregator (P2P)              │
    │  - Distributes to peers               │
    │  - Deduplicates errors                │
    │  - Bundles similar errors             │
    └────────┬──────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────┐
    │   GitHub Integration                    │
    │  1. Create Issue (via gh CLI)          │
    │  2. Create Draft PR (if enabled)       │
    │  3. Invoke Copilot (if enabled)        │
    └────────────────────────────────────────┘
```

## Features Implemented

### 1. Error Capture

- ✅ Full stack trace capture
- ✅ Command-line context
- ✅ Log context (last 50 lines by default)
- ✅ System information
- ✅ Severity determination
- ✅ Error signature generation for deduplication

### 2. GitHub Issue Creation

- ✅ Automatic issue creation via GitHub CLI
- ✅ Rich issue body with:
  - Error type and message
  - Full stack trace
  - Preceding log lines
  - System context
  - Error signature for deduplication
- ✅ Auto-generated labels
- ✅ Priority labeling based on severity

### 3. Draft PR Generation

- ✅ Branch name generation (`auto-fix/issue-{number}-{error-type}`)
- ✅ PR title and body generation
- ✅ Linking to the original issue
- ⚠️ **Note**: Actual PR creation requires branch creation and commits (placeholder implemented)

### 4. GitHub Copilot Integration

- ✅ Copilot SDK integration structure
- ✅ Prompt generation for error analysis
- ✅ Configuration for auto-healing
- ⚠️ **Note**: Requires `github-copilot-sdk` to be installed

### 5. P2P Error Aggregation

- ✅ Error distribution across peers
- ✅ Deduplication based on error signature
- ✅ Bundling (15-minute intervals)
- ✅ Minimum occurrence threshold (3 by default)
- ✅ Existing issue checking to prevent duplicates

### 6. Configuration

- ✅ Environment variable configuration
- ✅ Per-feature toggles
- ✅ Custom repository configuration
- ✅ Graceful fallback when features unavailable

## Configuration

### Environment Variables

| Variable | Purpose | Default | Values |
|----------|---------|---------|--------|
| `IPFS_AUTO_ISSUE` | Auto-create GitHub issues | `false` | `true`, `false`, `1`, `0`, `yes`, `no` |
| `IPFS_AUTO_PR` | Auto-create draft PRs | `false` | `true`, `false`, `1`, `0`, `yes`, `no` |
| `IPFS_AUTO_HEAL` | Invoke Copilot for fixes | `false` | `true`, `false`, `1`, `0`, `yes`, `no` |
| `IPFS_REPO` | Target GitHub repository | `endomorphosis/ipfs_accelerate_py` | `owner/repo` |

### Usage Examples

#### Enable Auto-Issue Creation

```bash
export IPFS_AUTO_ISSUE=true
ipfs-accelerate <command>
```

#### Enable Full Auto-Healing

```bash
export IPFS_AUTO_ISSUE=true
export IPFS_AUTO_PR=true
export IPFS_AUTO_HEAL=true
ipfs-accelerate <command>
```

#### One-Time Enable

```bash
IPFS_AUTO_ISSUE=true ipfs-accelerate inference generate --model bert-base
```

## Files Created/Modified

### New Files

1. **`ipfs_accelerate_py/error_handler.py`** (530 lines)
   - Main error handling class
   - GitHub integration
   - Copilot SDK wrapper
   - CLI function wrapping

2. **`docs/AUTO_HEALING_CONFIGURATION.md`** (300 lines)
   - Comprehensive documentation
   - Usage examples
   - Configuration guide
   - Troubleshooting

3. **`examples/auto_healing_demo.py`** (230 lines)
   - Demonstrates all features
   - Shows configuration options
   - Includes error simulation

4. **`test/test_error_handler.py`** (260 lines)
   - Unit tests for error handler
   - Mock-based testing
   - Integration test structure

### Modified Files

1. **`ipfs_accelerate_py/cli.py`**
   - Added error handler initialization in `main()`
   - Environment variable parsing
   - Error capture in exception handlers
   - Cleanup on exit

2. **`ipfs_accelerate_py/github_cli/error_aggregator.py`**
   - Added `enable_auto_pr_creation` parameter
   - Added `enable_copilot_autofix` parameter
   - Added `_create_draft_pr_from_issue()` method
   - Added `_invoke_copilot_autofix()` method

## Testing

### Run the Demo

```bash
cd /home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py
python3 examples/auto_healing_demo.py
```

### Run Unit Tests

```bash
pytest test/test_error_handler.py -v
```

### Manual Testing

1. **Test Error Capture**:
   ```bash
   python3 -c "from ipfs_accelerate_py.error_handler import CLIErrorHandler; handler = CLIErrorHandler('test/repo'); print('✓ Success')"
   ```

2. **Test with Real CLI** (requires dependencies):
   ```bash
   export IPFS_AUTO_ISSUE=false
   ipfs-accelerate --help
   ```

3. **Test with Auto-Issue** (requires `gh` CLI):
   ```bash
   gh auth login
   export IPFS_AUTO_ISSUE=true
   export IPFS_REPO=your-org/test-repo
   # Trigger an error intentionally
   ```

## Dependencies

### Required
- Python 3.8+
- `requests` (for GitHub API fallback)

### Optional (for full features)
- `gh` CLI (for GitHub integration)
- `github-copilot-sdk` (for auto-healing)
- `anyio` (for async operations)
- `libp2p` (for P2P error aggregation)

## Limitations & Future Work

### Current Limitations

1. **Draft PR Creation**: Stub implementation
   - Creates branch name and PR body
   - Doesn't create actual branch or commits
   - **Reason**: Requires decision on what changes to make

2. **Auto-Healing**: Partial implementation
   - Generates Copilot prompts
   - Doesn't apply fixes automatically
   - **Reason**: Requires approval workflow for code changes

3. **P2P Aggregation**: Requires connectivity
   - Works best with libp2p running
   - Gracefully degrades without it

### Future Enhancements

- [ ] Complete PR creation workflow
- [ ] Automated fix application (with approval)
- [ ] Integration with GitHub Actions
- [ ] Machine learning for error pattern detection
- [ ] Automatic rollback on critical errors
- [ ] Integration with monitoring services

## Security Considerations

1. **GitHub Authentication**: Uses existing `gh` CLI credentials
2. **Permissions**: Requires repo write access for issues/PRs
3. **Sensitive Data**: Stack traces may contain sensitive info
4. **Rate Limiting**: Implements delays to prevent API abuse

## How It Works

### Error Flow

1. **Error Occurs** in CLI command
2. **Error Handler** captures:
   - Stack trace
   - Command context
   - Log lines (last 50)
   - System info
3. **Error Aggregator** (if available):
   - Distributes to P2P peers
   - Deduplicates by signature
   - Bundles similar errors
4. **GitHub Integration** (if enabled):
   - Creates issue with details
   - Generates draft PR (if enabled)
   - Invokes Copilot (if enabled)

### Example Issue Created

```markdown
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
    result = generate_inference(args)
  File "inference.py", line 45, in generate_inference
    model = load_model(config)
  File "model.py", line 78, in load_model
    raise ValueError("Invalid model configuration: missing 'model_path' parameter")
ValueError: Invalid model configuration: missing 'model_path' parameter
```

## Preceding Logs
Last 50 log lines before error:
```
[12:34:50] INFO: Starting inference generation
[12:34:51] INFO: Loading model configuration
[12:34:52] WARNING: Model path not found in config
[12:34:55] ERROR: Configuration validation failed
```

---
*This issue was automatically created by the IPFS Accelerate error handler.*
```

## Troubleshooting

### "Error handler not available"

**Cause**: Import error or missing dependencies

**Solution**:
```bash
pip install requests
```

### "GitHub CLI not authenticated"

**Cause**: `gh` CLI not set up

**Solution**:
```bash
gh auth login
```

### "Could not initialize error aggregator"

**Cause**: P2P dependencies missing (optional feature)

**Solution**: This is expected and non-critical. Error capture still works.

## Support

For questions or issues:

1. Check `docs/AUTO_HEALING_CONFIGURATION.md`
2. Run `examples/auto_healing_demo.py`
3. Review test cases in `test/test_error_handler.py`
4. Create a GitHub issue (manually or auto-generated!)

## See Also

- [Auto-Healing Configuration Guide](../docs/AUTO_HEALING_CONFIGURATION.md)
- [Error Aggregator Source](../ipfs_accelerate_py/github_cli/error_aggregator.py)
- [CLI Integration](../ipfs_accelerate_py/cli.py)
- [Demo Example](../examples/auto_healing_demo.py)
