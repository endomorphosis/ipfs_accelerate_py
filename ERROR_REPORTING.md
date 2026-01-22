# Automated Error Reporting System

## Overview

The IPFS Accelerate error reporting system automatically converts runtime errors into GitHub issues for tracking and resolution. This applies to errors occurring in:

- **Python MCP Server**: Runtime errors during MCP server operation
- **JavaScript Dashboard**: Browser-side errors in the web dashboard
- **Docker Container**: Errors within containerized Python execution

## Features

- ✅ **Automatic Issue Creation**: Errors are automatically reported as GitHub issues
- ✅ **Duplicate Detection**: Prevents creating multiple issues for the same error
- ✅ **Rich Context**: Includes error details, stack traces, and system information
- ✅ **Smart Labeling**: Automatically categorizes issues with appropriate labels
- ✅ **Component Tracking**: Identifies which component (MCP server, dashboard, Docker) generated the error
- ✅ **Configurable**: Can be enabled/disabled via environment variables

## Setup

### 1. Create GitHub Personal Access Token

1. Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Give it a descriptive name (e.g., "IPFS Accelerate Error Reporting")
4. Select the `repo` scope (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't be able to see it again)

### 2. Configure Environment Variables

Create a `.env` file in the project root (or set environment variables):

```bash
# GitHub Configuration
GITHUB_TOKEN=ghp_your_token_here
GITHUB_REPO=your_username/ipfs_accelerate_py

# Optional: Error Reporting Settings
ERROR_REPORTING_ENABLED=true
ERROR_REPORTING_INCLUDE_SYSTEM_INFO=true
ERROR_REPORTING_AUTO_LABEL=true
```

**For Docker:**

```bash
docker run -e GITHUB_TOKEN=ghp_xxx -e GITHUB_REPO=user/repo ipfs_accelerate
```

**For Docker Compose:**

```yaml
services:
  ipfs_accelerate:
    environment:
      - GITHUB_TOKEN=ghp_xxx
      - GITHUB_REPO=user/repo
```

### 3. Verify Configuration

Test the error reporting system:

```bash
# Python
python -c "from utils.error_reporter import get_error_reporter; print(get_error_reporter().enabled)"

# Via API (if server is running)
curl http://localhost:5000/api/error-reporter/status
```

## Usage

### Python MCP Server

Error reporting is automatically enabled when you import the MCP server:

```python
from mcp.server import run_server

# Error reporting is automatically installed
# Any uncaught exceptions will be reported to GitHub
```

### JavaScript Dashboard

Error reporting is automatically initialized when the dashboard loads. No additional configuration needed:

```html
<!-- Already included in dashboard.html -->
<script src="/static/js/error-reporter.js"></script>
<script>
    installGlobalErrorHandler('dashboard');
</script>
```

### Manual Error Reporting

#### Python

```python
from utils.error_reporter import report_error

try:
    # Your code here
    risky_operation()
except Exception as e:
    # Manually report an error
    issue_url = report_error(
        exception=e,
        source_component='my-component',
        context={'additional': 'information'}
    )
    if issue_url:
        print(f"Error reported: {issue_url}")
```

#### JavaScript

```javascript
try {
    // Your code here
    riskyOperation();
} catch (error) {
    // Manually report an error
    reportError({
        error: error,
        sourceComponent: 'my-component',
        context: { additional: 'information' }
    }).then(issueUrl => {
        if (issueUrl) {
            console.log('Error reported:', issueUrl);
        }
    });
}
```

### Docker Container

The Docker container automatically wraps Python execution with error reporting:

```bash
# Run with error reporting
docker run \
  -e GITHUB_TOKEN=ghp_xxx \
  -e GITHUB_REPO=user/repo \
  ipfs_accelerate python script.py

# Or use the wrapper directly
docker exec container_name python /app/docker_error_wrapper.py -m mymodule
```

## Architecture

### Components

1. **Error Reporter (`utils/error_reporter.py`)**
   - Core error reporting logic
   - GitHub API integration
   - Duplicate detection
   - Error hash computation

2. **Error Reporting API (`utils/error_reporting_api.py`)**
   - REST API endpoints for JavaScript clients
   - `/api/report-error` - Report an error
   - `/api/error-reporter/status` - Check status
   - `/api/error-reporter/test` - Test reporting

3. **JavaScript Error Reporter (`static/js/error-reporter.js`)**
   - Browser-side error capture
   - Global error handlers
   - API client for backend

4. **Docker Error Wrapper (`docker_error_wrapper.py`)**
   - Wraps Python execution in containers
   - Captures and reports container errors

### Error Flow

```
┌─────────────────┐
│  Error Occurs   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Error Reporter  │
│  - Compute hash │
│  - Check cache  │
│  - Gather info  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GitHub API    │
│  Create Issue   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update Cache   │
│  Return URL     │
└─────────────────┘
```

## Issue Format

Created issues include:

- **Title**: `[Auto] ErrorType: component-name`
- **Labels**: `bug`, `automated-report`, component-specific labels
- **Body**:
  - Error type and message
  - Component that generated the error
  - Timestamp
  - Full stack trace
  - Additional context (if provided)
  - System information (if enabled)

Example:

```markdown
## Automated Error Report

**Error Type:** `ValueError`
**Component:** `mcp-server`
**Timestamp:** 2025-11-06T08:23:41.206Z

### Error Message
```
Invalid configuration: missing required field 'model_id'
```

### Traceback
```python
Traceback (most recent call last):
  File "server.py", line 42, in run
    validate_config(config)
  File "utils.py", line 15, in validate_config
    raise ValueError("Invalid configuration: missing required field 'model_id'")
ValueError: Invalid configuration: missing required field 'model_id'
```

### System Information
```json
{
  "python_version": "3.12.0",
  "platform": "Linux-5.15.0-1-amd64-x86_64",
  "architecture": "x86_64",
  "docker": true
}
```
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub personal access token | None (required) |
| `GITHUB_REPO` | Repository in `owner/repo` format | None (required) |
| `ERROR_REPORTING_ENABLED` | Enable/disable error reporting | `true` |
| `ERROR_REPORTING_INCLUDE_SYSTEM_INFO` | Include system info in reports | `true` |
| `ERROR_REPORTING_AUTO_LABEL` | Automatically add labels | `true` |

### Python Options

```python
from utils.error_reporter import ErrorReporter

reporter = ErrorReporter(
    github_token='ghp_xxx',
    github_repo='user/repo',
    enabled=True,
    include_system_info=True,
    auto_label=True
)
```

## Troubleshooting

### Error Reporting Not Working

1. **Check configuration**:
   ```bash
   curl http://localhost:5000/api/error-reporter/status
   ```

2. **Verify token has correct permissions**:
   - Token needs `repo` scope
   - Token must be valid and not expired

3. **Check logs**:
   ```bash
   # Python logs
   grep "Error reporter" /var/log/app.log
   
   # Docker logs
   docker logs container_name | grep "Error reporter"
   ```

### Duplicate Issues Not Being Prevented

The error reporter uses a hash of the error type, message, and first few stack frames to detect duplicates. If you're seeing duplicate issues:

- Check if the error messages are slightly different
- Check if the stack traces are different
- Clear the cache: `rm ~/.ipfs_accelerate/reported_errors.json`

### Permission Denied Creating Issues

- Verify `GITHUB_TOKEN` has `repo` scope
- Verify `GITHUB_REPO` format is `owner/repo`
- Check that the token has write access to the repository

## Security Considerations

1. **Token Security**:
   - Never commit tokens to version control
   - Use environment variables or secrets management
   - Rotate tokens periodically
   - Use minimal required permissions (`repo` scope only)

2. **Sensitive Information**:
   - The error reporter may include system information
   - Review error reports for sensitive data
   - Consider disabling `include_system_info` for sensitive environments
   - Add custom filtering if needed

3. **Rate Limiting**:
   - GitHub API has rate limits
   - Duplicate detection helps prevent hitting limits
   - Consider implementing additional throttling for high-volume applications

## API Reference

### Python API

#### `ErrorReporter.report_error()`

```python
def report_error(
    exception: Optional[Exception] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    traceback_str: Optional[str] = None,
    source_component: str = 'unknown',
    context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Report an error by creating a GitHub issue.
    
    Returns:
        URL of created issue, or None if not created
    """
```

### JavaScript API

#### `reportError()`

```javascript
async function reportError(options) {
    /**
     * Report an error by creating a GitHub issue.
     * 
     * @param {Error} options.error - The error object
     * @param {string} options.sourceComponent - Component name
     * @param {object} options.context - Additional context
     * @returns {Promise<string|null>} URL of created issue
     */
}
```

### REST API

#### `POST /api/report-error`

Report an error from a client.

**Request:**
```json
{
  "error_info": {
    "error_type": "TypeError",
    "error_message": "Cannot read property 'x' of undefined",
    "source_component": "dashboard",
    "stack": "Error: ...",
    "context": {}
  }
}
```

**Response:**
```json
{
  "success": true,
  "issue_url": "https://github.com/user/repo/issues/123",
  "message": "Error reported successfully"
}
```

#### `GET /api/error-reporter/status`

Get error reporter status.

**Response:**
```json
{
  "enabled": true,
  "github_repo": "user/repo",
  "has_token": true,
  "reported_errors_count": 42
}
```

## Examples

See the `examples/` directory for complete examples:

- `examples/error_reporting_demo.py` - Python error reporting examples
- `examples/error_reporting_demo.html` - JavaScript error reporting examples

## License

Same as IPFS Accelerate Python Framework
