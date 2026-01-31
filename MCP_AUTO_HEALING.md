# MCP Server and JavaScript SDK Auto-Healing

## Overview

The MCP (Model Context Protocol) server and JavaScript SDK now support automatic error reporting and self-healing through GitHub issue creation and Copilot integration.

## Features

### 1. MCP Server-Side Tool Error Handling

**What:** Automatically captures errors from MCP tool executions and resource accesses.

**How:** When a tool execution fails on the server:
- Error is captured with full stack trace
- Context includes tool name, parameters, and MCP server name
- GitHub issue is created automatically (if enabled)
- Draft PR is generated (if enabled)
- Copilot is invoked for fixes (if enabled)

**Example:**
```python
# MCP server running with auto-healing enabled
export IPFS_AUTO_ISSUE=true
export IPFS_AUTO_PR=true
python -m ipfs_accelerate_py.mcp.server
```

When a tool fails:
```
Tool: get_model
Error: ModelNotFoundError: Model 'bert-base' not found
→ GitHub Issue Created: #456
→ Draft PR Generated: PR #457
→ Copilot Analysis: Suggests checking model registry
```

### 2. JavaScript SDK Client-Side Error Handling

**What:** JavaScript SDK can report client-side errors to the server for auto-healing.

**How:** Enable error reporting when creating the MCP client:
```javascript
const client = new MCPClient('/jsonrpc', {
    reportErrors: true,  // Enable auto-healing
    errorReportEndpoint: '/report-error'
});
```

When an error occurs in the SDK:
- Error is captured with stack trace
- Context includes browser info, URL, and request details
- Error is sent to server's `/report-error` endpoint
- Server forwards to auto-healing system
- GitHub issue is created (if enabled)

**Example:**
```javascript
// Client-side code
const client = new MCPClient('/jsonrpc', { reportErrors: true });

try {
    const model = await client.getModel('invalid-model-id');
} catch (error) {
    // Error automatically reported to server
    // GitHub issue will be created with:
    // - JavaScript stack trace
    // - Browser information
    // - Request details
}
```

## Configuration

### Server-Side (Python)

Enable auto-healing for MCP server via environment variables:

```bash
# Enable GitHub issue creation
export IPFS_AUTO_ISSUE=true

# Enable draft PR generation
export IPFS_AUTO_PR=true

# Enable Copilot auto-fixing
export IPFS_AUTO_HEAL=true

# Specify target repository
export IPFS_REPO=owner/repo

# Start MCP server
python -m ipfs_accelerate_py.mcp.server
```

### Client-Side (JavaScript)

Enable error reporting in the SDK:

```javascript
const client = new MCPClient('/jsonrpc', {
    // Enable error reporting
    reportErrors: true,
    
    // Optional: custom error report endpoint
    errorReportEndpoint: '/report-error',
    
    // Other options
    timeout: 30000,
    retries: 3
});
```

Or enable globally:
```javascript
// Set as default for all clients
MCPClient.defaultOptions = {
    reportErrors: true
};
```

## Error Flow

### Server-Side Tool Error

```
MCP Tool Execution
        ↓
    Error Occurs
        ↓
Error Handler Captures:
  • Tool name
  • Parameters
  • Stack trace
  • MCP server name
        ↓
GitHub Issue Created
  • Title: [Auto-Generated] ToolExecutionError
  • Body: Full error details
  • Labels: auto-generated, bug, mcp-tool
        ↓
Draft PR Generated (if enabled)
        ↓
Copilot Analysis (if enabled)
```

### Client-Side SDK Error

```
JavaScript SDK Call
        ↓
    Error Occurs
        ↓
SDK Reports to Server:
  • Error type
  • Error message
  • JavaScript stack trace
  • Browser context
        ↓
Server Forwards to Error Handler
        ↓
GitHub Issue Created
  • Title: [Auto-Generated] JavaScript SDK Error
  • Body: Client error details + context
  • Labels: auto-generated, bug, mcp-sdk
        ↓
Draft PR Generated (if enabled)
```

## Error Types Captured

### MCP Server Errors

1. **Tool Execution Errors**
   - Tool not found
   - Invalid parameters
   - Tool runtime exceptions
   - Resource access failures

2. **Resource Errors**
   - Resource not found
   - Permission denied
   - Data fetch failures

### JavaScript SDK Errors

1. **Network Errors**
   - Connection timeouts
   - Failed requests
   - Network unavailable

2. **Protocol Errors**
   - JSON-RPC errors
   - Invalid responses
   - Malformed requests

3. **Application Errors**
   - Model not found
   - Invalid operations
   - Parameter validation errors

## GitHub Issue Example

**Title:** `[Auto-Generated] RuntimeError: Model processing failed`

**Body:**
```markdown
# Auto-Generated Error Report

**Error Type:** `RuntimeError`
**Error Source:** `mcp_tool`
**MCP Server:** `ipfs-accelerate`
**Tool Name:** `process_model`
**Timestamp:** 2024-01-31T12:34:56.789Z

## Error Message
```
Model processing failed: Insufficient memory
```

## Stack Trace
```python
Traceback (most recent call last):
  File "tools/models.py", line 123, in process_model
    result = model.process(data)
  File "model.py", line 456, in process
    raise RuntimeError("Insufficient memory")
RuntimeError: Model processing failed: Insufficient memory
```

## Context
```json
{
  "mcp_server": "ipfs-accelerate",
  "tool_name": "process_model",
  "tool_params": "{'model_id': 'bert-base', 'batch_size': 32}",
  "error_source": "mcp_tool"
}
```

Labels: auto-generated, bug, mcp-tool, high
```

## JavaScript SDK Error Example

**Title:** `[Auto-Generated] RuntimeError: [JavaScript SDK] NetworkError`

**Body:**
```markdown
# Auto-Generated Error Report

**Error Type:** `RuntimeError`
**Error Source:** `mcp_javascript_sdk`
**Timestamp:** 2024-01-31T12:34:56.789Z

## Error Message
```
[JavaScript SDK] NetworkError: Failed to fetch
```

## Client Stack Trace
```javascript
Error: Failed to fetch
    at MCPClient._makeHttpRequest (mcp-sdk.js:145)
    at MCPClient.request (mcp-sdk.js:42)
    at async listModels (app.js:234)
```

## Client Context
```json
{
  "timestamp": "2024-01-31T12:34:56.789Z",
  "userAgent": "Mozilla/5.0...",
  "url": "https://example.com/dashboard",
  "method": "list_models",
  "params": {}
}
```

Labels: auto-generated, bug, mcp-sdk
```

## Best Practices

### Development

```bash
# Keep auto-healing disabled during development
# (errors are logged but not reported)
python -m ipfs_accelerate_py.mcp.server
```

```javascript
// Disable error reporting for local development
const client = new MCPClient('/jsonrpc', {
    reportErrors: false
});
```

### Production

```bash
# Enable auto-healing in production
export IPFS_AUTO_ISSUE=true
export IPFS_AUTO_PR=true
python -m ipfs_accelerate_py.mcp.server
```

```javascript
// Enable error reporting for production
const client = new MCPClient('/jsonrpc', {
    reportErrors: true
});
```

### Testing

```bash
# Use test repository for auto-healing
export IPFS_REPO=my-org/test-repo
export IPFS_AUTO_ISSUE=true
```

## Security & Privacy

1. **Error Sanitization**: Stack traces are captured but can be sanitized
2. **Context Filtering**: Sensitive data can be filtered from context
3. **Rate Limiting**: Error reporting respects rate limits
4. **Authentication**: Uses existing GitHub CLI credentials

## Performance

- **Server-Side**: <100ms overhead per error (only on error path)
- **Client-Side**: Fire-and-forget reporting (no blocking)
- **Success Path**: Zero overhead (only runs on errors)

## Troubleshooting

### Error reporting not working

**Check server logs:**
```bash
# Look for error handler initialization
grep "MCP auto-healing enabled" server.log
```

**Check GitHub CLI:**
```bash
gh auth status
```

### JavaScript errors not reported

**Check browser console:**
```javascript
// Verify error reporting is enabled
console.log(client.options.reportErrors);  // Should be true
```

**Check network tab:**
- Look for POST to `/report-error`
- Check response status

### Issues not created

**Verify environment variables:**
```bash
echo $IPFS_AUTO_ISSUE  # Should be true
echo $IPFS_REPO        # Should be owner/repo
```

**Check GitHub authentication:**
```bash
gh auth login
gh auth status
```

## See Also

- [AUTO_HEALING_README.md](AUTO_HEALING_README.md) - CLI error handling
- [MCP_ERROR_HANDLING_VERIFICATION.md](MCP_ERROR_HANDLING_VERIFICATION.md) - MCP verification
- [QUICK_START_MCP.md](QUICK_START_MCP.md) - Quick start guide
