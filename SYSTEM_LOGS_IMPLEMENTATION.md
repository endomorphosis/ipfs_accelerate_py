# System Logs Access - Complete Implementation

The IPFS Accelerate system logs are now accessible through **4 different interfaces**, making it easy to access logs programmatically or interactively.

## 1. 📦 Python Package Import

Import and use system logs directly in your Python code:

```python
from ipfs_accelerate_py import get_system_logs, SystemLogs

# Quick access - get last 100 logs
logs = get_system_logs(service="ipfs-accelerate", lines=100)
for log in logs:
    print(f"[{log['timestamp']}] {log['level']}: {log['message']}")

# Advanced usage - use the SystemLogs class
logs_manager = SystemLogs("ipfs-accelerate")

# Get recent errors
errors = logs_manager.get_recent_errors(hours=24)
print(f"Found {len(errors)} errors in the last 24 hours")

# Get log statistics
stats = logs_manager.get_stats()
print(f"Total logs: {stats['total']}")
print(f"By level: {stats['by_level']}")

# Filter by level and time
logs = logs_manager.get_logs(
    lines=50,
    since="1 hour ago",
    level="ERROR"
)
```

## 2. 🖥️ Command Line Interface (CLI)

Access logs directly from the terminal:

```bash
# Basic usage - show last 100 logs
ipfs-accelerate mcp logs

# Specify number of lines
ipfs-accelerate mcp logs --lines 50

# Filter by time period
ipfs-accelerate mcp logs --since "1 hour ago"
ipfs-accelerate mcp logs --since "30 minutes ago"

# Filter by log level
ipfs-accelerate mcp logs --level ERROR
ipfs-accelerate mcp logs --level WARNING

# Show only recent errors
ipfs-accelerate mcp logs --errors

# Show log statistics
ipfs-accelerate mcp logs --stats

# Follow logs in real-time
ipfs-accelerate mcp logs --follow

# Output as JSON
ipfs-accelerate mcp logs --json

# Combine filters
ipfs-accelerate mcp logs --lines 20 --level ERROR --since "2 hours ago" --json
```

## 3. 🌐 HTTP API Endpoint

Access logs via HTTP REST API:

```bash
# Basic request
curl http://localhost:9000/api/mcp/logs

# With query parameters
curl "http://localhost:9000/api/mcp/logs?lines=50&level=ERROR&since=1+hour+ago"
```

**API Response Format:**
```json
{
  "logs": [
    {
      "timestamp": "2025-11-10 19:17:55",
      "level": "INFO",
      "message": "Service started successfully",
      "unit": "ipfs-accelerate.service",
      "pid": "353695"
    }
  ],
  "total": 100,
  "service": "ipfs-accelerate",
  "filters": {
    "lines": 100,
    "since": null,
    "level": null
  }
}
```

**Query Parameters:**
- `lines` (int): Number of log lines to retrieve (default: 100)
- `since` (string): Time period (e.g., "1 hour ago", "30 minutes ago")
- `level` (string): Filter by log level (INFO, WARNING, ERROR, CRITICAL, DEBUG)
- `service` (string): Service name (default: ipfs-accelerate)

## 4. 🛠️ MCP Server Tool

Access logs through the MCP server tools interface:

```python
# When connected to the MCP server, three tools are available:

# 1. get_system_logs - Get system logs with filters
result = mcp_server.call_tool("get_system_logs", {
    "lines": 100,
    "since": "1 hour ago",
    "level": "ERROR",
    "service": "ipfs-accelerate"
})

# 2. get_recent_errors - Get recent errors specifically
result = mcp_server.call_tool("get_recent_errors", {
    "hours": 24,
    "service": "ipfs-accelerate"
})

# 3. get_log_stats - Get log statistics
result = mcp_server.call_tool("get_log_stats", {
    "service": "ipfs-accelerate"
})
```

## 5. 🖱️ Web Dashboard UI

Access logs through the interactive web dashboard:

1. Open http://localhost:9000/ in your browser
2. Click on the "System Logs" tab
3. Click the "🔄 Refresh" button to load latest logs
4. Logs are displayed with color-coded levels and emojis:
   - 🔥 CRITICAL (red)
   - ❌ ERROR (red)
   - ⚠️ WARNING (yellow)
   - ℹ️ INFO (blue)
   - 🔍 DEBUG (gray)

## Log Entry Structure

Each log entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | Log entry timestamp (YYYY-MM-DD HH:MM:SS) |
| `level` | string | Log level (INFO, WARNING, ERROR, CRITICAL, DEBUG) |
| `message` | string | The log message content |
| `unit` | string | Systemd unit name (ipfs-accelerate.service) |
| `pid` | string | Process ID that generated the log |

## Examples

### Example 1: Get Recent Errors in Python

```python
from ipfs_accelerate_py import SystemLogs

logs = SystemLogs("ipfs-accelerate")
errors = logs.get_recent_errors(hours=24)

for error in errors:
    print(f"❌ [{error['timestamp']}] {error['message']}")
```

### Example 2: Monitor Logs in Real-time

```bash
# Follow logs as they come in
ipfs-accelerate mcp logs --follow
```

### Example 3: Get Log Statistics via API

```bash
curl http://localhost:9000/api/mcp/logs | jq '.total, .filters'
```

### Example 4: Export Logs to File

```bash
# Export as JSON
ipfs-accelerate mcp logs --json > logs-$(date +%Y%m%d).json

# Export as text
ipfs-accelerate mcp logs --lines 1000 > logs-$(date +%Y%m%d).txt
```

## Log Source

Logs are retrieved from systemd's `journalctl` command, which provides:
- Reliable structured logging
- Timestamp accuracy
- Process tracking
- Level detection
- Automatic log rotation

If `journalctl` is not available (non-systemd systems), the system falls back to reading from log files in:
- `/var/log/ipfs-accelerate.log`
- `~/.ipfs-accelerate/logs/service.log`
- `/tmp/ipfs-accelerate.log`

## Features

✅ **Real-time access** - Get logs as they're written
✅ **Flexible filtering** - By level, time period, and service
✅ **Multiple interfaces** - Python, CLI, HTTP API, MCP tools, Web UI
✅ **Structured output** - JSON format for easy parsing
✅ **Error tracking** - Dedicated endpoint for recent errors
✅ **Statistics** - Aggregate log counts by level
✅ **Backward compatible** - Works on systemd and non-systemd systems

## Files Created/Modified

| File | Description |
|------|-------------|
| `ipfs_accelerate_py/logs.py` | Core system logs module |
| `ipfs_accelerate_py/mcp/tools/system_logs.py` | MCP tools for logs |
| `ipfs_accelerate_py/mcp_dashboard.py` | HTTP API endpoint |
| `ipfs_accelerate_py/cli.py` | CLI commands |
| `ipfs_accelerate_py/__init__.py` | Package exports |
| `tools/comprehensive_mcp_server.py` | MCP tool registration |
| `ipfs_accelerate_py/static/js/dashboard.js` | Web UI functions |
| `ipfs_accelerate_py/templates/dashboard.html` | System Logs tab |

## Status

✅ **Package import** - Working
✅ **CLI tool** - Working
✅ **HTTP API** - Working
✅ **MCP server tool** - Registered
✅ **Web dashboard** - Functional

All four access methods are now fully implemented and tested!
