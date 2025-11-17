# MCP Server Autoscaler Integration

## Overview

The MCP (Model Control Plane) server now automatically starts the GitHub Actions runner autoscaler when it launches. This enables automatic provisioning of self-hosted runners for both GitHub workflows and P2P tasks without any manual intervention.

## Features

✅ **Automatic Startup** - Autoscaler starts automatically when MCP server starts  
✅ **Configurable** - Control autoscaler behavior via CLI arguments  
✅ **P2P Integration** - Supports P2P workflow discovery and execution  
✅ **GitHub Authentication** - Automatically detects GitHub CLI authentication  
✅ **Graceful Degradation** - Continues working even if autoscaler can't start  

## Usage

### Basic Usage (Default)

Start the MCP server with autoscaler enabled by default:

```bash
ipfs-accelerate mcp start
```

The autoscaler will:
- Start automatically in the background
- Monitor all accessible GitHub repositories
- Provision runners for GitHub workflows
- Discover and execute P2P workflows
- Poll every 60 seconds (default)

### Custom Configuration

Configure the autoscaler with specific settings:

```bash
ipfs-accelerate mcp start \
  --autoscaler-owner myorg \
  --autoscaler-interval 120 \
  --autoscaler-since-days 2 \
  --autoscaler-max-runners 8
```

### Disable Autoscaler

If you don't want the autoscaler to start:

```bash
ipfs-accelerate mcp start --disable-autoscaler
```

### Disable P2P Monitoring

Start autoscaler but disable P2P workflow monitoring:

```bash
ipfs-accelerate mcp start --no-p2p
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--disable-autoscaler` | Disable the autoscaler completely | Enabled |
| `--autoscaler-owner` | GitHub owner/org to monitor | All accessible repos |
| `--autoscaler-interval` | Poll interval in seconds | 60 |
| `--autoscaler-since-days` | Monitor repos from last N days | 1 |
| `--autoscaler-max-runners` | Maximum runners to provision | System cores |
| `--no-p2p` | Disable P2P workflow monitoring | P2P enabled |

## How It Works

### Startup Sequence

1. **MCP Server Starts** - Flask dashboard or integrated HTTP server launches
2. **Authentication Check** - Verifies GitHub CLI is authenticated (`gh auth login`)
3. **Autoscaler Creation** - Creates GitHubRunnerAutoscaler instance with configuration
4. **Background Thread** - Starts autoscaler in daemon thread
5. **Continuous Monitoring** - Autoscaler polls GitHub and P2P scheduler

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server Start                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Check GitHub Authentication                 │
│                  (gh auth status)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
                   ✓ Authenticated
                          ↓
┌─────────────────────────────────────────────────────────┐
│           Create GitHub Actions Autoscaler               │
│  • Configure with CLI arguments                         │
│  • Enable P2P discovery by default                      │
│  • Set poll interval and limits                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          Start Autoscaler in Background Thread          │
│  • Daemon thread (doesn't block shutdown)               │
│  • Graceful stop on MCP server shutdown                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Autoscaler Runs Continuously                │
│  • Monitor GitHub workflow queues                       │
│  • Discover P2P workflows across repos                  │
│  • Provision runners as needed                          │
│  • Balance GitHub + P2P workloads                       │
└─────────────────────────────────────────────────────────┘
```

### Integration Points

The autoscaler is integrated at two levels:

1. **Flask-based Dashboard** (`mcp_dashboard.py`)
   - MCPDashboard class has `_start_autoscaler()` method
   - Called automatically in `run()` method before Flask app starts
   - Autoscaler runs in background thread

2. **Integrated HTTP Server** (`cli.py`)
   - Already had autoscaler integration (lines 1304-1341)
   - Now both paths support autoscaler

## Requirements

### Authentication

The autoscaler requires GitHub CLI authentication:

```bash
# Authenticate (one time)
gh auth login

# Verify authentication
gh auth status
```

If not authenticated, the MCP server will start but the autoscaler will be disabled with a warning:

```
⚠ GitHub CLI not authenticated - autoscaler disabled
  To enable: gh auth login
```

### Permissions

The authenticated user needs:
- Read access to repositories
- Permission to create runner registration tokens
- Access to workflow run information

## Examples

### Example 1: Start MCP with Auto-scaling

```bash
# Start MCP server (autoscaler enabled by default)
ipfs-accelerate mcp start --port 9000

# Output:
# Starting MCP Dashboard on http://0.0.0.0:9000/mcp
# Starting GitHub Actions autoscaler in background...
# ✓ GitHub Actions autoscaler started
```

### Example 2: Configure for Organization

```bash
# Monitor specific organization with custom settings
ipfs-accelerate mcp start \
  --port 9000 \
  --autoscaler-owner myorg \
  --autoscaler-interval 120 \
  --autoscaler-max-runners 4 \
  --open-browser

# Autoscaler will:
# - Monitor only myorg repositories
# - Check every 2 minutes
# - Provision up to 4 runners
# - Browser opens automatically
```

### Example 3: Disable P2P but Keep Autoscaler

```bash
# Auto-scale GitHub workflows only (no P2P)
ipfs-accelerate mcp start \
  --autoscaler-owner myorg \
  --no-p2p

# Autoscaler will:
# - Monitor GitHub workflows only
# - Not discover P2P workflows
# - Provision runners for GitHub only
```

### Example 4: MCP Without Autoscaler

```bash
# Start MCP but disable autoscaler
ipfs-accelerate mcp start --disable-autoscaler

# Only MCP server runs, no autoscaler
```

## Monitoring

### Check Autoscaler Status

The autoscaler logs its activity:

```
--- Check #1 at 2025-11-17 14:30:00 ---
Checking workflow queues...
P2P workflows: 3 pending, 2 assigned
Found 2 repos with 4 workflows
  Running: 2, Failed: 1
Allocating 3 runners for P2P tasks, 5 for GitHub workflows
✓ Generated 5 runner token(s)
P2P Summary: 3 pending, 2 assigned, 3 runners allocated for P2P
```

### View Logs

```bash
# Start with logs visible
ipfs-accelerate mcp start 2>&1 | tee mcp-autoscaler.log

# Or check systemd logs if running as service
journalctl -u ipfs-accelerate-mcp -f
```

## Troubleshooting

### Autoscaler Not Starting

**Problem**: Autoscaler doesn't start when MCP launches.

**Solutions**:
1. Check GitHub authentication: `gh auth status`
2. Authenticate if needed: `gh auth login`
3. Check logs for error messages
4. Verify autoscaler not explicitly disabled: remove `--disable-autoscaler`

### No Runners Being Provisioned

**Problem**: Autoscaler runs but doesn't provision runners.

**Solutions**:
1. Check if there are workflows to provision for
2. Verify repository access
3. Check `--autoscaler-owner` matches your org/user
4. Increase `--autoscaler-since-days` to monitor more repos

### P2P Workflows Not Discovered

**Problem**: P2P workflows aren't being found.

**Solutions**:
1. Verify workflows have P2P tags (`WORKFLOW_TAGS: p2p-only`)
2. Check P2P not disabled: remove `--no-p2p` flag
3. Verify repositories are accessible
4. Check autoscaler logs for P2P discovery messages

## Configuration Files

### Environment Variables

You can also configure via environment variables:

```bash
# Set autoscaler owner
export GITHUB_AUTOSCALER_OWNER=myorg

# Set poll interval
export GITHUB_AUTOSCALER_INTERVAL=120

# Start MCP (will use env vars)
ipfs-accelerate mcp start
```

### Systemd Service

If running as a systemd service, configure in the service file:

```ini
[Service]
ExecStart=/usr/local/bin/ipfs-accelerate mcp start \
  --autoscaler-owner=myorg \
  --autoscaler-interval=120 \
  --autoscaler-max-runners=8
```

## Best Practices

### 1. Use Organization-Specific Monitoring

```bash
# Monitor your org specifically for better performance
ipfs-accelerate mcp start --autoscaler-owner myorg
```

### 2. Adjust Poll Interval Based on Activity

```bash
# Active development: shorter interval
ipfs-accelerate mcp start --autoscaler-interval 60

# Production/stable: longer interval
ipfs-accelerate mcp start --autoscaler-interval 300
```

### 3. Set Reasonable Runner Limits

```bash
# Limit based on your infrastructure capacity
ipfs-accelerate mcp start --autoscaler-max-runners 4
```

### 4. Enable P2P for Better Resource Usage

```bash
# Default (P2P enabled) is recommended
ipfs-accelerate mcp start --autoscaler-owner myorg

# P2P helps reduce GitHub Actions minutes
```

## Implementation Details

### Code Changes

1. **mcp_dashboard.py** - MCPDashboard class
   - Added `enable_autoscaler` parameter to `__init__`
   - Added `autoscaler_config` parameter to `__init__`
   - Added `_start_autoscaler()` method
   - Modified `run()` to call `_start_autoscaler()` before starting Flask

2. **cli.py** - IPFSAccelerateCLI class
   - Modified `run_mcp_start()` to pass autoscaler config to MCPDashboard
   - Added CLI arguments for autoscaler configuration
   - Integrated autoscaler in both Flask and integrated HTTP paths

### Thread Safety

- Autoscaler runs in daemon thread
- Doesn't block MCP server shutdown
- Gracefully stops when MCP server stops
- Thread-safe interaction with GitHub API

### Error Handling

- Autoscaler startup errors don't prevent MCP server from running
- Missing GitHub authentication logs warning but continues
- Import errors fall back gracefully
- All errors logged for debugging

## Related Documentation

- **P2P Workflow Discovery**: `P2P_WORKFLOW_DISCOVERY.md`
- **P2P Autoscaler Quick Reference**: `P2P_AUTOSCALER_QUICK_REF.md`
- **GitHub Autoscaler**: `AUTOSCALER.md`
- **MCP Server**: `README_MCP_INTEGRATION.md`

## Testing

Run the integration test:

```bash
python3 test_mcp_autoscaler_integration.py
```

Expected output:
```
✓ MCPDashboard __init__ accepts autoscaler parameters
✓ MCPDashboard signature supports autoscaler control
✓ CLI with autoscaler arguments loads successfully
✓ MCPDashboard has _start_autoscaler method
Results: 4 passed, 0 failed
```

## Summary

The MCP server now provides a complete, integrated solution for:
- Model serving and management
- GitHub Actions runner auto-scaling
- P2P workflow discovery and execution
- Web dashboard for monitoring

All started with a single command: `ipfs-accelerate mcp start`
