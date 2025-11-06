# 🔧 IPFS Accelerate MCP Server - Systemd Service Setup# IPFS Accelerate Service Setup - Complete



## Overview✅ **SUCCESS**: Your IPFS Accelerate project has been successfully configured as a systemd service with automatic startup and monitoring!

The IPFS Accelerate MCP Server is now configured as a systemd service with integrated GitHub Actions runner autoscaling capabilities.

## What Was Installed

## ✅ Service Status

### 1. Service Files Created

### Active Service- `ipfs-accelerate.service` - Systemd service definition

- **Service Name**: `ipfs-accelerate-mcp.service`- `install-service.sh` - Service installer script

- **Status**: ✅ Enabled and running- `uninstall-service.sh` - Service removal script

- **Command**: `ipfs-accelerate mcp start --dashboard --host 0.0.0.0 --port 9000 --keep-running`- `setup-cron.sh` - Monitoring setup script

- **User**: `barberb`- `setup-complete.sh` - Complete installation script

- **Working Directory**: `/home/barberb/ipfs_accelerate_py`- `test-service.sh` - Service testing script

- `check-service.sh` - Monitoring/health check script

### Integrated Features- `SERVICE_SETUP.md` - Detailed documentation

- **MCP Server**: HTTP server with JSON-RPC API

- **Web Dashboard**: Available at http://localhost:9000/dashboard### 2. System Configuration

- **GitHub Autoscaler**: Automatically provisions runners based on workflow demand- **Systemd Service**: `/etc/systemd/system/ipfs-accelerate.service`

- **Architecture**: ARM64 (with ARM64-specific workflow filtering)- **Service Status**: ✅ Enabled and Running

- **Docker Support**: Containerized runner isolation enabled- **Auto-start**: ✅ Enabled (starts on boot)

- **User**: barberb

## 🚀 Quick Start- **Working Directory**: /home/barberb/ipfs_accelerate_py

- **Command**: `ipfs-accelerate mcp start --host 0.0.0.0 --port 9000 --keep-running`

### Installation

```bash### 3. Monitoring Setup

cd /home/barberb/ipfs_accelerate_py- **Cron Job**: Runs every 5 minutes

./install-services.sh- **Monitor Script**: `/home/barberb/ipfs_accelerate_py/check-service.sh`

```- **Log File**: `/tmp/ipfs-accelerate-monitor.log`

- **Health Check**: Tests http://localhost:9000/health

### Service Management- **Auto-restart**: If service is down or not responding

```bash

# Start the service## Current Status

sudo systemctl start ipfs-accelerate-mcp

✅ **Service Running**: Active and healthy  

# Stop the service✅ **Dashboard Available**: http://localhost:9000/dashboard  

sudo systemctl stop ipfs-accelerate-mcp✅ **Health Endpoint**: http://localhost:9000/health  

✅ **API Available**: http://localhost:9000/api/  

# Restart the service✅ **Auto-start Enabled**: Will start on system boot  

sudo systemctl restart ipfs-accelerate-mcp✅ **Monitoring Active**: Cron job checking every 5 minutes  



# Check status## Service Management Commands

sudo systemctl status ipfs-accelerate-mcp

```bash

# Enable auto-start on boot# Check status

sudo systemctl enable ipfs-accelerate-mcpsudo systemctl status ipfs-accelerate



# Disable auto-start on boot# Start/Stop/Restart

sudo systemctl disable ipfs-accelerate-mcpsudo systemctl start ipfs-accelerate

```sudo systemctl stop ipfs-accelerate

sudo systemctl restart ipfs-accelerate

### View Logs

```bash# View logs (live)

# View live logssudo journalctl -u ipfs-accelerate -f

sudo journalctl -u ipfs-accelerate-mcp -f

# View monitoring logs

# View last 100 linestail -f /tmp/ipfs-accelerate-monitor.log

sudo journalctl -u ipfs-accelerate-mcp -n 100

# Test service health

# View logs from today./test-service.sh

sudo journalctl -u ipfs-accelerate-mcp --since today```



# View autoscaler-specific logs## Service Configuration Details

sudo journalctl -u ipfs-accelerate-mcp | grep autoscaler

```**Port**: 9000  

**Host**: 0.0.0.0 (all interfaces)  

## 📋 Service Configuration**Restart Policy**: Always (10 second delay)  

**Security**: Restricted permissions, private tmp, read-only system  

### Service File Location**Logging**: systemd journal + monitoring log  

`/etc/systemd/system/ipfs-accelerate-mcp.service`

## Startup Behavior

### Key Configuration

```ini1. **System Boot** → Service starts automatically

[Service]2. **Service Crash** → Systemd restarts after 10 seconds

Type=simple3. **Service Unresponsive** → Cron monitor restarts every 5 minutes

User=barberb4. **Port Conflict** → Service will attempt next available port

WorkingDirectory=/home/barberb/ipfs_accelerate_py

Environment="PATH=/home/barberb/ipfs_accelerate_py/.venv/bin:/usr/local/bin:/usr/bin:/bin"## Testing

Environment="PYTHONUNBUFFERED=1"

Environment="MCP_HOST=0.0.0.0"Run the test script to verify everything is working:

Environment="MCP_PORT=9000"

```bash

ExecStart=/home/barberb/ipfs_accelerate_py/.venv/bin/ipfs-accelerate mcp start \./test-service.sh

    --dashboard \```

    --host 0.0.0.0 \

    --port 9000 \## Next Steps

    --keep-running

Your service is now ready for production use. The system will:

Restart=always

RestartSec=101. ✅ Start the service automatically on boot

```2. ✅ Monitor the service health every 5 minutes

3. ✅ Restart the service if it stops or becomes unresponsive

### Resource Limits4. ✅ Log all activity for troubleshooting

- **Memory**: Maximum 2GB5. ✅ Provide web dashboard and API access

- **CPU**: Maximum 200% (2 cores)

## Support

## 🤖 GitHub Actions Autoscaler

- **Service Logs**: `sudo journalctl -u ipfs-accelerate -f`

### Autoscaler Features- **Monitor Logs**: `tail -f /tmp/ipfs-accelerate-monitor.log`

The MCP server includes an integrated GitHub Actions autoscaler that:- **Documentation**: `SERVICE_SETUP.md`

- ✅ Monitors workflow queues across accessible repositories- **Test Script**: `./test-service.sh`

- ✅ Automatically provisions self-hosted runners when needed

- ✅ Filters workflows by architecture (ARM64)---

- ✅ Uses Docker isolation for security

- ✅ Provides labels: `self-hosted`, `linux`, `arm64`, `docker`, `cuda`, `gpu`**Installation completed on**: $(date)  

**Service URL**: http://localhost:9000  

### Autoscaler Configuration**Dashboard URL**: http://localhost:9000/dashboard
- **Poll Interval**: 60 seconds
- **Max Runners**: 20
- **Monitor Window**: 1 day
- **Architecture**: ARM64 only
- **Docker Isolation**: Enabled

### Autoscaler Logs
```bash
# View autoscaler activity
sudo journalctl -u ipfs-accelerate-mcp | grep "github_autoscaler"

# Check for workflow monitoring
sudo journalctl -u ipfs-accelerate-mcp | grep "Checking workflow queues"

# View runner provisioning
sudo journalctl -u ipfs-accelerate-mcp | grep "Provisioning runner"
```

## 🌐 Network Access

### MCP Server Endpoints
- **Dashboard**: http://localhost:9000/dashboard
- **API**: http://localhost:9000/
- **Host**: 0.0.0.0 (accessible from network)
- **Port**: 9000

### Testing Connectivity
```bash
# Local test
curl http://localhost:9000/

# Network test (from another machine)
curl http://<server-ip>:9000/

# Check listening ports
ss -tulnp | grep :9000
```

## 🔒 Security Considerations

### Service Security
- **NoNewPrivileges**: Enabled (prevents privilege escalation)
- **PrivateTmp**: Enabled (isolated temporary directory)
- **User**: Non-root (`barberb`)
- **Resource Limits**: Memory and CPU caps applied

### Autoscaler Security
- **Docker Isolation**: Runners execute in isolated containers
- **Architecture Filtering**: Only provisions for ARM64 workflows
- **GitHub Authentication**: Requires `gh auth login`

## 🔧 Troubleshooting

### Service Won't Start
```bash
# Check service status
sudo systemctl status ipfs-accelerate-mcp

# Check logs for errors
sudo journalctl -u ipfs-accelerate-mcp -n 50

# Verify virtual environment
ls -la /home/barberb/ipfs_accelerate_py/.venv/bin/ipfs-accelerate

# Test manual start
/home/barberb/ipfs_accelerate_py/.venv/bin/ipfs-accelerate mcp start --help
```

### Autoscaler Not Working
```bash
# Check GitHub CLI authentication
gh auth status

# Re-authenticate if needed
gh auth login

# Restart service after authentication
sudo systemctl restart ipfs-accelerate-mcp
```

### Port Already in Use
```bash
# Check what's using port 9000
sudo ss -tulnp | grep :9000

# Kill process if needed
sudo kill <pid>

# Or change port in service file
sudo systemctl edit ipfs-accelerate-mcp
```

### High Memory/CPU Usage
```bash
# Check current resource usage
systemctl status ipfs-accelerate-mcp

# Adjust limits in service file
sudo systemctl edit ipfs-accelerate-mcp

# Add or modify:
[Service]
MemoryMax=1G
CPUQuota=100%
```

## 📊 Monitoring

### Service Health
```bash
# Check if service is active
systemctl is-active ipfs-accelerate-mcp

# Check if service is enabled
systemctl is-enabled ipfs-accelerate-mcp

# View service uptime
systemctl status ipfs-accelerate-mcp | grep Active
```

### Autoscaler Activity
```bash
# Count autoscaler checks
sudo journalctl -u ipfs-accelerate-mcp | grep -c "Check #"

# View recent workflow checks
sudo journalctl -u ipfs-accelerate-mcp --since "1 hour ago" | grep "workflow"

# Check for provisioned runners
sudo journalctl -u ipfs-accelerate-mcp | grep "Provisioning"
```

## 🔄 Updates and Maintenance

### Updating the Service
```bash
# Pull latest code
cd /home/barberb/ipfs_accelerate_py
git pull

# Update dependencies
source .venv/bin/activate
pip install -e .[minimal,mcp]

# Restart service
sudo systemctl restart ipfs-accelerate-mcp
```

### Modifying Service Configuration
```bash
# Edit service file
sudo systemctl edit --full ipfs-accelerate-mcp

# Reload systemd after changes
sudo systemctl daemon-reload

# Restart service
sudo systemctl restart ipfs-accelerate-mcp
```

## 📝 Integration Notes

### GitHub Actions Workflow Example
```yaml
name: CI with Auto-Scaled Runners

on: [push, pull_request]

jobs:
  test:
    runs-on: [self-hosted, ARM64]  # Autoscaler will provision if needed
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/
```

### Manual Runner Check
The autoscaler monitors these workflow labels:
- `self-hosted`
- `ARM64` or `arm64`
- `linux`

Workflows with these labels trigger automatic runner provisioning when queued.

## ✨ Summary

The IPFS Accelerate MCP Server systemd service provides:
- ✅ **Automatic startup** on system boot
- ✅ **Automatic restart** on failure
- ✅ **Integrated autoscaler** for GitHub Actions runners
- ✅ **Web dashboard** for monitoring
- ✅ **Secure isolation** with resource limits
- ✅ **Comprehensive logging** via journald

**Status**: 🎉 **FULLY OPERATIONAL**

---

**Created**: November 5, 2025  
**Last Updated**: November 5, 2025  
**Status**: ✅ ACTIVE AND RUNNING