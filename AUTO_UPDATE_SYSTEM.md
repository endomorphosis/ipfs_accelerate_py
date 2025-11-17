# Auto-Update System Documentation

This document describes the automatic update system for the ipfs_accelerate_py package.

## Overview

The auto-update system ensures that the ipfs_accelerate_py package is always up-to-date with the latest changes from the main branch on GitHub. It consists of three components:

1. **Auto-update script** - Pulls from main and installs packages
2. **Systemd service integration** - Runs update before service starts
3. **Cron job** - Periodically checks for updates

## Components

### 1. Auto-Update Script

**Location**: `/home/barberb/ipfs_accelerate_py/scripts/auto-update.sh`

**Features**:
- Fetches latest changes from GitHub main branch
- Checks if update is needed (compares commits)
- Stashes local changes if any
- Pulls latest code
- Activates or creates Python virtual environment
- Updates pip and installs/updates requirements
- Installs package in editable mode
- Logs all operations to `/home/barberb/ipfs_accelerate_py/logs/auto-update.log`

**Usage**:
```bash
/home/barberb/ipfs_accelerate_py/scripts/auto-update.sh
```

The script exits with code 0 if successful or if already up-to-date.

### 2. Systemd Service Integration

All systemd services now include an `ExecStartPre` directive that runs the auto-update script before starting the service:

**Updated Services**:
- `ipfs-accelerate.service`
- `ipfs-accelerate-mcp.service`
- `containerized-runner-launcher.service`

**Behavior**:
- Every time a service starts or restarts, it automatically updates the code
- If the update fails, the service won't start
- Services log update status to systemd journal

**Installing/Updating Service Files**:
```bash
sudo /home/barberb/ipfs_accelerate_py/scripts/install-updated-services.sh
```

**Viewing Service Logs**:
```bash
# View update logs in systemd journal
sudo journalctl -u ipfs-accelerate.service -n 50

# View auto-update log file
tail -f /home/barberb/ipfs_accelerate_py/logs/auto-update.log
```

### 3. Cron Job

**Schedule**: Every 6 hours (at 00:00, 06:00, 12:00, 18:00)

**Setup**:
```bash
/home/barberb/ipfs_accelerate_py/scripts/setup-auto-update-cron.sh
```

**Log File**: `/home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log`

**View Current Crontab**:
```bash
crontab -l
```

**Checking Cron Logs**:
```bash
# View cron execution log
tail -f /home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log

# View system cron logs
grep CRON /var/log/syslog
```

## Update Frequency

The system checks for updates in the following scenarios:

1. **Service Start/Restart** - Every time a systemd service starts
2. **Scheduled Updates** - Every 6 hours via cron
3. **Manual Trigger** - Run the script manually anytime

## Configuration

### Changing Cron Schedule

Edit the cron job schedule in `/home/barberb/ipfs_accelerate_py/scripts/setup-auto-update-cron.sh`:

```bash
# Current: every 6 hours
echo "0 */6 * * * ${UPDATE_SCRIPT} >> /home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log 2>&1"

# Examples:
# Every 4 hours:
echo "0 */4 * * * ${UPDATE_SCRIPT} >> /home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log 2>&1"

# Daily at 2 AM:
echo "0 2 * * * ${UPDATE_SCRIPT} >> /home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log 2>&1"

# Every hour:
echo "0 * * * * ${UPDATE_SCRIPT} >> /home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log 2>&1"
```

Then re-run the setup script:
```bash
/home/barberb/ipfs_accelerate_py/scripts/setup-auto-update-cron.sh
```

### Changing Repository Path

Edit the `REPO_DIR` variable in `/home/barberb/ipfs_accelerate_py/scripts/auto-update.sh`:

```bash
REPO_DIR="/home/barberb/ipfs_accelerate_py"  # Change this path if needed
```

## Troubleshooting

### Update Script Fails

Check the log file for errors:
```bash
tail -100 /home/barberb/ipfs_accelerate_py/logs/auto-update.log
```

Common issues:
- **Git conflicts**: The script stashes local changes, but manual intervention may be needed
- **Network issues**: Ensure GitHub is accessible
- **Permission issues**: Ensure the script has write access to the repository

### Service Won't Start

If a service fails to start after update:

```bash
# Check service status
sudo systemctl status ipfs-accelerate.service

# View recent logs
sudo journalctl -u ipfs-accelerate.service -n 100

# Manually run the update script to see errors
/home/barberb/ipfs_accelerate_py/scripts/auto-update.sh
```

### Cron Job Not Running

Check if cron job is installed:
```bash
crontab -l | grep auto-update
```

Check system cron logs:
```bash
grep CRON /var/log/syslog | grep auto-update
```

Manually run the cron setup script:
```bash
/home/barberb/ipfs_accelerate_py/scripts/setup-auto-update-cron.sh
```

## Disabling Auto-Updates

### Temporarily Disable

To temporarily disable auto-updates without removing the system:

1. **Disable Cron Job**:
   ```bash
   crontab -l | grep -v auto-update > /tmp/cron-temp
   crontab /tmp/cron-temp
   ```

2. **Remove ExecStartPre from Services**:
   Edit the service files and comment out the `ExecStartPre` line, then reload:
   ```bash
   sudo systemctl daemon-reload
   ```

### Permanently Remove

To completely remove the auto-update system:

```bash
# Remove cron job
crontab -l | grep -v auto-update > /tmp/cron-temp
crontab /tmp/cron-temp

# Remove ExecStartPre from all service files
# Edit each service file and remove the ExecStartPre line

# Reload systemd
sudo systemctl daemon-reload
```

## Manual Update

To manually trigger an update:

```bash
# Run the update script
/home/barberb/ipfs_accelerate_py/scripts/auto-update.sh

# Then restart services if needed
sudo systemctl restart ipfs-accelerate.service
```

## Monitoring

### Check Update History

View all updates from the log:
```bash
grep "Successfully updated to commit" /home/barberb/ipfs_accelerate_py/logs/auto-update.log
```

### Check Last Update Time

```bash
# Check last auto-update log entry
tail -1 /home/barberb/ipfs_accelerate_py/logs/auto-update.log

# Check current git commit
cd /home/barberb/ipfs_accelerate_py && git log -1 --oneline
```

## Best Practices

1. **Monitor Logs**: Regularly check update logs for issues
2. **Test Updates**: Test in a development environment before production
3. **Backup Configuration**: Keep backups of custom configurations
4. **Pin Versions**: For production, consider pinning to specific commits instead of always pulling main
5. **Service Restart Policy**: Configure appropriate restart policies in systemd for your use case

## Security Considerations

- The auto-update script has write access to the repository directory
- Updates are pulled from the main branch without verification
- Consider implementing:
  - GPG signature verification for commits
  - Staging environment testing before production updates
  - Rollback mechanism for failed updates
  - Alert system for update failures

## Future Enhancements

Potential improvements to the auto-update system:

1. **Rollback Support**: Automatic rollback on service failure
2. **Update Notifications**: Email or webhook notifications on updates
3. **Selective Updates**: Only update on tagged releases
4. **Pre/Post Update Hooks**: Custom scripts before/after updates
5. **Health Checks**: Verify service health after updates
6. **Update Windows**: Restrict updates to specific time windows

## Related Files

- `/home/barberb/ipfs_accelerate_py/scripts/auto-update.sh` - Main update script
- `/home/barberb/ipfs_accelerate_py/scripts/setup-auto-update-cron.sh` - Cron setup script
- `/home/barberb/ipfs_accelerate_py/scripts/install-updated-services.sh` - Service installation script
- `/home/barberb/ipfs_accelerate_py/logs/auto-update.log` - Update log file
- `/home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log` - Cron execution log
- `/home/barberb/ipfs_accelerate_py/*.service` - Systemd service files
