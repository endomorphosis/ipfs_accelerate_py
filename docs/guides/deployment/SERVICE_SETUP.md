# IPFS Accelerate Service Setup

This directory contains scripts to set up IPFS Accelerate as a system service that starts automatically on boot.

## Quick Setup

Run the complete setup script:

```bash
./scripts/setup/setup-complete.sh
```

This will:
1. Install the systemd service
2. Enable auto-start on boot
3. Set up monitoring via cron job
4. Start the service immediately

## Manual Setup

### 1. Install Service Only

```bash
sudo ./scripts/setup/install-service.sh
```

### 2. Setup Monitoring (Optional)

```bash
sudo ./scripts/setup/setup-cron.sh
```

### 3. Start Service

```bash
sudo systemctl start ipfs-accelerate
```

## Service Management

### Check Status
```bash
sudo systemctl status ipfs-accelerate
```

### View Logs
```bash
sudo journalctl -u ipfs-accelerate -f
```

### Start/Stop/Restart
```bash
sudo systemctl start ipfs-accelerate
sudo systemctl stop ipfs-accelerate
sudo systemctl restart ipfs-accelerate
```

### Disable Auto-start
```bash
sudo systemctl disable ipfs-accelerate
```

## Service Configuration

The service runs with the following configuration:

- **User**: barberb
- **Working Directory**: /home/barberb/ipfs_accelerate_py
- **Virtual Environment**: /home/barberb/ipfs_accelerate_py/.venv
- **Command**: `ipfs-accelerate mcp start --host 0.0.0.0 --port 9000 --keep-running`
- **Auto-restart**: Yes (10 second delay)
- **Logging**: systemd journal

## Monitoring

The cron job monitoring (if enabled) will:

- Check service status every 5 minutes
- Restart the service if it's not responding
- Log activity to `/tmp/ipfs-accelerate-monitor.log`

### View Monitoring Logs
```bash
tail -f /tmp/ipfs-accelerate-monitor.log
```

### View/Edit Cron Jobs
```bash
crontab -l          # List cron jobs
crontab -e          # Edit cron jobs
```

## Access Points

Once running, the service provides:

- **Dashboard**: http://localhost:9000/dashboard
- **Health Check**: http://localhost:9000/health
- **API**: http://localhost:9000/api/

## Uninstallation

To remove the service:

```bash
sudo ./uninstall-service.sh
```

To remove the cron job:

```bash
crontab -e  # Then delete the line containing check-service.sh
```

## Troubleshooting

### Service Won't Start

1. Check the service status:
   ```bash
   sudo systemctl status ipfs-accelerate
   ```

2. Check the logs:
   ```bash
   sudo journalctl -u ipfs-accelerate -n 50
   ```

3. Verify the virtual environment:
   ```bash
   ls -la /home/barberb/ipfs_accelerate_py/.venv/bin/ipfs-accelerate
   ```

4. Test the command manually:
   ```bash
   cd /home/barberb/ipfs_accelerate_py
   source .venv/bin/activate
   ipfs-accelerate mcp start
   ```

### Port Already in Use

If port 9000 is already in use, you can modify the service file:

1. Edit the service file:
   ```bash
   sudo nano /etc/systemd/system/ipfs-accelerate.service
   ```

2. Change `--port 9000` to a different port (e.g., `--port 9001`)

3. Reload and restart:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ipfs-accelerate
   ```

### Permission Issues

Ensure the correct ownership:

```bash
sudo chown -R barberb:barberb /home/barberb/ipfs_accelerate_py
```

## Files Created

- `/etc/systemd/system/ipfs-accelerate.service` - Service definition
- `/home/barberb/ipfs_accelerate_py/check-service.sh` - Monitoring script
- Cron job entry for user `barberb`

## Security Notes

The service runs with restricted permissions:
- No new privileges
- Private temporary directory
- Read-only access to most of the system
- Write access only to the project directory