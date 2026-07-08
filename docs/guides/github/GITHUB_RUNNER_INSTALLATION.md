# GitHub Actions Runner Installation Guide

This guide will help you install GitHub Actions runners on this machine to serve as backup runners for your existing setup.

## Prerequisites

- GitHub repository access with admin permissions
- sudo access on this machine
- GitHub Personal Access Token or runner registration token

## Quick Installation

### Step 1: Get Your GitHub Token

You need a registration token from GitHub:

1. Go to your repository: https://github.com/endomorphosis/ipfs_accelerate_py
2. Navigate to **Settings** > **Actions** > **Runners**
3. Click **New self-hosted runner**
4. Select **Linux** and your architecture
5. Copy the token from the configuration command

### Step 2: Install Primary Runner

For a primary runner with dependency installation:

```bash
cd /home/barberb/ipfs_accelerate_py
./scripts/setup-github-runner.sh --token YOUR_TOKEN_HERE --install-deps
```

### Step 3: Install Backup Runner (Recommended)

For a backup runner on the same machine:

```bash
cd /home/barberb/ipfs_accelerate_py
./scripts/setup-backup-runner.sh YOUR_TOKEN_HERE
```

## Advanced Installation Options

### Custom Runner Configuration

```bash
# Custom runner with specific labels
./scripts/setup-github-runner.sh \
    --token YOUR_TOKEN_HERE \
    --name my-custom-runner \
    --labels "self-hosted,linux,x64,docker,cuda,special" \
    --install-deps

# Runner without systemd service (manual start)
./scripts/setup-github-runner.sh \
    --token YOUR_TOKEN_HERE \
    --no-service

# Additional backup runner
./scripts/setup-github-runner.sh \
    --token YOUR_TOKEN_HERE \
    --additional \
    --name backup-runner-2
```

### Installation with GPU Support

The scripts automatically detect and configure GPU support:

- **NVIDIA GPU**: Adds `cuda,gpu` labels
- **AMD GPU**: Adds `rocm,gpu` labels  
- **Intel GPU**: Adds `openvino,gpu` labels
- **No GPU**: Adds `cpu-only` label

## System Requirements

### Automatic Dependencies (--install-deps)
- curl, wget, git
- build-essential
- Docker and Docker Compose
- Python 3 and pip

### Manual Dependencies
If you prefer to install dependencies manually:

```bash
# Update system
sudo apt-get update

# Install basic tools
sudo apt-get install -y curl wget git build-essential

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Python dependencies
python3 -m pip install --user --upgrade pip requests psutil
```

## Runner Management

### Service Control

```bash
# Check status
sudo systemctl status github-actions-runner
sudo systemctl status github-actions-runner-backup

# Start/stop/restart
sudo systemctl start github-actions-runner
sudo systemctl stop github-actions-runner
sudo systemctl restart github-actions-runner

# Enable/disable auto-start
sudo systemctl enable github-actions-runner
sudo systemctl disable github-actions-runner
```

### Manual Operation

```bash
# Run manually (if no service created)
cd ~/actions-runner
./run.sh

# Run backup runner manually
cd ~/actions-runner-backup
./run.sh
```

### Health Checks

```bash
# Quick health check
~/actions-runner/health-check.sh
~/actions-runner-backup/health-check.sh backup

# Detailed monitoring
~/actions-runner/monitor-runners.sh
```

## Troubleshooting

### Common Issues

1. **Permission Denied for Docker**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Service Won't Start**
   ```bash
   sudo journalctl -u github-actions-runner -f
   # Check logs for specific errors
   ```

3. **Runner Not Appearing in GitHub**
   - Check runner status: `sudo systemctl status github-actions-runner`
   - Verify token is valid and not expired
   - Check network connectivity

4. **Disk Space Issues**
   ```bash
   # Clean up Docker
   docker system prune -af
   
   # Clean up runner logs
   find ~/actions-runner/_diag -name "*.log" -mtime +7 -delete
   ```

### Uninstallation

```bash
# Stop and remove primary runner
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token YOUR_TOKEN_HERE

# Stop and remove backup runner
cd ~/actions-runner-backup
sudo ./svc.sh stop
sudo ./svc.sh uninstall  
./config.sh remove --token YOUR_TOKEN_HERE

# Remove directories
rm -rf ~/actions-runner ~/actions-runner-backup
```

## Security Considerations

- Runners have access to repository code and secrets
- Use dedicated machines for public repositories
- Regularly update runner software
- Monitor runner logs for suspicious activity
- Consider network isolation for sensitive workloads

## Configuration Files

After installation, you'll find:

- **Primary runner**: `~/actions-runner/`
- **Backup runner**: `~/actions-runner-backup/`
- **Logs**: `/var/log/github-actions/`
- **Service configs**: `/etc/systemd/system/github-actions-runner*.service`

## Hardware Detection

The setup scripts automatically detect and configure:

- **Architecture**: x64, arm64
- **GPU Support**: NVIDIA CUDA, AMD ROCm, Intel OpenVINO
- **Docker**: Automatically adds docker label if available
- **System Resources**: CPU, memory, disk space

## Labels Configuration

Default labels added:
- `self-hosted` - Always present
- `linux` - Operating system
- `x64`/`arm64` - Architecture
- `docker` - If Docker is available
- `cuda`/`rocm`/`openvino` - GPU type if detected
- `gpu` - Generic GPU support
- `cpu-only` - If no GPU detected
- `backup` - For backup runners

## Monitoring and Maintenance

### Automated Monitoring
- Health checks run every 5 minutes via cron
- Logs stored in `/var/log/github-actions/`
- Resource usage monitoring (CPU, memory, disk)

### Manual Checks
```bash
# System resources
df -h                    # Disk usage
free -h                  # Memory usage
top                      # CPU usage

# Runner status
sudo systemctl list-units | grep github-actions
sudo journalctl -u github-actions-runner --since "1 hour ago"
```

## Next Steps

1. **Verify Installation**: Check that runners appear in GitHub repository settings
2. **Test Workflows**: Run a simple workflow to test runner functionality  
3. **Configure Monitoring**: Set up additional monitoring if needed
4. **Documentation**: Update your team documentation with runner details

## Support

- **GitHub Actions Documentation**: https://docs.github.com/en/actions/hosting-your-own-runners
- **Project Documentation**: `/home/barberb/ipfs_accelerate_py/docs/SELF_HOSTED_RUNNER_SETUP.md`
- **Issues**: Create issues in the repository for runner-specific problems

---

**Note**: This installation creates backup runners to complement your existing GitHub Actions infrastructure. The runners will automatically register with your repository and begin accepting jobs based on their configured labels.