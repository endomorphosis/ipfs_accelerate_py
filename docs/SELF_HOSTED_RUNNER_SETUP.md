# Self-Hosted GitHub Actions Runner Setup Guide

This guide provides instructions for setting up self-hosted GitHub Actions runners for the ipfs_accelerate_py project, with a focus on Docker integration and hardware-specific configurations.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Docker Group Configuration](#docker-group-configuration)
- [Runner Installation](#runner-installation)
- [Hardware-Specific Setup](#hardware-specific-setup)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

Self-hosted runners allow you to run GitHub Actions workflows on your own infrastructure, which is particularly useful for:
- Hardware-specific testing (CUDA, ROCm, OpenVINO, etc.)
- Access to specialized resources (GPUs, NPUs, TPUs)
- Private network access
- Cost control for compute-intensive workloads

## Prerequisites

Before setting up a self-hosted runner, ensure you have:

1. A Linux machine with sudo access
2. Python 3.8 or later
3. Docker installed (for containerized testing)
4. Appropriate hardware drivers (for GPU/NPU testing)
5. GitHub repository admin access

## Docker Group Configuration

**IMPORTANT**: Tests on self-hosted runners that use Docker require the runner user to be added to the docker group.

### Add Runner User to Docker Group

```bash
# Replace <runner-user> with your actual runner username
sudo usermod -aG docker <runner-user>
```

### Common Runner User Names

Depending on your setup, the runner user might be:
- `runner` (default for many setups)
- `actions-runner` (common alternative)
- Your system username (if running as current user)
- A dedicated service account

### Verify Docker Group Membership

After adding the user to the docker group:

1. **Log out and log back in** (or start a new shell session)
2. Verify group membership:
   ```bash
   groups <runner-user>
   ```
   You should see `docker` in the list.

3. Test Docker access without sudo:
   ```bash
   docker ps
   ```
   This should work without requiring sudo.

### Apply Changes Without Logout (Alternative)

If you cannot log out, use `newgrp`:
```bash
newgrp docker
```

Or restart the runner service:
```bash
sudo systemctl restart actions-runner
```

## Runner Installation

### 1. Download and Configure Runner

```bash
# Navigate to your GitHub repository
# Go to Settings > Actions > Runners > New self-hosted runner

# Create a directory for the runner
mkdir actions-runner && cd actions-runner

# Download the latest runner package
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract the installer
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure the runner (use the token from GitHub)
./config.sh --url https://github.com/endomorphosis/ipfs_accelerate_py --token <YOUR_TOKEN>
```

### 2. Configure as a Service

Create a systemd service for automatic startup:

```bash
# Install the service
sudo ./svc.sh install

# Start the service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

### 3. Configure Runner Labels

When configuring the runner, add appropriate labels to identify hardware capabilities:

```bash
./config.sh --url https://github.com/endomorphosis/ipfs_accelerate_py \
  --token <YOUR_TOKEN> \
  --labels self-hosted,linux,x64,docker,cuda  # Add relevant labels
```

Common labels for this project:
- `docker` - Docker support enabled
- `cuda` - NVIDIA GPU with CUDA
- `rocm` - AMD GPU with ROCm
- `openvino` - Intel OpenVINO support
- `cpu-only` - CPU-only testing
- `gpu` - Generic GPU support

## Hardware-Specific Setup

### NVIDIA CUDA Setup

```bash
# Install NVIDIA drivers and CUDA toolkit
# Follow NVIDIA's official installation guide

# Install nvidia-docker for container support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Add runner user to docker and render groups
sudo usermod -aG docker <runner-user>
sudo usermod -aG video <runner-user>

# Verify GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### AMD ROCm Setup

```bash
# Install ROCm following AMD's official guide
# https://rocm.docs.amd.com/

# Add runner user to render and video groups
sudo usermod -aG docker <runner-user>
sudo usermod -aG render <runner-user>
sudo usermod -aG video <runner-user>

# Verify ROCm installation
rocm-smi
```

### Intel OpenVINO Setup

```bash
# Install OpenVINO Runtime
pip install openvino openvino-dev

# For GPU support, add user to render group
sudo usermod -aG docker <runner-user>
sudo usermod -aG render <runner-user>
```

## Security Considerations

### Important Security Notes

1. **Repository Access**: Self-hosted runners should only be used with private repositories or trusted code
2. **Secrets Management**: Be cautious with secrets on self-hosted runners
3. **Network Isolation**: Consider running runners in isolated networks
4. **Regular Updates**: Keep runner software and dependencies updated

### Docker Security

```bash
# Enable Docker content trust
export DOCKER_CONTENT_TRUST=1

# Limit Docker resources
cat > /etc/docker/daemon.json <<EOF
{
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

sudo systemctl restart docker
```

### Runner Isolation

For enhanced security, run each job in a clean environment:

```yaml
# In your workflow file
jobs:
  test:
    runs-on: self-hosted
    container:
      image: python:3.10
      options: --user 1000:1000
```

## Workflow Configuration

### Using Self-Hosted Runners in Workflows

Example workflow configuration:

```yaml
name: Hardware Tests

on: [push, pull_request]

jobs:
  cpu-tests:
    runs-on: [self-hosted, linux, docker, cpu-only]
    steps:
      - uses: actions/checkout@v4
      - name: Run CPU tests
        run: |
          docker run --rm -v $PWD:/workspace \
            python:3.10 python /workspace/test/run_tests.py --hardware cpu

  gpu-tests:
    runs-on: [self-hosted, linux, docker, cuda]
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU tests
        run: |
          docker run --rm --gpus all -v $PWD:/workspace \
            nvidia/cuda:11.8.0-runtime-ubuntu22.04 \
            python /workspace/test/run_tests.py --hardware cuda
```

## Troubleshooting

### Common Issues

#### Docker Permission Denied

**Error**: `Got permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
# Ensure user is in docker group
sudo usermod -aG docker <runner-user>

# Restart the runner service
sudo systemctl restart actions-runner

# Or log out and back in
```

#### Runner Not Picking Up Jobs

**Possible causes**:
1. Runner service not running: `sudo systemctl status actions-runner`
2. Incorrect labels: Check workflow `runs-on` matches runner labels
3. Runner offline: Check in GitHub repository settings

**Solution**:
```bash
# Check runner status
cd ~/actions-runner
./run.sh

# If it connects, then install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

#### GPU Not Accessible in Container

**Error**: `CUDA driver version is insufficient` or `No GPU devices found`

**Solution**:
```bash
# Verify nvidia-docker2 is installed
dpkg -l | grep nvidia-docker

# Install if missing
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### Disk Space Issues

**Error**: `No space left on device`

**Solution**:
```bash
# Clean up Docker resources
docker system prune -af

# Clean up old runner logs
cd ~/actions-runner/_diag
find . -name "*.log" -mtime +7 -delete

# Monitor disk usage
df -h
du -sh ~/actions-runner
```

## Monitoring and Maintenance

### Monitor Runner Health

```bash
# Check service status
sudo systemctl status actions-runner

# View recent logs
journalctl -u actions-runner -f

# Check runner connectivity
cd ~/actions-runner
./run.sh --check
```

### Regular Maintenance Tasks

1. **Weekly**: Review runner logs for errors
2. **Monthly**: Update runner software
3. **Quarterly**: Review and update dependencies
4. **As needed**: Clean up Docker images and volumes

### Update Runner Software

```bash
# Stop the runner service
sudo ./svc.sh stop

# Download latest version
curl -o actions-runner-linux-x64-latest.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64-latest.tar.gz

# Backup current installation
mv config.sh config.sh.bak

# Extract new version
tar xzf ./actions-runner-linux-x64-latest.tar.gz

# Restore configuration
mv config.sh.bak config.sh

# Restart service
sudo ./svc.sh start
```

## Additional Resources

- [GitHub Actions Self-Hosted Runner Documentation](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Docker Post-Installation Steps](https://docs.docker.com/engine/install/linux-postinstall/)
- [NVIDIA Docker Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Intel OpenVINO Documentation](https://docs.openvino.ai/)

## Summary

Key takeaways for self-hosted runner setup:

1. **Always add the runner user to the docker group** before running Docker-based tests
2. Use appropriate labels to match runners with workflow requirements
3. Keep runners updated and monitor their health regularly
4. Follow security best practices, especially for public repositories
5. Configure hardware-specific drivers and tools for specialized testing

For questions or issues, please open an issue in the repository or consult the GitHub Actions documentation.
