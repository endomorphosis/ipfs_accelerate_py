# GitHub Actions Self-Hosted Runner Setup Complete

**Date**: October 27, 2025  
**Machine**: workstation (devel@workstation)  
**Status**: Ready for configuration

## Setup Summary

A GitHub Actions self-hosted runner has been prepared for the `ipfs_accelerate_py` repository on this machine.

### System Specifications

- **OS**: Ubuntu 24.04.3 LTS (Noble Numbat)
- **Architecture**: x86_64
- **GPUs**: 2x NVIDIA GeForce RTX 3090 (24GB VRAM each)
- **CUDA Driver**: 575.57.08
- **User**: devel
- **Docker**: Installed and accessible (user added to docker group)

### Runner Directory

Location: `/home/devel/actions-runner-ipfs-accelerate`

### Setup Files Created

1. **`quick_setup.sh`** - Interactive setup script (RECOMMENDED)
   - Prompts for GitHub token
   - Configures runner automatically
   - Installs and starts service
   
2. **`setup_runner.sh`** - Command-line setup script
   - Usage: `./setup_runner.sh <GITHUB_TOKEN>`
   - Non-interactive automated setup
   
3. **`SETUP_INSTRUCTIONS.md`** - Detailed manual setup guide
   - Step-by-step instructions
   - Troubleshooting information
   - Configuration options

## Quick Start

### Option 1: Interactive Setup (Easiest)

```bash
cd ~/actions-runner-ipfs-accelerate
./quick_setup.sh
```

This will:
1. Ask you to get a token from GitHub
2. Configure the runner with the token you provide
3. Install it as a system service
4. Start the service
5. Verify it's running

### Option 2: Command-Line Setup

```bash
# Get token from: https://github.com/endomorphosis/ipfs_accelerate_py/settings/actions/runners/new
cd ~/actions-runner-ipfs-accelerate
./setup_runner.sh YOUR_GITHUB_TOKEN_HERE
```

### Option 3: Manual Setup

Follow the instructions in `SETUP_INSTRUCTIONS.md`:

```bash
cd ~/actions-runner-ipfs-accelerate
cat SETUP_INSTRUCTIONS.md
```

## Runner Configuration

When configured, the runner will have:

- **Name**: `gpu-runner-rtx3090-ipfs-accelerate`
- **Repository**: `endomorphosis/ipfs_accelerate_py`
- **Labels**: `self-hosted`, `linux`, `x64`, `docker`, `cuda`, `gpu`, `rtx3090`
- **Work Directory**: `_work`

## Using the Runner in Workflows

Once configured, use it in your GitHub Actions workflows:

```yaml
name: Test on Self-Hosted Runner
on: [push, pull_request]

jobs:
  test-cpu:
    runs-on: [self-hosted, linux, docker]
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          python -m pytest tests/
          
  test-gpu:
    runs-on: [self-hosted, linux, cuda]
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU tests
        run: |
          nvidia-smi
          python -m pytest tests/ --gpu
```

## Verification

After setup, verify the runner:

1. **Check service status**:
   ```bash
   cd ~/actions-runner-ipfs-accelerate
   sudo ./svc.sh status
   ```

2. **View logs**:
   ```bash
   journalctl -u actions.runner.endomorphosis-ipfs_accelerate_py.gpu-runner-rtx3090-ipfs-accelerate.service -f
   ```

3. **Check GitHub UI**:
   Go to: https://github.com/endomorphosis/ipfs_accelerate_py/settings/actions/runners
   
   You should see your runner listed with a green dot indicating it's online.

## Service Management

```bash
cd ~/actions-runner-ipfs-accelerate

# Start the service
sudo ./svc.sh start

# Stop the service
sudo ./svc.sh stop

# Check status
sudo ./svc.sh status

# Restart the service
sudo ./svc.sh stop && sudo ./svc.sh start

# Uninstall service (doesn't remove configuration)
sudo ./svc.sh uninstall
```

## Troubleshooting

### Docker Permission Issues

If you get "permission denied" errors with Docker:

```bash
# Verify docker group membership
groups

# If docker is not listed, log out and back in, or:
newgrp docker

# Restart the runner service
cd ~/actions-runner-ipfs-accelerate
sudo ./svc.sh stop
sudo ./svc.sh start
```

### Runner Not Showing in GitHub

1. Check service status: `sudo ./svc.sh status`
2. Check logs: `journalctl -u actions.runner.endomorphosis-ipfs_accelerate_py.* -f`
3. Verify network connectivity to GitHub
4. Check token hasn't expired

### GPU Not Accessible

Verify GPU access:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If docker GPU access fails, install nvidia-docker2:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Current Runner Status

You have the following runners configured on this machine:

1. **ipfs_datasets_py** - Running ✅
   - Name: `gpu-runner-rtx3090`
   - Service: Active

2. **ipfs_kit_py** - Failed/Stopped ❌
   - Multiple instances with issues

3. **ipfs_accelerate_py** - Ready to configure ⚙️
   - Directory prepared
   - Awaiting GitHub token

## Next Steps

1. **Run the setup script**:
   ```bash
   cd ~/actions-runner-ipfs-accelerate
   ./quick_setup.sh
   ```

2. **Verify runner appears in GitHub** (after setup):
   https://github.com/endomorphosis/ipfs_accelerate_py/settings/actions/runners

3. **Test with a workflow**:
   Create a simple workflow to test the runner works correctly

4. **Update CI/CD workflows**:
   Update existing workflows to use the self-hosted runner for appropriate jobs

## Security Notes

- This runner is configured for a specific repository
- It will only accept jobs from `endomorphosis/ipfs_accelerate_py`
- Jobs will run with `devel` user permissions
- Docker containers will have access to GPUs via `--gpus` flag
- Consider the security implications before accepting jobs from public forks

## Documentation References

- [Self-Hosted Runner Setup Guide](/home/devel/ipfs_accelerate_py/docs/SELF_HOSTED_RUNNER_SETUP.md)
- [CI/CD Integration Guide](/home/devel/ipfs_accelerate_py/test/distributed_testing/CI_CD_INTEGRATION_GUIDE.md)
- [Benchmark CI Integration](/home/devel/ipfs_accelerate_py/benchmarks/BENCHMARK_CI_INTEGRATION.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions/hosting-your-own-runners)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed setup guide: `SETUP_INSTRUCTIONS.md`
3. Check GitHub Actions runner logs
4. Consult the repository documentation in `docs/`
