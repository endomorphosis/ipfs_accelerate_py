# 🐳 Docker Group Configuration for GitHub Actions Self-Hosted Runners

## Overview
Self-hosted GitHub Actions runners require proper Docker group permissions to execute Docker-based CI/CD workflows.

## Problem
When GitHub Actions workflows run Docker commands (build, run, push, etc.), they may fail with permission errors if the runner user lacks Docker access.

## Solution: Add Runner User to Docker Group

### 1. Add User to Docker Group
```bash
sudo usermod -aG docker <runner-user>
```

For this setup:
```bash
sudo usermod -aG docker barberb
```

### 2. Verify Group Membership
```bash
groups $USER
# Should output: barberb : barberb sudo users docker ollama
```

### 3. Test Docker Access
```bash
docker ps
docker --version
```

### 4. Restart GitHub Actions Runner Service
**Important**: The runner service must be restarted to pick up the new group membership.

```bash
cd ~/actions-runner-ipfs
sudo ./svc.sh stop
sudo ./svc.sh start
```

Or for systemd service:
```bash
sudo systemctl restart actions.runner.endomorphosis-ipfs_accelerate_py.arm64-dgx-spark-gb10-ipfs.service
```

## Verification

### Test Docker Functionality
```bash
# Basic Docker access
docker ps

# Build test
docker build --platform linux/arm64 --target minimal -t test-image .

# Run test  
docker run --rm test-image echo "Docker working"

# Clean up
docker rmi test-image
```

### Test in CI/CD Context
The runner should now be able to execute workflows containing:
- `docker build`
- `docker run`
- `docker push`
- `docker-compose` commands
- Multi-platform builds

## Common Issues & Solutions

### Issue: "Permission denied while trying to connect to Docker daemon"
**Solution**: Ensure user is in docker group and runner service is restarted.

### Issue: "Cannot connect to the Docker daemon"
**Solution**: Verify Docker daemon is running:
```bash
sudo systemctl status docker
sudo systemctl start docker  # if not running
```

### Issue: Runner service doesn't pick up group changes
**Solution**: Always restart the runner service after adding user to docker group.

## Security Considerations

Adding a user to the docker group grants significant privileges since Docker commands can be run with root-level access. This is acceptable for CI/CD runners but should be done thoughtfully.

### Best Practices:
1. Only add necessary users to docker group
2. Use dedicated user accounts for CI/CD runners
3. Regularly audit group membership
4. Monitor Docker usage in CI/CD logs

## Validation

Use the provided validation script to ensure everything is working:
```bash
python validate_setup.py
```

This should show:
- ✅ Docker access: PASSED
- ✅ Docker build: PASSED  
- ✅ Docker container execution: PASSED
- ✅ Docker group membership: PASSED

## Current Status ✅

For this setup:
- **User**: `barberb` 
- **Groups**: `barberb sudo users docker ollama`
- **Docker Access**: ✅ Working
- **Runner Service**: ✅ Active with docker permissions
- **CI/CD Ready**: ✅ All Docker operations functional

---

**File**: `DOCKER_GROUP_SETUP.md`  
**Last Updated**: October 24, 2025  
**Status**: ✅ CONFIGURED