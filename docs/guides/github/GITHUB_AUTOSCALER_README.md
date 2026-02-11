# GitHub Actions Runner Autoscaler with Docker Isolation

## Overview

The GitHub Actions Runner Autoscaler automatically monitors GitHub workflow queues and provisions self-hosted runners as needed. It includes **architecture-based filtering** and **Docker container isolation** to ensure secure and appropriate job execution.

## Key Security Features

### 1. Architecture-Based Filtering 🎯

The autoscaler automatically detects the system architecture and **only provisions runners for compatible workflows**:

- **x64 (x86_64) systems**: Will NOT provision runners for ARM64-specific workflows
- **ARM64 (aarch64) systems**: Will NOT provision runners for x64-specific workflows
- **Detection**: Automatic based on workflow names, job labels, and runner requirements

This prevents:
- ❌ ARM64 machines from attempting to run x86-only jobs
- ❌ x86 machines from attempting to run ARM64-only jobs
- ✅ Resource waste on incompatible architectures
- ✅ Job failures due to architecture mismatch

### 2. Docker Container Isolation 🐳

All GitHub Actions workflows in this repository run inside **isolated Docker containers** on the runners. See [CONTAINERIZED_CI_SECURITY.md](../CONTAINERIZED_CI_SECURITY.md) for full details.

**Security Benefits:**
- **Process Isolation**: Tests cannot affect host system processes
- **Filesystem Isolation**: No access to host files outside mounted context
- **Network Isolation**: Controlled network access policies
- **Resource Limits**: CPU and memory limits prevent exhaustion attacks
- **Privilege Separation**: Containers run as non-root users

### 3. Automatic Runner Labeling 🏷️

The autoscaler automatically applies appropriate labels based on system capabilities:

```
Base labels: self-hosted, linux, <architecture>, docker
GPU labels: cuda, rocm, openvino (if hardware detected)
Fallback: cpu-only (if no GPU)
```

Example labels:
- x64 system with NVIDIA GPU: `self-hosted,linux,x64,docker,cuda,gpu`
- ARM64 system without GPU: `self-hosted,linux,arm64,docker,cpu-only`

## Quick Start

### Prerequisites

1. **GitHub CLI** installed and authenticated:
   ```bash
   # Install GitHub CLI
   # See: https://cli.github.com/
   
   # Authenticate
   gh auth login
   ```

2. **Docker** installed (for runner isolation):
   ```bash
   # Check Docker is available
   docker ps
   
   # If not accessible, add user to docker group
   sudo usermod -aG docker $USER
   # Log out and back in for changes to take effect
   ```

3. **Python 3.8+** with required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Installation as Service

#### Option 1: User Service (Recommended)

```bash
# Install the service
./scripts/manage-autoscaler.sh install --user

# Start the service
./scripts/manage-autoscaler.sh start

# Check status
./scripts/manage-autoscaler.sh status

# View logs
./scripts/manage-autoscaler.sh logs
```

#### Option 2: System Service

```bash
# Install the service (requires sudo)
./scripts/manage-autoscaler.sh install --system

# Start the service
sudo systemctl start github-autoscaler

# Check status
sudo systemctl status github-autoscaler

# View logs
sudo journalctl -u github-autoscaler -f
```

### Manual Testing

Test the autoscaler without installing as a service:

```bash
# Run in foreground
python3 github_autoscaler.py

# Run with specific organization
python3 github_autoscaler.py --owner myorg

# Custom settings
python3 github_autoscaler.py --interval 30 --max-runners 8
```

## Configuration

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--owner` | GitHub owner (user/org) to monitor | All accessible repos |
| `--interval` | Poll interval in seconds | 60 |
| `--since-days` | Monitor repos updated in last N days | 1 |
| `--max-runners` | Maximum runners to provision | System CPU cores |
| `--no-arch-filter` | Disable architecture filtering | Enabled |

### Environment Variables

The autoscaler respects standard GitHub CLI configuration:

```bash
# Use specific GitHub instance
export GH_HOST=github.company.com

# Use specific token (alternative to gh auth login)
export GH_TOKEN=ghp_yourtoken
```

## How It Works

### Architecture Filtering Process

1. **System Detection**: On startup, detect system architecture (x64 or arm64)
2. **Workflow Analysis**: For each workflow in the queue:
   - Check workflow name for architecture keywords (`arm64`, `aarch64`, `x64`, `amd64`)
   - Inspect job labels for architecture requirements
   - Compare against system architecture
3. **Filtering**: Only include compatible workflows in the provisioning queue
4. **Provisioning**: Generate runner tokens only for compatible workflows

### Example: x64 System Behavior

On an **x64 (x86_64) system**, the autoscaler will:

✅ **Provision runners for:**
- Workflows named `amd64-ci.yml` or `test-amd64-containerized`
- Workflows with job labels containing `x64` or `amd64`
- Generic workflows without specific architecture requirements

❌ **Skip provisioning for:**
- Workflows named `arm64-ci.yml` or `test-arm64-containerized`
- Workflows with job labels containing `arm64` or `aarch64`

### Logging Output

When architecture filtering is active, you'll see:

```
[2025-11-02 14:30:00] INFO: Auto-scaler configured:
[2025-11-02 14:30:00] INFO:   System architecture: x64
[2025-11-02 14:30:00] INFO:   Runner labels: self-hosted,linux,x64,docker,cpu-only
[2025-11-02 14:30:00] INFO:   Architecture filtering: enabled
[2025-11-02 14:30:00] INFO:   Docker isolation: enabled (see CONTAINERIZED_CI_SECURITY.md)

[2025-11-02 14:30:15] INFO: Checking workflow queues...
[2025-11-02 14:30:16] INFO: Found 3 repos with 7 workflows
[2025-11-02 14:30:16] INFO:   Running: 2, Failed: 1
[2025-11-02 14:30:16] INFO:   (Filtered for x64 architecture)
[2025-11-02 14:30:16] INFO:   Filtered 2 incompatible workflows for x64
```

## Service Management

### Start/Stop/Restart

```bash
# User service
./scripts/manage-autoscaler.sh start
./scripts/manage-autoscaler.sh stop
./scripts/manage-autoscaler.sh restart

# System service
sudo systemctl start github-autoscaler
sudo systemctl stop github-autoscaler
sudo systemctl restart github-autoscaler
```

### Check Status

```bash
# User service
./scripts/manage-autoscaler.sh status
systemctl --user status github-autoscaler

# System service
sudo systemctl status github-autoscaler
```

### View Logs

```bash
# User service
./scripts/manage-autoscaler.sh logs
journalctl --user -u github-autoscaler -f

# System service
sudo journalctl -u github-autoscaler -f
```

### Uninstall

```bash
# User service
./scripts/manage-autoscaler.sh uninstall

# System service
./scripts/manage-autoscaler.sh uninstall
```

## Security Best Practices

### 1. Runner Isolation

Ensure all runners execute jobs in Docker containers:

```yaml
# In your GitHub Actions workflow
jobs:
  test:
    runs-on: self-hosted
    container:
      image: python:3.11
    steps:
      - name: Run tests
        run: pytest tests/
```

### 2. Architecture Specification

Explicitly specify architecture requirements in workflow files:

```yaml
# For x64-specific workflows
jobs:
  test-x64:
    runs-on: [self-hosted, linux, x64, docker]
    
# For ARM64-specific workflows
jobs:
  test-arm64:
    runs-on: [self-hosted, linux, arm64, docker]
```

### 3. Resource Limits

The systemd service includes resource limits:

```ini
# CPU limit: 50% of one core
CPUQuota=50%

# Memory limit: 512MB
MemoryLimit=512M
```

### 4. Network Security

Review and restrict network access in Docker containers:

```yaml
# Example: Restrict network in container
container:
  image: python:3.11
  options: --network none  # No network access
```

## Troubleshooting

### Issue: "Not authenticated with GitHub CLI"

**Solution:**
```bash
gh auth login
```

### Issue: "No workflows need runner provisioning"

**Possible causes:**
1. No workflows are currently running or failed
2. All workflows are filtered out by architecture
3. Repositories haven't been updated recently

**Solutions:**
```bash
# Increase monitoring window
python3 github_autoscaler.py --since-days 7

# Disable architecture filtering (not recommended)
python3 github_autoscaler.py --no-arch-filter

# Check specific repository
gh run list --repo owner/repo
```

### Issue: Workflows filtered by architecture

**Expected behavior** on x64 system:
```
INFO:   Filtered 2 incompatible workflows for x64
```

This means the autoscaler correctly identified 2 workflows that require ARM64 architecture and skipped them.

**To verify:**
```bash
# Check workflow file
cat .github/workflows/arm64-ci.yml

# Look for:
runs-on: [self-hosted, linux, arm64]
```

### Issue: Docker not accessible

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker ps
```

## Monitoring

### Metrics to Watch

1. **Provisioned Runners**: Number of runner tokens generated
2. **Filtered Workflows**: Number of workflows skipped due to architecture
3. **System Architecture**: Current system architecture (x64 or arm64)
4. **Runner Labels**: Labels applied to provisioned runners

### Health Checks

```bash
# Check service is running
./scripts/manage-autoscaler.sh status

# Check recent activity
./scripts/manage-autoscaler.sh logs | tail -50

# Verify architecture detection
python3 -c "
from ipfs_accelerate_py.github_cli import RunnerManager
rm = RunnerManager()
print(f'Architecture: {rm.get_system_architecture()}')
print(f'Labels: {rm.get_runner_labels()}')
"
```

## API Usage

The autoscaler makes minimal GitHub API calls:

- 1 call to list repositories (if no `--owner`)
- N calls to list workflow runs (N = number of active repos)
- M calls to generate tokens (M = runners to provision)

**Example**: With 10 active repos and 60s interval:
- ~10-20 API calls per minute
- Well within GitHub's rate limits (5000/hour authenticated)

## Advanced Usage

### Disable Architecture Filtering

For testing or special cases, you can disable architecture filtering:

```bash
python3 github_autoscaler.py --no-arch-filter
```

**Warning**: This may result in job failures if workflows have architecture-specific requirements.

### Multiple Organizations

Run multiple autoscaler instances for different organizations:

```bash
# Terminal 1: Monitor org1
python3 github_autoscaler.py --owner org1

# Terminal 2: Monitor org2
python3 github_autoscaler.py --owner org2
```

### Custom Runner Labels

Modify `RunnerManager._generate_runner_labels()` in `ipfs_accelerate_py/github_cli/wrapper.py` to customize labels:

```python
def _generate_runner_labels(self) -> str:
    labels = ['self-hosted', 'linux', self._system_arch, 'docker']
    
    # Add custom labels
    labels.append('my-custom-label')
    
    return ','.join(labels)
```

## Testing

### Unit Tests

```bash
# Test architecture detection
python3 -c "
from ipfs_accelerate_py.github_cli import RunnerManager
rm = RunnerManager()
assert rm.get_system_architecture() in ['x64', 'arm64']
print('✓ Architecture detection works')
"

# Test workflow filtering
python3 -c "
from ipfs_accelerate_py.github_cli import WorkflowQueue
wq = WorkflowQueue()
workflow = {'workflowName': 'arm64-ci.yml'}
assert not wq._check_workflow_runner_compatibility(workflow, 'test/repo', 'x64')
print('✓ Workflow filtering works')
"
```

### Integration Tests

```bash
# Dry run (without actually starting runners)
python3 github_autoscaler.py --interval 300 &

# Wait for one cycle
sleep 310

# Check logs
./scripts/manage-autoscaler.sh logs | grep "Architecture filtering"
```

## See Also

- [CONTAINERIZED_CI_SECURITY.md](../CONTAINERIZED_CI_SECURITY.md) - Docker isolation details
- [AUTOSCALER.md](../AUTOSCALER.md) - General autoscaler documentation
- [GitHub Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [GitHub Actions Security](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

## Support

For issues or questions:

1. Check logs: `./scripts/manage-autoscaler.sh logs`
2. Verify GitHub CLI auth: `gh auth status`
3. Check Docker: `docker ps`
4. Open an issue on GitHub with:
   - System architecture (`uname -m`)
   - Service logs
   - GitHub workflow configuration
