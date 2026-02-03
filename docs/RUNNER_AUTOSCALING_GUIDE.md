# GitHub Actions Runner Autoscaling Guide

## Overview

The GitHub Actions runner autoscaling system automatically monitors workflow queues and provisions self-hosted runners as needed. It's fully integrated into the unified ipfs-accelerate architecture, using the same Docker provisioning methods across all interfaces.

## Architecture

```
ipfs_accelerate_py/kit/runner_kit.py
    ├─ Uses docker_kit.py (container operations)
    └─ Uses github_kit.py (GitHub operations)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
unified_cli.py      mcp/unified_tools.py
(CLI commands)      (MCP tools)
    ↓                   ↓
    └─────────┬─────────┘
              ↓
    MCP Server → JavaScript SDK → Dashboard
```

## Features

### Core Capabilities

- **Automatic Workflow Monitoring**: Continuously monitors GitHub repositories for pending workflows
- **Dynamic Runner Provisioning**: Launches Docker containers running GitHub Actions runners
- **Multi-Repository Support**: Handles workflows across multiple repositories
- **Resource Management**: Enforces CPU and memory limits on runner containers
- **Security**: Network isolation, read-only filesystems, no new privileges
- **Cleanup**: Automatic cleanup of completed runner containers
- **Status Reporting**: Real-time status and metrics

### Integration Points

1. **CLI Interface**: `ipfs-accelerate runner` commands
2. **MCP Tools**: 7 tools for programmatic access
3. **Python API**: Direct import of runner_kit module
4. **Dashboard**: Via JavaScript SDK integration

## Usage

### 1. CLI Commands

#### Start Autoscaler

Start monitoring workflows and provisioning runners:

```bash
# Basic usage
ipfs-accelerate runner start

# Monitor specific organization
ipfs-accelerate runner start --owner myorg

# Custom configuration
ipfs-accelerate runner start \
  --owner myorg \
  --interval 60 \
  --max-runners 8 \
  --image myoung34/github-runner:latest

# Run in background
ipfs-accelerate runner start --owner myorg --background
```

#### Get Status

Check autoscaler status:

```bash
ipfs-accelerate runner status
```

Output:
```json
{
  "running": true,
  "start_time": "2024-01-15T10:30:00Z",
  "iterations": 42,
  "active_runners": 3,
  "queued_workflows": 5,
  "repositories_monitored": 10,
  "last_check": "2024-01-15T11:45:00Z"
}
```

#### List Workflows

View workflow queues:

```bash
ipfs-accelerate runner list-workflows
```

Output:
```json
[
  {
    "repo": "owner/repo1",
    "total_workflows": 3,
    "running": 1,
    "failed": 0,
    "pending": 2
  },
  {
    "repo": "owner/repo2",
    "total_workflows": 2,
    "running": 2,
    "failed": 0,
    "pending": 0
  }
]
```

#### List Runner Containers

View active runner containers:

```bash
ipfs-accelerate runner list-containers
```

Output:
```json
[
  {
    "container_id": "abc123def456",
    "repo": "owner/repo1",
    "status": "running",
    "created_at": "2024-01-15T11:30:00Z"
  }
]
```

#### Manual Provisioning

Manually provision a runner for a specific repository:

```bash
# Provision for specific repo
ipfs-accelerate runner provision --repo owner/repo

# Provision for all queues
ipfs-accelerate runner provision
```

#### Stop Container

Stop a specific runner container:

```bash
ipfs-accelerate runner stop-container --container abc123def456
```

#### Stop Autoscaler

Stop the autoscaler:

```bash
ipfs-accelerate runner stop
```

### 2. MCP Tools (JavaScript SDK)

All CLI commands are available as MCP tools:

```javascript
const MCP = require('ipfs-accelerate-mcp-sdk');

const mcp = new MCP({
  url: 'http://localhost:8080'
});

// Start autoscaler
const startResult = await mcp.call_tool('runner_start_autoscaler', {
  owner: 'myorg',
  poll_interval: 60,
  max_runners: 8,
  background: true
});

// Get status
const status = await mcp.call_tool('runner_get_status', {});
console.log('Autoscaler status:', status.data);

// List workflows
const workflows = await mcp.call_tool('runner_list_workflows', {});
console.log('Workflow queues:', workflows.data);

// Provision for specific repo
const provision = await mcp.call_tool('runner_provision_for_workflow', {
  repo: 'owner/repo'
});

// List containers
const containers = await mcp.call_tool('runner_list_containers', {});

// Stop container
await mcp.call_tool('runner_stop_container', {
  container: 'abc123def456'
});

// Stop autoscaler
await mcp.call_tool('runner_stop_autoscaler', {});
```

### 3. Python API

Use runner_kit directly in Python code:

```python
from ipfs_accelerate_py.kit.runner_kit import get_runner_kit, RunnerConfig

# Create configuration
config = RunnerConfig(
    owner='myorg',
    poll_interval=60,
    max_runners=8,
    runner_image='myoung34/github-runner:latest',
    network_mode='host',
    memory_limit='4g',
    cpu_limit=4.0
)

# Get runner kit instance
kit = get_runner_kit(config)

# Start autoscaler in background
kit.start_autoscaler(background=True)

# Get status
status = kit.get_status()
print(f"Running: {status.running}")
print(f"Active runners: {status.active_runners}")

# List workflow queues
queues = kit.get_workflow_queues()
for queue in queues:
    print(f"{queue.repo}: {queue.total} workflows")

# Manually provision runner
token = kit.generate_runner_token('owner/repo')
if token:
    container_id = kit.launch_runner_container('owner/repo', token)
    print(f"Launched container: {container_id}")

# List runner containers
runners = kit.list_runner_containers()
for runner in runners:
    print(f"Container {runner.container_id}: {runner.status}")

# Stop autoscaler
kit.stop_autoscaler()
```

## Configuration

### RunnerConfig Options

```python
@dataclass
class RunnerConfig:
    owner: Optional[str] = None              # GitHub owner to monitor
    poll_interval: int = 120                 # Seconds between checks
    since_days: int = 1                      # Monitor repos updated in N days
    max_runners: int = 10                    # Maximum concurrent runners
    filter_by_arch: bool = True              # Filter by system architecture
    enable_p2p: bool = False                 # Enable P2P workflow monitoring
    runner_image: str = "myoung34/..."       # Docker image for runners
    runner_work_dir: str = "/tmp/_work"      # Work directory for runners
    token_file: str = "/var/lib/..."         # Token file path
    network_mode: str = "host"               # Docker network mode
    memory_limit: str = "4g"                 # Memory limit per runner
    cpu_limit: float = 4.0                   # CPU limit per runner
```

### Environment Variables

The runner containers support these environment variables:

- `GITHUB_TOKEN`: GitHub PAT for API access
- `CACHE_ENABLE_P2P`: Enable P2P cache (default: true)
- `CACHE_LISTEN_PORT`: P2P cache port
- `CACHE_BOOTSTRAP_PEERS`: P2P bootstrap peers
- `CACHE_P2P_SHARED_SECRET`: P2P encryption secret
- `RUNNER_DOCKER_NETWORK_MODE`: Network mode for runners

## Docker Provisioning

### Container Configuration

Runner containers are launched with:

**Security:**
- `--security-opt=no-new-privileges`
- `--cap-drop=ALL`
- `--cap-add=NET_ADMIN` (for runner networking)
- `--cap-add=NET_RAW`

**Resources:**
- `--memory=4g` (configurable)
- `--cpus=4` (configurable)

**Volumes:**
- Work directory: `/tmp/_work`
- Docker socket: `/var/run/docker.sock` (for Docker-in-Docker)

**Environment:**
- `REPO_URL`: GitHub repository URL
- `RUNNER_TOKEN`: Registration token
- `RUNNER_NAME`: Unique runner name
- `LABELS`: Runner labels
- `EPHEMERAL=true`: Self-destruct after one job

### Network Modes

**Host Mode (default):**
- Runners share host network
- Good for P2P cache access
- No port mapping needed

**Bridge Mode:**
- Isolated network
- Better security
- Requires port mapping for services

## Security Considerations

### Authentication

- **GitHub CLI Authentication**: Required for token generation
  ```bash
  gh auth login
  ```

### Token Management

- Tokens are generated on-demand per repository
- Tokens are ephemeral and expire quickly
- Tokens are not stored persistently

### Container Isolation

- Runners run in isolated Docker containers
- Network isolation by default (configurable)
- Resource limits enforced
- No new privileges granted
- Minimal capabilities

### Best Practices

1. **Use ephemeral runners**: Always set `EPHEMERAL=true`
2. **Limit resources**: Set appropriate CPU and memory limits
3. **Monitor containers**: Regularly check active containers
4. **Clean up**: Enable automatic cleanup of exited containers
5. **Rotate tokens**: Tokens are automatically regenerated for new runners
6. **Network isolation**: Use `network_mode="none"` for maximum isolation

## Monitoring

### Status Checks

Regular status checks provide:
- Number of active runners
- Queued workflows count
- Repositories being monitored
- Last check timestamp
- Iteration count

### Metrics

Track these metrics:
- `active_runners`: Current running containers
- `queued_workflows`: Workflows waiting for runners
- `repositories_monitored`: Repos being watched
- `iterations`: Number of check cycles
- `provisioned_runners`: Total runners provisioned

### Logging

Enable verbose logging:

```bash
ipfs-accelerate runner start --verbose
```

Logs include:
- Workflow queue status
- Runner provisioning events
- Container lifecycle events
- Errors and warnings

## Troubleshooting

### Common Issues

**1. "Not authenticated with GitHub CLI"**

Solution:
```bash
gh auth login
```

**2. "Docker is not accessible"**

Solution:
```bash
# Check Docker is running
docker version

# Check Docker permissions
sudo usermod -aG docker $USER
```

**3. "Failed to generate runner token"**

Solution:
- Verify GitHub authentication: `gh auth status`
- Check repository permissions
- Verify repository exists

**4. "Container failed to start"**

Solution:
- Check Docker image exists: `docker images`
- Pull runner image: `docker pull myoung34/github-runner:latest`
- Check Docker logs: `docker logs <container_id>`

**5. "At capacity: X/Y runners active"**

Solution:
- Increase `max_runners` setting
- Stop unused containers
- Check for stuck containers

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ipfs_accelerate_py.kit.runner_kit import get_runner_kit
kit = get_runner_kit()
kit.start_autoscaler()
```

## Migration Guide

### From Standalone Autoscaler

If you're using the standalone `github_autoscaler.py`:

**Before:**
```bash
python scripts/utils/github_autoscaler.py --owner myorg
```

**After (CLI):**
```bash
ipfs-accelerate runner start --owner myorg
```

**After (Python):**
```python
from ipfs_accelerate_py.kit.runner_kit import get_runner_kit, RunnerConfig

config = RunnerConfig(owner='myorg')
kit = get_runner_kit(config)
kit.start_autoscaler()
```

### Benefits of Migration

1. **Unified interface**: Same pattern as other ipfs-accelerate commands
2. **Code reuse**: Uses docker_kit and github_kit
3. **MCP integration**: Available via MCP tools
4. **Better testing**: Pure Python module
5. **Maintainability**: Single source of truth

## Examples

### Example 1: Basic Autoscaling

```bash
# Start monitoring all accessible repos
ipfs-accelerate runner start --background

# Check status
ipfs-accelerate runner status

# View workflows
ipfs-accelerate runner list-workflows

# Stop when done
ipfs-accelerate runner stop
```

### Example 2: Organization Monitoring

```bash
# Monitor specific organization
ipfs-accelerate runner start \
  --owner myorg \
  --interval 60 \
  --max-runners 20 \
  --background
```

### Example 3: Dashboard Integration

```javascript
// In your dashboard
async function monitorRunners() {
  const status = await mcp.call_tool('runner_get_status', {});
  
  document.getElementById('active-runners').textContent = 
    status.data.active_runners;
  
  document.getElementById('queued-workflows').textContent = 
    status.data.queued_workflows;
  
  const workflows = await mcp.call_tool('runner_list_workflows', {});
  renderWorkflowTable(workflows.data);
}

// Update every 30 seconds
setInterval(monitorRunners, 30000);
```

### Example 4: Custom Provisioning Logic

```python
from ipfs_accelerate_py.kit.runner_kit import get_runner_kit

kit = get_runner_kit()

# Get workflow queues
queues = kit.get_workflow_queues()

# Custom logic: provision extra runners for important repos
priority_repos = ['owner/critical-app', 'owner/main-service']

for queue in queues:
    if queue.repo in priority_repos and queue.pending > 0:
        # Provision multiple runners for priority repos
        for i in range(min(queue.pending, 3)):
            token = kit.generate_runner_token(queue.repo)
            if token:
                container_id = kit.launch_runner_container(queue.repo, token)
                print(f"Provisioned priority runner: {container_id}")
```

## API Reference

See the [runner_kit.py module documentation](../ipfs_accelerate_py/kit/runner_kit.py) for complete API reference.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs with `--verbose`
3. Check GitHub repository issues
4. Verify Docker and GitHub CLI are working

## License

Same as parent project (see LICENSE file).
