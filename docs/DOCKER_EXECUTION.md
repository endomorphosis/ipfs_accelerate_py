# Docker Execution in IPFS Accelerate MCP

## Overview

The Docker execution feature allows the IPFS Accelerate MCP server to execute arbitrary code in Docker containers. This enables running code from Docker Hub, building and executing GitHub repositories, and running custom payloads in isolated environments.

## Architecture

Following the IPFS Accelerate architecture pattern:

```
ipfs_accelerate_py.docker_executor (core module)
    ↓ (exposed as)
ipfs_accelerate_py.mcp.tools.docker_tools (MCP tool wrappers)
    ↓ (exposed to)
MCP server JavaScript SDK
```

## Features

### 1. Execute Docker Hub Containers

Run pre-built containers from Docker Hub with custom commands and configurations.

**Use Cases:**
- Run Python scripts in isolated environments
- Execute shell commands in specific OS distributions
- Test code in different runtime environments
- Run data processing tasks

### 2. Build and Execute from GitHub

Clone a GitHub repository, build a Docker image from its Dockerfile, and execute it.

**Use Cases:**
- Run applications directly from GitHub repositories
- Test code from pull requests
- Deploy and execute containerized applications
- CI/CD workflows

### 3. Execute with Custom Payload

Execute containers with custom code or data payloads.

**Use Cases:**
- Run dynamic Python/shell scripts
- Execute configuration-driven tasks
- Data processing with custom scripts
- Ad-hoc code execution

### 4. Container Management

List, monitor, and stop running containers.

**Use Cases:**
- Monitor active containers
- Clean up resources
- Manage long-running processes

## Core Module: `docker_executor.py`

### Classes

#### `DockerExecutionConfig`

Configuration for Docker container execution.

```python
from ipfs_accelerate_py.docker_executor import DockerExecutionConfig

config = DockerExecutionConfig(
    image="python:3.9",
    command=["python", "-c", "print('Hello')"],
    memory_limit="512m",
    cpu_limit=1.0,
    timeout=60,
    environment={"VAR": "value"},
    network_mode="none"
)
```

**Parameters:**
- `image`: Docker image name
- `command`: Command to run
- `entrypoint`: Custom entrypoint
- `working_dir`: Working directory
- `memory_limit`: Memory limit (e.g., "512m", "2g")
- `cpu_limit`: CPU limit (e.g., 1.0, 2.5)
- `timeout`: Execution timeout in seconds
- `environment`: Environment variables
- `volumes`: Volume mounts (host_path: container_path)
- `network_mode`: Network mode ("none", "bridge", "host")
- `read_only`: Read-only filesystem
- `no_new_privileges`: Security setting
- `user`: User to run as

#### `GitHubDockerConfig`

Configuration for building from GitHub repositories.

```python
from ipfs_accelerate_py.docker_executor import GitHubDockerConfig

config = GitHubDockerConfig(
    repo_url="https://github.com/user/repo",
    branch="main",
    dockerfile_path="Dockerfile",
    build_args={"PYTHON_VERSION": "3.9"}
)
```

#### `DockerExecutor`

Main executor class.

```python
from ipfs_accelerate_py.docker_executor import DockerExecutor

executor = DockerExecutor()

# Execute container
result = executor.execute_container(config)

# Build and execute from GitHub
result = executor.build_and_execute_github_repo(github_config, exec_config)

# List containers
containers = executor.list_running_containers()

# Stop container
success = executor.stop_container("container_id")
```

### Convenience Functions

```python
from ipfs_accelerate_py.docker_executor import (
    execute_docker_hub_container,
    build_and_execute_from_github
)

# Quick execution
result = execute_docker_hub_container(
    image="python:3.9",
    command=["python", "-c", "print('test')"],
    timeout=60
)

# Build and run from GitHub
result = build_and_execute_from_github(
    repo_url="https://github.com/user/repo",
    branch="main",
    command=["python", "app.py"]
)
```

## MCP Tools: `docker_tools.py`

### Available Tools

#### 1. `execute_docker_container`

Execute a pre-built Docker container.

**Parameters:**
- `image` (required): Docker image name
- `command`: Command to run
- `entrypoint`: Custom entrypoint
- `environment`: Environment variables
- `memory_limit`: Memory limit (default: "2g")
- `cpu_limit`: CPU limit
- `timeout`: Timeout in seconds (default: 300)
- `network_mode`: Network mode (default: "none")
- `working_dir`: Working directory

**Returns:**
```json
{
  "success": true,
  "exit_code": 0,
  "stdout": "output...",
  "stderr": "",
  "execution_time": 1.5,
  "error_message": null
}
```

**Example:**
```javascript
// Via MCP JavaScript SDK
const result = await mcp.call_tool("execute_docker_container", {
  image: "python:3.9",
  command: "python -c 'print(2+2)'",
  memory_limit: "512m",
  timeout: 60
});
```

#### 2. `build_and_execute_github_repo`

Build and execute a Docker image from a GitHub repository.

**Parameters:**
- `repo_url` (required): GitHub repository URL
- `branch`: Git branch (default: "main")
- `dockerfile_path`: Path to Dockerfile (default: "Dockerfile")
- `command`: Command to run
- `entrypoint`: Custom entrypoint
- `environment`: Environment variables
- `build_args`: Docker build arguments
- `memory_limit`: Memory limit (default: "2g")
- `timeout`: Timeout in seconds (default: 600)
- `context_path`: Build context path (default: ".")

**Example:**
```javascript
const result = await mcp.call_tool("build_and_execute_github_repo", {
  repo_url: "https://github.com/user/python-app",
  branch: "main",
  command: "python app.py",
  environment: {"ENV": "production"},
  build_args: {"PYTHON_VERSION": "3.9"}
});
```

#### 3. `execute_with_payload`

Execute a container with a custom payload.

**Parameters:**
- `image` (required): Docker image name
- `payload` (required): Payload content
- `payload_path`: Path in container (default: "/tmp/payload")
- `entrypoint`: Command to execute
- `environment`: Environment variables
- `memory_limit`: Memory limit (default: "2g")
- `timeout`: Timeout in seconds (default: 300)

**Example:**
```javascript
const result = await mcp.call_tool("execute_with_payload", {
  image: "python:3.9",
  payload: "print('Hello from payload!')\nprint(2+2)",
  payload_path: "/app/script.py",
  entrypoint: "python /app/script.py"
});
```

#### 4. `list_running_containers`

List currently running Docker containers.

**Returns:**
```json
{
  "success": true,
  "containers": [
    {"ID": "abc123", "Names": "container1", "Status": "Up 5 minutes"},
    {"ID": "def456", "Names": "container2", "Status": "Up 10 minutes"}
  ],
  "count": 2
}
```

#### 5. `stop_container`

Stop a running container.

**Parameters:**
- `container_id` (required): Container ID or name
- `timeout`: Timeout before force kill (default: 10)

**Example:**
```javascript
const result = await mcp.call_tool("stop_container", {
  container_id: "my_container",
  timeout: 10
});
```

#### 6. `pull_docker_image`

Pull a Docker image from Docker Hub.

**Parameters:**
- `image` (required): Docker image name

**Example:**
```javascript
const result = await mcp.call_tool("pull_docker_image", {
  image: "python:3.9-slim"
});
```

## Usage Examples

### Example 1: Run Python Script

```python
from ipfs_accelerate_py.mcp.tools.docker_tools import execute_docker_container

result = execute_docker_container(
    image="python:3.9",
    command="python -c 'import sys; print(sys.version)'",
    memory_limit="512m",
    timeout=30
)

print(f"Exit code: {result['exit_code']}")
print(f"Output: {result['stdout']}")
```

### Example 2: Build and Run from GitHub

```python
from ipfs_accelerate_py.mcp.tools.docker_tools import build_and_execute_github_repo

result = build_and_execute_github_repo(
    repo_url="https://github.com/user/python-app",
    branch="main",
    dockerfile_path="Dockerfile",
    command="python main.py",
    environment={"DATABASE_URL": "sqlite:///data.db"},
    build_args={"PYTHON_VERSION": "3.9"}
)

if result['success']:
    print(f"Application output: {result['stdout']}")
else:
    print(f"Error: {result['error_message']}")
```

### Example 3: Execute Shell Script with Payload

```python
from ipfs_accelerate_py.mcp.tools.docker_tools import execute_with_payload

script = """#!/bin/bash
echo "Processing data..."
date
echo "Done!"
"""

result = execute_with_payload(
    image="ubuntu:20.04",
    payload=script,
    payload_path="/tmp/process.sh",
    entrypoint="bash /tmp/process.sh"
)

print(result['stdout'])
```

### Example 4: Container Management

```python
from ipfs_accelerate_py.mcp.tools.docker_tools import (
    list_running_containers,
    stop_container
)

# List containers
containers_result = list_running_containers()
for container in containers_result['containers']:
    print(f"Container: {container['Names']} - {container['Status']}")
    
    # Stop container if needed
    if 'old' in container['Names']:
        stop_result = stop_container(container_id=container['ID'])
        print(f"Stopped: {stop_result['success']}")
```

## Security Considerations

### Default Security Settings

- **Network Isolation**: Containers run with `network_mode="none"` by default
- **No New Privileges**: `security-opt=no-new-privileges` is enabled
- **Resource Limits**: Memory and CPU limits are enforced
- **Automatic Cleanup**: Containers are automatically removed after execution (`--rm`)
- **Timeout Protection**: All executions have configurable timeouts

### Best Practices

1. **Always set resource limits** to prevent resource exhaustion
2. **Use network isolation** unless external connectivity is required
3. **Validate input** before executing user-provided code
4. **Use read-only filesystems** when possible
5. **Run as non-root user** when the image supports it
6. **Set appropriate timeouts** to prevent long-running processes

### Example: Secure Configuration

```python
from ipfs_accelerate_py.docker_executor import DockerExecutionConfig

secure_config = DockerExecutionConfig(
    image="python:3.9",
    command=["python", "script.py"],
    memory_limit="512m",
    cpu_limit=1.0,
    timeout=60,
    network_mode="none",
    read_only=True,
    no_new_privileges=True,
    user="1000:1000"  # Non-root user
)
```

## Testing

### Run Core Module Tests

```bash
# All Docker executor tests
python -m unittest test.test_docker_executor

# Specific test class
python -m unittest test.test_docker_executor.TestDockerExecutor

# Specific test
python -m unittest test.test_docker_executor.TestDockerExecutor.test_execute_container_success
```

### Run MCP Tool Tests

```bash
# All MCP Docker tool tests
python -m unittest ipfs_accelerate_py.mcp.tests.test_docker_tools

# Specific test class
python -m unittest ipfs_accelerate_py.mcp.tests.test_docker_tools.TestDockerMCPTools
```

### Test Coverage

- **Core Module**: 17 tests covering all Docker executor functionality
- **MCP Tools**: 15 tests covering all MCP tool wrappers
- **Total**: 32 tests, 100% passing ✅

## Troubleshooting

### Docker Not Available

**Error**: `RuntimeError: Docker is not available`

**Solution**: Ensure Docker is installed and running:
```bash
docker --version
docker ps
```

### Permission Denied

**Error**: `permission denied while trying to connect to the Docker daemon`

**Solution**: Add user to docker group or run with sudo:
```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Image Not Found

**Error**: Container execution fails with "image not found"

**Solution**: Pull the image first:
```python
from ipfs_accelerate_py.mcp.tools.docker_tools import pull_docker_image

pull_docker_image(image="python:3.9")
```

### Timeout Errors

**Error**: `Container execution exceeded timeout`

**Solution**: Increase the timeout or optimize the container execution:
```python
result = execute_docker_container(
    image="python:3.9",
    command="python long_script.py",
    timeout=600  # 10 minutes
)
```

## Future Enhancements

- [ ] Support for Docker Compose
- [ ] Advanced networking options
- [ ] Volume persistence across executions
- [ ] Container streaming logs
- [ ] GPU support for ML workloads
- [ ] Kubernetes backend option
- [ ] Container caching and reuse
- [ ] Multi-platform image support

## See Also

- [MCP Server Documentation](../README.md)
- [IPFS Accelerate Documentation](../../README.md)
- [Docker Documentation](https://docs.docker.com/)
- [Container Security Best Practices](https://docs.docker.com/engine/security/)
