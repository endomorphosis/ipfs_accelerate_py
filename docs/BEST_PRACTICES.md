# Best Practices Guide

## Overview

This guide provides best practices for using the IPFS Accelerate unified architecture, including kit modules, CLI, and MCP tools.

## General Principles

### 1. Use Kit Modules for Core Logic

**DO:**
```python
from ipfs_accelerate_py.kit.github_kit import GitHubKit

kit = GitHubKit()
result = kit.list_repos(owner='username')
if result.success:
    process_repos(result.data)
```

**DON'T:**
```python
import subprocess
result = subprocess.run(['gh', 'repo', 'list', 'username'], capture_output=True)
# Harder to test, no type hints, error handling fragile
```

**Why:** Kit modules provide:
- Consistent error handling
- Type hints for IDE support
- Easy testing with mocks
- Centralized logic

### 2. Handle Errors Gracefully

**DO:**
```python
result = kit.list_repos(owner='username')
if not result.success:
    logger.error(f"Failed to list repos: {result.error}")
    return default_value
return result.data
```

**DON'T:**
```python
result = kit.list_repos(owner='username')
# Assuming success without checking
data = result.data  # Could be None!
```

**Why:** Operations can fail for many reasons (network, auth, rate limits). Always check success status.

### 3. Use Appropriate Output Formats

#### CLI Usage

**For Humans:**
```bash
ipfs-accelerate hardware info --format text
```

**For Scripts:**
```bash
ipfs-accelerate hardware info --format json | jq '.cpu.count'
```

**Why:** JSON format is machine-readable and parseable, text format is human-readable.

### 4. Leverage Type Hints

**DO:**
```python
from ipfs_accelerate_py.kit.docker_kit import DockerKit, DockerResult

def run_container(kit: DockerKit, image: str) -> DockerResult:
    return kit.run_container(image=image, command='echo test')
```

**DON'T:**
```python
def run_container(kit, image):  # No type hints
    return kit.run_container(image=image, command='echo test')
```

**Why:** Type hints enable:
- IDE autocomplete
- Static type checking
- Better documentation
- Catching errors early

## Module-Specific Best Practices

### GitHub Kit

#### Authentication

```python
# Ensure gh CLI is authenticated before using
import subprocess
result = subprocess.run(['gh', 'auth', 'status'], capture_output=True)
if result.returncode != 0:
    logger.error("GitHub CLI not authenticated. Run: gh auth login")
```

#### Rate Limiting

```python
from time import sleep

def list_all_repos_paginated(kit: GitHubKit, owner: str):
    """List repos with rate limit handling"""
    repos = []
    page = 1
    per_page = 30
    
    while True:
        result = kit.list_repos(owner=owner, limit=per_page)
        if not result.success:
            if "rate limit" in result.error.lower():
                logger.warning("Rate limited, waiting 60s...")
                sleep(60)
                continue
            break
        
        if not result.data:
            break
            
        repos.extend(result.data)
        page += 1
        sleep(1)  # Be nice to API
    
    return repos
```

### Docker Kit

#### Resource Limits

**DO:**
```python
result = kit.run_container(
    image='python:3.9',
    command='python script.py',
    memory_limit='512m',  # Prevent OOM
    cpus=1.0,  # Limit CPU usage
    timeout=300  # Prevent hanging
)
```

**DON'T:**
```python
result = kit.run_container(
    image='python:3.9',
    command='python script.py'
    # No limits - could consume all resources!
)
```

#### Cleanup

```python
def run_temporary_container(kit: DockerKit, image: str, command: str):
    """Run container and ensure cleanup"""
    container_id = None
    try:
        result = kit.run_container(
            image=image,
            command=command,
            detach=False  # Wait for completion
        )
        container_id = result.container_id
        return result
    finally:
        if container_id:
            kit.remove_container(container_id, force=True)
```

### Hardware Kit

#### Caching Results

```python
import functools
import time

@functools.lru_cache(maxsize=1)
def get_cached_hardware_info():
    """Cache hardware info for 5 minutes"""
    kit = HardwareKit()
    return kit.get_hardware_info(), time.time()

def get_hardware_info_cached(max_age=300):
    """Get hardware info with caching"""
    info, timestamp = get_cached_hardware_info()
    if time.time() - timestamp > max_age:
        get_cached_hardware_info.cache_clear()
        info, _ = get_cached_hardware_info()
    return info
```

#### Graceful Degradation

```python
def select_best_device(kit: HardwareKit):
    """Select best available device"""
    info = kit.get_hardware_info()
    
    # Try CUDA first
    if 'cuda' in info.accelerators and info.accelerators['cuda'].get('available'):
        return 'cuda'
    
    # Fall back to CPU
    return 'cpu'
```

### Runner Kit

#### Background Autoscaling

```python
from ipfs_accelerate_py.kit.runner_kit import get_runner_kit, RunnerConfig

def start_autoscaler_daemon():
    """Start autoscaler as background service"""
    config = RunnerConfig(
        owner='myorg',
        poll_interval=60,
        max_runners=10,
        runner_image='ghcr.io/actions/actions-runner:latest'
    )
    
    kit = get_runner_kit(config)
    
    # Start in background
    kit.start_autoscaler(background=True)
    
    # Monitor
    import time
    while True:
        status = kit.get_status()
        logger.info(f"Autoscaler status: {status.running}, "
                   f"Active: {status.active_runners}, "
                   f"Queued: {status.queued_workflows}")
        time.sleep(60)
```

#### Manual Provisioning

```python
def provision_for_urgent_workflow(kit, repo: str):
    """Manually provision runner for urgent workflow"""
    # Generate token
    token = kit.generate_runner_token(repo)
    
    # Launch with high priority
    container_id = kit.launch_runner_container(
        repo=repo,
        token=token,
        labels=['urgent', 'priority'],
        config={
            'cpu': 4,
            'memory': '8g'
        }
    )
    
    return container_id
```

## CLI Best Practices

### 1. Use Scripts for Automation

**DO:**
```bash
#!/bin/bash
set -e

# Hardware check
HW_INFO=$(ipfs-accelerate hardware info --format json)
HAS_GPU=$(echo $HW_INFO | jq -r '.accelerators.cuda.available // false')

if [ "$HAS_GPU" = "true" ]; then
    echo "GPU available, using GPU backend"
    # Start GPU inference
else
    echo "No GPU, using CPU backend"
    # Start CPU inference
fi
```

### 2. Error Handling in Scripts

```bash
#!/bin/bash

# Run command and check exit code
if ! ipfs-accelerate docker run --image python:3.9 --command "python --version"; then
    echo "ERROR: Docker command failed" >&2
    exit 1
fi

echo "SUCCESS"
```

### 3. Logging

```bash
# Enable verbose logging
ipfs-accelerate --verbose hardware info

# Redirect logs
ipfs-accelerate hardware info 2>error.log
```

## MCP Tools Best Practices

### 1. Validate Input Parameters

```python
def github_list_repos_safe(owner: str, limit: int = 30):
    """Safely list repos with validation"""
    if not owner:
        raise ValueError("Owner cannot be empty")
    
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")
    
    from ipfs_accelerate_py.mcp.unified_tools import github_list_repos
    return github_list_repos(owner=owner, limit=limit)
```

### 2. Handle Async Operations

```javascript
// In JavaScript/TypeScript MCP client
async function listReposWithRetry(owner, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const result = await mcp.call_tool('github_list_repos', {
                owner: owner,
                limit: 30
            });
            return result;
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            console.log(`Retry ${i + 1}/${maxRetries}...`);
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
    }
}
```

### 3. Batch Operations

```javascript
// Process multiple repos efficiently
async function getMultipleRepos(repos) {
    const promises = repos.map(repo =>
        mcp.call_tool('github_get_repo', { repo })
    );
    
    // Parallel execution
    const results = await Promise.all(promises);
    return results;
}
```

## Testing Best Practices

### 1. Unit Tests for Kit Modules

```python
import unittest
from unittest.mock import Mock, patch

class TestMyFeature(unittest.TestCase):
    @patch('subprocess.run')
    def test_github_list_repos(self, mock_run):
        """Test with mocked subprocess"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"repos": []}',
            stderr=''
        )
        
        kit = GitHubKit()
        result = kit.list_repos(owner='test')
        
        self.assertTrue(result.success)
        mock_run.assert_called_once()
```

### 2. Integration Tests

```python
def test_real_hardware_detection():
    """Integration test with real hardware"""
    kit = HardwareKit()
    info = kit.get_hardware_info()
    
    # Should always have platform info
    assert info.platform_info is not None
    assert 'system' in info.platform_info
```

### 3. CLI Tests

```bash
#!/bin/bash
# test_cli.sh

# Test help
if ! python unified_cli.py --help > /dev/null; then
    echo "FAIL: CLI help failed"
    exit 1
fi

# Test hardware info
if ! python unified_cli.py hardware info --format json | jq . > /dev/null; then
    echo "FAIL: Hardware info not valid JSON"
    exit 1
fi

echo "PASS: All CLI tests passed"
```

## Performance Best Practices

### 1. Minimize External Calls

**DO:**
```python
# Get hardware info once
info = kit.get_hardware_info()
cpu_count = info.cpu['count']
memory = info.memory['total_gb']
```

**DON'T:**
```python
# Multiple calls for same info
cpu_count = kit.get_hardware_info().cpu['count']
memory = kit.get_hardware_info().memory['total_gb']
```

### 2. Use Timeouts

```python
result = kit.run_container(
    image='python:3.9',
    command='long_running_task.py',
    timeout=300  # Fail after 5 minutes
)
```

### 3. Parallel Operations When Safe

```python
import concurrent.futures

def check_multiple_repos(kit, repos):
    """Check multiple repos in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(kit.get_repo, repo)
            for repo in repos
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results
```

## Security Best Practices

### 1. Protect Credentials

**DO:**
```python
import os

token = os.environ.get('GITHUB_TOKEN')
if not token:
    raise ValueError("GITHUB_TOKEN not set")
```

**DON'T:**
```python
token = "ghp_xxxxxxxxxxxx"  # Hardcoded token!
```

### 2. Validate Input

```python
def safe_run_container(kit, image: str, command: str):
    """Run container with input validation"""
    # Validate image name
    if not image or '..' in image:
        raise ValueError("Invalid image name")
    
    # Validate command
    if not command or ';' in command:
        raise ValueError("Invalid command")
    
    return kit.run_container(image=image, command=command)
```

### 3. Resource Limits

```python
# Always set limits for untrusted code
result = kit.run_container(
    image='untrusted:latest',
    command='user_code.py',
    memory_limit='256m',
    cpus=0.5,
    network_mode='none',  # No network access
    timeout=60
)
```

## Maintenance Best Practices

### 1. Keep Dependencies Updated

```bash
pip install --upgrade ipfs-accelerate-py
```

### 2. Monitor Logs

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 3. Regular Health Checks

```python
def health_check():
    """Regular health check for services"""
    checks = {
        'github': check_github_access(),
        'docker': check_docker_available(),
        'hardware': check_hardware_detected()
    }
    return all(checks.values())
```

## Summary

Key best practices:

1. ✅ Use kit modules for core logic
2. ✅ Handle errors gracefully
3. ✅ Use appropriate output formats
4. ✅ Leverage type hints
5. ✅ Set resource limits
6. ✅ Cache when appropriate
7. ✅ Validate inputs
8. ✅ Protect credentials
9. ✅ Write tests
10. ✅ Monitor and log

Following these practices will help you build robust, maintainable applications with IPFS Accelerate.
