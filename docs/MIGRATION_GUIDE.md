# Migration Guide: Legacy to Unified Architecture

## Overview

This guide helps you migrate from legacy IPFS Accelerate tools to the new unified architecture with kit modules, unified CLI, and MCP tools.

## Why Migrate?

### Benefits of Unified Architecture

1. **Single Source of Truth**: Core logic in kit modules, no duplication
2. **Multiple Interfaces**: Same functionality via CLI, MCP, or Python API
3. **Better Testing**: Pure Python modules easy to unit test
4. **Consistency**: Same behavior across all interfaces
5. **Maintainability**: Changes in one place, propagates everywhere
6. **Type Safety**: Full type hints throughout
7. **Documentation**: Comprehensive docstrings

## Migration Paths

### 1. From Standalone Scripts to CLI

#### GitHub Operations

**Before:**
```bash
# Using gh CLI directly
gh repo list username --limit 10
gh pr list --repo owner/repo --state open
```

**After:**
```bash
# Using unified CLI
ipfs-accelerate github list-repos --owner username --limit 10
ipfs-accelerate github list-prs --repo owner/repo --state open
```

#### Docker Operations

**Before:**
```bash
# Using docker CLI directly
docker run python:3.9 python --version
docker ps -a
```

**After:**
```bash
# Using unified CLI
ipfs-accelerate docker run --image python:3.9 --command "python --version"
ipfs-accelerate docker list --all
```

#### Hardware Detection

**Before:**
```bash
# Using various system tools
lscpu
nvidia-smi
```

**After:**
```bash
# Using unified CLI
ipfs-accelerate hardware info --detailed
ipfs-accelerate hardware test --accelerator cuda
```

### 2. From Standalone Autoscaler to Runner Kit

#### GitHub Actions Runner Autoscaling

**Before:**
```bash
# Using standalone script
python scripts/utils/github_autoscaler.py --owner myorg --interval 60
```

**After (CLI):**
```bash
# Using unified CLI
ipfs-accelerate runner start --owner myorg --interval 60 --background
ipfs-accelerate runner status
ipfs-accelerate runner list-workflows
```

**After (Python API):**
```python
from ipfs_accelerate_py.kit.runner_kit import get_runner_kit, RunnerConfig

config = RunnerConfig(
    owner='myorg',
    poll_interval=60,
    max_runners=8
)

kit = get_runner_kit(config)
kit.start_autoscaler(background=True)

# Monitor status
status = kit.get_status()
print(f"Running: {status.running}")
print(f"Active runners: {status.active_runners}")
```

### 3. From Python Scripts to Kit Modules

#### Using GitHub Operations

**Before:**
```python
import subprocess
result = subprocess.run(['gh', 'repo', 'list', 'owner'], capture_output=True)
```

**After:**
```python
from ipfs_accelerate_py.kit.github_kit import GitHubKit

kit = GitHubKit()
result = kit.list_repos(owner='owner', limit=10)

if result.success:
    repos = result.data
else:
    print(f"Error: {result.error}")
```

#### Using Docker Operations

**Before:**
```python
import subprocess
result = subprocess.run(['docker', 'run', 'python:3.9', 'python', '--version'], 
                       capture_output=True)
```

**After:**
```python
from ipfs_accelerate_py.kit.docker_kit import DockerKit

kit = DockerKit()
result = kit.run_container(
    image='python:3.9',
    command='python --version',
    memory_limit='512m',
    timeout=30
)

if result.success:
    print(result.output)
```

#### Using Hardware Detection

**Before:**
```python
import platform
import psutil

system = platform.system()
cpu_count = psutil.cpu_count()
```

**After:**
```python
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit

kit = HardwareKit()
info = kit.get_hardware_info()

print(f"System: {info.platform_info['system']}")
print(f"CPUs: {info.cpu['count']}")
print(f"Memory: {info.memory['total_gb']:.2f} GB")
print(f"Accelerators: {list(info.accelerators.keys())}")
```

### 4. From Legacy MCP Tools to Unified Tools

#### Using MCP Tools

**Before:**
```python
from ipfs_accelerate_py.mcp.tools import some_legacy_tool

result = some_legacy_tool(param1, param2)
```

**After:**
```python
from ipfs_accelerate_py.mcp.unified_tools import (
    github_list_repos,
    docker_run_container,
    hardware_get_info
)

# All tools follow consistent patterns
repos = github_list_repos(owner='owner', limit=10)
container = docker_run_container(image='python:3.9', command='python --version')
hardware = hardware_get_info(include_detailed=True)
```

**Via MCP Server (JavaScript):**
```javascript
// Using MCP JavaScript SDK
const repos = await mcp.call_tool('github_list_repos', {
    owner: 'owner',
    limit: 10
});

const result = await mcp.call_tool('docker_run_container', {
    image: 'python:3.9',
    command: 'python --version'
});

const hardware = await mcp.call_tool('hardware_get_info', {
    include_detailed: true
});
```

## Step-by-Step Migration

### Step 1: Assess Current Usage

1. List all scripts/tools using legacy interfaces
2. Identify which kit modules they need
3. Check if equivalent functionality exists

### Step 2: Install/Update

```bash
# Ensure you have latest version
cd ipfs_accelerate_py
git pull
pip install -e .
```

### Step 3: Test New Interface

```bash
# Test CLI works
python ipfs_accelerate_py/unified_cli.py --help
python ipfs_accelerate_py/unified_cli.py hardware info

# Test in Python
python -c "from ipfs_accelerate_py.kit.hardware_kit import HardwareKit; print(HardwareKit().get_hardware_info())"
```

### Step 4: Migrate Scripts Incrementally

1. Start with simplest scripts
2. Migrate to kit modules
3. Test thoroughly
4. Update documentation
5. Keep legacy version temporarily

### Step 5: Update CI/CD

Update any CI/CD pipelines using old commands:

**Before:**
```yaml
- run: python scripts/utils/github_autoscaler.py --owner myorg
```

**After:**
```yaml
- run: python ipfs_accelerate_py/unified_cli.py runner start --owner myorg --background
```

### Step 6: Deprecate Old Code

1. Add deprecation warnings
2. Update documentation
3. Plan removal timeline
4. Communicate to users

## Common Patterns

### Pattern 1: Simple Command Replacement

Replace direct CLI calls with unified CLI:

```bash
# Old
gh repo list owner

# New
ipfs-accelerate github list-repos --owner owner
```

### Pattern 2: Python API Usage

Replace subprocess calls with kit modules:

```python
# Old
import subprocess
result = subprocess.run(['command'], capture_output=True)

# New
from ipfs_accelerate_py.kit.module_kit import ModuleKit
kit = ModuleKit()
result = kit.method()
```

### Pattern 3: MCP Tool Access

Use unified MCP tools:

```python
# Old - Direct tool function
from some_module import tool
result = tool(args)

# New - Unified tool
from ipfs_accelerate_py.mcp.unified_tools import tool_name
result = tool_name(args)
```

## Troubleshooting

### Issue: Module not found

**Solution:** Ensure you're importing from correct location:

```python
# Correct
from ipfs_accelerate_py.kit.github_kit import GitHubKit

# Not
from ipfs_kit_py.github_kit import GitHubKit  # Wrong package
```

### Issue: CLI command not found

**Solution:** Use full path or create alias:

```bash
# Full path
python /path/to/ipfs_accelerate_py/unified_cli.py hardware info

# Or create alias
alias ipfs-accelerate="python /path/to/ipfs_accelerate_py/unified_cli.py"
```

### Issue: Different behavior than before

**Solution:** Check new API documentation and adjust parameters. Kit modules may have enhanced validation or different defaults.

## Getting Help

1. **Documentation**: See `docs/UNIFIED_ARCHITECTURE.md`
2. **Examples**: Check `examples/` directory
3. **Tests**: Look at `test/test_*_kit.py` for usage patterns
4. **Issues**: Report on GitHub

## Timeline

**Current (v0.x):**
- ✅ Unified architecture available
- ✅ Both old and new interfaces work
- ⚠️ Migration recommended

**Future (v1.x):**
- Deprecation warnings added
- Old interfaces marked as deprecated

**Future (v2.x):**
- Old interfaces removed
- Unified architecture only

## Summary

The unified architecture provides:
- ✅ Better code organization
- ✅ Multiple access methods (CLI, MCP, Python)
- ✅ Improved testing
- ✅ Consistent interfaces
- ✅ Better documentation

Migration is straightforward and provides immediate benefits. Start migrating high-value scripts first, then gradually migrate remaining code.
