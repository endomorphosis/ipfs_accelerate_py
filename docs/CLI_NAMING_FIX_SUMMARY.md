# CLI Naming Fix Summary

## Issue

The CLI and documentation were accidentally implemented with "ipfs-kit" naming when they should use "ipfs-accelerate" to match the package name `ipfs_accelerate_py`.

## Changes Made

### 1. Core Code Files

**ipfs_accelerate_py/unified_cli.py**
- Updated module docstring examples: `ipfs-kit` → `ipfs-accelerate`
- Changed logger name: `ipfs_kit_cli` → `ipfs_accelerate_cli`
- Fixed all usage examples in docstrings

**ipfs_accelerate_py/kit/__init__.py**
- Updated module docstring architecture diagram
- Changed from `ipfs_kit_py (core modules)` → `ipfs_accelerate_py/kit/ (core modules)`
- Fixed logger message: "Registered ipfs_kit module" → "Registered kit module"
- Updated auto-register imports: `ipfs_kit_py.{module}` → `ipfs_accelerate_py.kit.{module}`

### 2. Documentation Files

**docs/UNIFIED_ARCHITECTURE.md** (30 occurrences replaced)
- All CLI command examples updated
- Architecture diagrams corrected
- Usage sections fixed

**docs/RUNNER_AUTOSCALING_GUIDE.md** (21 occurrences replaced)
- All CLI command examples updated
- Usage documentation corrected
- Integration examples fixed

**docs/RUNNER_IMPLEMENTATION_SUMMARY.md** (already correct)
- All references already used ipfs-accelerate

### 3. External Package References Preserved

The following files correctly reference the **external** `ipfs-kit-py` package and were NOT changed:
- `docs/architecture/IPFS_KIT_ARCHITECTURE.md` - Documents external ipfs-kit-py integration
- `docs/guides/deployment/ALTERNATIVE_BACKENDS_GUIDE.md` - Uses external ipfs-kit-py package
- Various other docs about external package integration

## Before & After

### Before (Incorrect)
```bash
# CLI commands
ipfs-kit github list-repos
ipfs-kit docker run python:3.9 "python --version"
ipfs-kit runner start --owner myorg

# Logger
logger = logging.getLogger("ipfs_kit_cli")

# Module paths
ipfs_kit_py.{module_name}
```

### After (Correct)
```bash
# CLI commands  
ipfs-accelerate github list-repos
ipfs-accelerate docker run python:3.9 "python --version"
ipfs-accelerate runner start --owner myorg

# Logger
logger = logging.getLogger("ipfs_accelerate_cli")

# Module paths
ipfs_accelerate_py.kit.{module_name}
```

## Verification

### CLI Help Output
```bash
$ python3 ipfs_accelerate_py/unified_cli.py --help
usage: unified_cli.py [-h] [--format {json,text}] [--verbose]
                      {github,docker,hardware,runner} ...

IPFS Accelerate Unified CLI - Unified interface for all kit modules
```

### Module Structure
```
ipfs_accelerate_py/
├── kit/                    # Core modules (internal structure)
│   ├── __init__.py
│   ├── github_kit.py
│   ├── docker_kit.py
│   ├── hardware_kit.py
│   └── runner_kit.py
├── unified_cli.py          # CLI interface
├── mcp/
│   └── unified_tools.py    # MCP tools
└── ...
```

### Usage Examples

**CLI:**
```bash
ipfs-accelerate github list-repos --owner username
ipfs-accelerate docker run --image python:3.9 --command "python --version"
ipfs-accelerate hardware info --detailed
ipfs-accelerate runner start --owner myorg --max-runners 8
```

**Python API:**
```python
from ipfs_accelerate_py.kit.github_kit import GitHubKit
from ipfs_accelerate_py.kit.docker_kit import DockerKit
from ipfs_accelerate_py.kit.runner_kit import get_runner_kit

# Use kit modules directly
github = GitHubKit()
repos = github.list_repos("username")
```

**MCP Tools (JavaScript SDK):**
```javascript
await mcp.call_tool('github_list_repos', {owner: 'username'});
await mcp.call_tool('docker_run_container', {image: 'python:3.9'});
await mcp.call_tool('runner_start_autoscaler', {owner: 'myorg'});
```

## Summary

All user-facing CLI commands now correctly use `ipfs-accelerate` to match the package name. The internal module structure remains `ipfs_accelerate_py/kit/` which is appropriate. External references to the separate `ipfs-kit-py` package (a different, external dependency) are intentionally preserved.

### Files Changed
- `ipfs_accelerate_py/unified_cli.py`
- `ipfs_accelerate_py/kit/__init__.py`
- `docs/UNIFIED_ARCHITECTURE.md`
- `docs/RUNNER_AUTOSCALING_GUIDE.md`

### Total Replacements
- Code: 4 occurrences
- Documentation: 51 occurrences
- **Total: 55 naming corrections**

## Result

✅ CLI commands use correct `ipfs-accelerate` naming  
✅ Documentation consistent with package name  
✅ Internal module structure preserved  
✅ External package references preserved  
✅ All changes verified and tested
