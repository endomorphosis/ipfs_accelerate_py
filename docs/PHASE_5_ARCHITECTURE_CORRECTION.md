# Phase 5 Architecture Correction

## Overview

This document clarifies the correct architecture for Phase 5 implementation based on the requirement that **all MCP tools should wrap ipfs_accelerate_py kit modules** (not the other way around), and that **ipfs_files_kit and network_kit should wrap the external ipfs_kit_py package**.

---

## The Correct Architecture

### Core Principle

```
External Package (ipfs_kit_py from git)
    ↓ (wrapped by)
Kit Modules (ipfs_accelerate_py/kit/)
    ↓ (wrapped by)
├─ CLI Tools (unified_cli.py)
└─ MCP Tools (mcp/unified_tools.py)
```

### Key Points

1. **Kit modules are the source of truth** - Core logic lives here
2. **Kit modules wrap external packages** - ipfs_kit_py is external dependency
3. **CLI wraps kit modules** - Not external packages directly
4. **MCP tools wrap kit modules** - Not external packages directly
5. **NOT the other way around** - MCP tools do NOT define core logic

---

## Implementation Status

### ✅ Completed

#### ipfs_files_kit.py

**Location:** `ipfs_accelerate_py/kit/ipfs_files_kit.py`

**What it does:**
- Wraps external `ipfs_kit_py` package for IPFS file operations
- Provides clean Python API for IPFS operations
- Includes graceful fallback to IPFS CLI
- Core logic for file operations

**Classes:**
- `IPFSFilesConfig` - Configuration
- `IPFSFileInfo` - File metadata
- `IPFSFileResult` - Operation results
- `IPFSFilesKit` - Main operations class

**Methods:**
- `add_file()` - Add file to IPFS
- `get_file()` - Get file by CID
- `cat_file()` - Read file content
- `pin_file()` / `unpin_file()` - Pin management
- `list_files()` - List files
- `validate_cid()` - CID validation

#### network_kit.py

**Location:** `ipfs_accelerate_py/kit/network_kit.py`

**What it does:**
- Wraps external `ipfs_kit_py` package for network operations
- Provides clean Python API for peer/network operations
- Includes graceful fallback to IPFS CLI
- Core logic for network operations

**Classes:**
- `NetworkConfig` - Configuration
- `PeerInfo` - Peer metadata
- `BandwidthStats` - Bandwidth statistics
- `NetworkResult` - Operation results
- `NetworkKit` - Main operations class

**Methods:**
- `list_peers()` - List connected peers
- `connect_peer()` / `disconnect_peer()` - Peer management
- `dht_put()` / `dht_get()` - DHT operations
- `get_swarm_info()` - Swarm statistics
- `get_bandwidth_stats()` - Bandwidth monitoring
- `ping_peer()` - Connectivity testing

### ⏳ Remaining Work

The following components need to be implemented to complete Phase 5:

#### 1. CLI Integration

**File to update:** `ipfs_accelerate_py/unified_cli.py`

**What to add:**

```python
# Add ipfs-files subcommand
parser_ipfs_files = subparsers.add_parser('ipfs-files', help='IPFS file operations')
ipfs_files_subparsers = parser_ipfs_files.add_subparsers(dest='ipfs_files_command')

# Commands: add, get, cat, pin, unpin, list, validate-cid
# Each command should import and call ipfs_files_kit methods

# Add network subcommand
parser_network = subparsers.add_parser('network', help='Network and peer operations')
network_subparsers = parser_network.add_subparsers(dest='network_command')

# Commands: list-peers, connect, disconnect, dht-put, dht-get, swarm-info, bandwidth, ping
# Each command should import and call network_kit methods
```

**Pattern to follow:** Look at how `github`, `docker`, `hardware`, and `runner` modules are integrated.

#### 2. MCP Tools Integration

**File to update:** `ipfs_accelerate_py/mcp/unified_tools.py`

**What to add:**

```python
# IPFS Files Tools (7 tools)
def ipfs_files_add(path: str, pin: bool = True) -> dict:
    """Add file to IPFS."""
    kit = get_ipfs_files_kit()
    result = kit.add_file(path, pin)
    return result.to_dict()

# Similar tools for: get, cat, pin, unpin, list, validate-cid

# Network Tools (8 tools)
def network_list_peers() -> dict:
    """List connected peers."""
    kit = get_network_kit()
    result = kit.list_peers()
    return result.to_dict()

# Similar tools for: connect, disconnect, dht-put, dht-get, swarm-info, bandwidth, ping
```

**Pattern to follow:** Look at how `github_*`, `docker_*`, `hardware_*`, and `runner_*` tools are implemented.

#### 3. Unit Tests

**Files to create:**
- `test/test_ipfs_files_kit.py` (~180 lines)
- `test/test_network_kit.py` (~170 lines)

**What to test:**
- Kit initialization
- Each method (success and failure scenarios)
- Dataclass validation
- Graceful fallback behavior
- Error handling

**Pattern to follow:** Look at `test/test_github_kit.py` and `test/test_hardware_kit.py`.

#### 4. Documentation Updates

**Files to update:**
- `docs/PHASES_5-7_COMPLETION_SUMMARY.md` - Update Phase 5 section with correct architecture
- `docs/UNIFIED_ARCHITECTURE.md` - Add ipfs_files and network modules to module list

**What to document:**
- Correct architecture (kit wraps ipfs_kit_py)
- Usage examples for each module
- CLI command examples
- MCP tool examples
- Integration with external ipfs_kit_py

---

## Why This Architecture Matters

### Wrong Approach (What NOT to do)

```
❌ MCP tools define core logic
❌ Kit modules wrap MCP tools
❌ Direct use of external packages in CLI/MCP
```

**Problems:**
- Duplication between CLI and MCP
- MCP tools become source of truth (wrong)
- Hard to test MCP-specific code
- Can't use functionality without MCP

### Correct Approach (What we're doing)

```
✅ Kit modules define core logic
✅ MCP tools wrap kit modules
✅ CLI wraps kit modules
✅ Kit modules wrap external packages
```

**Benefits:**
- Single source of truth (kit modules)
- No duplication (CLI and MCP both use kit)
- Easy to test (pure Python modules)
- Can use directly in Python code
- Consistent behavior across interfaces

---

## Usage Examples

### Direct Python API

```python
# Using ipfs_files_kit directly
from ipfs_accelerate_py.kit.ipfs_files_kit import get_ipfs_files_kit

kit = get_ipfs_files_kit()
result = kit.add_file("/path/to/file.txt")
if result.success:
    print(f"CID: {result.data['cid']}")

# Using network_kit directly
from ipfs_accelerate_py.kit.network_kit import get_network_kit

kit = get_network_kit()
result = kit.list_peers()
print(f"Connected peers: {result.data['count']}")
```

### CLI (Once integrated)

```bash
# IPFS files operations
ipfs-accelerate ipfs-files add --path file.txt
ipfs-accelerate ipfs-files get --cid Qm... --output output.txt
ipfs-accelerate ipfs-files list

# Network operations
ipfs-accelerate network list-peers
ipfs-accelerate network connect --peer /ip4/1.2.3.4/tcp/4001/p2p/QmPeer
ipfs-accelerate network bandwidth
```

### MCP Tools (Once integrated)

```javascript
// Via JavaScript SDK
const result = await mcp.call_tool('ipfs_files_add', {
  path: '/path/to/file.txt',
  pin: true
});

const peers = await mcp.call_tool('network_list_peers', {});
```

---

## Integration Checklist

To complete Phase 5, the following must be done:

- [x] Create ipfs_files_kit.py core module
- [x] Create network_kit.py core module
- [ ] Add ipfs-files CLI commands to unified_cli.py
- [ ] Add network CLI commands to unified_cli.py
- [ ] Add ipfs_files_* MCP tools to unified_tools.py
- [ ] Add network_* MCP tools to unified_tools.py
- [ ] Create test/test_ipfs_files_kit.py
- [ ] Create test/test_network_kit.py
- [ ] Update docs/PHASES_5-7_COMPLETION_SUMMARY.md
- [ ] Update docs/UNIFIED_ARCHITECTURE.md
- [ ] Run all tests to verify
- [ ] Test CLI commands manually
- [ ] Verify MCP tools registration

---

## Summary

**Completed:**
- ✅ Core kit modules created (ipfs_files_kit.py, network_kit.py)
- ✅ Correct architecture implemented (wrapping external ipfs_kit_py)
- ✅ Clean API with type hints and dataclasses
- ✅ Graceful fallback when ipfs_kit_py unavailable

**Remaining:**
- ⏳ CLI integration (15 commands)
- ⏳ MCP tools integration (15 tools)
- ⏳ Unit tests (2 test files, ~350 lines)
- ⏳ Documentation updates (2 files)

**Result:** When complete, Phase 5 will provide consistent IPFS and network operations across Python API, CLI, and MCP interfaces, all based on the same core kit modules that properly wrap the external ipfs_kit_py package.
