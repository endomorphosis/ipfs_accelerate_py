# IPFS Datasets Integration

Integration layer for `ipfs_datasets_py` distributed dataset manipulation services in `ipfs_accelerate_py`.

## Overview

This module provides a unified interface to `ipfs_datasets_py` services, enabling:

- **Distributed Dataset Management**: Store and retrieve datasets via IPFS with content-addressable storage
- **Decentralized Filesystem Operations**: UnixFS-based file operations with local-first design
- **Event & Provenance Logging**: Track data lineage and operation history immutably
- **P2P Workflow Coordination**: Distributed task scheduling for worker coordination
- **Model Manager Integration**: IPFS-based model storage with automatic deduplication
- **Graceful Fallbacks**: Works with or without ipfs_datasets_py (CI/CD friendly)

## Architecture

The integration consists of four main components:

### 1. DatasetsManager (`manager.py`)

High-level orchestrator for all ipfs_datasets_py services:

```python
from ipfs_accelerate_py.datasets_integration import DatasetsManager

manager = DatasetsManager({
    'cache_dir': '~/.cache/ipfs_accelerate',
    'enable_audit': True,
    'enable_provenance': True,
    'enable_p2p': False  # Disabled by default for safety
})

# Log events
manager.log_event("model_loaded", {"model": "bert-base", "size_mb": 420})

# Track provenance
cid = manager.track_provenance("inference", {
    "model": "bert-base",
    "input_cid": "Qm...",
    "output_cid": "Qm..."
})

# Check status
status = manager.get_status()
print(f"Enabled: {status['enabled']}")
```

### 2. FilesystemHandler (`filesystem.py`)

IPFS-based filesystem operations with local fallback:

```python
from ipfs_accelerate_py.datasets_integration import FilesystemHandler

fs = FilesystemHandler()

# Add file to IPFS
cid = fs.add_file("/path/to/model.bin", pin=True)
print(f"Model stored at: {cid}")

# Retrieve file from IPFS
fs.get_file(cid, "/path/to/output.bin")

# Add directory
dir_cid = fs.add_directory("/path/to/dataset/", recursive=True)

# List directory contents
entries = fs.list_directory(dir_cid)
for entry in entries:
    print(f"{entry['name']}: {entry['cid']}")
```

### 3. ProvenanceLogger (`provenance.py`)

Track data lineage and operation history:

```python
from ipfs_accelerate_py.datasets_integration import ProvenanceLogger

logger = ProvenanceLogger()

# Log model inference
logger.log_inference("bert-base", {
    "input": "Hello world",
    "output_embedding": [0.1, 0.2, ...],
    "duration_ms": 150,
    "hardware": "CUDA"
})

# Log data transformation
logger.log_transformation("tokenization", {
    "tokenizer": "bert-tokenizer",
    "max_length": 512
}, input_cid="Qm...", output_cid="Qm...")

# Log worker execution
logger.log_worker_execution("worker-001", "task-123", {
    "status": "completed",
    "duration_ms": 5000
})

# Log pull request activity
logger.log_pr_activity(123, "copilot_suggestion", {
    "file": "model.py",
    "suggestion": "Add error handling",
    "accepted": True
})

# Query logs
logs = logger.query_logs({"type": "inference", "model": "bert-base"})
```

### 4. WorkflowCoordinator (`workflow.py`)

P2P workflow scheduling and task distribution:

```python
from ipfs_accelerate_py.datasets_integration import WorkflowCoordinator

coordinator = WorkflowCoordinator({'enable_p2p': True})

# Submit task
coordinator.submit_task(
    task_id="infer-001",
    task_type="inference",
    data={"model": "bert", "batch_size": 32},
    priority=8,
    tags=["P2P_ELIGIBLE", "GPU_PREFERRED"]
)

# Get next task (for workers)
task = coordinator.get_next_task(
    worker_id="worker-001",
    capabilities=["GPU", "CUDA"]
)

# Complete task
coordinator.complete_task("infer-001", {
    "status": "success",
    "output_cid": "Qm...",
    "duration_ms": 5000
})

# Check status
status = coordinator.get_task_status("infer-001")
```

## Environment Variables

Control integration behavior via environment variables:

- **`IPFS_DATASETS_ENABLED`**: Enable/disable integration
  - `auto` (default): Auto-detect availability
  - `1`, `true`, `yes`, `on`: Force enable (warns if unavailable)
  - `0`, `false`, `no`, `off`: Force disable
  
- **`IPFS_DATASETS_PATH`**: Custom path to ipfs_datasets_py
  - Default: `external/ipfs_datasets_py` (submodule)

Examples:

```bash
# Disable for CI/CD
export IPFS_DATASETS_ENABLED=0
python run_tests.py

# Use custom path
export IPFS_DATASETS_PATH=/opt/ipfs_datasets_py
python run_app.py

# Force enable (will warn if unavailable)
export IPFS_DATASETS_ENABLED=1
python run_app.py
```

## Installation

### With Submodule (Recommended)

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/endomorphosis/ipfs_accelerate_py.git

# Or initialize submodule in existing clone
git submodule update --init external/ipfs_datasets_py

# Install with datasets extra
pip install -e ".[datasets]"
```

### Without Submodule (Fallback Mode)

```bash
# Install without ipfs_datasets_py
pip install ipfs-accelerate-py

# Integration will use local fallbacks
# All features work, but without IPFS distribution
```

## Usage Patterns

### Pattern 1: Check Before Use

```python
from ipfs_accelerate_py.datasets_integration import is_datasets_available

if is_datasets_available():
    from ipfs_accelerate_py.datasets_integration import DatasetsManager
    manager = DatasetsManager()
    manager.log_event("app_started", {})
else:
    print("Using local mode")
```

### Pattern 2: Graceful Degradation

```python
from ipfs_accelerate_py.datasets_integration import DatasetsManager

# Always create manager - it handles availability internally
manager = DatasetsManager()

# Methods return False/None if unavailable
if manager.log_event("inference", {"model": "bert"}):
    print("Logged to IPFS")
else:
    print("Logged locally")
```

### Pattern 3: Status Checking

```python
from ipfs_accelerate_py.datasets_integration import get_datasets_status

status = get_datasets_status()
print(f"Available: {status['available']}")
print(f"Path: {status['path']}")
print(f"Reason: {status.get('reason', 'N/A')}")
```

## Use Cases

### 1. Model Manager with IPFS Storage

```python
from ipfs_accelerate_py.datasets_integration import DatasetsManager, FilesystemHandler

manager = DatasetsManager()
fs = FilesystemHandler()

# Save model to IPFS
model_cid = fs.add_file("/models/bert-base.bin", pin=True)

# Track provenance
manager.track_provenance("model_upload", {
    "model": "bert-base",
    "version": "1.0",
    "cid": model_cid,
    "size_mb": 420
})

# Log event
manager.log_event("model_stored", {
    "model": "bert-base",
    "cid": model_cid
})
```

### 2. Distributed Inference Logging

```python
from ipfs_accelerate_py.datasets_integration import ProvenanceLogger

logger = ProvenanceLogger()

# Log inference operation
logger.log_inference("bert-base", {
    "input_text": "What is IPFS?",
    "input_tokens": 5,
    "output_embedding": [0.1, 0.2, ...],
    "output_dim": 768,
    "duration_ms": 150,
    "hardware": "CUDA",
    "batch_size": 1
})

# Query inference history
history = logger.query_logs({
    "type": "inference",
    "model": "bert-base"
}, limit=100)
```

### 3. Worker Coordination

```python
from ipfs_accelerate_py.datasets_integration import WorkflowCoordinator, ProvenanceLogger

coordinator = WorkflowCoordinator({'enable_p2p': True})
logger = ProvenanceLogger()

# Coordinator submits work
coordinator.submit_task("batch-001", "batch_inference", {
    "model": "bert-base",
    "inputs": ["text1", "text2", ...],
    "batch_size": 32
}, priority=7)

# Worker picks up task
task = coordinator.get_next_task("worker-001", ["GPU"])
if task:
    # Execute task
    result = execute_inference(task['data'])
    
    # Log execution
    logger.log_worker_execution("worker-001", task['task_id'], {
        "status": "completed",
        "outputs": result,
        "duration_ms": 5000
    })
    
    # Mark complete
    coordinator.complete_task(task['task_id'], result)
```

### 4. GitHub Copilot Activity Tracking

```python
from ipfs_accelerate_py.datasets_integration import ProvenanceLogger

logger = ProvenanceLogger()

# Track Copilot suggestion
logger.log_pr_activity(123, "copilot_suggestion", {
    "file": "model.py",
    "line": 42,
    "suggestion": "Add type hints",
    "accepted": True,
    "timestamp": "2024-01-28T10:30:00Z"
})

# Track PR merge
logger.log_pr_activity(123, "merge", {
    "commit": "abc123",
    "files_changed": 5,
    "lines_added": 100,
    "lines_removed": 20
})

# Query PR activity
activity = logger.query_logs({
    "type": "pr_activity",
    "pr_number": 123
})
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests with IPFS Datasets

on: [push, pull_request]

jobs:
  test-with-ipfs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install with datasets
        run: pip install -e ".[datasets,testing]"
      
      - name: Run tests
        run: pytest tests/

  test-without-ipfs:
    runs-on: ubuntu-latest
    env:
      IPFS_DATASETS_ENABLED: 0
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install without datasets
        run: pip install -e ".[testing]"
      
      - name: Run tests (fallback mode)
        run: pytest tests/
```

## Testing

The integration includes comprehensive tests:

```bash
# Test with ipfs_datasets_py enabled
pytest tests/test_datasets_integration.py

# Test with ipfs_datasets_py disabled
IPFS_DATASETS_ENABLED=0 pytest tests/test_datasets_integration.py

# Test fallback behavior
pytest tests/test_datasets_fallback.py
```

## Troubleshooting

### Issue: "ipfs_datasets_py not found"

**Solution**: Initialize the submodule:
```bash
git submodule update --init external/ipfs_datasets_py
```

### Issue: Import errors from ipfs_datasets_py

**Solution**: The integration handles this gracefully. Check status:
```python
from ipfs_accelerate_py.datasets_integration import get_datasets_status
print(get_datasets_status())
```

### Issue: P2P features not working

**Solution**: P2P is disabled by default. Enable explicitly:
```python
coordinator = WorkflowCoordinator({'enable_p2p': True})
```

### Issue: IPFS daemon not running

**Solution**: The integration falls back to local storage automatically. To use IPFS features:
```bash
# Start IPFS daemon
ipfs daemon &

# Or use specific API endpoint
fs = FilesystemHandler(ipfs_api='/ip4/127.0.0.1/tcp/5001')
```

## Performance Considerations

1. **Lazy Loading**: Components are only initialized when used
2. **Graceful Fallback**: No performance penalty when IPFS unavailable
3. **Local Caching**: All operations cached locally by default
4. **Async Support**: All I/O operations support async (when using underlying async APIs)

## Security

- **Content Addressing**: All IPFS content is cryptographically verified
- **Immutable Logs**: Provenance records are immutable once created
- **No Credentials**: IPFS uses public/private key cryptography (no passwords)
- **Local First**: All data cached locally before IPFS distribution

## Contributing

See main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for general guidelines.

For datasets integration:
1. Maintain backward compatibility
2. Always provide fallbacks
3. Document environment variables
4. Add tests for both enabled/disabled modes

## License

Same as ipfs_accelerate_py (AGPLv3+)
