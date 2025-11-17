# P2P Workflow Scheduler - Quick Start Guide

## What is it?

The P2P Workflow Scheduler allows GitHub Actions workflows to **bypass the GitHub API** and execute on a peer-to-peer IPFS network instead. This is useful for:

- 🔧 Code generation tasks
- 🌐 Web scraping workflows  
- 📊 Data processing tasks
- ⚡ Resource-intensive workflows

## Key Concepts

### 1. Merkle Clock
Determines which peer "owns" a task using distributed consensus.

### 2. Fibonacci Heap
Efficiently schedules tasks by priority (1-10).

### 3. Hamming Distance
Assigns tasks to peers based on hash similarity.

### 4. Workflow Tags
Control execution mode:
- `p2p-only` - Must use P2P (bypasses GitHub)
- `p2p-eligible` - Can use P2P or GitHub
- `code-generation` - Code generation task
- `web-scraping` - Web scraping task
- `unit-test` - Unit test (GitHub only)

## Installation

```bash
pip install ipfs_accelerate_py
```

## Quick Examples

### CLI: Check Status

```bash
ipfs-accelerate p2p-workflow status
```

### CLI: Submit Task

```bash
ipfs-accelerate p2p-workflow submit \
  --task-id task1 \
  --workflow-id workflow1 \
  --name "Generate Python Module" \
  --tags p2p-only code-generation \
  --priority 8
```

### CLI: Get Next Task

```bash
ipfs-accelerate p2p-workflow next
```

### CLI: Check Tags

```bash
# Should bypass GitHub?
ipfs-accelerate p2p-workflow check-tags --tags p2p-only code-generation

# Output:
# Should Bypass GitHub: Yes
# P2P Only: Yes
```

### Python: Basic Usage

```python
from ipfs_accelerate_py import P2PWorkflowScheduler, P2PTask, WorkflowTag
import time

# Create scheduler
scheduler = P2PWorkflowScheduler(peer_id="my-peer")

# Submit task
task = P2PTask(
    task_id="task1",
    workflow_id="workflow1",
    name="Code Generation",
    tags=[WorkflowTag.P2P_ONLY, WorkflowTag.CODE_GENERATION],
    priority=8,
    created_at=time.time()
)
scheduler.submit_task(task)

# Get next task
next_task = scheduler.get_next_task()
if next_task:
    print(f"Executing: {next_task.name}")
    # ... execute ...
    scheduler.mark_task_complete(next_task.task_id)
```

### GitHub Workflow Example

```yaml
name: P2P Code Generation

on:
  workflow_dispatch:
    inputs:
      description:
        required: true

env:
  WORKFLOW_TAGS: p2p-only,code-generation

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - name: Submit to P2P
        run: |
          ipfs-accelerate p2p-workflow submit \
            --task-id "${{ github.run_id }}" \
            --workflow-id "${{ github.workflow }}" \
            --name "${{ github.event.inputs.description }}" \
            --tags p2p-only code-generation \
            --priority 7
```

## Running a P2P Peer

```python
#!/usr/bin/env python3
from ipfs_accelerate_py import P2PWorkflowScheduler
import time

scheduler = P2PWorkflowScheduler(peer_id="worker-1")

while True:
    task = scheduler.get_next_task()
    if task:
        print(f"Executing: {task.name}")
        # ... execute task ...
        scheduler.mark_task_complete(task.task_id)
    else:
        time.sleep(10)
```

## All CLI Commands

```bash
# Status
ipfs-accelerate p2p-workflow status

# Submit task
ipfs-accelerate p2p-workflow submit \
  --task-id ID --workflow-id WID \
  --name NAME --tags TAG1 TAG2 --priority N

# Get next task
ipfs-accelerate p2p-workflow next

# Complete task
ipfs-accelerate p2p-workflow complete --task-id ID

# Check tags
ipfs-accelerate p2p-workflow check-tags --tags TAG1 TAG2

# Get clock state
ipfs-accelerate p2p-workflow clock [--json]
```

## Available Tags

| Tag | Use Case |
|-----|----------|
| `p2p-only` | Must use P2P network |
| `p2p-eligible` | Can use P2P or GitHub |
| `github-api` | Must use GitHub |
| `unit-test` | Unit tests (GitHub) |
| `code-generation` | Code generation |
| `web-scraping` | Web scraping |
| `data-processing` | Data processing |

## MCP Server Integration

Start MCP server with P2P tools:

```bash
ipfs-accelerate mcp start --port 9000
```

Available MCP tools:
- `p2p_scheduler_status()`
- `p2p_submit_task(...)`
- `p2p_get_next_task()`
- `p2p_mark_task_complete(task_id)`
- `p2p_check_workflow_tags(tags)`
- `p2p_get_merkle_clock()`
- `p2p_update_peer_state(peer_id, clock_data)`

## Testing

```bash
# Run all tests
pytest test_p2p_workflow_scheduler.py -v

# 27 tests covering:
# - Merkle clock operations
# - Fibonacci heap priority queue
# - Hamming distance calculation
# - Task submission and assignment
# - Workflow tag checking
# - Peer state management
```

## How It Works

1. **Workflow submits task** with P2P tags
2. **Merkle clock** + **task hash** determines owner
3. **Hamming distance** finds closest peer
4. **Fibonacci heap** orders by priority
5. **Peer executes** task outside GitHub
6. **Task marked complete** via merkle clock update

## Performance

| Operation | Complexity |
|-----------|------------|
| Submit Task | O(1) |
| Get Next Task | O(log n) |
| Peer Assignment | O(p) where p = peers |
| Priority Update | O(1) |

## Full Documentation

See [P2P_WORKFLOW_SCHEDULER.md](P2P_WORKFLOW_SCHEDULER.md) for:
- Complete API reference
- Architecture details
- Advanced examples
- Troubleshooting guide

## Support

- GitHub Issues: https://github.com/endomorphosis/ipfs_accelerate_py/issues
- Documentation: Repository README.md
