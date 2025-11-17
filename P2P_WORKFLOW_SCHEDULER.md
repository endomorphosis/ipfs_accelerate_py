# P2P Workflow Scheduler

## Overview

The P2P Workflow Scheduler enables GitHub Actions workflows to bypass the GitHub API by executing tasks across a peer-to-peer IPFS network. This is particularly useful for:

- **Code generation tasks** that don't need GitHub's infrastructure
- **Web scraping workflows** that can run on any peer
- **Data processing tasks** that benefit from distributed execution
- **Resource-intensive workflows** that would otherwise consume GitHub Actions minutes

## Key Features

### 1. Merkle Clock for Distributed Consensus

The scheduler uses a **Merkle Clock** (vector clock + merkle tree) to establish canonical ordering of events across the P2P network. This ensures:

- Deterministic task assignment
- Causal ordering of workflow events
- Conflict-free state synchronization between peers

### 2. Fibonacci Heap for Priority Scheduling

Tasks are managed using a **Fibonacci Heap** data structure, providing:

- O(1) insertion time
- O(log n) extract-minimum (amortized)
- Efficient priority-based task scheduling
- Support for dynamic priority updates

### 3. Hamming Distance-Based Peer Selection

Task assignment uses **Hamming Distance** between:
- Hash of (merkle clock head + task hash)
- Hash of peer IDs

This creates deterministic, load-balanced task distribution where:
- Each task has a canonical "owner" peer
- Failed peers are automatically detected
- Tasks are reassigned to the next closest peer

### 4. Workflow Tagging System

Workflows can be tagged to control their execution mode:

| Tag | Description |
|-----|-------------|
| `github-api` | Standard GitHub Actions workflow |
| `p2p-eligible` | Can run on P2P network or GitHub |
| `p2p-only` | Must run on P2P network (bypasses GitHub) |
| `unit-test` | Unit test workflows (typically GitHub-only) |
| `code-generation` | Code generation tasks (P2P-eligible) |
| `web-scraping` | Web scraping tasks (P2P-eligible) |
| `data-processing` | Data processing tasks (P2P-eligible) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    P2P Workflow Scheduler                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Merkle Clock │  │Fibonacci Heap│  │   Peer Pool  │      │
│  │              │  │              │  │              │      │
│  │ • Vector     │  │ • Priority   │  │ • Peer IDs   │      │
│  │   Clock      │  │   Queue      │  │ • Health     │      │
│  │ • Merkle     │  │ • O(1)       │  │   Status     │      │
│  │   Root       │  │   Insert     │  │ • Clocks     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Task Assignment (Hamming Distance)           │   │
│  │                                                       │   │
│  │  hash(merkle_clock + task) ⊕ hash(peer_id)          │   │
│  │  → Closest peer handles task                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

The P2P Workflow Scheduler is included in the `ipfs_accelerate_py` package:

```bash
pip install ipfs_accelerate_py
```

## Usage

### CLI Commands

#### 1. Check Scheduler Status

```bash
ipfs-accelerate p2p-workflow status
```

Output:
```
✓ P2P Workflow Scheduler Status:
  Peer ID: peer-hostname-abc123
  Pending Tasks: 5
  Assigned Tasks: 2
  Completed Tasks: 10
  Queue Size: 5
  Known Peers: 3
  Merkle Clock Hash: 343dbca40f4c433b04a5f818cdfdaa1ffb8d3b2c383017db54ecba3ea8972d9e
```

#### 2. Submit a Task

```bash
ipfs-accelerate p2p-workflow submit \
  --task-id task1 \
  --workflow-id workflow1 \
  --name "Code Generation Task" \
  --tags p2p-only code-generation \
  --priority 8
```

Output:
```
✓ Task submitted successfully
  Task ID: task1
  Task Hash: 702173a32abe2e88c2d1334db2ec96835d4e331b7fab6cb85bd9655a1fcee7b6
  Priority: 8
```

#### 3. Get Next Task to Execute

```bash
ipfs-accelerate p2p-workflow next
```

Output:
```
✓ Next task:
  Task ID: task1
  Workflow ID: workflow1
  Name: Code Generation Task
  Tags: p2p-only, code-generation
  Priority: 8
  Assigned Peer: peer-hostname-abc123
```

#### 4. Mark Task as Complete

```bash
ipfs-accelerate p2p-workflow complete --task-id task1
```

Output:
```
✓ Task task1 marked complete
```

#### 5. Check Workflow Tags

```bash
# Check if workflow should bypass GitHub
ipfs-accelerate p2p-workflow check-tags --tags p2p-only code-generation
```

Output:
```
✓ Tag Check Results:
  Tags: p2p-only, code-generation
  Should Bypass GitHub: Yes
  P2P Only: Yes
```

```bash
# Check unit test tags (should use GitHub)
ipfs-accelerate p2p-workflow check-tags --tags unit-test github-api
```

Output:
```
✓ Tag Check Results:
  Tags: unit-test, github-api
  Should Bypass GitHub: No
  P2P Only: No
```

#### 6. Get Merkle Clock State

```bash
ipfs-accelerate p2p-workflow clock --json
```

Output:
```json
{
  "node_id": "peer-hostname-abc123",
  "vector": {
    "peer-hostname-abc123": 5,
    "peer-other-def456": 3
  },
  "merkle_root": "011185da0ecfe5770c445546bdd7250eaabe79e4e8a511d41a992159ca2b9ed5"
}
```

### Python Package API

#### Basic Usage

```python
import time
from ipfs_accelerate_py import (
    P2PWorkflowScheduler,
    P2PTask,
    WorkflowTag
)

# Create scheduler
scheduler = P2PWorkflowScheduler(peer_id="my-peer")

# Submit a task
task = P2PTask(
    task_id="task1",
    workflow_id="workflow1",
    name="Code Generation",
    tags=[WorkflowTag.P2P_ONLY, WorkflowTag.CODE_GENERATION],
    priority=8,
    created_at=time.time()
)
scheduler.submit_task(task)

# Get next task to execute
next_task = scheduler.get_next_task()
if next_task:
    print(f"Executing: {next_task.name}")
    # ... execute task ...
    scheduler.mark_task_complete(next_task.task_id)
```

#### Check Workflow Tags

```python
from ipfs_accelerate_py import P2PWorkflowScheduler, WorkflowTag

scheduler = P2PWorkflowScheduler(peer_id="my-peer")

# Check if workflow should bypass GitHub
tags = [WorkflowTag.P2P_ONLY, WorkflowTag.CODE_GENERATION]
should_bypass = scheduler.should_bypass_github(tags)
is_p2p_only = scheduler.is_p2p_only(tags)

print(f"Should bypass GitHub: {should_bypass}")  # True
print(f"P2P only: {is_p2p_only}")  # True
```

#### Working with Merkle Clock

```python
from ipfs_accelerate_py import MerkleClock

# Create clock
clock = MerkleClock(node_id="my-peer")

# Increment clock on local events
clock.tick()

# Update from another peer's clock
other_clock = MerkleClock.from_dict(received_clock_data)
clock.update(other_clock)

# Get merkle root hash
merkle_hash = clock.get_hash()
print(f"Clock hash: {merkle_hash}")

# Serialize for transmission
clock_data = clock.to_dict()
```

#### Using Fibonacci Heap Directly

```python
from ipfs_accelerate_py import FibonacciHeap

# Create heap
heap = FibonacciHeap()

# Insert tasks with priorities
heap.insert(5, {"name": "task1"})
heap.insert(3, {"name": "task2"})  # Higher priority (lower number)
heap.insert(7, {"name": "task3"})

# Extract in priority order
while not heap.is_empty():
    priority, task = heap.extract_min()
    print(f"Priority {priority}: {task['name']}")

# Output:
# Priority 3: task2
# Priority 5: task1
# Priority 7: task3
```

#### Calculate Hamming Distance

```python
from ipfs_accelerate_py import calculate_hamming_distance

hash1 = "a1b2c3d4"
hash2 = "a1b2c3d5"

distance = calculate_hamming_distance(hash1, hash2)
print(f"Hamming distance: {distance}")  # 1 bit different
```

### MCP Server Tools

The P2P Workflow Scheduler is also exposed as MCP Server tools:

1. `p2p_scheduler_status()` - Get scheduler status
2. `p2p_submit_task()` - Submit a task
3. `p2p_get_next_task()` - Get next task to execute
4. `p2p_mark_task_complete()` - Mark task as complete
5. `p2p_check_workflow_tags()` - Check if tags should bypass GitHub
6. `p2p_update_peer_state()` - Update peer state information
7. `p2p_get_merkle_clock()` - Get merkle clock state

These tools are automatically registered when starting the MCP server:

```bash
ipfs-accelerate mcp start --port 9000
```

## GitHub Workflow Integration

### Example: P2P Code Generation Workflow

Create `.github/workflows/p2p-code-gen.yml`:

```yaml
name: P2P Code Generation

on:
  workflow_dispatch:
    inputs:
      task_description:
        description: 'Code to generate'
        required: true

# Tag this workflow as P2P-only to bypass GitHub API
# P2P peers will pick up and execute this workflow
env:
  WORKFLOW_TAGS: p2p-only,code-generation

jobs:
  generate:
    # This won't actually run on GitHub
    # P2P peers will detect the tags and handle execution
    runs-on: ubuntu-latest
    
    steps:
      - name: Submit to P2P Scheduler
        run: |
          ipfs-accelerate p2p-workflow submit \
            --task-id "${{ github.run_id }}" \
            --workflow-id "${{ github.workflow }}" \
            --name "Code Generation: ${{ github.event.inputs.task_description }}" \
            --tags p2p-only code-generation \
            --priority 7
```

### Example: P2P Web Scraping Workflow

```yaml
name: P2P Web Scraping

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

env:
  WORKFLOW_TAGS: p2p-eligible,web-scraping

jobs:
  scrape:
    runs-on: ubuntu-latest
    
    steps:
      - name: Check if P2P should handle
        id: check
        run: |
          RESULT=$(ipfs-accelerate p2p-workflow check-tags \
            --tags p2p-eligible web-scraping --json)
          echo "should_bypass=$(echo $RESULT | jq -r '.should_bypass_github')" >> $GITHUB_OUTPUT
      
      - name: Submit to P2P or run locally
        run: |
          if [ "${{ steps.check.outputs.should_bypass }}" = "true" ]; then
            ipfs-accelerate p2p-workflow submit \
              --task-id "${{ github.run_id }}" \
              --workflow-id "${{ github.workflow }}" \
              --name "Web Scraping Task" \
              --tags p2p-eligible web-scraping \
              --priority 5
          else
            # Run locally
            ./scripts/scrape.sh
          fi
```

## P2P Peer Implementation

To run a P2P peer that executes workflows:

```python
#!/usr/bin/env python3
"""
P2P Workflow Peer
Continuously polls for and executes P2P workflow tasks
"""

import time
import logging
from ipfs_accelerate_py import P2PWorkflowScheduler, WorkflowTag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_task(task):
    """Execute a workflow task"""
    logger.info(f"Executing task: {task.name}")
    
    # Execute based on task type
    if WorkflowTag.CODE_GENERATION in task.tags:
        # Handle code generation
        pass
    elif WorkflowTag.WEB_SCRAPING in task.tags:
        # Handle web scraping
        pass
    elif WorkflowTag.DATA_PROCESSING in task.tags:
        # Handle data processing
        pass
    
    logger.info(f"Task {task.task_id} completed")

def main():
    # Create scheduler
    scheduler = P2PWorkflowScheduler(peer_id="peer-worker-1")
    
    logger.info("P2P Workflow Peer started")
    logger.info(f"Peer ID: {scheduler.peer_id}")
    
    # Main loop
    while True:
        try:
            # Get next task
            task = scheduler.get_next_task()
            
            if task:
                # Execute task
                execute_task(task)
                
                # Mark complete
                scheduler.mark_task_complete(task.task_id)
            else:
                # No tasks available, wait
                time.sleep(10)
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
```

Run the peer:

```bash
python p2p_workflow_peer.py
```

## Technical Details

### Merkle Clock Algorithm

1. Each peer maintains a vector clock: `{peer_id: timestamp}`
2. On local event: increment own timestamp
3. On receiving message: merge clocks (take max of each entry)
4. Calculate merkle root from sorted vector entries
5. Use merkle root as canonical clock "hash"

### Task Assignment Algorithm

1. Combine merkle clock head with task hash:
   ```
   combined_hash = SHA256(clock_hash + ":" + task_hash)
   ```

2. For each known peer, calculate hamming distance:
   ```
   distance = hamming(combined_hash, peer_id_hash)
   ```

3. Assign task to peer with minimum distance

4. If assigned peer fails to respond within timeout:
   - Reassign to next closest peer
   - Update failure tracking

### Priority Scheduling

Tasks are ordered by:
1. **Priority** (1-10, higher = more important)
2. **Creation time** (earlier = higher priority as tiebreaker)

The Fibonacci heap ensures:
- O(1) insertion of new tasks
- O(log n) extraction of highest priority task
- Efficient reordering when priorities change

## Testing

Run the comprehensive test suite:

```bash
# Run all P2P scheduler tests
pytest test_p2p_workflow_scheduler.py -v

# Run specific test class
pytest test_p2p_workflow_scheduler.py::TestMerkleClock -v

# Run with coverage
pytest test_p2p_workflow_scheduler.py --cov=ipfs_accelerate_py.p2p_workflow_scheduler
```

## Performance Considerations

### Fibonacci Heap Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Insert | O(1) | Amortized |
| Extract Min | O(log n) | Amortized |
| Find Min | O(1) | Direct access |
| Decrease Key | O(1) | Amortized |

### Hamming Distance

- **Calculation**: O(n) where n = hash length in bits
- **Typical**: O(256) for SHA-256 hashes
- **Optimization**: Can be cached for frequently compared peers

### Merkle Clock

- **Update**: O(p) where p = number of peers
- **Compare**: O(p)
- **Serialize**: O(p)
- **Memory**: O(p) for vector clock storage

## Future Enhancements

1. **Persistent Task Storage** - Save tasks to IPFS
2. **Task Result Storage** - Store results in distributed database
3. **Reputation System** - Track peer reliability
4. **Dynamic Priority** - Adjust priorities based on system load
5. **Multi-stage Workflows** - Support for task dependencies
6. **Fault Tolerance** - Automatic task retry on failure
7. **Load Balancing** - Consider peer capacity in assignment

## Troubleshooting

### Task Not Being Assigned

**Problem**: Task submitted but never gets assigned to any peer.

**Solution**:
1. Check that peers are running and connected
2. Verify peer IDs are being shared
3. Check merkle clock synchronization:
   ```bash
   ipfs-accelerate p2p-workflow clock
   ```

### Tasks Assigned to Wrong Peer

**Problem**: Tasks consistently go to the same peer despite multiple peers available.

**Solution**:
1. Ensure all peers are registered with the scheduler
2. Check that peer health monitoring is working
3. Verify merkle clock is advancing:
   ```bash
   ipfs-accelerate p2p-workflow status
   ```

### Priority Not Working

**Problem**: Lower priority tasks being executed before higher priority.

**Solution**:
1. Verify priority values (1-10, higher = more important)
2. Check that tasks are being inserted into the heap
3. Ensure extract_min is being called correctly

## License

This component is part of ipfs_accelerate_py and follows the same license terms.

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest test_p2p_workflow_scheduler.py`
2. Code follows PEP 8 style guidelines
3. New features include tests
4. Documentation is updated

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/endomorphosis/ipfs_accelerate_py/issues
- Documentation: See README.md in the repository root
