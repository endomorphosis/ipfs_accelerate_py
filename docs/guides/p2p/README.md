# P2P and Distributed Computing Guides

Comprehensive guides for peer-to-peer networking, distributed workflows, and libp2p integration.

## Quick Links

- [P2P Setup Guide](../../P2P_SETUP_GUIDE.md) - Main P2P setup guide (root level)
- [P2P & MCP Architecture](../../P2P_AND_MCP.md) - Complete P2P and MCP documentation
- [P2P Workflow Scheduler](../../P2P_WORKFLOW_SCHEDULER.md) - Workflow scheduling guide
- [P2P Workflow Quick Start](../../P2P_WORKFLOW_QUICK_START.md) - Quick start guide

## Specialized Guides (in this directory)

### P2P Cache

- **[P2P Cache Deadlock Fix](P2P_CACHE_DEADLOCK_FIX.md)** - Troubleshooting deadlocks
- **[P2P Cache Encryption](P2P_CACHE_ENCRYPTION.md)** - Secure P2P communications

### P2P Workflow Management

- **[P2P Workflow Discovery](P2P_WORKFLOW_DISCOVERY.md)** - Discover and join P2P workflows
- **[P2P Workflow Scheduler](P2P_WORKFLOW_SCHEDULER.md)** - Advanced scheduling features

### Autoscaling

- **[P2P Autoscaler Quick Reference](P2P_AUTOSCALER_QUICK_REF.md)** - Quick reference for autoscaling

### libp2p Integration

- **[libp2p Universal Connectivity](LIBP2P_UNIVERSAL_CONNECTIVITY.md)** - Universal connectivity setup
- **[MCP P2P Setup Guide](MCP_P2P_SETUP_GUIDE.md)** - Integrate MCP with P2P

## Core Concepts

### Peer-to-Peer Networking

IPFS Accelerate uses libp2p for peer-to-peer networking, enabling:

- **Distributed Inference**: Share compute across network peers
- **Content Addressing**: Cryptographically secure model distribution
- **Fault Tolerance**: Automatic failover and recovery
- **Load Balancing**: Distribute work across available peers

### Merkle Clock Consensus

The P2P workflow scheduler uses Merkle clocks for distributed consensus:

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import MerkleClock

# Create clock for this node
clock = MerkleClock(node_id="node-123")

# Synchronize with other nodes
clock.update(other_node_clock)

# Get consensus hash
consensus_hash = clock.get_hash()
```

### Fibonacci Heap Scheduling

Efficient priority-based task scheduling:

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler

scheduler = P2PWorkflowScheduler(node_id="worker-01")
await scheduler.start()

# Submit high-priority task
await scheduler.submit_workflow({
    "name": "urgent-task",
    "priority": 1,  # Lower = higher priority
    "tasks": [...]
})
```

## Quick Start

### Basic P2P Setup

```bash
# Install P2P dependencies
pip install ipfs-accelerate-py[p2p]

# Start P2P node
ipfs-accelerate p2p start --node-id my-node

# Join network
ipfs-accelerate p2p join --bootstrap /ip4/...
```

### Distributed Workflow

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import (
    P2PWorkflowScheduler,
    WorkflowTag
)

async def main():
    # Create scheduler
    scheduler = P2PWorkflowScheduler(node_id="worker-01")
    await scheduler.start()
    
    # Submit distributed workflow
    workflow_id = await scheduler.submit_workflow({
        "name": "batch-inference",
        "tag": WorkflowTag.P2P_ELIGIBLE,
        "tasks": [
            {"model": "bert-base", "input": "text1"},
            {"model": "bert-base", "input": "text2"}
        ]
    })
    
    # Monitor progress
    status = await scheduler.get_workflow_status(workflow_id)
    print(f"Progress: {status['completed']}/{status['total']}")
```

## Network Architecture

```
┌─────────────────────────────────────────────┐
│         P2P Network Topology                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐      ┌──────────┐            │
│  │  Node A  │◄────►│  Node B  │            │
│  │ (Worker) │      │ (Worker) │            │
│  └──────────┘      └──────────┘            │
│       ▲                 ▲                   │
│       │                 │                   │
│       ▼                 ▼                   │
│  ┌──────────┐      ┌──────────┐            │
│  │  Node C  │◄────►│  Node D  │            │
│  │(Scheduler)│      │(Bootstrap)│           │
│  └──────────┘      └──────────┘            │
│                                             │
└─────────────────────────────────────────────┘
```

## See Also

- [Main Documentation](../../README.md)
- [P2P & MCP Architecture](../../P2P_AND_MCP.md)
- [GitHub Guides](../github/)
- [Deployment Guides](../deployment/)

---

**Last Updated**: January 2026
