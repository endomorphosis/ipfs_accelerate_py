# Advanced Recovery Strategies

This document describes the advanced recovery strategies implemented for the Distributed Testing Framework's coordinator redundancy system.

## Overview

The Advanced Recovery Strategies module provides automated detection and recovery mechanisms for various failure scenarios in coordinator clusters. While the basic Raft algorithm provides fault tolerance for simple node failures, the Advanced Recovery Strategies extend this with sophisticated handling of complex failure modes.

## Failure Modes

The system can detect and recover from the following failure modes:

### 1. Process Crash

**Detection:**
- Heartbeat response failures
- Process ID verification

**Recovery:**
- Automatic restart of the failed process
- Log-based recovery to restore state
- Rejoin the cluster as a follower

### 2. Network Partition

**Detection:**
- Connectivity checks between nodes
- Split vote patterns
- Inconsistent leader views

**Recovery:**
- Wait for natural partition healing
- Attempt alternative network paths
- Support for multiple network interfaces

### 3. Database Corruption

**Detection:**
- Database integrity checks
- Checksum verification
- Schema validation

**Recovery:**
- Backup the corrupted database
- Restore from a healthy replica
- Full state synchronization

### 4. Log Corruption

**Detection:**
- Log integrity checks
- Index verification
- Term sequence validation

**Recovery:**
- Backup the corrupted log
- Reset log state
- Full state synchronization from current leader

### 5. Split Brain

**Detection:**
- Multiple nodes claiming leadership
- Conflicting term numbers
- Divergent log entries

**Recovery:**
- Force all nodes to step down
- Staggered restart sequence
- Term number reconciliation

### 6. Term Divergence

**Detection:**
- Inconsistent term numbers across nodes
- Abnormal term progression

**Recovery:**
- Force term alignment
- Re-run election with highest term

### 7. State Divergence

**Detection:**
- State checksums comparison
- Command application verification
- Inconsistent query results

**Recovery:**
- Identify the authoritative state (usually leader's)
- Full state transfer to divergent nodes
- Force sync flag to prioritize consistency

### 8. Deadlock

**Detection:**
- Long-running operations
- Request queue backlog
- Timeout patterns

**Recovery:**
- Cancel blocked operations
- Restart deadlocked components
- Circuit breaker pattern implementation

### 9. Resource Exhaustion

**Detection:**
- CPU, memory, disk usage monitoring
- Growth trend analysis
- Resource leak detection

**Recovery:**
- Resource cleanup operations
- Memory limit enforcement
- Graceful degradation modes

### 10. Slow Follower

**Detection:**
- Log replication lag metrics
- Apply index comparison
- Response time monitoring

**Recovery:**
- Snapshot-based catch-up
- Direct state transfer
- Temporary removal from quorum

## Recovery Strategy Implementation

The recovery strategies are implemented in the `recovery_strategies.py` module, which provides:

### Recovery Strategy Interface

```python
class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, cluster_config, data_dir="/tmp/distributed_testing_recovery"):
        """Initialize the recovery strategy."""
        
    async def detect_failures(self):
        """Detect failures in the cluster."""
        
    async def recover(self, failure_type, affected_nodes):
        """Recover from a failure."""
        
    def log_recovery_action(self, action, details):
        """Log a recovery action."""
        
    async def save_recovery_log(self):
        """Save the recovery log to a file."""
```

### Coordinator Recovery Strategy

The `CoordinatorRecoveryStrategy` class extends the base class with coordinator-specific recovery implementations:

```python
class CoordinatorRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for coordinator redundancy failures."""
    
    async def _recover_from_process_crash(self, failures):
        """Recover from process crashes."""
        
    async def _recover_from_network_partition(self, failures):
        """Recover from network partitions."""
        
    async def _recover_from_database_corruption(self, failures):
        """Recover from database corruption."""
        
    async def _recover_from_log_corruption(self, failures):
        """Recover from log corruption."""
        
    async def _recover_from_split_brain(self, failures):
        """Recover from split brain condition."""
        
    async def _recover_from_term_divergence(self, failures):
        """Recover from term divergence."""
        
    async def _recover_from_state_divergence(self, failures):
        """Recover from state divergence."""
        
    async def _recover_from_deadlock(self, failures):
        """Recover from deadlock conditions."""
        
    async def _recover_from_resource_exhaustion(self, failures):
        """Recover from resource exhaustion."""
        
    async def _recover_from_slow_follower(self, failures):
        """Recover from slow follower condition."""
```

## Running the Recovery Tool

### Command-Line Usage

```bash
python -m distributed_testing.monitoring.recovery_strategies \
  --config cluster_config.json \
  --interval 30 \
  --daemon
```

### Configuration File Format

```json
{
  "nodes": [
    {
      "id": "node-1",
      "host": "coordinator1.example.com",
      "port": 8080,
      "data_dir": "/home/coordinator/distributed_testing/node1"
    },
    {
      "id": "node-2",
      "host": "coordinator2.example.com",
      "port": 8080,
      "data_dir": "/home/coordinator/distributed_testing/node2"
    },
    {
      "id": "node-3",
      "host": "coordinator3.example.com",
      "port": 8080,
      "data_dir": "/home/coordinator/distributed_testing/node3"
    }
  ],
  "recovery_dir": "/home/coordinator/distributed_testing/recovery",
  "recovery_options": {
    "auto_restart": true,
    "backup_before_recovery": true,
    "max_restart_attempts": 5,
    "restart_cooldown": 60
  }
}
```

### Running as a System Service

To run the recovery tool as a system service, create a systemd service file:

```ini
# /etc/systemd/system/coordinator-recovery.service
[Unit]
Description=Distributed Testing Coordinator Recovery Service
After=network.target

[Service]
User=coordinator
WorkingDirectory=/home/coordinator/ipfs_accelerate_py
ExecStart=/usr/bin/python -m distributed_testing.monitoring.recovery_strategies --config /home/coordinator/recovery_config.json --daemon
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable coordinator-recovery
sudo systemctl start coordinator-recovery
```

## Monitoring Recovery Actions

The recovery tool logs all actions to:

1. Standard log files
2. A structured JSON recovery log
3. The DuckDB database (optional)

### Recovery Log Format

```json
[
  {
    "timestamp": "2025-03-10T14:30:25.123456",
    "action": "restart_node",
    "details": {
      "node_id": "node-2",
      "reason": "process_crash",
      "pid": 12345
    }
  },
  {
    "timestamp": "2025-03-10T14:35:12.654321",
    "action": "recover_split_brain",
    "details": {
      "node_ids": ["node-1", "node-3"],
      "resolution": "force_term_increment"
    }
  }
]
```

## Best Practices

1. **Regular Health Checks**: Configure frequent health checks for early detection
2. **Backup Before Recovery**: Always enable backup before performing recovery actions
3. **Multiple Recovery Instances**: Run recovery tools on multiple hosts for redundancy
4. **Graduated Recovery**: Implement least-invasive recovery strategies first
5. **Recovery Testing**: Regularly test recovery strategies with failure simulations
6. **State Verification**: Always verify state consistency after recovery

## Advanced Configuration

### Customizing Recovery Behavior

Create a custom recovery configuration:

```python
# custom_recovery_config.py
from distributed_testing.monitoring.recovery_strategies import CoordinatorRecoveryStrategy

class CustomRecoveryStrategy(CoordinatorRecoveryStrategy):
    """Custom recovery strategy with specialized behaviors."""
    
    async def _recover_from_split_brain(self, failures):
        """Custom split brain recovery."""
        # Implement custom recovery logic
        
# Register the custom strategy
recovery_strategies.register("custom", CustomRecoveryStrategy)
```

Use the custom strategy:

```bash
python -m distributed_testing.monitoring.recovery_strategies \
  --config cluster_config.json \
  --strategy custom \
  --daemon
```

### Integration with Monitoring Systems

The recovery system can integrate with:

- **Prometheus**: Export recovery metrics
- **Grafana**: Visualize recovery actions
- **AlertManager**: Send alerts for recovery actions
- **PagerDuty**: Escalate serious failures
- **Slack/Teams**: Send notifications

## Conclusion

The Advanced Recovery Strategies module provides comprehensive protection against various failure modes in coordinator clusters. By automatically detecting and recovering from failures, it ensures high availability of the Distributed Testing Framework with minimal human intervention.