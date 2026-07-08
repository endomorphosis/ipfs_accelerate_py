# Coordinator Redundancy and Failover

This document provides a comprehensive overview of the coordinator redundancy and failover feature in the Distributed Testing Framework.

## Introduction

The coordinator redundancy and failover feature implements a high-availability solution for the Distributed Testing Coordinator. Using a simplified [Raft consensus algorithm](https://raft.github.io/), it ensures the distributed testing system remains operational even when coordinator nodes fail.

## Core Features

- **Leader Election**: Automatic election of a leader coordinator among a cluster of nodes
- **Log Replication**: Consistent replication of all operations across all nodes
- **State Synchronization**: Full state transfer to ensure consistency
- **Failure Detection**: Heartbeat-based detection of node failures
- **Automatic Failover**: Seamless transition to a new leader when the current leader fails
- **Crash Recovery**: Persistence mechanisms to recover from node crashes
- **Client Request Forwarding**: Automatic forwarding of requests to the current leader

## Architecture

The coordinator redundancy system uses a leader-follower architecture:

![Coordinator Redundancy Architecture](../images/coordinator_redundancy_architecture.png)

### Components

1. **RedundancyManager**: Core implementation of the Raft consensus algorithm
2. **NodeRole enum**: Defines the possible roles for a node (LEADER, FOLLOWER, CANDIDATE)
3. **Raft State**: Persistent state information (term, voted_for, log)
4. **Coordinator Integration**: Methods to integrate with the main coordinator
5. **API Routes**: HTTP endpoints for Raft protocol communication

### Data Structures

**Raft Log Entry**:
```python
{
    "term": <term_number>,
    "command": {
        "type": <command_type>,
        ... command-specific fields ...
    }
}
```

**Vote Request**:
```python
{
    "term": <candidate_term>,
    "candidate_id": <candidate_node_id>,
    "last_log_index": <index_of_candidates_last_log_entry>,
    "last_log_term": <term_of_candidates_last_log_entry>
}
```

**Append Entries Request**:
```python
{
    "term": <leader_term>,
    "leader_id": <leader_node_id>,
    "prev_log_index": <index_of_log_entry_before_new_ones>,
    "prev_log_term": <term_of_prev_log_entry>,
    "entries": [<log_entries>],
    "leader_commit": <leader_commit_index>
}
```

## Implementation Details

### RedundancyManager Class

The `RedundancyManager` class implements the Raft consensus algorithm:

```python
class RedundancyManager:
    def __init__(self, node_id, host, port, peers, data_dir=None, coordinator=None):
        # Initialize Raft state
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peers = peers
        self.data_dir = data_dir or os.path.join(tempfile.gettempdir(), f"coordinator_{node_id}")
        self.coordinator = coordinator
        
        # Raft state
        self.role = NodeRole.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        self.current_leader = None
        
        # Leader state
        self.next_index = {}
        self.match_index = {}
        
        # Timers
        self.election_timer = None
        self.heartbeat_timer = None
        
        # State machine
        self.state_machine = {}
```

### Raft Protocol Implementation

#### Leader Election

The implementation includes a timeout-based leader election mechanism:

```python
async def _start_election(self):
    """Start a new election."""
    self.role = NodeRole.CANDIDATE
    self.current_term += 1
    self.voted_for = self.node_id
    await self._save_persistent_state()
    
    # Vote for self
    votes_received = 1
    
    # Request votes from all peers
    for peer in self.peers:
        try:
            last_log_index = len(self.log) - 1
            last_log_term = self.log[last_log_index]["term"] if last_log_index >= 0 else 0
            
            request = {
                "term": self.current_term,
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term
            }
            
            response = await self.request_vote(peer["id"], request)
            
            if response and response.get("vote_granted"):
                votes_received += 1
                
                # Check if we have majority
                if votes_received > (len(self.peers) + 1) // 2:
                    await self._become_leader()
                    return True
                    
        except Exception as e:
            logger.warning(f"Error requesting vote from {peer['id']}: {e}")
    
    return False
```

#### Log Replication

Ensures all nodes have a consistent view of the log:

```python
async def append_log(self, command):
    """Append a command to the log and replicate to followers."""
    if self.role != NodeRole.LEADER:
        return False
        
    # Create log entry
    entry = {
        "term": self.current_term,
        "command": command
    }
    
    # Append to local log
    self.log.append(entry)
    log_index = len(self.log) - 1
    
    # Save persistent state
    await self._save_persistent_state()
    
    # Replicate to followers
    replication_success = await self._replicate_to_followers(log_index)
    
    if replication_success:
        # Update commit index
        self._update_commit_index()
        
        # Apply to state machine
        await self._apply_logs()
        
    return replication_success
```

#### State Synchronization

Ensures all nodes have a consistent view of the system state:

```python
async def _sync_state_to_followers(self):
    """Synchronize full state to followers."""
    if self.role != NodeRole.LEADER:
        return
        
    # Get full state from coordinator
    state = self.coordinator.get_full_state()
    
    # Send state to all followers
    for peer in self.peers:
        try:
            url = f"http://{peer['host']}:{peer['port']}/api/raft/sync_state"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    "term": self.current_term,
                    "leader_id": self.node_id,
                    "state": state,
                    "last_applied": self.last_applied
                }) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"State sync to {peer['id']} successful: {result}")
                    else:
                        logger.warning(f"State sync to {peer['id']} failed: HTTP {response.status}")
                        
        except Exception as e:
            logger.warning(f"Error syncing state to {peer['id']}: {e}")
```

### Failure Detection and Recovery

The system uses heartbeats to detect failures:

```python
async def _send_heartbeats(self):
    """Send heartbeats to all followers."""
    if self.role != NodeRole.LEADER:
        return
        
    for peer in self.peers:
        try:
            prev_log_index = self.next_index[peer["id"]] - 1
            prev_log_term = 0
            
            if prev_log_index >= 0 and prev_log_index < len(self.log):
                prev_log_term = self.log[prev_log_index]["term"]
                
            entries = []
            
            # Include entries that the follower might be missing
            if prev_log_index < len(self.log) - 1:
                entries = self.log[prev_log_index + 1:]
                
            request = {
                "term": self.current_term,
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": entries,
                "leader_commit": self.commit_index
            }
            
            response = await self.append_entries(peer["id"], request)
            
            if response:
                if response.get("success"):
                    # Update follower state
                    if entries:
                        self.next_index[peer["id"]] = prev_log_index + len(entries) + 1
                        self.match_index[peer["id"]] = prev_log_index + len(entries)
                else:
                    # Follower is behind, decrement next_index and retry
                    self.next_index[peer["id"]] = max(1, self.next_index[peer["id"]] - 1)
                    
        except Exception as e:
            logger.warning(f"Error sending heartbeat to {peer['id']}: {e}")
```

## Integration with Coordinator

The coordinator integrates with the redundancy system as follows:

```python
def _init_redundancy_manager(self):
    """Initialize the redundancy manager."""
    if not self.enable_redundancy:
        return None
        
    try:
        # Import redundancy module
        from .coordinator_redundancy import RedundancyManager, NodeRole
        
        # Parse peers
        peers = []
        for peer in self.peers.split(","):
            parts = peer.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 8080
            peer_id = f"node-{len(peers) + 1}"
            peers.append({"id": peer_id, "host": host, "port": port})
            
        # Create redundancy manager
        manager = RedundancyManager(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            peers=peers,
            data_dir=self.data_dir,
            coordinator=self
        )
        
        # Register callbacks
        manager.register_on_leadership_change(self._on_leadership_change)
        
        return manager
        
    except ImportError:
        logger.warning("Redundancy manager not available - running in single node mode")
        return None
```

## API Routes

The coordinator includes API routes for Raft protocol communication:

```python
def _setup_routes(self):
    """Set up API routes."""
    # ... existing routes ...
    
    # Raft protocol routes
    if self.redundancy_manager:
        self.app.router.add_post('/api/raft/request_vote', self._handle_raft_request_vote)
        self.app.router.add_post('/api/raft/append_entries', self._handle_raft_append_entries)
        self.app.router.add_post('/api/raft/sync_state', self._handle_raft_sync_state)
        self.app.router.add_get('/api/status', self._handle_status)
```

## Command-Line Options

Command-line options for configuring redundancy:

```
  --enable-redundancy        Enable redundancy features
  --peers PEERS              Comma-separated list of peer coordinator nodes in format host:port
  --node-id NODE_ID          Unique identifier for this coordinator node
  --election-timeout-min ELECTION_TIMEOUT_MIN
                             Minimum election timeout in milliseconds (default: 150)
  --election-timeout-max ELECTION_TIMEOUT_MAX
                             Maximum election timeout in milliseconds (default: 300)
  --heartbeat-interval HEARTBEAT_INTERVAL
                             Heartbeat interval in milliseconds (default: 50)
```

## Testing

The implementation includes comprehensive test coverage:

1. **Unit Tests**: Testing individual components of the Raft algorithm
2. **Integration Tests**: Testing coordinator integration with redundancy
3. **Failover Tests**: Testing automatic failover upon leader failure
4. **Recovery Tests**: Testing recovery from various failure scenarios
5. **Performance Benchmarks**: Measuring overhead of the redundancy system

## Deployment

For deployment guidelines, see the [Deployment Guide](deployment_guide.md).

## Monitoring

For monitoring redundant coordinator clusters, use the provided monitoring tools:

1. **Cluster Health Monitor**: Visual dashboard for cluster status
2. **Recovery Strategies**: Automatic recovery from various failure scenarios

## Performance Considerations

The coordinator redundancy implementation has the following performance characteristics:

1. **Write Operations**: Slightly reduced throughput due to consensus requirements
2. **Read Operations**: Improved throughput through load distribution
3. **Failover Time**: Typically 2-4 seconds to elect a new leader
4. **Resource Usage**: Minimal overhead compared to single-node deployment

## Conclusion

The coordinator redundancy and failover feature completes the Fault Tolerance phase of the Distributed Testing Framework. It provides high availability, fault tolerance, and automatic recovery, ensuring that the testing system remains operational even in the presence of failures.

## References

1. [Raft Consensus Algorithm](https://raft.github.io/)
2. [In Search of an Understandable Consensus Algorithm](https://raft.github.io/raft.pdf)
3. [Distributed Testing Framework Documentation](../README.md)
4. [Deployment Guide](deployment_guide.md)
5. [API Reference](api_reference.md)