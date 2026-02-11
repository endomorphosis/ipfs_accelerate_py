# Worker Reconnection Monitoring Guide

This guide describes how to monitor the Worker Reconnection System's performance and health metrics.

## Performance Metrics

The Enhanced Worker Reconnection System collects detailed performance metrics that can be used to evaluate system health, diagnose issues, and optimize performance.

### Available Metrics

#### Connection Metrics

| Metric | Description | Normal Range |
|--------|-------------|--------------|
| `connection_attempts` | Total number of connection attempts | Increases with network disruptions |
| `successful_connections` | Number of successful connections | Should be close to `connection_attempts` |
| `avg_reconnect_time` | Average time to reconnect (seconds) | 0.5-3.0s (varies by network) |
| `max_reconnect_time` | Maximum reconnection time observed (seconds) | <10s in normal conditions |
| `connection_stability` | Connection uptime ratio (0-1) | >0.95 is good |

#### Message Metrics

| Metric | Description | Normal Range |
|--------|-------------|--------------|
| `messages_sent` | Total messages sent | Increases with usage |
| `messages_received` | Total messages received | Should be similar to `messages_sent` |
| `avg_message_latency_ms` | Average message round-trip time (ms) | <100ms in good conditions |
| `avg_message_size_bytes` | Average message size (bytes) | Varies by message type |
| `message_errors` | Number of message delivery errors | Should be low (<1% of sent) |

#### Task Metrics

| Metric | Description | Normal Range |
|--------|-------------|--------------|
| `tasks_executed` | Number of tasks executed | Increases with usage |
| `task_success_rate` | Ratio of successful tasks (0-1) | >0.98 is good |
| `avg_task_execution_time_ms` | Average task execution time (ms) | Varies by task type |
| `checkpoints_created` | Number of task checkpoints created | Proportional to long tasks |
| `checkpoints_restored` | Number of tasks restored from checkpoints | Should be ≤ checkpoints_created |

> **Note**: With the recent fix to the task execution recursion error (March 13, 2025), these task metrics are now fully reliable for all worker types.

#### Compression Metrics

| Metric | Description | Normal Range |
|--------|-------------|--------------|
| `compression_ratio` | Average message compression ratio | 2.0-5.0x for typical messages |
| `compression_time_ms` | Average time to compress messages (ms) | <10ms |
| `decompression_time_ms` | Average time to decompress messages (ms) | <5ms |

### Accessing Metrics

#### Enhanced Worker Client

The Enhanced Worker Client reports metrics at regular intervals in its logs:

```
====== Performance Metrics for worker-1 ======
Message count: 120
Average message size: 1024.5 bytes
Average message latency: 42.3 ms
Message errors: 0
Reconnections: 2
Average reconnection duration: 1.2 s
Task execution count: 15
Task success rate: 1.00
Average task duration: 250.3 ms
Checkpoints created: 5
Checkpoints resumed: 1
Compression ratio: 3.2x
Uptime: 300.0 s
=============================================
```

You can also access these metrics programmatically:

```python
from worker_reconnection_enhancements import EnhancedWorkerReconnectionPlugin

# Get an existing plugin instance
plugin = worker.reconnection_plugin

# Get metrics
metrics = plugin.get_performance_metrics()
print(f"Message latency: {metrics['avg_message_latency_ms']}ms")
print(f"Task success rate: {metrics['task_success_rate']}")
```

#### Coordinator

The Coordinator Server collects metrics from all connected workers. You can access these metrics through the Coordinator's WebSocket API:

```python
import aiohttp
import json

async def get_worker_metrics(coordinator_url, worker_id, api_key):
    """Get metrics for a specific worker from the coordinator."""
    url = f"{coordinator_url}/api/v1/workers/{worker_id}/metrics"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Error: {response.status}")
                return None
```

## Monitoring Dashboard Integration

The Worker Reconnection System integrates with the Distributed Testing Framework's monitoring dashboard to provide visual monitoring of reconnection metrics.

### Dashboard Views

- **Worker Status**: Shows the connection status of all workers
- **Reconnection Metrics**: Visualizes reconnection attempts, success rates, and times
- **Message Performance**: Displays message throughput, latency, and error rates
- **Task Execution**: Shows task success rates and checkpoint usage
- **System Health**: Aggregates metrics to show overall system health

### Accessing the Dashboard

The monitoring dashboard is available at the Coordinator's HTTP interface:

```
http://<coordinator_host>:<coordinator_port>/dashboard
```

## Setting Up Alerts

You can configure alerts based on Worker Reconnection metrics to be notified of potential issues:

### Alert Conditions

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Reconnection Rate | >3 reconnections per minute | WARNING |
| Connection Failures | Failed reconnection attempts | CRITICAL |
| High Message Latency | avg_message_latency > 200ms | WARNING |
| Task Failures | task_success_rate < 0.95 | CRITICAL |
| Low Compression | compression_ratio < 1.5 | INFO |

### Configuring Alerts

Alerts can be configured in the Coordinator's configuration file:

```yaml
alerts:
  worker_reconnection:
    high_reconnection_rate:
      condition: "reconnections_per_minute > 3"
      severity: "WARNING"
      notification: ["dashboard", "email"]
    connection_failures:
      condition: "failed_reconnections > 0"
      severity: "CRITICAL"
      notification: ["dashboard", "email", "slack"]
```

## Logging and Debugging

The Enhanced Worker Reconnection System includes detailed logging for monitoring and debugging purposes.

### Log Levels

- **DEBUG**: Detailed debugging information, including all messages
- **INFO**: General information about normal operation
- **WARNING**: Potential issues that don't prevent operation
- **ERROR**: Issues that prevent specific operations
- **CRITICAL**: Issues that prevent system operation

### Enabling Debug Logging

For more detailed logs to diagnose issues:

```bash
# Run coordinator with debug logging
./run_coordinator_server.py --host localhost --port 8765 --log-level DEBUG

# Run worker with debug logging
./run_enhanced_worker_client.py --worker-id test-worker --coordinator-host localhost --coordinator-port 8765 --log-level DEBUG
```

### Log Analysis

The reconnection system logs include markers for important events:

- `[CONNECT]`: Connection establishment
- `[RECONNECT]`: Reconnection attempts
- `[SYNC]`: State synchronization events
- `[CHECKPOINT]`: Checkpoint creation and restoration
- `[TASK]`: Task execution events
- `[SECURITY]`: Security-related events
- `[COMPRESS]`: Compression-related events

You can filter logs to focus on specific areas:

```bash
# View only reconnection-related events
grep "\[RECONNECT\]" worker-1.log

# View task execution events
grep "\[TASK\]" worker-1.log

# View security events
grep "\[SECURITY\]" worker-1.log
```

## Performance Tuning

You can tune the Worker Reconnection System for optimal performance based on your network conditions:

### Connection Parameters

- `--initial-reconnect-delay`: Initial delay before reconnection (default: 1.0s)
- `--max-reconnect-delay`: Maximum reconnection delay (default: 60.0s)
- `--heartbeat-interval`: Interval between heartbeat messages (default: 5.0s)

### Compression Settings

- `--compression-level`: ZLib compression level (0-9, default: 6)
- `--min-compress-size`: Minimum message size for compression (default: 1024 bytes)

### Priority Queue Settings

- `--enable-priority-queue`: Enable priority-based message handling (default: enabled)
- `--queue-high-watermark`: Maximum queue size before applying backpressure (default: 1000)

### Example Tuning

For high-latency networks:
```bash
./run_enhanced_worker_client.py --worker-id test-worker --coordinator-host remote-server --coordinator-port 8765 --heartbeat-interval 10.0 --initial-reconnect-delay 2.0 --max-reconnect-delay 120.0
```

For message-intensive applications:
```bash
./run_enhanced_worker_client.py --worker-id test-worker --coordinator-host localhost --coordinator-port 8765 --compression-level 1 --queue-high-watermark 5000
```

## Conclusion

Effective monitoring of the Worker Reconnection System is essential for maintaining reliable communication in the Distributed Testing Framework. By tracking performance metrics, setting up alerts, and analyzing logs, you can ensure optimal operation and quickly diagnose any issues that arise.

## Recent Updates

### March 13, 2025: Task Execution Metrics Reliability

The task execution metrics are now fully reliable with the fix of the task execution recursion error that previously affected the Enhanced Worker Reconnection System. See [Task Execution Recursion Fix](TASK_EXECUTION_RECURSION_FIX.md) for details on this important improvement.

This fix ensures that:
- Task success rate metrics accurately reflect actual task execution outcomes
- Task duration measurements are correct and not affected by recursion
- Checkpoint creation and restoration metrics are accurate
- Task execution performance can be reliably analyzed for optimization