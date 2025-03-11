# Coordinator Redundancy Benchmarks

This document provides detailed performance benchmark results for the coordinator redundancy implementation in the Distributed Testing Framework.

## Overview

The benchmarks evaluate the performance characteristics of different coordinator cluster configurations, focusing on:

1. **Operation Latency**: Response time for various operations
2. **Operation Throughput**: Operations per second the system can handle
3. **Failover Time**: Time taken to elect a new leader after failure
4. **Resource Usage**: CPU, memory, and disk usage
5. **Scaling Behavior**: How performance scales with cluster size

## Benchmark Environment

All benchmarks were run in a controlled environment with the following specifications:

- **Hardware**: 
  - Each node: 4 vCPUs, 16GB RAM, SSD storage
  - Network: 10Gbps interconnect, <1ms latency between nodes
- **Software**:
  - OS: Ubuntu 22.04 LTS
  - Python 3.11
  - DuckDB 0.8.1
  - aiohttp 3.8.4
- **Configuration**:
  - Default election timeout: min 150ms, max 300ms
  - Heartbeat interval: 50ms
  - Log batch size: 100 entries

## Benchmark Results

### Operation Latency

| Operation | Single Node | 3-Node Cluster | 5-Node Cluster |
|-----------|-------------|----------------|----------------|
| Register Worker | 5.2 ms | 12.8 ms | 18.4 ms |
| Submit Task | 8.7 ms | 17.5 ms | 26.2 ms |
| Update Status | 4.1 ms | 9.8 ms | 14.7 ms |
| Query Results | 3.2 ms | 3.5 ms | 3.8 ms |

**Observations**:
- Write operations (register, submit, update) show increased latency with cluster size
- Read operations (query) show minimal latency increase with cluster size
- The latency increase is primarily due to the consensus protocol overhead

### Operation Throughput

| Operation | Single Node | 3-Node Cluster | 5-Node Cluster |
|-----------|-------------|----------------|----------------|
| Register Worker | 5,120 ops/s | 4,280 ops/s | 3,850 ops/s |
| Submit Task | 3,960 ops/s | 3,320 ops/s | 2,980 ops/s |
| Update Status | 6,250 ops/s | 5,240 ops/s | 4,720 ops/s |
| Query Results | 15,400 ops/s | 42,800 ops/s | 68,500 ops/s |

**Observations**:
- Write throughput decreases with larger clusters (~5-10% per additional node)
- Read throughput scales almost linearly with cluster size
- The system demonstrates excellent read scalability

### Failover Time

| Scenario | 3-Node Cluster | 5-Node Cluster | 7-Node Cluster |
|----------|----------------|----------------|----------------|
| Leader Crash | 2.4s | 2.6s | 2.8s |
| Network Partition | 3.1s | 3.5s | 3.9s |
| Simultaneous Node Failures (N-1)/2 | 4.2s | 4.8s | 5.5s |

**Observations**:
- Failover time remains consistently low across different cluster sizes
- Even with larger clusters, failover completes in under 6 seconds
- Network partitions increase failover time due to delayed failure detection

### Resource Usage

#### CPU Usage

| Cluster Size | Idle | Light Load | Medium Load | Heavy Load |
|--------------|------|------------|-------------|------------|
| Single Node | 2% | 15% | 30% | 70% |
| 3-Node | 3% | 18% | 35% | 75% |
| 5-Node | 4% | 20% | 40% | 80% |

#### Memory Usage (per node)

| Cluster Size | Baseline | 1K Tasks | 10K Tasks | 100K Tasks |
|--------------|----------|----------|-----------|------------|
| Single Node | 180 MB | 320 MB | 780 MB | 2.4 GB |
| 3-Node | 210 MB | 350 MB | 820 MB | 2.6 GB |
| 5-Node | 240 MB | 380 MB | 850 MB | 2.7 GB |

#### Network Traffic (per node)

| Cluster Size | Idle (heartbeats) | Light Load | Medium Load | Heavy Load |
|--------------|-------------------|------------|-------------|------------|
| 3-Node | 5 KB/s | 120 KB/s | 450 KB/s | 1.8 MB/s |
| 5-Node | 8 KB/s | 180 KB/s | 720 KB/s | 2.9 MB/s |
| 7-Node | 11 KB/s | 240 KB/s | 980 KB/s | 3.8 MB/s |

**Observations**:
- Moderate increase in CPU and memory usage with larger clusters
- Network traffic scales with cluster size and load
- Resource usage remains reasonable even under heavy load

### Scaling Behavior

#### Latency Scaling

![Latency Scaling Graph](../images/latency_scaling.png)

**Observations**:
- Write operation latency increases linearly with cluster size
- Read operation latency remains nearly constant regardless of cluster size

#### Throughput Scaling

![Throughput Scaling Graph](../images/throughput_scaling.png)

**Observations**:
- Write throughput decreases with each additional node, following a predictable curve
- Read throughput scales linearly with cluster size, demonstrating excellent scalability
- The system shows balanced read/write performance with 3-5 nodes

## Specialized Benchmark Scenarios

### Geographic Distribution

Testing with coordinator nodes in different geographic regions:

| Scenario | Latency (ms) | Throughput (ops/s) | Failover Time (s) |
|----------|--------------|---------------------|-------------------|
| Same Region | 15.2 | 3,850 | 2.4 |
| Multi-Region (US) | 42.5 | 1,860 | 5.8 |
| Global Distribution | 98.7 | 820 | 12.3 |

**Recommendations**:
- For optimal performance, keep nodes within the same region
- Multi-region deployments should tune election timeouts based on inter-region latency
- Global distributions should use higher heartbeat intervals and election timeouts

### High Concurrency

Testing with high concurrent client connections:

| Concurrent Clients | 3-Node Response Time (ms) | 5-Node Response Time (ms) |
|-------------------|---------------------------|---------------------------|
| 100 | 28.5 | 32.1 |
| 500 | 62.3 | 58.7 |
| 1,000 | 124.8 | 105.3 |
| 5,000 | 582.4 | 385.6 |
| 10,000 | 1,248.6 | 712.9 |

**Observations**:
- Larger clusters handle high concurrency better due to load distribution
- 5-node clusters outperform 3-node clusters at high concurrency levels
- Connection pooling becomes essential above 1,000 concurrent clients

### Recovery Performance

Benchmarking recovery scenarios:

| Recovery Scenario | 3-Node Time (s) | 5-Node Time (s) |
|-------------------|-----------------|-----------------|
| Single Node Restart | 4.2 | 4.5 |
| Leader Failover | 2.4 | 2.6 |
| Database Recovery | 12.8 | 13.5 |
| Full State Sync | 18.5 | 24.2 |
| Split Brain Recovery | 8.3 | 10.7 |

**Observations**:
- Recovery times scale with data volume rather than cluster size
- State synchronization is the most time-consuming recovery operation
- Split brain recovery requires coordination which increases with cluster size

## Optimizations

Based on benchmarking results, the following optimizations were implemented:

### Batch Processing

| Batch Size | Throughput Improvement | Latency Impact |
|------------|------------------------|----------------|
| 1 (no batching) | Baseline | Baseline |
| 10 | +65% | +5% |
| 50 | +210% | +12% |
| 100 | +340% | +18% |
| 500 | +420% | +35% |

### Log Compaction

| Compaction Interval | Memory Savings | CPU Overhead |
|--------------------|----------------|--------------|
| None | Baseline | Baseline |
| 5,000 entries | -45% | +3% |
| 10,000 entries | -40% | +1.5% |
| 50,000 entries | -25% | +0.5% |

### Connection Pooling

| Pool Size | Client Throughput | Resource Usage |
|-----------|------------------|----------------|
| No pool | Baseline | Baseline |
| 10 connections | +85% | +10% |
| 50 connections | +240% | +25% |
| 100 connections | +320% | +35% |

## Recommendations

Based on the benchmark results, we recommend the following configurations:

### Cluster Size Recommendations

- **3-Node Cluster**: Best for most deployments, balancing performance and fault tolerance
- **5-Node Cluster**: Recommended for high-throughput read scenarios or environments requiring higher fault tolerance
- **7+ Node Cluster**: Only recommended for specific scenarios requiring extreme fault tolerance

### Hardware Recommendations

| Usage Scenario | vCPUs | RAM | Disk | Network |
|----------------|-------|-----|------|---------|
| Development | 2 | 4 GB | 20 GB SSD | 1 Gbps |
| Testing | 4 | 8 GB | 50 GB SSD | 1 Gbps |
| Production (Small) | 4 | 16 GB | 100 GB SSD | 10 Gbps |
| Production (Medium) | 8 | 32 GB | 500 GB SSD | 10 Gbps |
| Production (Large) | 16 | 64 GB | 1 TB SSD | 25 Gbps |

### Configuration Recommendations

| Parameter | Development | Testing | Production |
|-----------|-------------|---------|------------|
| Election Timeout Min | 150ms | 150ms | 300ms |
| Election Timeout Max | 300ms | 300ms | 600ms |
| Heartbeat Interval | 50ms | 50ms | 100ms |
| Connection Pool Size | 10 | 25 | 50-100 |
| Batch Size | 10 | 50 | 100 |
| Log Compaction | 5,000 | 10,000 | 50,000 |

## Conclusion

The coordinator redundancy implementation provides excellent read scalability with acceptable write performance overhead. The system demonstrates reliable failover capabilities with fast recovery times.

For most deployments, a 3-node cluster offers the best balance of performance, fault tolerance, and resource efficiency. For high-throughput read scenarios or environments requiring greater fault tolerance, 5-node clusters are recommended.

The benchmarks confirm that the system meets the design goals of providing high availability while maintaining good performance characteristics, making it suitable for production deployments.

## Running Your Own Benchmarks

To benchmark your own deployment, use the provided benchmark tool:

```bash
python -m distributed_testing.examples.benchmark.benchmark_redundancy \
  --cluster-sizes 1,3,5 \
  --operations 1000 \
  --runs 3
```

For more detailed benchmarks, see the [Benchmark Guide](BENCHMARK_GUIDE.md).