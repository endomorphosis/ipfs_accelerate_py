# Distributed Testing Framework - Worker Guide

## Overview

The distributed testing framework worker node is a key component of the IPFS Accelerate Distributed Testing Framework. Workers register with a coordinator, receive tasks, execute them, and report results back to the coordinator.

This guide explains how to set up, configure, and run worker nodes to participate in distributed testing.

## Features

- **Automated Hardware Detection**: Workers automatically detect CPU, GPU (CUDA, ROCm, MPS), and other hardware capabilities
- **Secure Communication**: All communications use signed messages for security
- **Robust Authentication**: API key and JWT token-based authentication
- **Fault Tolerance**: Automatic reconnection and task recovery
- **Health Monitoring**: Self-monitoring of CPU, memory, and hardware utilization
- **Multi-task Support**: Execute benchmarks, tests, and custom tasks
- **Batched Execution**: Efficiently run multiple tasks as batches when possible
- **Database Integration**: Optional result caching using DuckDB

## Requirements

- Python 3.8 or higher
- Required Python packages:
  - websockets
  - aiohttp
  - pyjwt
  - psutil
  - duckdb (optional)
  - torch (optional, for GPU detection)
  - openvino (optional, for OpenVINO detection)

## Quick Start

1. Install required packages:
   ```bash
   pip install websockets aiohttp pyjwt psutil duckdb
   ```

2. Run a worker with default settings:
   ```bash
   python run_worker_example.py
   ```

3. Run a worker with specific coordinator and API key:
   ```bash
   python run_worker_example.py --coordinator http://coordinator.example.com:8080 --api-key YOUR_API_KEY
   ```

## Configuration Options

### Basic Configuration

- `--coordinator`: URL of the coordinator server (default: http://localhost:8080)
- `--name`: Friendly name for the worker (default: system hostname)
- `--worker-id`: Worker ID (default: generated UUID)

### Authentication

- `--api-key`: API key for authentication with coordinator
- `--token`: JWT token for authentication (alternative to API key)
- `--token-file`: Path to file containing JWT token

### Storage

- `--db-path`: Path to DuckDB database for caching results (optional)

### Capability Tagging

- `--tags`: Comma-separated list of capability tags (e.g., 'gpu,cuda,transformers')

### Advanced Options

- `--heartbeat-interval`: Heartbeat interval in seconds (default: 10)
- `--reconnect-attempts`: Maximum number of reconnection attempts (default: 10)

## Task Types and Execution

Workers can execute three primary types of tasks:

### 1. Benchmark Tasks

Benchmark tasks measure model performance across different batch sizes and precision settings.

Example benchmark task:
```json
{
  "task_id": "benchmark-bert-001",
  "type": "benchmark",
  "config": {
    "model": "bert-base-uncased",
    "batch_sizes": [1, 2, 4, 8, 16],
    "precision": "fp32",
    "iterations": 10
  }
}
```

### 2. Test Tasks

Test tasks execute test files and report pass/fail results.

Example test task:
```json
{
  "task_id": "test-bert-001",
  "type": "test",
  "config": {
    "test_file": "test_bert.py",
    "test_args": ["--batch-size", "4"]
  }
}
```

### 3. Custom Tasks

Custom tasks can perform arbitrary operations defined by the framework.

Example custom task:
```json
{
  "task_id": "custom-001",
  "type": "custom",
  "config": {
    "name": "setup_environment",
    "parameters": {
      "clear_cache": true,
      "prepare_models": ["bert", "vit"]
    }
  }
}
```

## Worker Lifecycle

1. **Initialization**: Worker initializes and detects hardware capabilities
2. **Connection**: Worker connects to coordinator via WebSocket
3. **Authentication**: Worker authenticates using API key or token
4. **Registration**: Worker registers hardware capabilities with coordinator
5. **Heartbeat**: Worker sends periodic heartbeats to maintain connection
6. **Task Reception**: Worker receives tasks from coordinator
7. **Task Execution**: Worker executes tasks and reports results
8. **Health Monitoring**: Worker monitors its own health and reports issues

## Advanced Usage

### Hardware Capability Tags

The `--tags` option allows you to add custom capability tags that aren't automatically detected. This is useful for:

- Indicating support for specific models: `--tags transformers,diffusion,bert`
- Marking specialized hardware: `--tags npu,dsp,qualcomm`
- Indicating software features: `--tags openvino,onnxruntime,tensorrt`

### Running Multiple Workers

You can run multiple worker instances on the same machine with different configurations:

```bash
# Run CPU worker
python run_worker_example.py --name cpu-worker-01 --tags cpu --coordinator http://coordinator:8080 --api-key CPU_KEY

# Run GPU worker
python run_worker_example.py --name gpu-worker-01 --tags gpu,cuda --coordinator http://coordinator:8080 --api-key GPU_KEY
```

### Environment Variables

You can use environment variables instead of command-line arguments:

- `DT_COORDINATOR_URL`: Coordinator URL
- `DT_API_KEY`: API key for authentication
- `DT_WORKER_NAME`: Worker name
- `DT_DB_PATH`: Database path

Example:
```bash
export DT_COORDINATOR_URL="http://coordinator:8080"
export DT_API_KEY="your-api-key"
python run_worker_example.py
```

## Security

All communication between workers and the coordinator is secured:

- API keys for initial authentication
- JWT tokens for ongoing authentication
- Message signing for all messages
- Timestamp verification to prevent replay attacks

## Advanced Configuration

### High Availability Mode

Workers support high availability mode, which enhances their ability to reconnect and recover from coordinator failures. This is particularly important in distributed environments where coordinators may be running in a cluster.

To enable high availability mode:

```bash
python run_worker_example.py \
  --coordinator http://coordinator.example.com:8080 \
  --api-key YOUR_API_KEY \
  --high-availability \
  --reconnect-attempts 20 \
  --reconnect-interval 15
```

In high availability mode, workers:

1. Maintain a registry of all coordinator nodes in the cluster
2. Detect coordinator failures and automatically switch to backup coordinators
3. Maintain task state across coordinator switches
4. Retry failed connections with exponential backoff
5. Report their health metrics more frequently

### Task Execution Profiles

Workers can be configured with different task execution profiles to optimize for specific workloads:

```bash
python run_worker_example.py \
  --profile benchmark \
  --max-concurrent-tasks 2 \
  --task-timeout 3600
```

Available profiles:
- `benchmark`: Optimized for running benchmark tasks (low concurrency, high resource allocation)
- `test`: Balanced profile for running test tasks (medium concurrency)
- `light`: Lightweight profile for simple tasks (high concurrency, low resource allocation)

### Resource Limits

Configure resource limits to prevent workers from becoming overwhelmed:

```bash
python run_worker_example.py \
  --max-cpu-percent 80 \
  --max-memory-percent 75 \
  --max-gpu-percent 90
```

When these limits are reached, the worker will:
1. Stop accepting new tasks
2. Report its status as resource-constrained
3. Enter a cool-down period before accepting new tasks

## Troubleshooting

### Connection Issues

If the worker fails to connect to the coordinator:

1. Verify the coordinator URL is correct
2. Check that the coordinator is running and accessible
3. Ensure the API key or token is valid
4. Check network connectivity and firewall settings
5. Look for TLS certificate issues if using HTTPS

### Authentication Failures

If authentication fails:

1. Verify the API key or token is correct
2. Check if the API key has expired or been revoked
3. Ensure the worker has the required roles
4. Verify the system time is synchronized (JWT validation can fail with mismatched clocks)

### Task Execution Failures

If tasks fail to execute:

1. Check worker logs for detailed error messages
2. Verify the worker has the required hardware for the task
3. Ensure all required dependencies are installed
4. Check disk space and memory availability
5. Look for environment variable misconfiguration
6. Verify the task parameters are correct

### Recovery and Self-Healing

The worker includes self-healing capabilities:

1. **Task Recovery**: Recovers interrupted tasks when possible
2. **State Recovery**: Maintains task state in a local database for recovery
3. **Resource Recovery**: Automatically releases resources if they are leaked
4. **Dependency Recovery**: Automatically reinstalls missing dependencies
5. **Connection Recovery**: Uses exponential backoff for reconnection attempts

To trigger manual recovery:

```bash
python run_worker_example.py --recover --reset-state
```

## Performance Tuning

### Memory Management

Optimize memory usage for different workloads:

```bash
python run_worker_example.py \
  --memory-buffer 0.2 \
  --enable-swap \
  --cache-models
```

The `--memory-buffer` option reserves a percentage of memory to prevent OOM errors, `--enable-swap` allows using disk space as memory overflow, and `--cache-models` keeps frequently used models in memory.

### GPU Optimization

For GPU workers, optimize performance with:

```bash
python run_worker_example.py \
  --cuda-streams 4 \
  --optimize-for throughput \
  --mixed-precision
```

These options configure CUDA stream concurrency, optimize for throughput vs. latency, and enable automatic mixed precision where supported.

## Next Steps

- Configure multiple workers across your infrastructure using the [Deployment Guide](DEPLOYMENT_GUIDE.md)
- Set up specialized workers for different hardware types
- Integrate with your CI/CD pipeline for continuous testing
- Create custom task types for your specific needs
- Explore advanced monitoring using the [dashboard](dashboard_server.py)
- Configure workers for high availability mode in a production environment