# Distributed Testing Worker Examples

This directory contains example implementations and configurations for the worker component of the IPFS Accelerate Distributed Testing Framework.

## Files

- `worker.py`: Full implementation of the worker node component
- `run_worker_example.py`: Example script for running a worker with command-line options
- `run_multiple_workers.sh`: Script to run multiple worker instances with different configurations
- `worker_config_example.json`: Example JSON configuration for a worker
- `task_examples.json`: Example task definitions that can be sent to the coordinator
- `WORKER_GUIDE.md`: Comprehensive documentation for setting up and running workers

## Quick Start

### Run a single worker

```bash
# Basic usage with default settings
python run_worker_example.py

# Connect to a specific coordinator
python run_worker_example.py --coordinator http://coordinator.example.com:8080

# Add authentication
python run_worker_example.py --coordinator http://coordinator.example.com:8080 --api-key YOUR_API_KEY

# Specify hardware capabilities
python run_worker_example.py --tags gpu,cuda,transformers
```

### Run multiple workers

```bash
# Run multiple workers with different configurations
./run_multiple_workers.sh
```

## Worker Configuration

Workers can be configured with various options:

### Basic Configuration

```bash
python run_worker_example.py \
  --coordinator http://coordinator.example.com:8080 \
  --name gpu-worker-01 \
  --db-path ./worker_results.duckdb
```

### Authentication

```bash
# Using API key
python run_worker_example.py --api-key YOUR_API_KEY

# Using token file
python run_worker_example.py --token-file ./worker_token.txt
```

### Custom Capabilities

```bash
# Specify hardware and software capabilities
python run_worker_example.py --tags gpu,cuda,transformers,bert,diffusion
```

## Advanced Configuration

For advanced configurations, use a JSON configuration file:

```json
{
  "coordinator_url": "http://coordinator.example.com:8080",
  "authentication": {
    "api_key": "YOUR_API_KEY_HERE",
    "token_file": "./worker_token.txt"
  },
  "worker": {
    "name": "gpu-worker-01",
    "heartbeat_interval": 10,
    "max_concurrent_tasks": 4
  }
}
```

See `worker_config_example.json` for a complete example.

## Task Examples

Workers can execute various types of tasks:

### Benchmark Task

```json
{
  "type": "benchmark",
  "name": "BERT Base Benchmark",
  "priority": 10,
  "config": {
    "model": "bert-base-uncased",
    "batch_sizes": [1, 2, 4, 8, 16, 32],
    "precision": "fp32",
    "iterations": 10,
    "hardware_requirements": {
      "gpu": true,
      "cuda_compute": 7.0,
      "memory_gb": 8
    }
  }
}
```

### Test Task

```json
{
  "type": "test",
  "name": "BERT Unit Tests",
  "priority": 10,
  "config": {
    "test_file": "test_bert.py",
    "test_args": ["--batch-size", "4"],
    "hardware_requirements": {
      "gpu": true
    }
  }
}
```

See `task_examples.json` for more examples.

## For More Information

See the comprehensive [Worker Guide](WORKER_GUIDE.md) for detailed documentation on all worker features and options.