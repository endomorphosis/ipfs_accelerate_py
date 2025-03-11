# Distributed Testing Framework

A high-performance, fault-tolerant framework for parallel execution of tests and benchmarks across heterogeneous hardware.

## Overview

The Distributed Testing Framework enables parallel execution of benchmarks and tests across multiple machines with heterogeneous hardware. It provides several key benefits:

1. **Scalability**: Run thousands of tests in parallel across multiple machines
2. **Hardware Efficiency**: Automatically match tests to machines with appropriate hardware
3. **Centralized Results**: Aggregate all test results in a single database
4. **Test Prioritization**: Schedule tests based on importance and dependencies
5. **Fault Tolerance**: Automatically recover from worker and coordinator failures

## Architecture

The framework uses a coordinator-worker architecture:

```
                             ┌────────────┐
                             │            │
                             │  DuckDB    │
                             │  Database  │
                             │            │
                             └─────┬──────┘
                                   │
                                   ▼
┌───────────────┐           ┌────────────┐           ┌───────────────┐
│               │           │            │           │               │
│   Web UI      │◄─────────►│ Coordinator │◄─────────►│  REST API     │
│               │           │            │           │               │
└───────────────┘           └─────┬──────┘           └───────────────┘
                                  │
                                  │
         ┌──────────────┬─────────┴──────────┬──────────────┐
         │              │                    │              │
         ▼              ▼                    ▼              ▼
┌─────────────┐  ┌─────────────┐    ┌─────────────┐  ┌─────────────┐
│             │  │             │    │             │  │             │
│  Worker 1   │  │  Worker 2   │    │  Worker 3   │  │  Worker N   │
│ (CPU, CUDA) │  │ (ROCm, MPS) │    │ (CPU, QNN)  │  │ (WebNN, CPU)│
│             │  │             │    │             │  │             │
└─────────────┘  └─────────────┘    └─────────────┘  └─────────────┘
```

## Features

### Core Features

- ✅ Worker registration and capability reporting
- ✅ Task submission and scheduling
- ✅ Result collection and aggregation
- ✅ REST API for client interaction
- ✅ DuckDB integration for result storage

### Advanced Features

- ✅ Hardware-aware task scheduling
- ✅ Priority-based queue management
- ✅ Task dependencies and DAG execution
- ✅ Resource-aware scheduling
- ✅ Real-time performance monitoring
- ✅ Database optimization for high throughput
- ✅ Worker failure detection and recovery
- ✅ Coordinator redundancy with automatic failover
- ✅ Secure communication (TLS)
- ✅ API authentication and authorization

## Installation

### Prerequisites

- Python 3.8 or higher
- aiohttp
- DuckDB
- psutil

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Coordinator

```bash
# Start a coordinator
python -m distributed_testing.coordinator --port 8080 --db-path ./coordinator.duckdb
```

### Starting a Worker

```bash
# Start a worker
python -m distributed_testing.worker --coordinator http://localhost:8080 --worker-id worker-1
```

### Submitting Tasks

```bash
# Submit a task
python -m distributed_testing.client submit-task \
  --coordinator http://localhost:8080 \
  --type benchmark \
  --model bert-base-uncased \
  --hardware cuda \
  --batch-sizes 1,2,4,8,16
```

### Querying Results

```bash
# Get results for a task
python -m distributed_testing.client get-results \
  --coordinator http://localhost:8080 \
  --task-id 12345
```

## High-Availability Deployment

For high-availability deployments, you can run a cluster of coordinator nodes with automatic failover:

```bash
# Start a coordinator with redundancy enabled
python -m distributed_testing.coordinator \
  --port 8080 \
  --db-path ./coordinator1.duckdb \
  --node-id node-1 \
  --enable-redundancy \
  --peers localhost:8081,localhost:8082

# Start a second coordinator
python -m distributed_testing.coordinator \
  --port 8081 \
  --db-path ./coordinator2.duckdb \
  --node-id node-2 \
  --enable-redundancy \
  --peers localhost:8080,localhost:8082

# Start a third coordinator
python -m distributed_testing.coordinator \
  --port 8082 \
  --db-path ./coordinator3.duckdb \
  --node-id node-3 \
  --enable-redundancy \
  --peers localhost:8080,localhost:8081
```

For an easier setup, use the provided script:

```bash
./distributed_testing/examples/high_availability_cluster.sh start
```

## Monitoring and Management

### Cluster Health Monitor

Monitor the health of your coordinator cluster:

```bash
python -m distributed_testing.monitoring.cluster_health_monitor \
  --nodes localhost:8080,localhost:8081,localhost:8082
```

### Recovery Strategies

Automatically detect and recover from failures:

```bash
python -m distributed_testing.monitoring.recovery_strategies \
  --config cluster_config.json \
  --daemon
```

### Performance Benchmarks

Benchmark different cluster configurations:

```bash
python -m distributed_testing.examples.benchmark.benchmark_redundancy \
  --cluster-sizes 1,3,5 \
  --operations 1000 \
  --runs 3
```

## Development

### Running Tests

```bash
# Run unit tests
python -m unittest discover -s distributed_testing/tests

# Run failover tests
python -m distributed_testing.tests.test_coordinator_failover
```

### Building Documentation

```bash
# Generate documentation
make docs
```

## Implementation Status

The framework is implemented in multiple phases:

- ✅ **Phase 1: Core Functionality** - COMPLETED
- ✅ **Phase 2: Advanced Scheduling** - COMPLETED
- ✅ **Phase 3: Performance and Monitoring** - COMPLETED
- ✅ **Phase 4: Scalability** - COMPLETED
- ✅ **Phase 5: Fault Tolerance** - COMPLETED
  - Coordinator Redundancy with consensus-based leader election
  - Distributed State Management with transaction-based updates
  - Comprehensive Error Recovery Strategies with categorized handling
- 🔲 **Phase 6: Monitoring Dashboard** - DEFERRED
- 🔲 **Phase 7: Security and Access Control** - DEFERRED
- ✅ **Phase 8: Integration and Extensibility** - COMPLETED
  - ✅ Plugin Architecture for extending framework functionality
  - ✅ WebGPU/WebNN Resource Pool Integration with fault tolerance
  - ✅ CI/CD Integration with popular pipeline tools
  - ✅ External system integrations using standardized interfaces

## Documentation

For more detailed documentation, refer to:

- [Distributed Testing Design](../DISTRIBUTED_TESTING_DESIGN.md): Overall framework design
- [Fault Tolerance](README_FAULT_TOLERANCE.md): Comprehensive fault tolerance features
- [Auto Recovery](README_AUTO_RECOVERY.md): Auto recovery system details
- [Coordinator Redundancy](docs/COORDINATOR_REDUNDANCY.md): Redundancy implementation details
- [Plugin Architecture](README_PLUGIN_ARCHITECTURE.md): Plugin system for extending functionality
- [WebGPU/WebNN Resource Pool](README_WEBGPU_RESOURCE_POOL.md): Browser-based resource pool integration
- [Integration & Extensibility](docs/INTEGRATION_GUIDE.md): External system integration
- [Deployment Guide](docs/deployment_guide.md): Deployment instructions
- [API Reference](docs/api_reference.md): API documentation
- [Security Guide](SECURITY.md): Security features and best practices
- [Implementation Status](docs/IMPLEMENTATION_STATUS.md): Current development status and roadmap

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- The Raft consensus algorithm for coordinator redundancy
- The DuckDB team for their excellent database engine
- The aiohttp community for their robust async framework