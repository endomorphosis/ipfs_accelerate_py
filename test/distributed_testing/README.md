# Distributed Testing Framework

This is a high-performance distributed testing framework for the IPFS Accelerate Python package. It enables parallel execution of benchmarks and tests across multiple machines with heterogeneous hardware, with intelligent workload distribution and result aggregation.

## Components

The distributed testing framework consists of the following components:

1. **Coordinator**: Manages worker nodes, distributes tasks, and aggregates results
2. **Worker**: Executes tasks and reports results back to the coordinator
3. **DuckDB Integration**: Stores test results in a DuckDB database
4. **Test Runner**: Helps run and test the distributed testing framework
5. **Security Manager**: Provides authentication, authorization, and secure communication

## Requirements

- Python 3.7+
- Dependencies: `aiohttp`, `websockets`, `duckdb`, `psutil`, `torch` (optional for GPU detection), `pyjwt` (for security features)

Install dependencies:
```bash
pip install aiohttp websockets duckdb psutil torch pyjwt
```

## Usage

### Running the Coordinator

The coordinator server manages worker nodes, distributes tasks, and aggregates results.

```bash
python coordinator.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb
```

Options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8080)
- `--db-path`: Path to the DuckDB database (default: ./benchmark_db.duckdb)

### Running a Worker

Workers connect to the coordinator, execute tasks, and report results.

```bash
python worker.py --coordinator http://localhost:8080 --db-path ./benchmark_db.duckdb
```

Options:
- `--coordinator`: URL of the coordinator server (required)
- `--hostname`: Hostname of the worker node (default: system hostname)
- `--db-path`: Path to the DuckDB database (optional)
- `--worker-id`: Worker ID (default: generated UUID)

### Running the Test Runner

The test runner helps run and test the distributed testing framework components.

```bash
# Run both coordinator and workers for testing
python run_test.py --mode all --db-path ./test_db.duckdb --num-workers 2 --run-time 60

# Run coordinator only
python run_test.py --mode coordinator --db-path ./test_db.duckdb

# Run worker only
python run_test.py --mode worker --coordinator http://localhost:8080
```

Options:
- `--mode`: Which component(s) to run (`coordinator`, `worker`, or `all`) (default: `all`)
- `--db-path`: Path to the DuckDB database (default: ./test_db.duckdb)
- `--host`: Host for coordinator (default: localhost)
- `--port`: Port for coordinator (default: 8080)
- `--coordinator`: URL of coordinator (for worker mode)
- `--worker-id`: Worker ID (for worker mode)
- `--num-workers`: Number of workers to start (for all mode) (default: 2)
- `--run-time`: How long to run the test in seconds (for all mode) (default: 60)

## API Endpoints

The coordinator exposes the following API endpoints:

### Worker Management

- `GET /api/workers`: List all registered workers
- `GET /api/workers/{worker_id}`: Get information about a specific worker
- `POST /api/workers/register`: Register a new worker

### Task Management

- `GET /api/tasks`: List all tasks
- `GET /api/tasks/{task_id}`: Get information about a specific task
- `POST /api/tasks`: Create a new task
- `POST /api/tasks/{task_id}/cancel`: Cancel a running task

### Status

- `GET /status`: Get coordinator status information

## Task Types

The framework supports the following task types:

1. **Benchmark Tasks**: Execute model benchmarks with specified configurations
```json
{
  "type": "benchmark",
  "priority": 1,
  "config": {
    "model": "bert-base-uncased",
    "batch_sizes": [1, 2, 4, 8, 16],
    "precision": "fp16",
    "iterations": 100
  },
  "requirements": {
    "hardware": ["cuda"],
    "min_memory_gb": 8,
    "min_cuda_compute": 7.5
  }
}
```

2. **Test Tasks**: Execute test files
```json
{
  "type": "test",
  "priority": 1,
  "config": {
    "test_file": "/path/to/test.py",
    "test_args": ["--verbose", "--no-cache"]
  },
  "requirements": {
    "hardware": ["cpu"],
    "min_memory_gb": 4
  }
}
```

3. **Custom Tasks**: Execute custom tasks
```json
{
  "type": "custom",
  "priority": 1,
  "config": {
    "name": "custom-task",
    "args": ["arg1", "arg2"]
  },
  "requirements": {
    "hardware": ["cpu"],
    "min_memory_gb": 2
  }
}
```

## DuckDB Schema

The framework uses the following DuckDB schema to store data:

### Worker Nodes Table

```sql
CREATE TABLE worker_nodes (
    worker_id VARCHAR PRIMARY KEY,
    hostname VARCHAR,
    registration_time TIMESTAMP,
    last_heartbeat TIMESTAMP,
    status VARCHAR,
    capabilities JSON,
    hardware_metrics JSON,
    tags JSON
)
```

### Distributed Tasks Table

```sql
CREATE TABLE distributed_tasks (
    task_id VARCHAR PRIMARY KEY,
    type VARCHAR,
    priority INTEGER,
    status VARCHAR,
    create_time TIMESTAMP,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    worker_id VARCHAR,
    attempts INTEGER,
    config JSON,
    requirements JSON
)
```

### Task Execution History Table

```sql
CREATE TABLE task_execution_history (
    id INTEGER PRIMARY KEY,
    task_id VARCHAR,
    worker_id VARCHAR,
    attempt INTEGER,
    status VARCHAR,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    execution_time_seconds FLOAT,
    error_message VARCHAR,
    hardware_metrics JSON
)
```

## Implementation Progress

The current implementation includes:

- ✅ Phase 1: Core Infrastructure (May 8-15, 2025)
  - ✅ Basic coordinator server with WebSocket API
  - ✅ Worker registration and capability tracking
  - ✅ Simple task distribution logic
  - ✅ Basic result aggregation
  - ✅ Worker client with auto-registration
  - ✅ Hardware capability reporting
  - ✅ Basic task execution
  - ✅ Result reporting

- ⏳ Phase 2: Security and Worker Management (May 15-22, 2025)
  - ✅ Security module with API key and JWT token authentication
  - ✅ Message signing and verification
  - ✅ Role-based access control
  - ✅ Security configuration persistence
  - ⏳ Integration of security module with coordinator and worker
  - ⏳ Health monitoring and status tracking
  - ⏳ Auto-recovery mechanisms

- 🔲 Phase 3: Intelligent Task Distribution (May 22-29, 2025)
- 🔲 Phase 4: Adaptive Load Balancing (May 29-June 5, 2025)
- 🔲 Phase 5: Fault Tolerance (June 5-12, 2025)
- 🔲 Phase 6: Monitoring Dashboard (June 12-19, 2025)

## Next Steps

1. Complete Phase 2 implementation
2. Add real benchmark and test execution (currently simulated)
3. Implement advanced task distribution algorithms
4. Add comprehensive fault tolerance
5. Create monitoring dashboard
6. Integrate with CI/CD pipeline

## Documentation

- [README.md](./README.md): Main documentation for the distributed testing framework
- [DISTRIBUTED_TESTING_DESIGN.md](../DISTRIBUTED_TESTING_DESIGN.md): Detailed design document
- [SECURITY.md](./SECURITY.md): Documentation for security implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.