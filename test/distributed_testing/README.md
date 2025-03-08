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

## Security Features

The framework includes comprehensive security features to ensure secure communication between coordinator and workers:

- **API Key Authentication**: Secure initial registration with API keys
- **JWT Token Authentication**: Ongoing secure communication with short-lived tokens
- **Message Signing**: All WebSocket messages are signed with HMAC
- **Role-Based Access Control**: Different permission levels for workers and admins
- **Secure Configuration Storage**: Security settings stored in a configuration file

For detailed information, see [SECURITY.md](./SECURITY.md).

## Usage

### Running the Coordinator

The coordinator server manages worker nodes, distributes tasks, and aggregates results.

```bash
# Start coordinator with security enabled
python coordinator.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb --security-config ./security_config.json

# Generate new API keys for admin and worker nodes
python coordinator.py --generate-admin-key --generate-worker-key --security-config ./security_config.json
```

Options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8080)
- `--db-path`: Path to the DuckDB database (default: ./benchmark_db.duckdb)
- `--security-config`: Path to security configuration file (default: ./security_config.json)
- `--generate-admin-key`: Generate a new admin API key
- `--generate-worker-key`: Generate a new worker API key

### Running a Worker

Workers connect to the coordinator, execute tasks, and report results.

```bash
# Connect to coordinator with API key authentication
python worker.py --coordinator http://localhost:8080 --api-key YOUR_API_KEY

# Connect to coordinator with token authentication (after obtaining a token)
python worker.py --coordinator http://localhost:8080 --token YOUR_JWT_TOKEN

# Connect with token stored in a file
python worker.py --coordinator http://localhost:8080 --token-file /path/to/token.txt
```

Options:
- `--coordinator`: URL of the coordinator server (required)
- `--hostname`: Hostname of the worker node (default: system hostname)
- `--db-path`: Path to the DuckDB database (optional)
- `--worker-id`: Worker ID (default: generated UUID)
- `--api-key`: API key for authentication with coordinator
- `--token`: JWT token for authentication (alternative to API key)
- `--token-file`: Path to file containing JWT token

### Running the Test Runner

The test runner helps run and test the distributed testing framework components.

```bash
# Run both coordinator and workers for testing with security
python run_test.py --mode all --db-path ./test_db.duckdb --security-config ./test_security_config.json

# Run coordinator only with security
python run_test.py --mode coordinator --security-config ./test_security_config.json --generate-keys

# Run worker with API key
python run_test.py --mode worker --coordinator http://localhost:8080 --api-key YOUR_API_KEY
```

Options:
- `--mode`: Which component(s) to run (`coordinator`, `worker`, or `all`) (default: `all`)
- `--db-path`: Path to the DuckDB database (default: ./test_db.duckdb)
- `--host`: Host for coordinator (default: localhost)
- `--port`: Port for coordinator (default: 8080)
- `--coordinator`: URL of coordinator (for worker mode)
- `--worker-id`: Worker ID (for worker mode)
- `--api-key`: API key for worker authentication
- `--security-config`: Path to security configuration file
- `--num-workers`: Number of workers to start (for all mode) (default: 2)
- `--run-time`: How long to run the test in seconds (for all mode) (default: 60)
- `--generate-keys`: Generate new API keys for testing

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

- ✅ Phase 2: Security and Worker Management (May 15-22, 2025)
  - ✅ Security module with API key and JWT token authentication
  - ✅ Message signing and verification
  - ✅ Role-based access control
  - ✅ Security configuration persistence
  - ✅ Integration of security module with coordinator and worker
  - ⏳ Health monitoring and status tracking
  - ⏳ Auto-recovery mechanisms

- 🔲 Phase 3: Intelligent Task Distribution (May 22-29, 2025)
- 🔲 Phase 4: Adaptive Load Balancing (May 29-June 5, 2025)
- 🔲 Phase 5: Fault Tolerance (June 5-12, 2025)
- 🔲 Phase 6: Monitoring Dashboard (June 12-19, 2025)

## Security Configuration

The security configuration is stored in a JSON file (`security_config.json`) with the following structure:

```json
{
  "secret_key": "random_secret_key_for_signing_tokens",
  "token_expiry": 3600,
  "required_roles": ["worker"],
  "api_keys": {
    "api_key_1": {
      "name": "admin-user",
      "roles": ["admin"],
      "created": "2025-05-15T10:00:00"
    },
    "api_key_2": {
      "name": "worker-node",
      "roles": ["worker"],
      "created": "2025-05-15T10:00:00"
    }
  }
}
```

## Authentication Flow

1. **Initial Authentication**:
   - Worker connects to coordinator WebSocket
   - Worker sends authentication message with API key
   - Coordinator validates API key and generates JWT token
   - Coordinator sends token to worker

2. **Ongoing Communication**:
   - All messages are signed with HMAC for integrity
   - Messages include timestamps to prevent replay attacks
   - Worker token is refreshed periodically

## Next Steps

1. Complete remaining Phase 2 implementation:
   - Implement health monitoring and status tracking
   - Add auto-recovery mechanisms
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