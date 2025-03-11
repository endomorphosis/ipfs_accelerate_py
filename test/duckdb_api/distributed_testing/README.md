# Distributed Testing Framework

A high-performance distributed testing system that enables parallel execution of benchmarks and tests across multiple machines with heterogeneous hardware. This framework provides intelligent workload distribution and centralized result aggregation.

## New Features (March 2025)

The framework has been enhanced with the following new features:

1. **Coordinator Redundancy and Failover**: Multiple coordinator instances can now operate in a high-availability configuration with automatic leader election and state synchronization.

2. **Performance Trend Analysis**: Long-term performance trend tracking with statistical analysis, anomaly detection, and predictive forecasting for worker and task performance.

3. **Advanced Fault Tolerance**: Enhanced recovery mechanisms for worker and coordinator failures with automatic state recovery.

4. **Visualization System**: Visual representations of performance metrics and trends with interactive charts and reports.

5. **Time-Series Data Storage**: Historical performance data storage with efficient querying and aggregation capabilities.

6. **CI/CD Integration**: Seamless integration with GitHub Actions, GitLab CI, and Jenkins pipelines for automated distributed testing and reporting.

## Features

- **Coordinator-Worker Architecture**: Central coordinator server distributes tasks to worker nodes
- **DuckDB Integration**: Centralized storage of distributed test results
- **Security**: Comprehensive JWT-based authentication and message signing
- **Intelligent Task Distribution**: Routes tasks to worker nodes with appropriate hardware
- **Resource Monitoring**: Tracks worker node health, capabilities, and resource usage
- **Fault Tolerance**: Automatic task retry and worker node recovery
- **Scalability**: Supports dynamic addition and removal of worker nodes
- **Real-time Dashboard**: Web-based monitoring and management interface
- **High Availability**: Support for multiple coordinators with automatic leader election and failover
- **Performance Analytics**: Long-term trend analysis and anomaly detection for performance metrics
- **CI/CD Integration**: Support for GitHub Actions, GitLab CI, and Jenkins with automated test discovery, requirement analysis, and reporting

## Components

### Coordinator Server

The coordinator server is responsible for:

- Managing worker node registration and capabilities
- Distributing test tasks based on worker capabilities
- Monitoring worker health and task execution
- Handling result aggregation and storage
- Providing administration API for test orchestration
- Participating in leader election for high availability (when auto recovery is enabled)

### Worker Nodes

Worker nodes are responsible for:

- Self-registration with the coordinator
- Reporting hardware capabilities and status
- Executing assigned test tasks
- Reporting results and execution metrics
- Implementing local caching for efficient testing

### Task Scheduler

The task scheduler implements advanced task distribution algorithms:

- Hardware-aware task assignment
- Test-specific requirements matching
- Priority-based scheduling
- Workload balancing across workers

### Load Balancer

The adaptive load balancer optimizes resource utilization:

- Dynamic worker capability reassessment
- Real-time performance monitoring
- Workload redistribution based on performance
- Automatic task migration between workers

### Health Monitor

The health monitoring system ensures reliable operation:

- Worker node health monitoring and status tracking
- Failure detection and handling
- Automatic recovery mechanisms
- Performance anomaly detection

### Auto Recovery System

The auto recovery system provides high availability:

- Coordinator redundancy with automatic leader election
- State replication between coordinator instances
- Automatic failover when the leader fails
- Task state persistence and recovery during failover
- Worker reassignment during coordinator transitions

### Performance Trend Analyzer

The performance trend analyzer tracks and analyzes performance data:

- Long-term time series tracking of performance metrics
- Statistical trend analysis with significance testing
- Anomaly detection based on statistical methods
- Performance prediction and forecasting
- Visualization of performance trends and anomalies
- Comprehensive reporting with actionable insights

### CI/CD Integration

The CI/CD integration component enables automated testing in CI/CD pipelines:

- Automatic test discovery and submission 
- Test requirements analysis and hardware matching
- Integration with GitHub Actions, GitLab CI, and Jenkins
- Comprehensive test result reporting
- Support for parallel test execution
- Report generation in multiple formats (JSON, MD, HTML)

### Dashboard Server

The web-based dashboard provides real-time monitoring:

- Worker status visualization
- Task execution status and history
- System health metrics and alerts
- Performance metrics and statistics
- Coordinator cluster status
- Performance trend visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/example/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Coordinator Server

```bash
# Start the coordinator server on default port (8080)
python -m duckdb_api.distributed_testing.coordinator --host 0.0.0.0

# Start with a specific database file
python -m duckdb_api.distributed_testing.coordinator --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb

# Generate a worker API key
python -m duckdb_api.distributed_testing.coordinator --generate-worker-key --security-config ./security_config.json

# Start with auto recovery (high availability)
python -m duckdb_api.distributed_testing.coordinator --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb --auto-recovery --coordinator-id coordinator-1

# Start with performance analyzer
python -m duckdb_api.distributed_testing.coordinator --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb --performance-analyzer --visualization-path ./visualizations

# Generate a performance report
python -m duckdb_api.distributed_testing.coordinator --db-path ./benchmark_db.duckdb --performance-analyzer --report --report-output performance_report.html
```

### Running Worker Nodes

```bash
# Start a worker node
python -m duckdb_api.distributed_testing.worker --coordinator http://localhost:8080 --api-key YOUR_API_KEY

# Start a worker with a specific ID
python -m duckdb_api.distributed_testing.worker --coordinator http://localhost:8080 --api-key YOUR_API_KEY --worker-id worker1

# Start a worker with a specific working directory
python -m duckdb_api.distributed_testing.worker --coordinator http://localhost:8080 --api-key YOUR_API_KEY --work-dir ./worker_tasks
```

### Running the Dashboard

```bash
# Start the dashboard server
python -m duckdb_api.distributed_testing.dashboard_server --host 0.0.0.0 --port 8081 --coordinator-url http://localhost:8080

# Start and automatically open in browser
python -m duckdb_api.distributed_testing.dashboard_server --host localhost --port 8081 --coordinator-url http://localhost:8080 --auto-open
```

### Integrated Testing Environment

For easy testing, you can use the run_test.py script to start all components:

```bash
# Start all components (coordinator, workers, dashboard)
python -m duckdb_api.distributed_testing.run_test --mode all --host localhost

# Start with more workers
python -m duckdb_api.distributed_testing.run_test --mode all --host localhost --worker-count 4

# Start with a specific database
python -m duckdb_api.distributed_testing.run_test --mode all --host localhost --db-path ./benchmark_db.duckdb
```

### Submitting Tasks

```bash
# Submit a test task
python -m duckdb_api.distributed_testing.run_test --mode client --coordinator http://localhost:8080 --test-file /path/to/test_file.py

# Submit with arguments and priority
python -m duckdb_api.distributed_testing.run_test --mode client --coordinator http://localhost:8080 --test-file /path/to/test_file.py --test-args "--verbose --batch-size 4" --priority 1
```

### Using CI/CD Integration

The framework includes ready-to-use integration with common CI/CD systems:

```bash
# GitHub Actions integration
python -m duckdb_api.distributed_testing.cicd_integration --provider github \
  --coordinator http://coordinator-url:8080 --api-key YOUR_API_KEY \
  --test-dir ./tests --output-dir ./test_reports --report-formats json md html

# GitLab CI integration
python -m duckdb_api.distributed_testing.cicd_integration --provider gitlab \
  --coordinator http://coordinator-url:8080 --api-key YOUR_API_KEY \
  --test-pattern "test_*.py" --output-dir ./test_reports

# Jenkins integration
python -m duckdb_api.distributed_testing.cicd_integration --provider jenkins \
  --coordinator http://coordinator-url:8080 --api-key YOUR_API_KEY \
  --test-files test_file1.py test_file2.py --output-dir ./test_reports
```

Example files for each CI/CD system are available in the `examples/` directory:
- `github_workflow.yml`: Ready-to-use GitHub Actions workflow
- `gitlab-ci.yml`: Complete GitLab CI configuration
- `Jenkinsfile`: Jenkins pipeline definition

The CI/CD integration automatically:
1. Discovers test files based on patterns or explicit lists
2. Analyzes test requirements (hardware, browsers, memory)
3. Submits tests to the coordinator with appropriate priorities
4. Monitors test execution with configurable timeouts
5. Generates comprehensive reports in multiple formats
6. Returns appropriate exit codes for CI/CD pipeline success/failure

## API Reference

### Coordinator API Endpoints

- `GET /api/status`: Get coordinator status
- `GET /api/workers`: Get list of registered workers
- `GET /api/workers/{worker_id}`: Get worker information
- `GET /api/tasks`: Get list of tasks
- `GET /api/tasks/{task_id}`: Get task information
- `POST /api/tasks`: Submit a new task
- `DELETE /api/tasks/{task_id}`: Cancel a task
- `GET /api/dashboard`: Get dashboard data

### WebSocket API

- `ws://{host}:{port}/ws`: WebSocket endpoint for worker communication
- `ws://{host}:{port}/ws/dashboard`: WebSocket endpoint for dashboard real-time updates

## Task Format

Tasks are defined in JSON format:

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
  },
  "timeout_seconds": 1800,
  "retry_policy": {
    "max_retries": 3,
    "retry_delay_seconds": 60
  },
  "dependencies": []
}
```

## Implementation Status

- [x] Phase 1: Core Infrastructure (March 12, 2025)
- [x] Phase 2: Security and Worker Management (March 12, 2025)
- [x] Phase 3: Intelligent Task Distribution (March 16, 2025)
- [x] Phase 4: Adaptive Load Balancing (March 16, 2025)
- [x] Phase 5: Fault Tolerance (March 10, 2025)
- [x] Phase 6: Monitoring Dashboard (March 12, 2025)
- [x] Phase 7: Coordinator Redundancy (March 10, 2025)
- [x] Phase 8: Performance Trend Analysis (March 10, 2025)
- [x] Phase 9: CI/CD Integration (March 15, 2025)

Implementation completed: March 15, 2025

## Architecture

The distributed testing framework follows a coordinator-worker architecture:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Coordinator  │     │  Database     │     │  Dashboard    │
│  Server       │◄────┤  (DuckDB)     │────►│  Server       │
└───────┬───────┘     └───────────────┘     └───────────────┘
        │
        │ (Secure WebSocket)
        │
┌───────┴───────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │
│  Worker Node  │     │  Worker Node  │     │  Worker Node  │
│  (Machine 1)  │     │  (Machine 2)  │     │  (Machine N)  │
│               │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.