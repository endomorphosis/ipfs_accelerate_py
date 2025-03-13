# Distributed Testing Framework

A high-performance distributed testing system that enables parallel execution of benchmarks and tests across multiple machines with heterogeneous hardware. This framework provides intelligent workload distribution and centralized result aggregation.

![Combined Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/combined-status.json)
![Integration Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/integration-status.json)
![Fault Tolerance Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/fault-status.json)
![Monitoring Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/monitoring-status.json)
![Stress Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/stress-status.json)
![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/coverage.json)

## New Features (March 17, 2025)

The framework has been enhanced with the following new features:

1. **Comprehensive Monitoring Dashboard** ✅ (NEW): A full-featured web-based monitoring system with:
   - Real-time monitoring of distributed worker nodes and task execution
   - Interactive system topology visualization with drill-down capabilities
   - Live metrics visualization with WebSocket-based updates
   - Hardware utilization tracking across the distributed system
   - Fault tolerance monitoring and visualization
   - Integrated alert system with multiple severity levels
   - Historical trend analysis with interactive charts
   - Customizable dashboards with theme support (light/dark)
   - Comprehensive API for programmatic access to monitoring data

2. **Hardware-Aware Fault Tolerance System** ✅: Advanced fault tolerance with hardware-specific strategies:
   - Specialized recovery strategies for different hardware types (CPU, GPU, TPU, WebGPU/WebNN)
   - Machine learning-based failure pattern detection and prevention
   - Intelligent retry policies with exponential backoff and jitter
   - Task state persistence and recovery mechanisms
   - Checkpoint and resume support for long-running tasks
   - Comprehensive visualization and reporting system
   - Integration with heterogeneous hardware scheduler

3. **Intelligent Result Aggregation System**: Advanced result aggregation and analysis pipeline with:
   - Comprehensive statistical analysis of test results
   - Automated performance regression detection
   - Multi-dimensional result analysis across hardware, models, and configurations
   - Customizable visualization and reporting
   - Integration with the coordinator for real-time result processing

4. **Comprehensive CI/CD Integration** ✅: Complete GitHub Actions, GitLab CI, and Jenkins support with:
   - Automatic test discovery and hardware requirement analysis
   - Parallel test execution across test types (integration, fault tolerance, monitoring, stress)
   - Advanced report generation in multiple formats (JSON, Markdown, HTML)
   - Status badge system for real-time test status visualization
   - Coverage tracking and reporting
   - Secure credential management for coordinator access

5. **Coordinator Redundancy and Failover**: Multiple coordinator instances can now operate in a high-availability configuration with automatic leader election and state synchronization.

6. **Performance Trend Analysis**: Long-term performance trend tracking with statistical analysis, anomaly detection, and predictive forecasting for worker and task performance.

7. **Time-Series Data Storage**: Historical performance data storage with efficient querying and aggregation capabilities.

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
- **Intelligent Result Aggregation**: Advanced statistical analysis and reporting of distributed test results
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

### Result Aggregator

The result aggregator provides advanced analysis of test results:

- Comprehensive statistical analysis of distributed test results
- Multi-dimensional analysis across hardware, models, and configurations
- Performance regression detection with statistical significance testing
- Automated identification of optimal configurations
- Visualizations and reports of aggregated results
- Integration with the coordinator for real-time result processing

### CI/CD Integration

The CI/CD integration component enables automated testing in CI/CD pipelines:

- Automatic test discovery and submission 
- Test requirements analysis and hardware matching
- Integration with GitHub Actions, GitLab CI, and Jenkins
- Comprehensive test result reporting
- Support for parallel test execution
- Report generation in multiple formats (JSON, MD, HTML)

### Monitoring Dashboard

The comprehensive monitoring dashboard provides real-time system visualization and monitoring:

- Real-time worker node monitoring with status indicators and metric tracking
- Interactive system topology visualization with drill-down capabilities
- Live task execution tracking with progress visualization
- Hardware utilization charts across the distributed system
- Fault tolerance monitoring with recovery action tracking
- Integrated alert system with multiple severity levels (critical, warning, info)
- Historical trend analysis with interactive charts
- WebSocket-based live updates for real-time metrics
- Customizable dashboard with light/dark themes
- Comprehensive API for programmatic access to monitoring data
- Mobile-responsive design for monitoring on any device

### Dashboard Server

The web-based dashboard server manages the monitoring dashboard:

- Serves interactive HTML dashboards with responsive design
- Provides REST API for accessing system data
- Implements WebSocket server for real-time updates
- Generates visualizations of system metrics
- Integrates with result aggregator for test performance data
- Monitors system health and generates alerts
- Tracks performance trends and anomalies
- Provides detailed views of workers, tasks, and system components

## Implementation Status

- [x] Phase 1: Core Infrastructure (March 12, 2025)
- [x] Phase 2: Security and Worker Management (March 12, 2025)
- [x] Phase 3: Intelligent Task Distribution (March 16, 2025)
- [x] Phase 4: Adaptive Load Balancing (March 16, 2025)
- [x] Phase 5: Fault Tolerance (March 13, 2025) ✅
- [x] Phase 6: Monitoring Dashboard (March 17, 2025) ✅
- [x] Phase 7: Coordinator Redundancy (March 10, 2025)
- [x] Phase 8: Performance Trend Analysis (March 10, 2025)
- [x] Phase 9: CI/CD Integration (March 13, 2025) ✅
- [✅] Phase 10: Intelligent Result Aggregation (March 17, 2025 - COMPLETED)

Target completion for all phases: June 26, 2025 (currently ahead of schedule)

## CI/CD Integration Guide

The CI/CD integration enables automated testing of the Distributed Testing Framework in continuous integration pipelines. It supports GitHub Actions, GitLab CI, and Jenkins with comprehensive test discovery, requirement analysis, result reporting, and badge generation.

### GitHub Actions Integration

The framework includes a GitHub Actions workflow file (`.github/workflows/distributed-testing.yml`) that runs all test types in parallel:

1. **Integration Tests**: Validate component interactions using the `run_integration_tests.py` script
2. **Fault Tolerance Tests**: Verify system behavior under failure conditions using `test_load_balancer_fault_tolerance.py`
3. **Monitoring Tests**: Test monitoring components using `test_load_balancer_monitoring.py`
4. **Stress Tests**: Validate system performance under load using `test_load_balancer_stress.py`

The workflow generates test reports, coverage data, and status badges automatically.

### Status Badges

Status badges are automatically generated and updated for each test type:

- **Combined Status**: Overall test status across all test types
- **Integration Tests**: Status of integration test suite
- **Fault Tolerance Tests**: Status of fault tolerance test suite
- **Monitoring Tests**: Status of monitoring test suite
- **Stress Tests**: Status of stress test suite
- **Coverage**: Code coverage percentage across all tests

These badges are displayed at the top of this README and are updated automatically after each test run.

### Running Tests Locally

To run tests locally before committing:

```bash
# Run all test types
./run_all_tests.sh

# Run a specific test type
./run_all_tests.sh --type integration

# Run tests with a filter
./run_all_tests.sh --type fault --filter test_worker_failure

# Run with verbose output
./run_all_tests.sh --verbose
```

### Running the Monitoring Dashboard

To start the comprehensive monitoring dashboard:

```bash
# Start the monitoring dashboard with default settings
python run_monitoring_dashboard.py

# Start with custom host and port
python run_monitoring_dashboard.py --host 0.0.0.0 --port 8085

# Connect to a specific coordinator
python run_monitoring_dashboard.py --coordinator http://coordinator-server:8080

# Use dark theme with 10-second refresh
python run_monitoring_dashboard.py --theme dark --refresh 10

# Open in browser automatically
python run_monitoring_dashboard.py --browser

# Enable real-time updates via WebSockets
python run_monitoring_dashboard.py --real-time

# Disable alerts
python run_monitoring_dashboard.py --no-alerts

# Connect to a specific database
python run_monitoring_dashboard.py --db-path ./my_benchmark_db.duckdb

# Specify time range for result aggregation (in days)
python run_monitoring_dashboard.py --time-range 14

# Disable result aggregator integration
python run_monitoring_dashboard.py --disable-aggregator

# Enable debug logging
python run_monitoring_dashboard.py --debug
```

### Dashboard Features

The monitoring dashboard provides comprehensive real-time visibility into the distributed testing framework:

1. **System Overview**: Real-time status of all system components including workers, tasks, and resources
2. **Worker Monitoring**: Detailed view of all worker nodes with status, capabilities, and current workload
3. **Task Execution**: Live tracking of task execution across the distributed system
4. **Performance Metrics**: Real-time performance metrics and historical trends
5. **Test Results Analysis**: Comprehensive visualizations of test results including:
   - Performance trends across different model-hardware pairs
   - Compatibility matrices showing which models work with which hardware
   - Integration test pass rates by test module
   - Web platform success rates across browsers and platforms
   - Performance anomaly detection and visualization
   - Historical performance comparisons with improvement/regression tracking
6. **Alerts**: Real-time alerting for critical events with severity levels
7. **System Topology**: Interactive visualization of the distributed system structure
8. **Fault Tolerance**: Monitoring of failure patterns and recovery actions

### Docker Testing Environment

A Docker Compose configuration is provided for running tests in an isolated environment:

```bash
# Start testing environment
docker-compose -f docker-compose.test.yml up

# Run tests in Docker environment
./run_docker_tests.sh
```

The Docker environment includes:
- Coordinator container (with dashboard)
- CPU worker container
- GPU worker container (simulated)
- WebGPU worker container (simulated)
- Tester container for executing test suites

### Using the CI/CD Integration API

For custom CI/CD integrations, use the `cicd_integration.py` module:

```python
# Submit tests to a coordinator
python -m duckdb_api.distributed_testing.cicd_integration \
    --provider github \
    --coordinator http://coordinator-url:8080 \
    --api-key KEY \
    --test-dir ./tests \
    --output-dir ./reports \
    --report-formats json md html
```

The module automatically detects test requirements based on file content, submits tests to the coordinator, waits for results, and generates comprehensive reports.

### Troubleshooting CI/CD Issues

If tests fail in CI/CD pipelines:

1. Check the detailed reports in the "test-results-*" artifacts
2. Examine coordinator and worker logs for error messages
3. Verify that the coordinator URL and API key are correctly configured
4. Check if the tests require specific hardware that's not available in the CI environment
5. For timeout errors, increase the timeout value in the workflow configuration