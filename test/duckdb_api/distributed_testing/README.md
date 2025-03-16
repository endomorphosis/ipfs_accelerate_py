# Distributed Testing Framework

A high-performance distributed testing system that enables parallel execution of benchmarks and tests across multiple machines with heterogeneous hardware. This framework provides intelligent workload distribution and centralized result aggregation.

![Combined Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/combined-status.json)
![Integration Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/integration-status.json)
![Fault Tolerance Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/fault-status.json)
![Monitoring Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/monitoring-status.json)
![Stress Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/stress-status.json)
![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/coverage.json)

## New Features (March 20, 2025)

The framework has been enhanced with the following new features:

1. **Cross-Platform Worker Support** ✅ (NEW): A comprehensive cross-platform support system with:
   - Unified interface for Linux, Windows, macOS, and containers
   - Platform-specific hardware detection (CPU, memory, GPU, disk)
   - Deployment script generation for each supported platform
   - Path conversion utilities for cross-platform compatibility
   - Dependency management with platform-specific implementations
   - Comprehensive test suite with mocked platform support
   - See [CROSS_PLATFORM_WORKER_README.md](CROSS_PLATFORM_WORKER_README.md) for details

2. **Error Visualization System** ✅: A comprehensive error visualization system with:
   - Interactive dashboard for error monitoring and analysis
   - Hierarchical sound notifications (system-critical, critical, warning, info) with distinct audio signatures
   - Advanced system-critical alerts for highest-priority infrastructure failures
   - Automated error pattern detection to identify recurring issues
   - Worker error analysis with status indicators and statistics
   - Hardware-specific error analysis and visualization
   - Database integration for historical error tracking
   - Comprehensive testing suite with unit and end-to-end tests
   - Full integration with monitoring dashboard and error reporting
   - See [ERROR_VISUALIZATION_GUIDE.md](ERROR_VISUALIZATION_GUIDE.md) for details

3. **Enhanced Worker Error Reporting** ✅: A worker-side error reporting system with:
   - Comprehensive system context collection during error conditions
   - Detailed hardware metrics and status reporting
   - Error categorization aligned with coordinator error handling
   - Error frequency analysis and pattern detection
   - Integration with enhanced worker reconnection system
   - Error simulation capabilities for testing
   - Seamless integration with existing worker implementations
   - See [WORKER_ERROR_REPORTING_GUIDE.md](WORKER_ERROR_REPORTING_GUIDE.md) for details

4. **Enhanced Error Handling System** ✅: A comprehensive error handling framework with:
   - Standardized error categorization (resource, network, hardware, worker, test execution)
   - Intelligent retry policies with exponential backoff and jitter
   - Specialized recovery strategies for different error types
   - Error pattern detection through similarity analysis
   - Automatic recovery action execution
   - Hardware-aware error recovery
   - Seamless coordinator integration
   - Integration with monitoring dashboard for visualization
   - Comprehensive documentation and usage guide
   - See [ENHANCED_ERROR_HANDLING_GUIDE.md](ENHANCED_ERROR_HANDLING_GUIDE.md) for details

2. **Comprehensive Monitoring Dashboard** ✅: A full-featured web-based monitoring system with:
   - Real-time monitoring of distributed worker nodes and task execution
   - Interactive system topology visualization with drill-down capabilities
   - Live metrics visualization with WebSocket-based updates
   - Hardware utilization tracking across the distributed system
   - Fault tolerance monitoring and visualization
   - Integrated alert system with multiple severity levels
   - Historical trend analysis with interactive charts
   - Customizable dashboards with theme support (light/dark)
   - Comprehensive API for programmatic access to monitoring data

3. **Hardware-Aware Fault Tolerance System** ✅: Advanced fault tolerance with hardware-specific strategies:
   - Specialized recovery strategies for different hardware types (CPU, GPU, TPU, WebGPU/WebNN)
   - Machine learning-based failure pattern detection and prevention
   - Intelligent retry policies with exponential backoff and jitter
   - Task state persistence and recovery mechanisms
   - Checkpoint and resume support for long-running tasks
   - Comprehensive visualization and reporting system
   - Integration with heterogeneous hardware scheduler

4. **Intelligent Result Aggregation System** ✅: Advanced result aggregation and analysis pipeline with:
   - Comprehensive statistical analysis of test results
   - Automated performance regression detection
   - Multi-dimensional result analysis across hardware, models, and configurations
   - Customizable visualization and reporting
   - Integration with the coordinator for real-time result processing

5. **Comprehensive CI/CD Integration** ✅: Complete GitHub Actions, GitLab CI, and Jenkins support with:
   - Automatic test discovery and hardware requirement analysis
   - Parallel test execution across test types (integration, fault tolerance, monitoring, stress)
   - Advanced report generation in multiple formats (JSON, Markdown, HTML)
   - Status badge system for real-time test status visualization
   - Coverage tracking and reporting
   - Secure credential management for coordinator access

6. **Coordinator Redundancy and Failover**: Multiple coordinator instances can now operate in a high-availability configuration with automatic leader election and state synchronization.

7. **Performance Trend Analysis**: Long-term performance trend tracking with statistical analysis, anomaly detection, and predictive forecasting for worker and task performance.

8. **Time-Series Data Storage**: Historical performance data storage with efficient querying and aggregation capabilities.

## Features

- **Coordinator-Worker Architecture**: Central coordinator server distributes tasks to worker nodes
- **DuckDB Integration**: Centralized storage of distributed test results
- **Cross-Platform Support** ✅ (NEW): Worker nodes that run on Linux, Windows, macOS, and containers
- **Security**: Comprehensive JWT-based authentication and message signing
- **Intelligent Task Distribution**: Routes tasks to worker nodes with appropriate hardware
- **Resource Monitoring**: Tracks worker node health, capabilities, and resource usage
- **Fault Tolerance**: Automatic task retry and worker node recovery
- **Scalability**: Supports dynamic addition and removal of worker nodes
- **Real-time Dashboard**: Web-based monitoring and management interface
- **High Availability**: Support for multiple coordinators with automatic leader election and failover
- **Enhanced Error Handling**: Comprehensive error handling framework with specialized recovery strategies
- **Error Reporting**: Detailed worker-side error reporting with system context collection
- **Error Visualization**: Interactive dashboard for error monitoring, pattern detection, and analysis with hierarchical sound notifications
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

### Error Visualization System

The error visualization system provides comprehensive error monitoring and analysis:

- Interactive error summary dashboard with counts by category and severity
- Hierarchical sound notification system with four distinct severity levels:
  - System-critical: Highest-priority alerts for coordinator failure, database corruption, and security issues
  - Critical: High-priority alerts for hardware issues, resource allocation failures, and worker crashes
  - Warning: Medium-priority alerts for network issues, timeout errors, and resource cleanup issues
  - Info: Low-priority notifications for test execution errors and other non-critical issues
- Specialized sound design with acoustic properties matched to error urgency:
  - System-critical: Rising frequency pattern (880Hz → 1046.5Hz → 1318.5Hz) with accelerating pulse rate
  - Critical: Higher frequency (880Hz/440Hz) with consistent pulsing effect
  - Warning: Medium frequency (660Hz/330Hz) with moderate decay
  - Info: Lower frequency (523Hz) with quick decay
- Error distribution visualization with interactive charts and filtering
- Automated error pattern detection to identify recurring issues
- Worker-specific error analysis with status indicators and statistics
- Hardware-specific error analysis to identify problematic hardware types
- Detailed error context information with system metrics at error time
- Time range selection for historical error analysis (1h, 6h, 24h, 7d)
- Database integration for long-term error tracking and historical analysis
- Pattern similarity analysis using advanced text processing techniques
- Critical error highlighting for urgent attention
- Visual indicators for error severity with color coding and animations
- User-controlled notification settings (volume, mute)
- Supporting infrastructure for end-to-end testing with simulated errors
- Comprehensive API for programmatic access to error data
- Integration with worker error reporting and enhanced error handling

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
- [x] Phase 9: Enhanced Error Handling (March 18, 2025) ✅
- [x] Phase 10: Enhanced Worker Error Reporting (March 18, 2025) ✅
- [x] Phase 11: Error Visualization System (March 19, 2025) ✅
- [x] Phase 12: CI/CD Integration (March 13, 2025) ✅
- [✅] Phase 13: Intelligent Result Aggregation (March 17, 2025 - COMPLETED)
- [✅] Phase 14: Cross-Platform Worker Support (March 20, 2025 - COMPLETED) ✅ NEW

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

### Running the Coordinator with Enhanced Error Handling

To start the coordinator with enhanced error handling capabilities:

```bash
# Start the coordinator with enhanced error handling
python run_coordinator_with_error_handling.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb

# Enable comprehensive features
python run_coordinator_with_error_handling.py --dashboard --load-balancer --result-aggregator

# Set custom API key
python run_coordinator_with_error_handling.py --api-key YOUR_API_KEY

# Enable debug logging
python run_coordinator_with_error_handling.py --log-level DEBUG
```

### Running the Worker with Enhanced Error Reporting

To start a worker with enhanced error reporting capabilities:

```bash
# Start worker with enhanced error reporting
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY

# Simulate errors for testing the error reporting system
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --simulate-error \
  --error-type "ResourceAllocationError" \
  --error-message "Simulated out of memory error for testing"

# Configure error history size
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --error-history-size 200

# Run with verbose logging
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --log-level DEBUG

# Simulate network disconnection
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --simulate-disconnect 30
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

# Enable error tracking visualization
python run_monitoring_dashboard.py --error-tracking
```

### Running the Error Visualization Dashboard

To start the monitoring dashboard with the new Error Visualization system:

```bash
# Start the dashboard with error visualization enabled
python run_monitoring_dashboard_with_error_visualization.py

# Run with custom settings
python run_monitoring_dashboard_with_error_visualization.py \
  --host localhost \
  --port 8080 \
  --coordinator-url http://localhost:8000 \
  --db-path ./benchmark_db.duckdb \
  --refresh-interval 5 \
  --theme dark

# Run with error visualization and result aggregator integration
python run_monitoring_dashboard_with_error_visualization.py \
  --enable-result-aggregator \
  --result-aggregator-url http://localhost:8085

# Run with error visualization and E2E test integration
python run_monitoring_dashboard_with_error_visualization.py \
  --enable-e2e-test

# Run with error visualization and visualization integration
python run_monitoring_dashboard_with_error_visualization.py \
  --enable-visualization

# Specify custom dashboard directory
python run_monitoring_dashboard_with_error_visualization.py \
  --dashboard-dir ./custom_dashboard_dir
```

### Running Cross-Platform Worker Support Tools

To use the Cross-Platform Worker Support module:

```bash
# Run the example with all features
python -m duckdb_api.distributed_testing.examples.cross_platform_worker_example --all

# Run platform and hardware detection
python -m duckdb_api.distributed_testing.examples.cross_platform_worker_example --detect

# Generate deployment scripts for the current platform
python -m duckdb_api.distributed_testing.examples.cross_platform_worker_example --scripts \
  --coordinator-url "http://coordinator-server:8080" \
  --api-key "your_api_key" \
  --output-dir "./workers"

# Generate startup scripts
python -m duckdb_api.distributed_testing.examples.cross_platform_worker_example --startup \
  --coordinator-url "http://coordinator-server:8080" \
  --api-key "your_api_key" \
  --output-dir "./workers"

# Demonstrate path conversion
python -m duckdb_api.distributed_testing.examples.cross_platform_worker_example --paths
```

You can also use the module directly in your code:

```python
from duckdb_api.distributed_testing.cross_platform_worker_support import CrossPlatformWorkerSupport

# Initialize for your platform
support = CrossPlatformWorkerSupport()

# Detect hardware capabilities
hardware_info = support.detect_hardware()
print(f"Platform: {hardware_info['platform']}")
print(f"CPU: {hardware_info['cpu']['model']} with {hardware_info['cpu']['cores']} cores")
print(f"Memory: {hardware_info['memory']['total_gb']} GB")
print(f"GPUs: {hardware_info['gpu']['count']}")

# Create deployment scripts for distributed workers
config = {
    "coordinator_url": "http://coordinator.example.com:8080",
    "api_key": "your_api_key",
    "worker_id": "worker_123"
}
script_path = support.create_deployment_script(config, "deploy_worker")
```

### Running Error Visualization Tests

To run tests for the Error Visualization system:

```bash
# Run all error visualization tests
python run_error_visualization_tests.py

# Test specific components
python run_error_visualization_tests.py --type sound     # Test sound notification system
python run_error_visualization_tests.py --type severity  # Test severity detection
python run_error_visualization_tests.py --type websocket # Test WebSocket integration
python run_error_visualization_tests.py --type dashboard # Test dashboard integration
python run_error_visualization_tests.py --type html      # Test HTML templates

# Test system-critical sound notifications specifically
python run_error_visualization_tests.py --test-system-critical

# Run only unit tests
python run_error_visualization_tests.py --unit-only

# Run only end-to-end tests
python run_error_visualization_tests.py --e2e-only

# Generate test reports
python run_error_visualization_tests.py --report --report-format html

# Run as part of the complete test suite
./run_all_tests.sh --type errorviz

# Run only specific test type with filter
./run_all_tests.sh --type errorviz --filter unit

# Run the interactive error notification demo
firefox ./dashboard/static/sounds/error_notification_demo.html
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
6. **Error Visualization** ✅ (NEW): Comprehensive error monitoring and analysis:
   - Error summary with total counts and category distribution
   - Hierarchical sound notification system with four distinct severity levels:
     - System-critical alerts with rising tone pattern for infrastructure failures
     - Critical alerts with pulsing tones for hardware issues and worker crashes
     - Warning alerts with moderate tones for network issues and timeouts
     - Info notifications with subtle tones for non-critical issues
   - Interactive error distribution visualization with drill-down capabilities
   - Automated error pattern detection to identify recurring issues
   - Worker-specific error analysis with status indicators and statistics
   - Hardware-specific error analysis to identify problematic hardware
   - Detailed error context information including system metrics at error time
   - Time range selection for historical error analysis
   - Integration with database for long-term error tracking
   - User controls for sound notification preferences
7. **Alerts**: Real-time alerting for critical events with severity levels
8. **System Topology**: Interactive visualization of the distributed system structure
9. **Fault Tolerance**: Monitoring of failure patterns and recovery actions

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