# Hardware Monitoring for Distributed Testing Framework

This document outlines the hardware monitoring capabilities added to the distributed testing framework. These capabilities enable resource-aware task scheduling, hardware utilization tracking, and performance optimization based on real-time resource metrics.

## Overview

The hardware monitoring system integrates with the existing distributed testing framework to provide:

1. Real-time hardware utilization monitoring (CPU, memory, GPU, disk, network)
2. Resource-aware task scheduling based on current hardware load
3. Database integration for persistent storage of utilization metrics
4. Task-specific resource tracking during execution
5. Threshold-based alerts for resource overutilization
6. Historical performance tracking for predictive scheduling
7. HTML and JSON reporting for resource utilization analysis
8. Comprehensive testing suite for verification and validation
9. Quality assurance through automated tests
10. CI/CD integration for continuous testing and validation
11. Artifact management and test result tracking
12. Status badge generation for real-time quality indicators
13. Multi-channel notification system for test failures

## Components

The hardware monitoring system consists of the following key components:

### Hardware Utilization Monitor (`hardware_utilization_monitor.py`)

The core monitoring component that tracks hardware resource utilization on worker nodes:

- Real-time monitoring of CPU, memory, GPU, disk, and network resources
- Database integration with DuckDB for persistent storage of metrics
- Threshold-based alerts for overutilization
- Task-specific resource tracking with peak/average metrics
- HTML and JSON report generation
- Configurable monitoring levels (basic, standard, detailed, intensive)
- Historical metrics tracking with time-based analysis

### Coordinator Hardware Monitoring Integration (`coordinator_hardware_monitoring_integration.py`)

Integrates the hardware utilization monitor with the coordinator and task scheduler:

- Resource-aware task scheduling based on current hardware load
- Worker registration with hardware capability detection
- Task monitoring during execution
- Performance history tracking for predictive scheduling
- Scheduler method patching for hardware-aware scheduling
- Resource-based worker selection with utilization scoring
- Overload prevention with automatic alternative worker selection

### Hardware Capability Detector (`hardware_capability_detector.py`)

Detects hardware capabilities on worker nodes and stores them in the database:

- Comprehensive hardware detection (CPU, GPU, TPU, NPU, WebGPU, WebNN)
- Database integration for capability storage
- Hardware fingerprinting for unique identification
- WebGPU/WebNN detection with browser automation
- DuckDB integration for optimization

### Demo Script (`run_coordinator_with_hardware_monitoring.py`)

Demonstrates the hardware monitoring system in action:

- Coordinator setup with hardware monitoring integration
- Worker registration with capability detection
- Task creation and hardware-aware scheduling
- Real-time resource utilization monitoring
- Resource-aware task assignment
- Task execution with hardware utilization tracking
- HTML report generation with utilization metrics

### Test Suite (`tests/test_hardware_utilization_monitor.py`)

Comprehensive test suite for verifying the functionality of the hardware monitoring system:

- Unit tests for the hardware utilization monitor component
- Integration tests for the coordinator hardware monitoring integration
- Validation of database operations and error handling
- Testing of resource-aware scheduling algorithms
- Validation of HTML and JSON report generation
- Testing of task-specific resource tracking
- Verification of alert generation and handling

### Test Runner (`run_hardware_monitoring_tests.py`)

A dedicated script for running the hardware monitoring test suite:

- Configurable test execution with various options
- HTML report generation for test results
- Support for both quick and comprehensive testing modes
- Detailed logging of test execution and results
- Database path configuration for testing
- Skip capability for long-running tests

### CI Integration Files

Files for continuous integration and automated testing:

- **`.github/workflows/hardware_monitoring_tests.yml`**: Local GitHub Actions workflow for testing
- **`.github/workflows/hardware_monitoring_integration.yml`**: Global GitHub Actions workflow
- **`run_hardware_monitoring_ci_tests.sh`**: Local script for simulating the CI environment
- **Hardware monitoring integration with global CI/CD pipeline**
- **Integration with artifact handling system for test results storage**
- **DuckDB integration for test metrics and results**

### Notification System

Components for automated alerting and reporting:

- **`ci_notification.py`**: Script for sending notifications when tests fail
- **`notification_config.json`**: Configuration file for notification channels
- **Email, Slack, and GitHub integration** for comprehensive alerting
- **Customizable notification templates** for different channels

### Status Badge Generator

Components for generating status indicators:

- **`generate_status_badge.py`**: Script for creating SVG status badges
- **Auto-updating badges** showing current test status
- **Multiple badge styles** (flat, square, etc.) for different uses
- **JSON status export** for custom integrations

## Usage

### Running the Demo

The demo script simulates a distributed testing environment with multiple workers and tasks. It demonstrates how hardware metrics influence task scheduling and how resource utilization is tracked during task execution.

```bash
# Basic usage
python run_coordinator_with_hardware_monitoring.py

# With custom parameters
python run_coordinator_with_hardware_monitoring.py --num-workers 5 --num-tasks 20 --duration 120

# With custom database path and report location
python run_coordinator_with_hardware_monitoring.py --db-path ./my_metrics.duckdb --report ./my_report.html

# Export results to JSON
python run_coordinator_with_hardware_monitoring.py --export-json ./results.json
```

#### Command-Line Arguments

- `--db-path`: Path to DuckDB database for metrics storage (default: `./hardware_metrics.duckdb`)
- `--num-workers`: Number of workers to simulate (default: 3)
- `--num-tasks`: Number of tasks to create (default: 10)
- `--duration`: Simulation duration in seconds (default: 60)
- `--max-tasks-per-worker`: Maximum tasks per worker (default: 2)
- `--report`: Path for HTML report (default: `hardware_utilization_report.html`)
- `--export-json`: Export results to JSON file (optional)

### Running the Test Suite

A comprehensive test suite is included to verify the functionality of the hardware monitoring system. The test suite includes unit tests, integration tests, and end-to-end tests of the hardware monitoring components.

```bash
# Basic usage
python run_hardware_monitoring_tests.py

# Run tests with verbose output
python run_hardware_monitoring_tests.py --verbose

# Run long-running tests (including end-to-end demo)
python run_hardware_monitoring_tests.py --run-long-tests

# Run tests with custom database path
python run_hardware_monitoring_tests.py --db-path ./test_metrics.duckdb

# Generate HTML test report
python run_hardware_monitoring_tests.py --html-report ./test_report.html

# Run all tests with verbose output and generate HTML report
python run_hardware_monitoring_tests.py --verbose --run-long-tests --html-report ./complete_test_report.html
```

### Running CI Tests

The hardware monitoring system includes CI integration for continuous testing. You can run these tests locally to simulate the CI environment:

```bash
# Run standard tests in CI mode
./run_hardware_monitoring_ci_tests.sh

# Run full tests in CI mode
./run_hardware_monitoring_ci_tests.sh --mode full

# Run long tests with specific Python version
./run_hardware_monitoring_ci_tests.sh --mode long --python python3.9

# Run with CI integration tests
./run_hardware_monitoring_ci_tests.sh --mode full --ci-integration

# Generate status badge
./run_hardware_monitoring_ci_tests.sh --mode full --generate-badge

# Send notifications about test results
./run_hardware_monitoring_ci_tests.sh --mode full --send-notifications

# Run CI tests using a different test database path
BENCHMARK_DB_PATH=./custom_metrics.duckdb ./run_hardware_monitoring_ci_tests.sh --mode full

# Run full CI simulation with all features
./run_hardware_monitoring_ci_tests.sh --mode full --ci-integration --generate-badge --send-notifications
```

The CI test script generates HTML reports and database files similar to the GitHub Actions workflows, allowing you to verify that your changes will pass in the CI environment.

#### Test Command-Line Arguments

- `--verbose`: Display verbose test output with detailed logging
- `--run-long-tests`: Run long-running tests (including end-to-end demo) that validate more complex scenarios
- `--db-path`: Path to test database (default: temporary file) for persistent testing metrics
- `--html-report`: Generate HTML test report at specified path with detailed test results

#### Test Classes and Methods

The test suite consists of the following main test classes:

1. **TestHardwareUtilizationMonitor**: Tests for the hardware utilization monitor component
   - `test_basic_monitoring`: Tests basic hardware monitoring functionality
   - `test_task_monitoring`: Tests task-specific resource tracking
   - `test_alert_generation`: Tests threshold-based alerting
   - `test_database_integration`: Tests metrics storage in DuckDB
   - `test_report_generation`: Tests HTML report generation
   - `test_json_export`: Tests JSON export of metrics

2. **TestCoordinatorHardwareMonitoringIntegration**: Tests for the coordinator integration
   - `test_worker_monitors_created`: Tests worker monitor creation
   - `test_worker_registration_callback`: Tests worker registration callbacks
   - `test_update_worker_utilization`: Tests worker utilization updates
   - `test_task_monitoring_integration`: Tests task monitoring
   - `test_hardware_aware_find_best_worker`: Tests hardware-aware worker selection
   - `test_report_generation`: Tests HTML report generation

3. **TestHardwareMonitoringEndToEnd**: End-to-end tests for the complete system
   - `test_end_to_end_demo`: Tests the entire system with a demo script

#### Test Coverage

The test suite covers the following aspects of the hardware monitoring system:

1. **Basic Hardware Monitoring**: Tests basic hardware utilization monitoring functionality
   - CPU, memory, GPU, disk, and network monitoring
   - Metrics collection and storage
   - Historical metrics tracking

2. **Task Monitoring**: Tests task-specific resource tracking during execution
   - Peak and average metrics
   - Resource usage profiles
   - Task completion tracking

3. **Alert Generation**: Tests threshold-based alerting for high resource utilization
   - Different severity levels
   - Multi-resource alerts
   - Alert callback system

4. **Database Integration**: Tests metrics storage in the DuckDB database
   - Schema validation
   - Data insertion and retrieval
   - Query functionality

5. **Report Generation**: Tests HTML and JSON report generation
   - Format verification
   - Content validation
   - Visualization elements

6. **Coordinator Integration**: Tests integration with the coordinator component
   - Worker monitoring management
   - Event handling
   - Status updates

7. **Resource-Aware Scheduling**: Tests hardware-aware task scheduling algorithm
   - Utilization scoring
   - Alternative worker selection
   - Overload prevention

8. **End-to-End Functionality**: Tests the entire system in an end-to-end scenario
   - Full workflow validation
   - Component interaction
   - System stability

#### Test Error Handling

The test suite is designed to handle various error scenarios gracefully:

- Database connectivity issues
- Missing hardware components
- High resource utilization
- Worker registration/deregistration
- Task execution failures

This robust error handling ensures that the hardware monitoring system can operate reliably even under non-ideal conditions.

### Integrating with Existing Coordinator

To integrate hardware monitoring with an existing coordinator:

```python
from coordinator_hardware_monitoring_integration import CoordinatorHardwareMonitoringIntegration
from hardware_utilization_monitor import MonitoringLevel

# Create integration with coordinator
integration = CoordinatorHardwareMonitoringIntegration(
    coordinator=your_coordinator_instance,
    db_path="./hardware_metrics.duckdb",
    monitoring_level=MonitoringLevel.STANDARD,
    enable_resource_aware_scheduling=True,
    utilization_threshold=80.0,
    update_interval_seconds=5.0
)

# Initialize integration
integration.initialize()

# When a task is assigned to a worker, start monitoring
integration.start_task_monitoring(task_id, worker_id)

# When a task completes or fails, stop monitoring
integration.stop_task_monitoring(task_id, worker_id, success=True)

# Generate HTML report
integration.generate_html_report("hardware_utilization_report.html")

# When shutting down, shutdown integration
integration.shutdown()
```

## Resource-Aware Task Scheduling

The integration enhances the task scheduler with resource-aware scheduling capabilities:

1. **Hardware Match Scoring**: Tasks are matched to workers based on hardware requirements (GPU, memory)
2. **Utilization Scoring**: Workers with lower resource utilization are preferred
3. **Performance History**: Worker performance from past executions influences scheduling
4. **Overload Prevention**: Critically overloaded workers are excluded from scheduling
5. **Alternative Selection**: If a selected worker has high utilization, an alternative worker may be chosen

Utilization metrics that affect scheduling:

- **CPU Utilization**: Workers with high CPU load (>70%) are penalized in scoring
- **Memory Utilization**: Workers with high memory usage (>70%) are penalized in scoring
- **GPU Utilization**: For GPU tasks, workers with high GPU load (>60%) are penalized more aggressively
- **Critical Overload**: Workers with extremely high (>95%) utilization of any resource are temporarily excluded

## Database and Reporting

The monitoring system uses DuckDB for persistent storage of metrics and capabilities:

### Database Tables

- `resource_utilization`: Detailed hardware metrics over time
- `task_resource_usage`: Resource usage statistics for each task execution
- `hardware_alerts`: Threshold-based alerts for resource overutilization
- `hardware_capabilities`: Worker hardware capabilities and specifications

### HTML Reports

The system can generate comprehensive HTML reports with:

- Worker utilization metrics with visual indicators for high utilization
- Task resource usage statistics (peak, average, total)
- Scheduler statistics and worker performance metrics
- Visual alerts for resource overutilization

### JSON Export

Results can be exported to JSON for further analysis or integration with other systems.

## Benefits

The hardware monitoring integration provides several key benefits:

1. **Better Resource Utilization**: More efficient use of available hardware resources
2. **Reduced Task Failures**: Avoids overloading workers, reducing task failures
3. **Performance Insights**: Identifies hardware bottlenecks and performance issues
4. **Predictive Scheduling**: Uses historical data to make better scheduling decisions
5. **Hardware-Aware Optimization**: Matches tasks to the most appropriate hardware
6. **Real-Time Monitoring**: Provides visibility into resource utilization during execution
7. **Performance History**: Tracks resource usage patterns for continuous improvement

## Implementation Details

### Monitoring Integration Flow

1. Coordinator initializes with hardware monitoring integration
2. Workers register with hardware capabilities
3. Tasks are created with hardware requirements
4. Task scheduler uses hardware metrics for task assignment
5. Task execution is monitored for resource utilization
6. Resource metrics are stored in the database
7. Reports are generated for analysis

### Monitoring Level Options

The hardware utilization monitor supports different levels of monitoring detail:

- `BASIC`: CPU and memory only, less frequent collection
- `STANDARD`: All main resources, moderate frequency
- `DETAILED`: Comprehensive metrics, high frequency
- `INTENSIVE`: Maximum metrics, highest frequency

### Alert Thresholds

The system generates alerts when resource utilization exceeds thresholds:

- CPU utilization above 90% (warning) or 95% (critical)
- Memory utilization above 90% (warning) or 95% (critical)
- GPU utilization above 95% (warning) or 98% (critical)
- Disk utilization above 95% (warning) or 98% (critical)

### Resource-Aware Scheduling Algorithm

The integration enhances the task scheduler with a hardware-aware scheduling algorithm:

1. Original scheduler selects a worker based on standard criteria
2. Hardware utilization metrics are retrieved for the selected worker
3. Score is adjusted based on current resource utilization
4. If utilization is high, penalties are applied to the score
5. If worker is critically overloaded, it's excluded from selection
6. If adjusted score is significantly worse, an alternate worker is found
7. The worker with the best adjusted score is selected

## Extending the System

### Adding New Resource Metrics

To add monitoring for new resource types:

1. Add the new metric to the `ResourceUtilization` data class
2. Update the `_collect_metrics` method to collect the new metric
3. Update the `_store_metrics` method to store the new metric in the database
4. Update the alert system if threshold-based alerts are needed for the new metric

### Custom Scheduling Policies

To implement custom scheduling policies based on hardware metrics:

1. Extend the `_hardware_aware_find_best_worker` method in the integration
2. Implement custom scoring logic based on specific hardware metrics
3. Adjust weights in the `hardware_scoring_factors` dictionary

### Adding New Hardware Types

To add support for new hardware types:

1. Update the `HardwareType` enum in the hardware capability detector
2. Add detection logic for the new hardware type
3. Update the hardware compatibility scoring in the task scheduler

## Conclusion

The hardware monitoring system provides a comprehensive solution for resource-aware task scheduling and hardware utilization tracking in the distributed testing framework. By integrating real-time hardware metrics into the scheduling process, it optimizes resource utilization, reduces task failures, and provides valuable insights into system performance.