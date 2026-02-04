# Distributed Testing Framework - End-to-End Integration Testing

This document provides a comprehensive guide to the end-to-end integration testing framework for the Distributed Testing Framework. This framework enables thorough testing of all components working together, including the Coordinator, Load Balancer, Monitoring Dashboard, and Workers.

## Overview

The integration testing framework has been designed to provide:

1. **Comprehensive Testing**: Tests that validate all components working together as a cohesive system
2. **Isolated Environment**: Each test creates its own isolated environment to prevent interference
3. **Real-World Scenarios**: Tests simulate real-world usage patterns and failure scenarios
4. **Extensibility**: Easy to add new test cases for future features
5. **Automation**: Can be run as part of CI/CD workflows

## Key Components Tested

The integration tests validate the following components and their interactions:

1. **Coordinator**: Central server managing tasks and workers
2. **Load Balancer**: Adaptive workload distribution system
3. **Monitoring Dashboard**: Real-time visualization of system metrics
4. **Worker Nodes**: Task execution units with different capabilities
5. **Task Scheduler**: Intelligent task scheduling and distribution
6. **Health Monitor**: Worker health monitoring and recovery
7. **Metrics Collection**: Gathering and storing system performance data

## Test Types

The integration testing framework includes several types of tests:

1. **Basic Integration Tests**: Verify that components can initialize and communicate
2. **Task Distribution Tests**: Validate that tasks are distributed correctly to appropriate workers
3. **Fault Tolerance Tests**: Verify system behavior when workers fail or become unavailable
4. **Performance Tests**: Measure system performance under various conditions
5. **Dashboard Integration Tests**: Verify that the monitoring dashboard collects and displays metrics correctly
6. **WebSocket Tests**: Validate real-time updates via WebSocket connections
7. **API Endpoint Tests**: Verify that REST API endpoints function correctly

## Running the Tests

### Combined Test Script

The most convenient way to run all tests is using the combined test script that supports all test types with unified logging and reporting:

```bash
# Run all tests
./run_all_tests.sh

# Run only fault tolerance tests
./run_all_tests.sh --type fault

# Run only monitoring tests
./run_all_tests.sh --type monitoring

# Run only integration tests with a filter
./run_all_tests.sh --type integration --filter load_balancer

# Run tests with verbose output
./run_all_tests.sh --type all --verbose

# Specify a log directory
./run_all_tests.sh --log-dir my_test_results

# Show all available options
./run_all_tests.sh --help
```

The combined test script provides:
- A unified interface for running all types of tests
- Organized logging with a timestamp-based directory structure
- Detailed test summaries showing pass/fail status
- The ability to run specific test types or all tests
- Test filtering for more focused test runs
- Colorized output for improved readability

### Running All Integration Tests

To run all integration tests directly:

```bash
python -m duckdb_api.distributed_testing.tests.run_integration_tests
```

For detailed output:

```bash
python -m duckdb_api.distributed_testing.tests.run_integration_tests --verbose
```

### Running Specific Test Files

To run specific test modules:

```bash
# Run only load balancer tests
python -m duckdb_api.distributed_testing.tests.run_integration_tests --test load_balancer

# Run only monitoring dashboard tests
python -m duckdb_api.distributed_testing.tests.run_integration_tests --test load_balancer_monitoring

# Run only fault tolerance tests
python -m duckdb_api.distributed_testing.tests.test_load_balancer_fault_tolerance
```

### Running Specific Test Cases

To run a specific test case:

```bash
python -m duckdb_api.distributed_testing.tests.run_integration_tests --case LoadBalancerMonitoringIntegrationTest.test_03_metrics_collection

# Run a specific fault tolerance test
python -m duckdb_api.distributed_testing.tests.test_load_balancer_fault_tolerance LoadBalancerFaultToleranceTest.test_06_worker_recovery
```

### Listing Available Tests

To see a list of all available tests:

```bash
python -m duckdb_api.distributed_testing.tests.run_integration_tests --list
```

### Specialized Test Scripts

The framework includes specialized scripts for specific test types:

```bash
# Run fault tolerance tests
./run_fault_tolerance_tests.sh --all

# Run the integrated system (coordinator, load balancer, dashboard)
./run_integrated_system.sh --mock-workers 5 --open-browser
```

## Test Environment

Each test creates its own isolated environment, including:

1. **Temporary Database**: Each test creates its own temporary database files
2. **In-Memory Coordinator**: A standalone coordinator instance
3. **Mock Workers**: Simulated worker processes with different capabilities
4. **Isolated Dashboard**: A dedicated monitoring dashboard instance
5. **Security Configuration**: Test-specific API keys and security settings

This isolation ensures that tests don't interfere with each other and can be run concurrently.

## Key Test Files

### Basic Integration Tests

- `test_integration.py`: Tests the core functionality of the Distributed Testing Framework
- `test_scheduler.py`: Tests task scheduling algorithms
- `test_health_monitor.py`: Tests worker health monitoring and recovery

### Load Balancer Tests

- `test_load_balancer.py`: Tests the core functionality of the Load Balancer
- `test_work_stealing.py`: Tests the work stealing algorithm
- `test_task_analyzer.py`: Tests the task analyzer component
- `test_load_balancer_fault_tolerance.py`: Tests the fault tolerance capabilities of the Load Balancer

### Monitoring Dashboard Tests

- `test_load_balancer_monitoring.py`: Tests the Monitoring Dashboard integration
- This comprehensive test validates that metrics are properly collected, stored, and displayed in the dashboard

## Key Integration Tests

### Load Balancer Monitoring Integration Test

The `test_load_balancer_monitoring.py` file contains a comprehensive end-to-end test for the Load Balancer Monitoring Dashboard. This test validates:

1. **Initialization**: Verifies that all components initialize correctly
2. **API Access**: Tests access to dashboard API endpoints
3. **Metrics Collection**: Validates that metrics are collected and stored
4. **WebSocket Updates**: Tests real-time updates via WebSocket connection
5. **Anomaly Detection**: Verifies the anomaly detection system
6. **Task Tracking**: Tests task tracking in the dashboard
7. **Worker Performance Scoring**: Validates the worker performance scoring system
8. **Historical Metrics**: Tests retrieval of historical metrics
9. **Load Balancer Integration**: Verifies that load balancer events are properly monitored
10. **HTML Interface**: Tests that the dashboard HTML interface is accessible

### Load Balancer Fault Tolerance Test

The `test_load_balancer_fault_tolerance.py` file contains a comprehensive test suite for the fault tolerance capabilities of the Load Balancer. This test validates:

1. **Basic Initialization**: Verifies that fault tolerance is properly configured
2. **Worker Registration**: Tests that workers are registered with capabilities
3. **Task Assignment**: Validates task assignment based on worker capabilities
4. **Worker Failure Detection**: Tests that worker failures are detected correctly
5. **Task Reassignment**: Validates that tasks are reassigned after worker failure
6. **Worker Recovery**: Tests worker recovery and task continuity after reconnection
7. **Simultaneous Failures**: Tests handling of multiple simultaneous worker failures
8. **Work Stealing**: Validates that work stealing redistributes load after recovery
9. **Task Completion**: Tests that tasks complete successfully despite worker failures
10. **Repeated Failures**: Tests system resiliency to repeated worker failures

The fault tolerance test uses multiple worker processes with different capabilities and simulates various failure scenarios:

- **Controlled Worker Termination**: Simulates worker crashes by terminating processes
- **Worker Restart**: Tests recovery by restarting workers with the same ID
- **Resource Redistribution**: Tests automatic rebalancing after failures
- **Task Migration**: Validates that tasks are migrated to healthy workers
- **Staggered Failures**: Tests handling of failures that occur at different times
- **Simultaneous Failures**: Tests handling of multiple failures at once
- **High-Capacity Recovery**: Tests intelligent work distribution after adding a high-capacity worker

### Test Execution Flow

The test follows this execution flow:

1. **Setup**: Creates temporary databases, security configuration, and coordinator
2. **Coordinator Initialization**: Starts the coordinator with load balancer enabled
3. **Dashboard Initialization**: Starts the monitoring dashboard connected to the coordinator
4. **Worker Creation**: Launches mock workers with different capabilities
5. **Test Execution**: Runs various test cases to validate different aspects of the system
6. **Task Submission**: Submits test tasks to generate metrics
7. **Verification**: Verifies that metrics are correctly collected and displayed
8. **Cleanup**: Stops all components and removes temporary files

## Creating New Tests

When adding new integration tests, follow these guidelines:

1. **Isolated Environment**: Always create a temporary environment for your test
2. **Clean Up**: Make sure to clean up all resources in the `tearDownClass` method
3. **Test Organization**: Use descriptive method names prefixed with `test_` and numbered for execution order
4. **Documentation**: Add detailed docstrings explaining the purpose of each test
5. **Error Handling**: Handle potential errors to prevent them from affecting other tests
6. **Assert Messages**: Include helpful messages in assertions to make failures easier to diagnose

Example test case structure:

```python
def test_01_component_initialization(self):
    """Test that all components are properly initialized."""
    # Test initialization logic here
    self.assertTrue(self.component.initialized, "Component should be initialized")
```

## Advanced Testing

### Mock Workers

The integration tests use mock worker processes that simulate real workers with different capabilities. These mock workers:

1. Connect to the coordinator
2. Register with different hardware capabilities
3. Accept and execute tasks
4. Report task results back to the coordinator

This allows testing task distribution and load balancing algorithms with heterogeneous worker pools.

### Metrics Validation

To validate metrics collection and storage, the tests:

1. Generate known metrics through test tasks
2. Query the metrics database to verify storage
3. Access metrics through API endpoints
4. Verify metrics values match expected values

### WebSocket Testing

Testing real-time updates via WebSocket involves:

1. Establishing a WebSocket connection to the dashboard
2. Subscribing to specific metric channels
3. Generating metrics through test tasks
4. Verifying received messages match expected values

## Troubleshooting Tests

If integration tests fail, check the following:

1. **Port Conflicts**: Ensure the coordinator and dashboard ports are available
2. **Database Access**: Verify the test can create and access temporary databases
3. **Dependencies**: Make sure all required packages are installed
4. **Process Cleanup**: Ensure no processes from previous test runs are still running
5. **Timing Issues**: Some tests may fail due to timing; try increasing sleep intervals

## Conclusion

The end-to-end integration testing framework ensures that all components of the Distributed Testing Framework work together seamlessly. By validating the system as a whole, these tests provide confidence that the framework will function correctly in production environments.

For more information about specific components, refer to the following documents:

- [LOAD_BALANCER_IMPLEMENTATION_STATUS.md](../LOAD_BALANCER_IMPLEMENTATION_STATUS.md): Status of the Load Balancer implementation
- [WEBNN_WEBGPU_DATABASE_INTEGRATION.md](../../../WEBNN_WEBGPU_DATABASE_INTEGRATION.md): Database integration details
- [WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md](../../../WEB_RESOURCE_POOL_MAY2025_ENHANCEMENTS.md): Resource pool enhancements