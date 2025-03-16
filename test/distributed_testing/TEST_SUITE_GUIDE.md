# Hardware Monitoring Test Suite Guide

This document provides detailed information about the test suite for the hardware monitoring system. The test suite ensures that all components of the system work correctly and integrates properly with the distributed testing framework.

## Overview

The hardware monitoring test suite consists of the following components:

1. **Test Classes**: Classes that contain individual test methods
2. **Test Runner**: A script that executes the tests and reports results
3. **Test Fixtures**: Setup code that prepares the test environment
4. **Test Utilities**: Helper functions used by multiple tests
5. **Test Reports**: HTML and text output showing test results

## Test Structure

The test suite is organized into three main classes:

### 1. TestHardwareUtilizationMonitor

Tests for the `HardwareUtilizationMonitor` class, which is responsible for collecting and storing hardware metrics.

Test methods:
- `test_basic_monitoring`: Verifies basic hardware metric collection
- `test_task_monitoring`: Tests task-specific resource tracking
- `test_alert_generation`: Validates alert generation for high resource usage
- `test_database_integration`: Tests database operations for metrics storage
- `test_report_generation`: Tests HTML report generation
- `test_json_export`: Validates JSON export functionality

### 2. TestCoordinatorHardwareMonitoringIntegration

Tests for the `CoordinatorHardwareMonitoringIntegration` class, which connects the hardware monitor to the coordinator.

Test methods:
- `test_worker_monitors_created`: Verifies monitor creation for workers
- `test_worker_registration_callback`: Tests worker registration events
- `test_update_worker_utilization`: Validates worker utilization updates
- `test_task_monitoring_integration`: Tests task monitoring integration
- `test_hardware_aware_find_best_worker`: Tests hardware-aware worker selection
- `test_report_generation`: Validates HTML report generation

### 3. TestHardwareMonitoringEndToEnd

End-to-end tests that validate the entire system working together.

Test methods:
- `test_end_to_end_demo`: Tests the full system with a demo script

## Test Runner

The `run_hardware_monitoring_tests.py` script executes the test suite. It provides several command-line options:

```bash
python run_hardware_monitoring_tests.py [options]
```

Options:
- `--verbose`: Display detailed test output
- `--run-long-tests`: Include long-running tests
- `--db-path PATH`: Use a specific database path
- `--html-report PATH`: Generate HTML test report

## Running Tests

### Basic Test Execution

To run the basic test suite:

```bash
python run_hardware_monitoring_tests.py
```

This runs all tests except for long-running tests, using a temporary database that is deleted after tests complete.

### Verbose Output

For detailed test output:

```bash
python run_hardware_monitoring_tests.py --verbose
```

This shows all test details including setup, execution, and teardown phases.

### Long-Running Tests

To include tests that take more time to execute:

```bash
python run_hardware_monitoring_tests.py --run-long-tests
```

This includes end-to-end tests that simulate the full system.

### Using a Specific Database

To use a specific database for tests:

```bash
python run_hardware_monitoring_tests.py --db-path ./test_metrics.duckdb
```

This stores test metrics in the specified database, which persists after tests complete.

### HTML Test Report

To generate an HTML report of test results:

```bash
python run_hardware_monitoring_tests.py --html-report ./test_report.html
```

This creates an HTML file with test results, including passed, failed, and skipped tests.

## Test Database

The test suite uses DuckDB for storing metrics during tests. By default, it creates a temporary database that is deleted after tests complete. You can specify a persistent database with the `--db-path` option.

Database tables used during testing:
- `resource_utilization`: Hardware metrics collected during monitoring
- `task_resource_usage`: Resource usage for specific tasks
- `hardware_alerts`: Alerts generated for high resource usage

## Error Handling

The test suite is designed to handle various error scenarios:

- **Database Errors**: Tests continue even if database operations fail
- **Monitoring Failures**: Tests gracefully handle monitoring failures
- **Missing Hardware**: Tests adapt to missing hardware components
- **Network Issues**: Tests handle network connectivity problems

## Test Design Principles

The hardware monitoring test suite follows these design principles:

1. **Isolation**: Tests are isolated from each other to prevent interference
2. **Independence**: Tests do not depend on the order of execution
3. **Repeatability**: Tests give the same results on repeated execution
4. **Performance**: Tests execute quickly to avoid long test times
5. **Robustness**: Tests handle errors and edge cases gracefully

## Adding Tests

To add a new test to the suite:

1. Identify the appropriate test class (monitor, integration, or end-to-end)
2. Add a new test method following the naming convention `test_*`
3. Write setup code to prepare the test environment
4. Execute the code to be tested
5. Add assertions to verify expected behavior
6. Add teardown code to clean up the test environment

Example:

```python
def test_new_feature(self):
    """Test description here."""
    # Setup
    monitor = HardwareUtilizationMonitor(worker_id="test-worker")
    monitor.start_monitoring()
    
    # Test execution
    result = monitor.some_feature()
    
    # Assertions
    self.assertIsNotNone(result)
    self.assertEqual(result.some_property, expected_value)
    
    # Teardown
    monitor.stop_monitoring()
```

## Continuous Integration

The test suite can be integrated with CI pipelines to run tests automatically:

1. Add a CI configuration file (e.g., GitHub Actions workflow)
2. Configure the CI to install dependencies
3. Run the test suite with appropriate options
4. Generate test reports for review
5. Fail the CI if tests fail

Example GitHub Actions workflow:

```yaml
name: Hardware Monitoring Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python test/distributed_testing/run_hardware_monitoring_tests.py --verbose --html-report test_report.html
      - name: Upload test report
        uses: actions/upload-artifact@v2
        with:
          name: test-report
          path: test_report.html
```

## Troubleshooting Tests

If tests fail, check the following:

1. **Database Issues**: Ensure DuckDB is installed and the database path is accessible
2. **Hardware Access**: Some tests may require specific hardware to be available
3. **Dependencies**: Make sure all dependencies are installed and up to date
4. **File Permissions**: Ensure the test has permission to create and modify files
5. **Network Access**: Some tests may require network connectivity

## Conclusion

The hardware monitoring test suite provides comprehensive validation of the hardware monitoring system. By running these tests regularly, you can ensure that the system continues to function correctly as changes are made to the codebase.

For more information about the hardware monitoring system itself, see:
- [README_HARDWARE_MONITORING.md](README_HARDWARE_MONITORING.md): Usage guide
- [HARDWARE_MONITORING_IMPLEMENTATION_SUMMARY.md](HARDWARE_MONITORING_IMPLEMENTATION_SUMMARY.md): Implementation details