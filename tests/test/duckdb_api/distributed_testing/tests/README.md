# Testing Framework for Distributed Testing

This directory contains comprehensive testing tools for the Distributed Testing Framework, including end-to-end tests and component-specific unit tests.

## Overview

The end-to-end testing framework creates a complete testing environment that:

1. Launches all framework components (Result Aggregator, Coordinator, Monitoring Dashboard)
2. Starts multiple simulated worker nodes with various hardware profiles
3. Submits diverse test workloads to exercise all system components
4. Optionally injects failures to test fault tolerance
5. Validates that all components function correctly and are properly integrated
6. Generates comprehensive test reports

## Main Components

### End-to-End Testing
- `test_end_to_end_framework.py`: Core testing framework that sets up the complete environment and runs tests
- `run_e2e_tests.py`: Helper script to run test suites with various configurations
- `e2e_test_reports/`: Directory containing generated test reports

### Dynamic Resource Management (DRM) Tests
- `test_dynamic_resource_manager.py`: Unit tests for the DynamicResourceManager component
- `test_resource_performance_predictor.py`: Unit tests for the ResourcePerformancePredictor component
- `test_cloud_provider_manager.py`: Unit tests for the CloudProviderManager component
- `run_drm_tests.py`: Helper script to run all DRM-related tests

## Running the Tests

### Quick Test

To run a quick validation test (useful for CI/CD):

```bash
python -m duckdb_api.distributed_testing.tests.run_e2e_tests --quick
```

### Basic Test Suite

To run the basic test suite:

```bash
python -m duckdb_api.distributed_testing.tests.run_e2e_tests
```

### Comprehensive Test Suite

To run the comprehensive test suite, including more workers and longer duration:

```bash
python -m duckdb_api.distributed_testing.tests.run_e2e_tests --comprehensive
```

### Fault Tolerance Testing

To include fault tolerance tests (with simulated failures):

```bash
python -m duckdb_api.distributed_testing.tests.run_e2e_tests --fault-tolerance
```

### Generate HTML Report

To generate an HTML report with test results:

```bash
python -m duckdb_api.distributed_testing.tests.run_e2e_tests --generate-report
```

### Running Individual Test

To run an individual test with custom configuration:

```bash
python -m duckdb_api.distributed_testing.tests.test_end_to_end_framework \
  --workers 5 \
  --test-duration 60 \
  --hardware-profiles all \
  --include-failures
```

## Test Validation

The framework validates:

1. **Dashboard Accessibility**: Verifies that the monitoring dashboard is accessible
2. **Results Page**: Validates that the results page is accessible and displays proper tabs
3. **Result Aggregation**: Confirms that test results are being properly aggregated
4. **Integration**: Validates the integration between the dashboard and result aggregator
5. **Visualization Data**: Checks that visualization data is properly generated and accessible

## Test Reports

Reports are generated in JSON format with a summary of validation results. When using `--generate-report` with the test runner, an HTML report is also created with:

- Overall test success/failure summary
- Details for each test configuration
- Test duration and exit code information
- Configuration details for each test

## Fault Tolerance Testing

When using `--include-failures` or running fault tolerance tests, the framework:

1. Terminates a random worker node during test execution
2. Sends malformed requests to the coordinator and result aggregator
3. Validates that the system properly recovers and continues functioning

## Hardware Profiles

Tests can be run with the following hardware profiles:

- **cpu**: CPU-only hardware
- **gpu**: GPU hardware
- **webgpu**: WebGPU hardware (browser-based)
- **webnn**: WebNN hardware (browser-based)
- **multi**: Multi-device hardware
- **all**: All hardware types (default for comprehensive tests)

## Advanced Configuration

See `--help` on either script for additional configuration options:

```bash
python -m duckdb_api.distributed_testing.tests.test_end_to_end_framework --help
python -m duckdb_api.distributed_testing.tests.run_e2e_tests --help
```

## Running Dynamic Resource Management Tests

To run all DRM unit tests:

```bash
python -m duckdb_api.distributed_testing.tests.run_drm_tests
```

To run specific DRM component tests:

```bash
# Run only the DynamicResourceManager tests
python -m duckdb_api.distributed_testing.tests.run_drm_tests --pattern dynamic_resource_manager

# Run only the ResourcePerformancePredictor tests
python -m duckdb_api.distributed_testing.tests.run_drm_tests --pattern resource_performance_predictor

# Run only the CloudProviderManager tests
python -m duckdb_api.distributed_testing.tests.run_drm_tests --pattern cloud_provider_manager
```

For verbose output:

```bash
python -m duckdb_api.distributed_testing.tests.run_drm_tests --verbose
```

The DRM tests validate:

1. **Resource Tracking**: Proper tracking of CPU, memory, and GPU resources
2. **Resource Allocation**: Accurate reservation and release of resources
3. **Task Scheduling**: Correct fitness calculation and task-worker matching
4. **Performance Prediction**: ML-based prediction of resource requirements
5. **Cloud Integration**: Proper deployment and management of workers across cloud providers
6. **Scaling Logic**: Intelligent scaling decisions based on resource utilization

### DRM Integration Tests

The DRM integration tests validate the entire system working together:

```bash
# Run only the DRM integration tests
python -m duckdb_api.distributed_testing.tests.run_drm_tests --pattern drm_integration
```

Integration tests cover:

1. **Coordinator-DRM Integration**: Proper initialization and management of DRM components within the coordinator
2. **Worker Registration with Resources**: Registration of workers with resource information
3. **Resource-Aware Task Scheduling**: Task assignment based on resource requirements and availability
4. **Resource Reservation Lifecycle**: Reservation and release of resources during task execution
5. **Resource Usage Recording**: Recording of execution metrics for future prediction
6. **Adaptive Scaling**: Worker pool scaling based on resource utilization
7. **End-to-End Workflow**: Complete workflow from worker registration to task execution with resource management

These integration tests ensure that all components of the DRM system work correctly together and with the broader Distributed Testing Framework.