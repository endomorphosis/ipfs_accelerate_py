# Distributed Testing Framework - Integration Testing

This directory contains integration tests for the distributed testing framework, ensuring all components work together seamlessly.

## Overview

The distributed testing framework consists of several key components:

- **Coordinator** (`coordinator.py`): Central server managing tasks and workers
- **Worker** (`worker.py`): Worker nodes that execute tasks
- **Task Scheduler** (`task_scheduler.py`): Intelligent task scheduling and distribution
- **Load Balancer** (`load_balancer.py`): Adaptive workload distribution
- **Health Monitor** (`health_monitor.py`): Worker health monitoring and recovery
- **Dashboard** (`dashboard_server.py`): Web-based monitoring interface

These tests verify that these components integrate correctly and handle various scenarios appropriately.

## Test Structure

- **`test_integration.py`**: End-to-end tests of the entire framework
- **`test_scheduler.py`**: Tests of the task scheduler component
- **`test_health_monitor.py`**: Tests of the health monitoring component
- **`test_load_balancer.py`**: Tests of the load balancer component
- **`test_benchmark.py`**: Performance benchmarks for all components
- **`run_integration_tests.py`**: Script to run all tests

## Running the Tests

### Running All Tests

To run all integration tests:

```bash
python run_integration_tests.py
```

For detailed output:

```bash
python run_integration_tests.py --verbose
```

### Running Individual Test Files

To run specific test files:

```bash
python test_integration.py
python test_scheduler.py
python test_health_monitor.py
python test_load_balancer.py
python test_benchmark.py
```

### Running Performance Benchmarks

The benchmark tests measure the performance of various components:

```bash
# Run all benchmarks
python test_benchmark.py

# Run a specific benchmark test
python test_benchmark.py DistributedFrameworkBenchmark.test_03_benchmark_task_assignment
```

## Test Categories

### Integration Tests (`test_integration.py`)

End-to-end tests that ensure all components work together correctly:

1. **Initialization Tests**: Verify all components initialize correctly
2. **Worker Registration Tests**: Verify workers register with the coordinator
3. **Task Distribution Tests**: Verify tasks are distributed to appropriate workers
4. **Task Execution Tests**: Verify workers execute tasks and report results
5. **Worker Health Tests**: Verify health monitoring detects worker failures
6. **Dashboard Connectivity Tests**: Verify dashboard connects to coordinator and displays data
7. **Fault Tolerance Tests**: Verify system recovers from worker failures

### Task Scheduler Tests (`test_scheduler.py`)

Tests focusing on the task scheduler component:

1. **Task Matching Tests**: Verify tasks are matched to appropriate workers
2. **Hardware Requirement Tests**: Verify hardware requirements are enforced
3. **Priority-Based Scheduling Tests**: Verify tasks are scheduled according to priority

### Health Monitor Tests (`test_health_monitor.py`)

Tests focusing on the health monitoring component:

1. **Heartbeat Timeout Tests**: Verify timeouts are detected correctly
2. **Recovery Tests**: Verify workers can recover from failures
3. **Alert Tests**: Verify alerts are generated for failures

### Load Balancer Tests (`test_load_balancer.py`)

Tests focusing on the load balancer component:

1. **Worker Scoring Tests**: Verify workers are scored correctly based on capabilities
2. **Task Assignment Tests**: Verify tasks are assigned to the most appropriate workers
3. **Load Balancing Tests**: Verify workloads are balanced across workers
4. **Workload Detection Tests**: Verify overloaded and underutilized workers are identified
5. **Task Migration Tests**: Verify tasks are migrated between workers to balance load
6. **Performance-Based Balancing Tests**: Verify balancing considers worker performance

### Benchmark Tests (`test_benchmark.py`)

Performance benchmarks for various components:

1. **Database Benchmarks**: Measure database operation performance
2. **Worker Registration Benchmarks**: Measure worker registration performance
3. **Task Creation Benchmarks**: Measure task creation performance
4. **Task Assignment Benchmarks**: Measure task assignment performance with different loads
5. **Load Balancing Benchmarks**: Measure load balancing performance with different imbalances
6. **Health Monitoring Benchmarks**: Measure health monitoring performance
7. **Concurrent Operation Benchmarks**: Measure performance with multiple components running

## Test Environment

The tests create a controlled environment with:

- Temporary database files
- In-memory coordinator server
- Simulated worker processes
- Web dashboard

The environment is cleaned up after each test run.

## Requirements

- Python 3.8+
- DuckDB
- websockets
- requests
- pytest (optional, for additional test features)

## Adding New Tests

When adding new tests, follow these patterns:

1. Create appropriate setup in `setUp()` method
2. Clean up resources in `tearDown()` method
3. Use meaningful test method names like `test_worker_registration()`
4. Include detailed docstrings explaining the test purpose
5. Use assertions to verify expected behavior

Example:

```python
def test_something_important(self):
    """Test that something important works correctly."""
    # Setup specific to this test
    
    # Perform the action being tested
    
    # Assert expected outcomes
    self.assertEqual(actual, expected, "Something important should work")
```