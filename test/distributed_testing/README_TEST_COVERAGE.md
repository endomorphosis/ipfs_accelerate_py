# Distributed Testing Framework Test Coverage Guide

This guide documents the test coverage for the Distributed Testing Framework, explaining the testing approach, available tests, and how to run them.

## Testing Approach

The Distributed Testing Framework uses a comprehensive testing approach that includes:

- **Unit Tests**: Tests for individual components in isolation with mocked dependencies
- **Integration Tests**: Tests for interacting components and subsystems
- **Asynchronous Tests**: Tests for asynchronous functionality using `pytest-asyncio`
- **⛔ NOTE**: Security and authentication features are now **OUT OF SCOPE** (see SECURITY_DEPRECATED.md)

The tests are organized by component, with each component having its own test file in the `tests/` directory.

## Test Components

The framework includes tests for the following critical components:

| Component | Test File | Description |
|-----------|-----------|-------------|
| Worker | `test_worker.py` | Tests the worker node component, including task execution, heartbeat mechanism, and health monitoring |
| Coordinator | `test_coordinator.py` | Tests the coordinator server component, including worker management, task distribution, and result aggregation |
| ~~Security~~ | ~~`test_security.py`~~ | ⛔ **OUT OF SCOPE** - Security features moved to separate module |
| Load Balancer | `test_load_balancer.py` | Tests the adaptive load balancer for efficient task distribution |
| Task Scheduler | `test_task_scheduler.py` | Tests the task scheduler for optimal task assignment |
| Distributed Error Handler | `test_distributed_error_handler.py` | Tests error classification, recovery strategies, and retry policies |
| Health Monitor | `test_health_monitor.py` | Tests system health monitoring and reporting |
| Coordinator Redundancy | `test_coordinator_redundancy.py` | Tests failover and high-availability features |
| CI Integration | `test_ci_integration.py` | Tests CI/CD system integration |
| Fault Tolerance | `test_fault_tolerance.py` | Tests system behavior during failures |
| Browser Recovery | `test_selenium_browser_integration.py` | Tests browser recovery strategies with Selenium |
| E2E Browser Testing | `selenium_e2e_browser_recovery_test.py` | End-to-end browser testing with recovery capabilities |
| Real Browser Testing | `run_real_browser_test.py` | Direct testing with real browsers and WebGPU/WebNN detection |

## Running Tests

The framework includes a test runner script (`run_test_distributed_framework.py`) that can run all tests or specific test types, with or without coverage reporting.

### Prerequisites

Install the required packages for testing:

```bash
pip install pytest pytest-asyncio coverage
```

### Running All Tests

To run all tests (unit and integration):

```bash
python run_test_distributed_framework.py
```

### Running Specific Test Types

To run only unit tests:

```bash
python run_test_distributed_framework.py --unit
```

To run only integration tests:

```bash
python run_test_distributed_framework.py --integration
```

### Running with Coverage

To run tests with coverage reporting:

```bash
python run_test_distributed_framework.py --coverage
```

This will generate a coverage report in the terminal and an HTML report in the `htmlcov/` directory.

### Running Individual Test Files

You can also run individual test files directly:

```bash
# Run with unittest
python -m unittest tests/test_worker.py

# Run with pytest
pytest tests/test_worker.py
```

### Running Browser Recovery Tests

The framework includes extensive browser recovery tests:

```bash
# Run real browser test with Chrome
./run_real_browser_test.sh --chrome --bert --webgpu

# Run real browser test with Firefox optimized for audio
./run_real_browser_test.sh --firefox --whisper --webgpu

# Run real browser test with Edge optimized for text with WebNN
./run_real_browser_test.sh --edge --bert --webnn

# Run comprehensive browser tests
./run_comprehensive_browser_tests.sh --standard

# Run end-to-end browser recovery tests
./run_selenium_e2e_tests.sh --chrome-only --text-only
```

These tests validate the browser recovery strategies with different browser types and model configurations, providing a comprehensive verification of the WebGPU/WebNN integration and recovery capabilities.

## Test Coverage Goals

The framework aims for high test coverage with these specific goals:

1. **Code Coverage**: At least 80% overall code coverage
2. **Critical Component Coverage**: At least 90% coverage for critical components
3. **~~Security Components~~**: ⛔ **OUT OF SCOPE** - Security features moved to separate module
4. **Error Handling**: Comprehensive testing of error paths and edge cases

## Test Organization

Tests are organized according to these principles:

1. **Component Isolation**: Unit tests test components in isolation with mocks
2. **Integration Testing**: Integration tests verify component interactions
3. **Error Path Testing**: Tests verify correct behavior for both success and failure paths
4. **Edge Case Coverage**: Tests include boundary conditions and edge cases
5. **~~Security Verification~~**: ⛔ **OUT OF SCOPE** - Security features moved to separate module

## Advanced Testing Features

The test framework supports these advanced features:

### Async Testing Support

For testing asynchronous code (like WebSocket communication):

```python
@pytest.mark.asyncio
async def test_async_functionality():
    # Test async code
    result = await async_function()
    assert result == expected_value
```

### Mocking External Dependencies

Tests use `unittest.mock` to isolate components:

```python
# Mock database
with patch('coordinator.duckdb.connect') as mock_db:
    # Configure mock
    mock_db.return_value.execute.return_value = mock_result
    
    # Test with mocked database
    result = function_under_test()
    
    # Verify interactions with mock
    mock_db.assert_called_once()
```

### Testing WebSocket Communication

For testing WebSocket-based communication:

```python
# Mock WebSocket
mock_ws = AsyncMock()
mock_ws.send = AsyncMock()
mock_ws.recv = AsyncMock(return_value=json.dumps({"status": "success"}))

# Test WebSocket communication
result = await communicate_over_websocket(mock_ws)
```

## Adding New Tests

When adding new components or features to the framework, follow these guidelines for testing:

1. **Create a Test File**: Create a new test file in the `tests/` directory named `test_<component>.py`
2. **Test Both Success and Failure**: Include tests for both success and failure paths
3. **Mock Dependencies**: Use mocks to isolate the component being tested
4. **Test Async Code**: Use `pytest.mark.asyncio` for testing asynchronous code
5. **Mock Authentication**: Use simple mocks for authentication (actual security is out of scope)
6. **Run with Coverage**: Check coverage to ensure adequate testing

## Continuous Integration

The test framework is integrated with CI/CD systems to ensure consistent testing:

- Tests run automatically for all pull requests
- Coverage reports are generated and tracked
- Test failures block merging of changes

## Troubleshooting Tests

Common issues and solutions:

1. **Async Test Timeouts**: Increase timeout or fix event loop behavior
2. **Mock Configuration**: Ensure mocks are configured correctly for expected behavior
3. **WebSocket Test Failures**: Verify WebSocket message formats and sequences
4. **Database Test Issues**: Ensure database mocks return expected data structures
5. **Auth Mock Issues**: Ensure authentication mocks are configured to allow tests to pass (actual security is out of scope)