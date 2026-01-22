# Test Suite API

This directory contains the API implementation for the refactored test suite. The API provides a consistent interface for running tests, checking status, and retrieving results.

## Architecture

The Test Suite API follows a layered architecture:

```
┌─────────────────────────────────────┐
│           FastAPI Server            │
│      (test_api_server.py)           │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│           Test Runner               │
│         (test_runner.py)            │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│         Test Suite Core             │
│       (model_test_base.py)          │
└─────────────────────────────────────┘
```

### Components

1. **FastAPI Server (test_api_server.py)**:
   - Provides RESTful endpoints and WebSocket support
   - Handles HTTP requests and responses
   - Manages API models and validation

2. **Test Runner (test_runner.py)**:
   - Bridges between API and test execution
   - Manages test execution and status tracking
   - Provides progress reporting and result handling

3. **Test Suite Core**:
   - Executes actual tests using BaseModelTest
   - Handles various test types and hardware platforms
   - Responsible for the actual test logic

## API Endpoints

### Test Execution

- `POST /api/test/run` - Run a test on a model
  - Parameters: `model_name`, `hardware`, `test_type`, `timeout`, `save_results`
  - Returns: `run_id`, `status`, `message`, `started_at`

### Status and Results

- `GET /api/test/status/{run_id}` - Get the status of a test run
  - Returns: Test status information including progress and current step

- `GET /api/test/results/{run_id}` - Get the results of a completed test run
  - Returns: Detailed test results including performance metrics

- `GET /api/test/runs` - List recent test runs
  - Parameters: `limit`, `status` (optional)
  - Returns: List of test runs with basic information

### Metadata

- `GET /api/test/models` - Get a list of available models for testing
  - Returns: List of model information including name, type, and modality

- `GET /api/test/hardware` - Get a list of available hardware platforms
  - Returns: List of hardware platforms with availability information

- `GET /api/test/types` - Get a list of available test types
  - Returns: List of test types with descriptions and included test methods

### Control

- `POST /api/test/cancel/{run_id}` - Cancel a running test
  - Returns: Whether the test was successfully cancelled

### Real-time Updates

- `WebSocket /api/test/ws/{run_id}` - WebSocket endpoint for real-time updates
  - Provides streaming updates on test progress
  - Supports interactive commands (status, cancel)

## Usage

### Starting the API Server

```bash
# Start the API server on the default port (8000)
python -m refactored_test_suite.api.test_api_server

# Start with custom configuration
python -m refactored_test_suite.api.test_api_server --port 8000 --host 0.0.0.0 --results-dir ./test_results
```

### Using the API Client

```python
from refactored_test_suite.api.api_client import ApiClient

# Create client
client = ApiClient(base_url="http://localhost:8000")

# Run a test
response = client.run_test("bert-base-uncased", hardware=["cpu"], test_type="comprehensive")
run_id = response["run_id"]

# Monitor the test until completion
result = client.monitor_test(run_id)

# Print test results
print(f"Test completed with status: {result['status']}")
print(f"Tests passed: {result['results']['tests_passed']}")
print(f"Tests failed: {result['results']['tests_failed']}")
```

### Asynchronous API Usage

```python
from refactored_test_suite.api.api_client import AsyncApiClient
import asyncio

async def run_async_example():
    client = AsyncApiClient(base_url="http://localhost:8000")
    
    # Define update callback
    async def on_update(data):
        print(f"Progress: {data['progress']:.1%} - {data['current_step']}")
    
    # Run a test
    response = await client.run_test("bert-base-uncased")
    run_id = response["run_id"]
    
    # Monitor via WebSocket with updates
    result = await client.monitor_test_ws(run_id, callback=on_update)
    
    print(f"Test completed with status: {result['status']}")

# Run the async example
asyncio.run(run_async_example())
```

## Test Types

### Basic Tests

Basic tests verify fundamental model functionality:
- `test_load_model` - Tests model loading
- `test_basic_inference` - Tests basic inference with minimal input

### Comprehensive Tests

Comprehensive tests provide thorough testing of model capabilities:
- All basic tests
- `test_batch_inference` - Tests inference with batched inputs
- `test_model_attributes` - Tests model attributes and configuration
- `test_save_load` - Tests saving and loading model state

### Fault Tolerance Tests

Fault tolerance tests focus on error handling and recovery:
- All basic tests
- `test_error_handling` - Tests handling of invalid inputs
- `test_resource_cleanup` - Tests proper resource cleanup

## Integration with Other Components

The Test Suite API is designed to integrate with other components through the unified API server:

```
┌───────────────────────────────────────────────────────────┐
│                Unified API Server (Gateway)               │
└───────────┬───────────────────┬───────────────────┬───────┘
            │                   │                   │
┌───────────▼─────┐   ┌─────────▼─────────┐   ┌─────▼───────────┐
│  Test Suite API  │   │  Generator API   │   │  Benchmark API  │
└─────────────────┘   └───────────────────┘   └─────────────────┘
```

This allows for seamless workflows like:
1. Generate model implementation using Generator API
2. Run tests on the generated model using Test Suite API
3. Benchmark the model performance using Benchmark API

See the unified API server documentation for details on these integrations.