# IPFS Accelerate Documentation Index

This document serves as a central index to all documentation for the IPFS Accelerate project.

## API Metrics and Simulation Validation

- [API Management UI README](API_MANAGEMENT_UI_README.md): Documentation for the API Management UI dashboard
- [API Monitoring README](API_MONITORING_README.md): Guide to monitoring API performance and reliability
- [API Distributed Testing Guide](API_DISTRIBUTED_TESTING_GUIDE.md): Guide to running distributed tests for APIs
- [Predictive Analytics README](PREDICTIVE_ANALYTICS_README.md): Documentation for the predictive analytics system

## Hardware Optimization and Visualization

- [HARDWARE_OPTIMIZATION_GUIDE.md](HARDWARE_OPTIMIZATION_GUIDE.md): Guide to hardware-specific optimizations
- [OPTIMIZATION_EXPORTER_README.md](OPTIMIZATION_EXPORTER_README.md): Documentation for the Hardware Optimization Exporter
- [ENHANCED_VISUALIZATION_EXPORT_GUIDE.md](ENHANCED_VISUALIZATION_EXPORT_GUIDE.md): Guide to the Enhanced Visualization UI for exports
- [DuckDB Simulation Validation](../duckdb_api/simulation_validation/README.md): Documentation for the simulation validation module
- [API Metrics Validation Guide](API_METRICS_VALIDATION_GUIDE.md): Comprehensive guide to API metrics validation
- [API Metrics Validation Tool](run_api_metrics_validation.py): CLI tool for validating API metrics
- [Calibration DuckDB Integration Guide](CALIBRATION_DUCKDB_INTEGRATION_GUIDE.md): Guide to using DuckDB with simulation calibration
- [Calibration DuckDB CLI Tool](run_calibration_with_duckdb.py): Command-line tool for calibration with DuckDB integration
- [Predictive Performance DuckDB Integration Guide](PREDICTIVE_PERFORMANCE_DUCKDB_INTEGRATION_GUIDE.md): Guide to using DuckDB with predictive performance modeling
- [Predictive Performance DuckDB CLI Tool](run_predictive_performance_with_duckdb.py): Command-line tool for predictive performance with DuckDB integration
- [Predictive Performance API Integration Guide](PREDICTIVE_PERFORMANCE_API_INTEGRATION_GUIDE.md): Guide to the FastAPI integration for predictive performance

## Mobile Edge Support

### Planning Documentation
- [MOBILE_EDGE_EXPANSION_PLAN.md](MOBILE_EDGE_EXPANSION_PLAN.md) - Strategic plan for mobile and edge device support
- [MOBILE_EDGE_CI_INTEGRATION_PLAN.md](MOBILE_EDGE_CI_INTEGRATION_PLAN.md) - Plan for CI/CD integration of mobile testing

### Implementation Components
- [CROSS_PLATFORM_ANALYSIS_GUIDE.md](CROSS_PLATFORM_ANALYSIS_GUIDE.md) - Guide to cross-platform mobile analysis tool
- Android Test Harness - See `android_test_harness` directory
  - `android_test_harness/README.md` - Android test harness documentation
  - `android_test_harness/run_ci_benchmarks.py` - CI benchmark runner for Android
- iOS Test Harness - See `ios_test_harness` directory
  - `ios_test_harness/README.md` - iOS test harness documentation
  - `ios_test_harness/run_ci_benchmarks.py` - CI benchmark runner for iOS

### CI/CD Integration Tools
- `merge_benchmark_databases.py` - Utility for merging benchmark databases from different platforms
- `check_mobile_regressions.py` - Tool for detecting performance regressions in mobile benchmarks
- `generate_mobile_dashboard.py` - Dashboard generator for mobile performance visualization
- `test_mobile_ci_integration.py` - Test script for the mobile CI integration components

## API Documentation

### FastAPI Integration

- [FASTAPI_INTEGRATION_GUIDE.md](FASTAPI_INTEGRATION_GUIDE.md) - Comprehensive guide to FastAPI integration
- [API_INTEGRATION_PLAN.md](refactored_test_suite/integration/API_INTEGRATION_PLAN.md) - Detailed API refactoring plan
- [API_DUCKDB_INTEGRATION.md](API_DUCKDB_INTEGRATION.md) - DuckDB integration for API components
- [API_UNIFIED_DB_INTEGRATION.md](API_UNIFIED_DB_INTEGRATION.md) - Unified API Server with cross-component database operations
- [BENCHMARK_FASTAPI_DASHBOARD.md](refactored_benchmark_suite/BENCHMARK_FASTAPI_DASHBOARD.md) - Benchmark API dashboard documentation

### API Components

- **Test Suite API** - Endpoints for running tests and retrieving results
  - API Server: `/test/refactored_test_suite/api/test_api_server.py`
  - Test Runner: `/test/refactored_test_suite/api/test_runner.py`
  - API Client: `/test/refactored_test_suite/api/api_client.py`
  - Base URL: `http://localhost:8000/api/test/`
  - Documentation: `/test/refactored_test_suite/api/README.md`
  - Database Integration: `/test/refactored_test_suite/database/`
  - DB Documentation: `/test/refactored_test_suite/database/README.md`

- **Generator API** - Endpoints for generating model implementations
  - Location: `/test/refactored_generator_suite/generator_api_server.py`
  - Base URL: `http://localhost:8001/api/generator/`
  - Database Integration: `/test/refactored_generator_suite/database/`
  - Documentation: [GENERATOR_DUCKDB_INTEGRATION.md](GENERATOR_DUCKDB_INTEGRATION.md)

- **Benchmark API** - Endpoints for running benchmarks and retrieving metrics
  - Location: `/test/refactored_benchmark_suite/benchmark_api_server.py`
  - Base URL: `http://localhost:8002/api/benchmark/`

- **Predictive Performance API** - Endpoints for hardware recommendations and performance predictions
  - API Server: `/test/api_server/predictive_performance_api_server.py`
  - Integration Module: `/test/api_server/integrations/predictive_performance_integration.py`
  - API Client: `/test/api_client/predictive_performance_client.py`
  - Base URL: `http://localhost:8500/api/predictive-performance/`
  - Gateway URL: `http://localhost:8080/api/predictive-performance/`
  - Documentation: [PREDICTIVE_PERFORMANCE_API_INTEGRATION_GUIDE.md](PREDICTIVE_PERFORMANCE_API_INTEGRATION_GUIDE.md)
  - Example Usage: `/test/demo_predictive_performance_api.py`
  - Run Script: `/test/run_integrated_api_servers.py`

- **Unified API Server** - Gateway to all API components (✅ COMPLETED - 100%)
  - Location: `/test/unified_api_server.py`
  - Base URL: `http://localhost:8080/api/`
  - Database Operations: `http://localhost:8080/api/db/`
  - Cross-Component Database: `http://localhost:8080/api/db/overview` and `http://localhost:8080/api/db/model/{model_name}`
  - Authentication: API key required via `X-API-Key` header for database operations
  - Documentation: [API_UNIFIED_DB_INTEGRATION.md](API_UNIFIED_DB_INTEGRATION.md)
  - Example Usage: 
    - `/test/examples/api_integration_example.py` - General API integration example
    - `/test/examples/unified_db_example.py` - Unified database operations example

## Project Modules

### Core Modules

- [CLAUDE.md](CLAUDE.md) - Central project status and development guide
- [README.md](../README.md) - Main project documentation

### Distributed Testing Framework

- [DISTRIBUTED_TESTING_DESIGN.md](DISTRIBUTED_TESTING_DESIGN.md) - Architecture design
- [DISTRIBUTED_TESTING_GUIDE.md](DISTRIBUTED_TESTING_GUIDE.md) - Usage guide
- [DISTRIBUTED_TESTING_COMPLETION.md](DISTRIBUTED_TESTING_COMPLETION.md) - Completion report

### Fault Tolerance and High Availability

- [HARDWARE_FAULT_TOLERANCE_GUIDE.md](HARDWARE_FAULT_TOLERANCE_GUIDE.md) - Fault tolerance implementation
- [REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md](REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md) - Performance monitoring
- [DYNAMIC_RESOURCE_MANAGEMENT.md](DYNAMIC_RESOURCE_MANAGEMENT.md) - Resource management

### Refactored Components

- [README_TEST_REFACTORING_IMPLEMENTATION.md](README_TEST_REFACTORING_IMPLEMENTATION.md) - Test suite refactoring
- [refactored_test_suite/README.md](refactored_test_suite/README.md) - Test suite documentation
- [refactored_generator_suite/README.md](refactored_generator_suite/README.md) - Generator suite documentation
- [refactored_benchmark_suite/README.md](refactored_benchmark_suite/README.md) - Benchmark suite documentation

### Cross-Browser Features

- [CROSS_BROWSER_MODEL_SHARDING_TESTING_GUIDE.md](CROSS_BROWSER_MODEL_SHARDING_TESTING_GUIDE.md) - Model sharding implementation
- [WEB_RESOURCE_POOL_FAULT_TOLERANCE_README.md](WEB_RESOURCE_POOL_FAULT_TOLERANCE_README.md) - Resource pool fault tolerance
- [IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md](IPFS_CROSS_MODEL_TENSOR_SHARING_GUIDE.md) - Tensor sharing

## Guides

### Installation and Setup

- [install/README.md](../install/README.md) - Installation instructions
- [setup.py](../setup.py) - Package setup

### API Usage

- [integration_workflow_example.py](integration_workflow_example.py) - End-to-end API workflow example
- [examples/unified_db_example.py](examples/unified_db_example.py) - Unified database operations example with cross-component queries

### Development Guides

- [COMPREHENSIVE_TEST_REFACTORING_PLAN.md](COMPREHENSIVE_TEST_REFACTORING_PLAN.md) - Test refactoring plan
- [TEST_REFACTORING_SUMMARY.md](TEST_REFACTORING_SUMMARY.md) - Test refactoring summary

## Running the API Servers

### Unified API Server

To run the unified API server that provides access to all component APIs and database operations:

```bash
python test/unified_api_server.py --gateway-port 8080
```

This starts a gateway on port 8080 that forwards requests to the component APIs and provides cross-component database operations.

For database operations, you need to provide an API key in the `X-API-Key` header:

```bash
# Example curl request to get database overview
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/db/overview

# Example curl request to get unified model data
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/db/model/bert-base-uncased
```

The unified API server also provides a configuration file option for more detailed setup:

```bash
python test/unified_api_server.py --config config.json
```

See [API_UNIFIED_DB_INTEGRATION.md](API_UNIFIED_DB_INTEGRATION.md) for more details on database operations.

### Individual Component Servers

To run the individual API servers directly:

```bash
# Test Suite API (with DuckDB integration)
python -m test.refactored_test_suite.api.test_api_server --db-path ./data/test_runs.duckdb

# Generator API (with DuckDB integration)
python -m test.refactored_generator_suite.generator_api_server --db-path ./data/generator_tasks.duckdb

# Benchmark API
python -m test.refactored_benchmark_suite.benchmark_api_server

# Predictive Performance API
python -m test.api_server.predictive_performance_api_server --db-path ./data/predictive_performance.duckdb
```

### Integrated API Servers

To run both the Predictive Performance API and the Unified API Server together:

```bash
# Run both servers with default settings
python test/run_integrated_api_servers.py

# Run with custom ports
python test/run_integrated_api_servers.py --gateway-port 8080 --predictive-port 8500

# Run with custom database path
python test/run_integrated_api_servers.py --db-path ./data/predictive_performance.duckdb

# Run only the Predictive Performance API
python test/run_integrated_api_servers.py --predictive-only

# Run only the Unified API Server (assumes Predictive Performance API is running)
python test/run_integrated_api_servers.py --unified-only
```

### Demo the Predictive Performance API

To run the Predictive Performance API demo:

```bash
# Generate sample data and run all demos
python test/demo_predictive_performance_api.py --setup --all

# Run specific demos
python test/demo_predictive_performance_api.py --hardware  # Hardware recommendations demo
python test/demo_predictive_performance_api.py --performance  # Performance predictions demo
python test/demo_predictive_performance_api.py --batch  # Batch size analysis demo
```

### Integration Workflow Example

To run the end-to-end integration workflow example:

```bash
python test/integration_workflow_example.py --model bert-base-uncased
```

### Unified Database Example

To run the unified database operations example:

```bash
python test/examples/unified_db_example.py --api-key your-api-key --model bert-base-uncased
```

This example shows how to access cross-component database operations, including:
- Database overview across all components
- Unified model data from all components
- Cross-component search operations

## API Client Usage

### Test Suite API Client

The Test Suite API client provides a convenient way to interact with the Test API:

```python
from test.refactored_test_suite.api.api_client import ApiClient

# Create client
client = ApiClient(base_url="http://localhost:8000")

# Run a test
response = client.run_test("bert-base-uncased")
run_id = response["run_id"]

# Monitor the test until completion
result = client.monitor_test(run_id)

# Print result
print(f"Test completed with status: {result['status']}")
```

For asynchronous usage:

```python
from test.refactored_test_suite.api.api_client import AsyncApiClient
import asyncio

async def run_async_example():
    client = AsyncApiClient(base_url="http://localhost:8000")
    response = await client.run_test("bert-base-uncased")
    result = await client.monitor_test_ws(response["run_id"])
    print(f"Test completed with status: {result['status']}")

# Run the async example
asyncio.run(run_async_example())
```

### Predictive Performance API Client

The Predictive Performance API client provides a simple interface for hardware recommendations and performance predictions:

```python
from test.api_client.predictive_performance_client import (
    PredictivePerformanceClient,
    HardwarePlatform,
    PrecisionType,
    ModelMode
)

# Create client (pointing to Unified API Server gateway)
client = PredictivePerformanceClient(base_url="http://localhost:8080")

# Predict optimal hardware
result = client.predict_hardware(
    model_name="bert-base-uncased",
    batch_size=8,
    available_hardware=[HardwarePlatform.CPU, HardwarePlatform.CUDA],
    predict_performance=True,
    wait=True  # Wait for task completion
)
print(f"Hardware recommendation: {result['result']['primary_recommendation']}")

# Predict performance metrics
result = client.predict_performance(
    model_name="bert-base-uncased",
    hardware=[HardwarePlatform.CPU, HardwarePlatform.CUDA],
    batch_size=8,
    wait=True
)
print(f"Predicted throughput on CPU: {result['result']['predictions']['cpu']['throughput']}")

# Record a measurement
client.record_measurement(
    model_name="bert-base-uncased",
    hardware_platform=HardwarePlatform.CUDA,
    batch_size=8,
    throughput=120.5,
    latency=8.3,
    memory_usage=1024.0,
    wait=True
)
```

For asynchronous usage:

```python
from test.api_client.predictive_performance_client import AsyncPredictivePerformanceClient
import asyncio

async def run_async_example():
    client = AsyncPredictivePerformanceClient(base_url="http://localhost:8080")
    try:
        result = await client.predict_hardware(
            model_name="bert-base-uncased",
            batch_size=8,
            wait=True
        )
        print(f"Hardware recommendation: {result['result']['primary_recommendation']}")
    finally:
        await client.close()

# Run the async example
asyncio.run(run_async_example())
```
