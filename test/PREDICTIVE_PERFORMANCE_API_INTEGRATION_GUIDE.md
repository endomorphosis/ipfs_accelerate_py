# Predictive Performance API Integration Guide

This guide explains how to use the Predictive Performance API integrated with the Unified API Server. The integration enables access to hardware recommendations, performance predictions, measurement tracking, and analysis through a consistent API.

## Overview

The Predictive Performance API extends the IPFS Accelerate platform with predictive modeling capabilities for hardware selection and performance optimization. It integrates with the Unified API Server to provide a cohesive interface alongside other components like the Test Suite API, Generator API, and Benchmark API.

### Key Features

- **Hardware Recommendations**: Get optimal hardware recommendations for models based on their characteristics
- **Performance Predictions**: Predict performance metrics (throughput, latency, memory usage) for models on different hardware
- **Measurement Tracking**: Record actual performance measurements and compare with predictions
- **Analysis**: Analyze prediction accuracy and identify improvement opportunities
- **Machine Learning Models**: Store and use ML models for performance prediction
- **WebSocket Support**: Real-time progress tracking for long-running operations
- **DuckDB Integration**: Persistent storage of predictions, measurements, and ML models

## Architecture

The integration consists of several components:

1. **Predictive Performance API Server**: Standalone FastAPI server with RESTful endpoints and WebSocket support
   - Located at `/test/api_server/predictive_performance_api_server.py`
   - Provides comprehensive RESTful API for all predictive performance functionality
   - Supports background task processing for long-running operations
   - Includes WebSocket endpoints for real-time progress tracking

2. **DuckDB Integration**: Repository and adapters for persistent storage
   - Located at `/duckdb_api/predictive_performance/predictor_repository.py`
   - Provides a repository layer for storing predictions, measurements, and ML models
   - Supports querying and analyzing prediction accuracy

3. **Integration Module**: Connects the Predictive Performance API with the Unified API Server
   - Located at `/test/api_server/integrations/predictive_performance_integration.py`
   - Provides service management for starting/stopping the Predictive Performance API
   - Supplies patch code for the Unified API Server integration

4. **API Gateway**: Routes in the Unified API Server gateway for Predictive Performance endpoints
   - Routes are automatically added to the Unified API Server by the update script
   - Provides unified access through `http://localhost:8080/api/predictive-performance/`
   - Forwards requests to the actual Predictive Performance API server

5. **Client Library**: Python library for easy access to the API
   - Located at `/test/api_client/predictive_performance_client.py`
   - Provides both synchronous and asynchronous client implementations
   - Includes comprehensive error handling and WebSocket support

## Running the Integrated Servers

To run the Predictive Performance API with the Unified API Server, use the `run_integrated_api_servers.py` script:

```bash
# Run both servers
python test/run_integrated_api_servers.py

# Run with custom ports
python test/run_integrated_api_servers.py --gateway-port 8080 --predictive-port 8500

# Run with custom database path
python test/run_integrated_api_servers.py --db-path /path/to/database.duckdb

# Run only the Predictive Performance API server
python test/run_integrated_api_servers.py --predictive-only

# Run only the Unified API server (assumes Predictive Performance API is running)
python test/run_integrated_api_servers.py --unified-only
```

## API Endpoints

### Hardware Recommendations

```
POST /api/predictive-performance/predict-hardware
```

Predicts the optimal hardware for a given model and configuration.

Request:
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "batch_size": 8,
  "sequence_length": 128,
  "mode": "inference",
  "precision": "fp16",
  "available_hardware": ["cuda", "cpu", "rocm"],
  "predict_performance": true
}
```

### Performance Predictions

```
POST /api/predictive-performance/predict-performance
```

Predicts performance metrics for a model on specified hardware.

Request:
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "hardware": ["cuda", "cpu"],
  "batch_size": 8,
  "sequence_length": 128,
  "mode": "inference",
  "precision": "fp16"
}
```

### Record Measurements

```
POST /api/predictive-performance/record-measurement
```

Records an actual performance measurement and compares with predictions if available.

Request:
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "hardware_platform": "cuda",
  "batch_size": 8,
  "sequence_length": 128,
  "precision": "fp16",
  "mode": "inference",
  "throughput": 123.45,
  "latency": 7.89,
  "memory_usage": 1024.5,
  "source": "benchmark"
}
```

### Analyze Predictions

```
POST /api/predictive-performance/analyze-predictions
```

Analyzes the accuracy of performance predictions compared to actual measurements.

Request:
```json
{
  "model_name": "bert-base-uncased",
  "hardware_platform": "cuda",
  "metric": "throughput",
  "days": 30
}
```

### List Resources

```
GET /api/predictive-performance/recommendations
GET /api/predictive-performance/predictions
GET /api/predictive-performance/measurements
GET /api/predictive-performance/models
```

List various resources with optional filters.

### Task Management Endpoints

```
GET /api/predictive-performance/task/{task_id}
```

Get the status of a specific task. Includes progress, status, and result if completed.

Response:
```json
{
  "status": "completed",
  "progress": 1.0,
  "message": "Task completed successfully",
  "timestamp": "2025-08-15T14:30:45.123456",
  "result": {
    "primary_recommendation": "cuda",
    "alternative_recommendations": ["rocm", "cpu"],
    "model_family": "embedding",
    "performance": {
      "throughput": 256.78,
      "latency": 3.9,
      "memory_usage": 1280.5
    }
  }
}
```

### WebSocket Endpoint

```
WebSocket /api/predictive-performance/ws/{task_id}
```

Real-time progress tracking for long-running tasks. Connect to this endpoint to receive push updates as tasks progress, without having to poll the task status endpoint.

Example WebSocket message:
```json
{
  "task_id": "hardware-f8a7b3c2-d1e0-4f6a-9b8a-7c6d5e4f3a2b",
  "status": "running",
  "progress": 0.65,
  "message": "Predicting performance metrics",
  "timestamp": "2025-08-15T14:30:40.123456"
}
```

### Database Access Endpoints

Through the Unified API Server, database operations are accessible with API key authentication:

```
GET /api/db/predictive-performance/measurements
GET /api/db/predictive-performance/predictions
GET /api/db/predictive-performance/recommendations
GET /api/db/predictive-performance/models
```

These endpoints provide direct access to database records with advanced filtering.

## Client Library

The Python client library provides an easy way to interact with the API:

```python
from test.api_client.predictive_performance_client import (
    PredictivePerformanceClient,
    HardwarePlatform,
    PrecisionType,
    ModelMode
)

# Create client
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

# List recommendations
recommendations = client.list_recommendations(limit=5)
print(f"Found {recommendations['count']} recommendations")
```

The library also provides an asynchronous client for use with asyncio:

```python
import asyncio
from test.api_client.predictive_performance_client import AsyncPredictivePerformanceClient

async def main():
    client = AsyncPredictivePerformanceClient()
    try:
        result = await client.predict_hardware(
            model_name="bert-base-uncased",
            batch_size=8,
            wait=True
        )
        print(result)
    finally:
        await client.close()

asyncio.run(main())
```

## Modifying the Integration

If you need to modify the integration, use the `update_unified_api.py` script:

```bash
# Update the Unified API Server with Predictive Performance API integration
python test/api_server/update_unified_api.py

# Restore from backup
python test/api_server/update_unified_api.py --restore
```

## Database Schema

The DuckDB integration provides tables for:

- Predictions: Store performance predictions for model-hardware pairs
- Measurements: Record actual performance measurements
- Recommendations: Hardware recommendations for models
- Prediction Models: Machine learning models for predicting performance
- Feedback: User feedback on recommendations
- Feature Importance: Feature importance scores for ML models

## Configuration

The Predictive Performance API can be configured with environment variables or command-line arguments:

- `PREDICTIVE_PERFORMANCE_DB`: Path to DuckDB database file
- `BENCHMARK_DIR`: Path to benchmark results directory
- `BENCHMARK_DB`: Path to benchmark database

Command-line arguments:
- `--host`: Host to bind server (default: 127.0.0.1)
- `--port`: Port to bind server (default: 8500)
- `--db`: Path to DuckDB database
- `--benchmark-dir`: Path to benchmark results directory
- `--benchmark-db`: Path to benchmark database

## Security

The Unified API Server applies its API key validation to database operations. To authenticate, include the `X-API-Key` header with your requests.

## Troubleshooting

Common issues:

1. **Connection refused**: Ensure the servers are running on the expected ports
2. **Missing dependencies**: Install required packages with `pip install fastapi uvicorn websockets requests duckdb`
3. **Database errors**: Check the database path and ensure it exists with proper permissions
4. **Task timeout**: Increase the timeout value for long-running operations

## Examples

### Batch Size Performance Analysis

```python
# Analyze performance across batch sizes
client = PredictivePerformanceClient()
batch_sizes = [1, 2, 4, 8, 16, 32]
results = {}

for batch in batch_sizes:
    result = client.predict_performance(
        model_name="bert-base-uncased",
        hardware=HardwarePlatform.CUDA,
        batch_size=batch,
        wait=True
    )
    if "result" in result and "predictions" in result["result"]:
        results[batch] = result["result"]["predictions"]["cuda"]["throughput"]

# Print results
for batch, throughput in results.items():
    print(f"Batch size {batch}: {throughput:.2f} samples/sec")
```

### Finding the Best Hardware for a Model

```python
# Find the best hardware for a model
client = PredictivePerformanceClient()
result = client.predict_hardware(
    model_name="bert-base-uncased",
    available_hardware=[
        HardwarePlatform.CPU,
        HardwarePlatform.CUDA,
        HardwarePlatform.ROCM,
        HardwarePlatform.WEBGPU
    ],
    predict_performance=True,
    wait=True
)

# Print recommendations
if "result" in result:
    primary = result["result"]["primary_recommendation"]
    print(f"Primary recommendation: {primary}")
    
    alternatives = result["result"].get("alternative_recommendations", [])
    if alternatives:
        print("Alternative recommendations:")
        for alt in alternatives:
            print(f"- {alt}")
    
    if "performance" in result["result"]:
        perf = result["result"]["performance"]
        print(f"Predicted performance on {primary}:")
        print(f"- Throughput: {perf.get('throughput', 'N/A')} samples/sec")
        print(f"- Latency: {perf.get('latency', 'N/A')} ms")
        print(f"- Memory usage: {perf.get('memory_usage', 'N/A')} MB")
```

## Integration with Other Components

The Predictive Performance API is designed to integrate with other components in the IPFS Accelerate ecosystem:

### Integration with Benchmark API

You can automatically feed benchmark results into the Predictive Performance system:

```python
from test.api_client.predictive_performance_client import PredictivePerformanceClient
from test.refactored_benchmark_suite.api_client import BenchmarkApiClient

# Create clients
benchmark_client = BenchmarkApiClient(base_url="http://localhost:8080")
predictive_client = PredictivePerformanceClient(base_url="http://localhost:8080")

# Run a benchmark
benchmark_result = benchmark_client.run_benchmark(
    model_name="bert-base-uncased",
    hardware="cuda",
    batch_size=8,
    wait=True
)

# Record the benchmark result in the Predictive Performance system
if benchmark_result["status"] == "completed":
    metrics = benchmark_result["result"]["metrics"]
    predictive_client.record_measurement(
        model_name="bert-base-uncased",
        hardware_platform="cuda",
        batch_size=8,
        throughput=metrics["throughput"],
        latency=metrics["latency"],
        memory_usage=metrics["memory_usage"],
        source="benchmark",
        wait=True
    )
```

### Integration with Test API

You can use hardware recommendations when running tests:

```python
from test.api_client.predictive_performance_client import PredictivePerformanceClient
from test.refactored_test_suite.api.api_client import ApiClient

# Create clients
test_client = ApiClient(base_url="http://localhost:8080")
predictive_client = PredictivePerformanceClient(base_url="http://localhost:8080")

# Get hardware recommendation
hw_result = predictive_client.predict_hardware(
    model_name="bert-base-uncased",
    wait=True
)

if "result" in hw_result:
    recommended_hw = hw_result["result"]["primary_recommendation"]
    
    # Run test on recommended hardware
    test_response = test_client.run_test(
        model_name="bert-base-uncased",
        hardware=recommended_hw
    )
```

### Cross-Component Database Queries

Through the Unified API Server, you can perform cross-component queries:

```python
import requests

# Query the unified database view for a specific model
response = requests.get(
    "http://localhost:8080/api/db/model/bert-base-uncased",
    headers={"X-API-Key": "your-api-key"}
)

if response.status_code == 200:
    data = response.json()
    
    # Access data from different components
    test_data = data["recent_test_runs"]
    benchmark_data = data["recent_benchmark_runs"]
    prediction_data = data["recent_predictions"]
    
    # Use the combined data for analysis
    print(f"Test success rate: {data['overview']['test_success_rate']:.2f}")
    print(f"Benchmark performance: {benchmark_data[0]['throughput'] if benchmark_data else 'N/A'}")
    print(f"Predicted performance: {prediction_data[0]['throughput'] if prediction_data else 'N/A'}")
```

## Next Steps

1. **Create prediction models**: Train ML models for specific hardware platforms using data collected from benchmarks
2. **Integrate with benchmarking**: Automatically feed benchmark results into the measurement database
3. **Visualize predictions**: Create interactive dashboards for monitoring prediction accuracy and analyzing trends
4. **Expand hardware support**: Add support for new hardware platforms like specialized AI accelerators and mobile processors
5. **Optimize recommendations**: Improve the recommendation algorithm based on collected feedback
6. **Implement automated feature importance analysis**: Identify which model and hardware characteristics most affect performance
7. **Add cross-component analysis tools**: Develop tools that combine insights from tests, benchmarks, and predictions
8. **Create a visualization dashboard**: Build a web interface for exploring prediction accuracy and performance trends