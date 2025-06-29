# OpenVINO Model Server (OVMS) API Testing Guide

This document outlines how to test the OpenVINO Model Server (OVMS) API backend implementation in the IPFS Accelerate framework.

## Overview

OpenVINO Model Server is a high-performance solution for serving deep learning models optimized for Intel hardware. The IPFS Accelerate framework provides a fully-featured OVMS API client implementation with the following capabilities:

1. **Complete API Implementation**: Provides full access to all OVMS features
2. **Per-Endpoint API Key Support**: Allows multiple authentication configurations
3. **Thread-Safe Request Queuing**: Manages concurrent requests efficiently
4. **Exponential Backoff**: Handles rate limits and transient errors
5. **Request Tracking**: Monitors requests with unique IDs
6. **Performance Metrics**: Tracks statistics and processing times

The test suite validates all these aspects and ensures the implementation meets production standards.

## Test Files

- `test_ovms.py` - Core API implementation tests
- `test_ovms_unified.py` - Unified test runner with performance and advanced feature tests
- `OVMS_TESTING_README.md` - This documentation file

## Requirements

- Python 3.8+
- OVMS server (for real connection tests)
- Python packages: requests, numpy, unittest

## Running Tests

### Standard API Tests

Tests the implementation of the OVMS API client:

```bash
python generators/models/test_ovms_unified.py --standard
```

This validates:
- Endpoint handler creation
- API request formatting
- Response parsing
- Error handling
- Model compatibility checks
- Model metadata retrieval
- Inference methods

### Performance Tests

Benchmarks the performance of the API:

```bash
python generators/models/test_ovms_unified.py --performance
```

This measures:
- Single inference time
- Batch inference time
- Instances per second throughput
- Batch speedup factor
- Request formatting efficiency
- Different input format performance

### Real Connection Tests

Tests connection to an actual OVMS server:

```bash
python generators/models/test_ovms_unified.py --real
```

This verifies:
- Server availability
- Model availability
- Basic inference functionality
- Version-specific inference
- Batch inference capabilities
- Model configuration retrieval
- Server status checks

### Advanced Feature Tests

Tests specialized OVMS features:

```bash
python generators/models/test_ovms_unified.py --advanced
```

This tests:
- Different input formats
- Model execution modes (latency vs. throughput)
- Model configuration options
- Model reload functionality
- Server statistics retrieval
- Prediction explanations
- Specialized metadata features

### All Tests

Run all test suites:

```bash
python generators/models/test_ovms_unified.py --all
```

## Configuration Options

You can customize the tests with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model to use for testing | model |
| `--api-url` | URL for OVMS API | http://localhost:9000 |
| `--timeout` | Timeout in seconds for API requests | 30 |
| `--version` | Specific model version to use | latest |
| `--precision` | Model precision to test (FP32, FP16, INT8) | FP32 |

Example with custom configuration:
```bash
python generators/models/test_ovms_unified.py --all --model resnet --api-url http://ovms-server:9000 --timeout 60 --version 2 --precision FP16
```

## Environment Variables

You can also set configuration via environment variables:

- `OVMS_API_URL` - URL for the OVMS API
- `OVMS_MODEL` - Model to use for testing
- `OVMS_VERSION` - Model version to use for testing
- `OVMS_PRECISION` - Precision to use for testing (FP32, FP16, INT8)
- `OVMS_TIMEOUT` - Timeout in seconds
- `SKIP_PERFORMANCE_TESTS` - Set to "true" to skip performance tests
- `SKIP_REAL_TESTS` - Set to "true" to skip real connection tests
- `SKIP_ADVANCED_TESTS` - Set to "true" to skip advanced feature tests

## Test Results

Test results are saved in the `collected_results/` directory:

- `ovms_test_results.json` - Standard API test results
- `ovms_performance_*.json` - Performance test results
- `ovms_connection_*.json` - Real connection test results
- `ovms_advanced_*.json` - Advanced feature test results
- `ovms_summary_*.json` - Test summary reports

## Extended API Features

The OVMS implementation includes the following advanced features that have been added in the latest version:

### Per-Endpoint Configuration

The OVMS implementation supports per-endpoint configuration, allowing different settings for each endpoint:

```python
# Create an endpoint with custom settings
endpoint_id = ovms_client.create_endpoint(
    api_key="my_custom_key",
    max_retries=3,
    initial_retry_delay=2,
    backoff_factor=3,
    max_concurrent_requests=10
)

# Use the endpoint for inference
response = ovms_client.make_request_with_endpoint(
    endpoint_id=endpoint_id,
    data={"instances": [{"data": [1.0, 2.0, 3.0, 4.0]}]},
    model="my_model"
)
```

### Request Tracking and Statistics

The implementation provides comprehensive request tracking and statistics:

```python
# Get statistics for a specific endpoint
stats = ovms_client.get_stats(endpoint_id)
print(f"Total requests: {stats['total_requests']}")
print(f"Successful requests: {stats['successful_requests']}")
print(f"Failed requests: {stats['failed_requests']}")
print(f"Total processing time: {stats['total_processing_time']}s")

# Get aggregate statistics across all endpoints
all_stats = ovms_client.get_stats()
print(f"Total endpoints: {all_stats['endpoints_count']}")
print(f"Total requests across all endpoints: {all_stats['total_requests']}")
```

### The test suite also checks for these standard features:

### Input Format Handling

OVMS supports multiple input formats. The test suite verifies handling of:

```python
# Simple array format
infer_result = ovms.infer("model", [1.0, 2.0, 3.0, 4.0])

# 2D array format
infer_result = ovms.infer("model", [[1.0, 2.0], [3.0, 4.0]])

# Object with data field
infer_result = ovms.infer("model", {"data": [1.0, 2.0, 3.0, 4.0]})

# Standard OVMS format
infer_result = ovms.infer("model", {"instances": [{"data": [1.0, 2.0, 3.0, 4.0]}]})

# Named tensor format
infer_result = ovms.infer("model", {"input": [1.0, 2.0, 3.0, 4.0]})

# Multi-input format
infer_result = ovms.infer("model", {"input1": [1.0, 2.0], "input2": [3.0, 4.0]})

# TensorFlow Serving format
infer_result = ovms.infer("model", {
   "signature_name": "serving_default", 
   "inputs": {"input": [1.0, 2.0, 3.0, 4.0]}
})

# NumPy array format (if NumPy is available)
import numpy as np
infer_result = ovms.infer("model", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
```

### Batch Processing

Methods for efficient batch processing:

```python
# Process multiple inputs in a single request
batch_result = ovms.batch_infer("model", [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
])

# With batch processing metrics
batch_result, metrics = ovms.process_batch_with_metrics(
    endpoint_url, batch_data=[[1.0, 2.0], [3.0, 4.0]], model="model"
)
```

### Model Management

Enhanced methods for interacting with models:

```python
# Get detailed information about a specific model
model_info = ovms.get_model_info("model")

# Get model metadata with input/output shapes
metadata = ovms.get_model_metadata_with_shapes("model")

# Get available model versions
versions = ovms.get_model_versions("model")

# Check if a model is compatible
is_compatible = ovms.is_model_compatible("model")

# Get detailed model status 
status = ovms.get_model_status("model")

# Infer with a specific model version
result = ovms.infer_with_version("model", version="2", data=[1.0, 2.0, 3.0, 4.0])
```

### Advanced Configuration

Methods for advanced model configuration:

```python
# Set model configuration options
config_result = ovms.set_model_config("model", {
    "batch_size": 4,
    "instance_count": 2,
    "execution_mode": "throughput"
})

# Set specific execution mode (latency vs throughput)
mode_result = ovms.set_execution_mode("model", mode="throughput")

# Reload model (e.g., after config changes)
reload_result = ovms.reload_model("model")
```

### Explainability and Analytics

Advanced features for model explanation and statistics:

```python
# Get server-wide statistics
stats = ovms.get_server_statistics()

# Get explanation for a prediction
explanation = ovms.explain_prediction("model", data=[1.0, 2.0, 3.0, 4.0])
```

### Tensor Handling

Specialized methods for tensor formatting:

```python
# Format tensor input (from numpy array)
import numpy as np
tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor_result = ovms.format_tensor_request(handler, tensor)

# Format batched tensor input
batched_result = ovms.format_batch_request(handler, 
    [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
)
```

## Running a Local OVMS Server

For real connection tests, you need an OVMS server:

1. Run OVMS using Docker:
   ```bash
   docker run -d --rm -p 9000:9000 \
       -v /models:/models \
       openvino/model_server:latest \
       --model_path /models/my_model \
       --model_name my_model \
       --port 9000
   ```

2. To prepare models for OVMS, you need to convert them to OpenVINO IR format using the Model Optimizer.

3. The server should now be available at http://localhost:9000

## OVMS API Request/Response Formats

### Standard Request Format

```json
{
  "instances": [
    {
      "data": [1.0, 2.0, 3.0, 4.0]
    }
  ]
}
```

### Standard Response Format

```json
{
  "predictions": [
    [0.1, 0.2, 0.3, 0.4, 0.5]
  ]
}
```

### Request with Named Inputs

```json
{
  "inputs": {
    "input1": [1.0, 2.0],
    "input2": [3.0, 4.0]
  }
}
```

### Batched Request Format

```json
{
  "instances": [
    {"data": [1.0, 2.0, 3.0, 4.0]},
    {"data": [5.0, 6.0, 7.0, 8.0]}
  ]
}
```

### TensorFlow Serving Format

```json
{
  "signature_name": "serving_default",
  "inputs": {
    "input": [1.0, 2.0, 3.0, 4.0]
  }
}
```

## Model Configuration Options

OVMS allows advanced model configuration for performance optimization:

| Configuration Option | Description | Example Value |
|----------------------|-------------|---------------|
| `batch_size` | Fixed batch size for the model | 4 |
| `instance_count` | Number of model instances to load | 2 |
| `execution_mode` | Optimization target | "throughput" or "latency" |
| `preferred_batch` | Preferred batch size for dynamic batching | 8 |

## Troubleshooting

### Common Issues

- **Connection errors**: Verify OVMS is running with `curl http://localhost:9000/v1/models`
- **Model not found**: Verify the model path is set correctly in the server
- **Timeout errors**: Increase timeout with `--timeout` parameter
- **Shape mismatch**: Ensure the input data matches the model's expected shape
- **Version errors**: Verify the model version exists with `curl http://localhost:9000/v1/models/my_model`
- **Precision issues**: Make sure the model is available in the requested precision

### Debugging Commands

```bash
# Check if OVMS server is running and get all models
curl http://localhost:9000/v1/models

# Get information about a specific model
curl http://localhost:9000/v1/models/my_model

# Get model metadata
curl http://localhost:9000/v1/models/my_model/metadata

# Get status for all models
curl http://localhost:9000/v1/status

# Check available model versions
curl http://localhost:9000/v1/models/my_model/versions

# Test simple inference
curl -X POST http://localhost:9000/v1/models/my_model:predict \
    -d '{"instances": [{"data": [1.0, 2.0, 3.0, 4.0]}]}' \
    -H "Content-Type: application/json"

# Test inference with specific version
curl -X POST http://localhost:9000/v1/models/my_model/versions/1:predict \
    -d '{"instances": [{"data": [1.0, 2.0, 3.0, 4.0]}]}' \
    -H "Content-Type: application/json"
```

## Development Notes

When implementing new features in the OVMS backend:

1. Add corresponding tests in `test_ovms.py`
2. Update the `test_ovms_unified.py` for performance and advanced feature testing
3. Document new features in this README

The test suite is designed to gracefully handle missing features, so it will not fail if a feature is not yet implemented. The test reports will show which features are available and which are marked as "Not implemented".

### Adding New Input Formats

To add support for a new input format:

1. Add test cases to `self.input_formats` in `test_ovms.py`
2. Implement format handling in the `format_request` or `format_input` method
3. Add performance testing for the new format
4. Document the format in this README