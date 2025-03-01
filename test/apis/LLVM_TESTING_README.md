# LLVM API Testing Guide

This document outlines how to test the LLVM API backend implementation in the IPFS Accelerate framework.

## Overview

LLVM (Low Level Virtual Machine) API provides a high-performance interface for running deep learning models with LLVM optimizations. LLVM is particularly valuable for model execution across different architectures because it:

1. Provides hardware-specific optimizations through its compilation infrastructure
2. Enables just-in-time (JIT) compilation for runtime optimization
3. Supports various precision modes (FP32, FP16, INT8) for performance/accuracy tradeoffs
4. Facilitates efficient batch processing for higher throughput

The test suite validates three main aspects:

1. **Standard API Implementation**: Verifies the core API functionality and interfaces
2. **Performance Benchmarks**: Measures throughput, latency, and optimization effectiveness
3. **Real Connection**: Tests actual connection to a running LLVM server instance

## Test Files

- `test_llvm.py` - Core API implementation tests
- `test_llvm_unified.py` - Unified test runner with performance tests
- `LLVM_TESTING_README.md` - This documentation file

## Requirements

- Python 3.8+
- LLVM server (for real connection tests)
- Python packages: requests, numpy, unittest

## Running Tests

### Standard API Tests

Tests the implementation of the LLVM API client:

```bash
python test_llvm_unified.py --standard
```

This validates:
- Endpoint handler creation
- API request formatting
- Response parsing
- Error handling
- Model metadata retrieval
- Inference methods
- Batch processing capabilities

### Performance Tests

Benchmarks the performance of the API:

```bash
python test_llvm_unified.py --performance
```

This measures:
- Single inference time
- Batch inference time
- Inputs per second throughput
- Batch speedup factor
- Precision mode performance comparison

### Real Connection Tests

Tests connection to an actual LLVM server:

```bash
python test_llvm_unified.py --real
```

This verifies:
- Server availability
- Model availability
- Inference functionality

### All Tests

Run all test suites:

```bash
python test_llvm_unified.py --all
```

## Configuration Options

You can customize the tests with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model to use for testing | default |
| `--api-url` | URL for LLVM API | http://localhost:8090 |
| `--timeout` | Timeout in seconds for API requests | 30 |

Example with custom configuration:
```bash
python test_llvm_unified.py --all --model resnet50 --api-url http://llvm-server:8090 --timeout 60
```

## Environment Variables

You can also set configuration via environment variables:

- `LLVM_API_URL` - URL for the LLVM API
- `LLVM_MODEL` - Model to use for testing
- `LLVM_TIMEOUT` - Timeout in seconds
- `SKIP_PERFORMANCE_TESTS` - Set to "true" to skip performance tests
- `SKIP_REAL_TESTS` - Set to "true" to skip real connection tests

## Test Results

Test results are saved in the `collected_results/` directory:

- `llvm_test_results.json` - Standard API test results
- `llvm_performance_*.json` - Performance test results
- `llvm_connection_*.json` - Real connection test results
- `llvm_summary_*.json` - Test summary reports

## Extended API Features

The test suite checks for these advanced features if implemented:

### Parameterized Endpoint Handlers

Create handlers with different precision and optimization settings:

```python
# Create a handler with fp16 precision
handler = llvm.create_llvm_endpoint_handler_with_params(
    "http://localhost:8090",
    model="resnet50",
    parameters={"precision": "fp16"}
)

# Create a handler optimized for batch processing
handler = llvm.create_llvm_endpoint_handler_with_params(
    "http://localhost:8090",
    model="resnet50",
    parameters={"batch_size": 8}
)
```

### Request Formatting with Parameters

Control inference parameters for each request:

```python
# Format request with specific parameters
result = llvm.format_request_with_params(
    handler,
    "Input data",
    {"precision": "fp16", "optimize": True}
)

# Format a structured request
result = llvm.format_structured_request(
    handler,
    {
        "text": "Input text",
        "context": "Additional context",
        "options": {"return_details": True}
    }
)
```

### Batch Processing

Methods for efficient batch processing:

```python
# Process a batch of inputs
results = llvm.process_batch(
    "http://localhost:8090",
    ["Input 1", "Input 2", "Input 3"],
    "resnet50"
)

# Process a batch with parameters
results = llvm.process_batch_with_params(
    "http://localhost:8090",
    ["Input 1", "Input 2", "Input 3"],
    "resnet50",
    {"batch_size": 8}
)

# Process a batch and retrieve metrics
results, metrics = llvm.process_batch_with_metrics(
    "http://localhost:8090",
    ["Input 1", "Input 2", "Input 3"],
    "resnet50"
)
```

### Model Information and Statistics

Methods for getting model metadata and performance stats:

```python
# Get model information
info = llvm.get_model_info("http://localhost:8090", "resnet50")

# Get model performance statistics
stats = llvm.get_model_statistics("http://localhost:8090", "resnet50")
```

### Model Optimization

Methods for optimizing model execution:

```python
# Optimize model for speed
result = llvm.optimize_model(
    "http://localhost:8090",
    "resnet50",
    optimization_type="speed"
)

# Optimize model for memory efficiency
result = llvm.optimize_model(
    "http://localhost:8090",
    "resnet50",
    optimization_type="memory"
)

# Balanced optimization
result = llvm.optimize_model(
    "http://localhost:8090",
    "resnet50",
    optimization_type="balanced"
)
```

## Running a Local LLVM Server

For real connection tests, you need an LLVM server. Here's a simple way to run one:

1. Pull and run the LLVM server Docker image:
   ```bash
   docker run -d --name llvm-server \
       -p 8090:8090 \
       -v /path/to/models:/models \
       llvm/api-server:latest
   ```

2. Load your model:
   ```bash
   curl -X POST http://localhost:8090/models/load \
       -H "Content-Type: application/json" \
       -d '{"model_path": "/models/resnet50", "model_name": "resnet50"}'
   ```

3. The server should now be available at http://localhost:8090

## LLVM API Request Format

LLVM generally expects requests in this format:

```json
{
  "input": "Your input data here",
  "parameters": {
    "precision": "fp32",
    "batch_size": 1
  }
}
```

Responses follow this format:

```json
{
  "result": "Inference result",
  "status": "success",
  "metrics": {
    "inference_time": 0.0234,
    "throughput": 42.5
  }
}
```

For batch processing:

```json
{
  "results": ["Result 1", "Result 2"],
  "status": "success",
  "metrics": {
    "batch_size": 2,
    "inference_time": 0.0456
  }
}
```

## Troubleshooting

### Common Issues

- **Connection errors**: Verify LLVM server is running with `curl http://localhost:8090/status`
- **Model not found**: Ensure the model is properly loaded on the server
- **Timeout errors**: Increase timeout with `--timeout` parameter
- **Memory issues**: For large models, try optimizing with lower precision (fp16/int8)

### Debugging Commands

```bash
# Check if LLVM server is running
curl http://localhost:8090/status

# Check available models
curl http://localhost:8090/models

# Get information about a specific model
curl http://localhost:8090/models/resnet50

# Test simple inference
curl -X POST http://localhost:8090/infer \
    -H "Content-Type: application/json" \
    -d '{"input": "test input", "model": "resnet50"}'
```

## Development Notes

When implementing new features in the LLVM backend:

1. Add corresponding tests in `test_llvm.py`
2. Update the `test_llvm_unified.py` for performance testing
3. Document new features in this README

The test suite is designed to gracefully handle missing features, so it will not fail if a feature is not yet implemented.