# FastAPI Integration Guide

## Introduction

This guide outlines our approach to integrating FastAPI interfaces across the IPFS Accelerate project. By establishing consistent API patterns, we can reduce code debt, improve maintainability, and provide a more cohesive user experience.

## Why FastAPI?

FastAPI provides several advantages that make it an ideal choice for our API implementation:

1. **Performance**: Built on Starlette and Pydantic, FastAPI offers high performance
2. **Type checking**: Automatic request validation using Python type hints
3. **OpenAPI documentation**: Automatic generation of Swagger and ReDoc documentation
4. **Async support**: Native support for asynchronous request handling
5. **WebSockets**: Built-in WebSocket support for real-time communication
6. **Middleware**: Easy integration of CORS, authentication, and other middleware

## Architecture Overview

Our FastAPI integration follows a layered architecture:

```
┌───────────────────────────────────────────────────────────┐
│                  Unified API Server                       │
├─────────────┬─────────────────────────┬──────────────────┤
│ Test API    │ Generator API           │ Benchmark API    │
├─────────────┼─────────────────────────┼──────────────────┤
│ Test Core   │ Generator Core          │ Benchmark Core   │
└─────────────┴─────────────────────────┴──────────────────┘
```

Each component (Test Suite, Generator Suite, Benchmark Suite) exposes its functionality through a dedicated API layer, which is then combined in a unified API server.

## API Design Principles

### 1. Consistent Endpoint Structure

All APIs follow a consistent endpoint structure:

- `POST /api/{component}/operation`: Initiate an operation
- `GET /api/{component}/status/{id}`: Check operation status
- `GET /api/{component}/results/{id}`: Get operation results
- `WebSocket /api/{component}/ws/{id}`: Real-time updates

### 2. Common Request/Response Models

All APIs use consistent request and response models:

- Request models define the operation parameters
- Response models include operation status and relevant data
- All models use Python type hints for validation

### 3. Background Task Processing

Long-running operations are handled as background tasks:

- Operations are initiated with a POST request
- A unique ID is returned immediately
- The operation runs asynchronously
- Progress can be monitored via GET requests or WebSockets

### 4. Real-time Updates

All long-running operations support real-time updates via WebSockets:

- Updates include progress information
- Clients can receive streaming updates
- WebSocket connections are managed efficiently

### 5. Error Handling

All APIs implement consistent error handling:

- Standard HTTP status codes for different error types
- Detailed error messages in response bodies
- Graceful handling of exceptions

## Component APIs

### Test Suite API

The Test Suite API provides endpoints for running tests, checking test status, and retrieving test results:

```python
@app.post("/api/test/run")
async def run_test(request: TestRunRequest, background_tasks: BackgroundTasks):
    """Run a test for a specific model."""
    # Implementation details...

@app.get("/api/test/status/{run_id}")
async def get_test_status(run_id: str):
    """Get the status of a test run."""
    # Implementation details...

@app.get("/api/test/results/{run_id}")
async def get_test_results(run_id: str):
    """Get the results of a completed test run."""
    # Implementation details...

@app.websocket("/api/test/ws/{run_id}")
async def test_websocket(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time test updates."""
    # Implementation details...
```

### Generator API

The Generator API provides endpoints for generating model implementations:

```python
@app.post("/api/generator/model")
async def generate_model(request: GenerateModelRequest, background_tasks: BackgroundTasks):
    """Generate a model implementation."""
    # Implementation details...

@app.get("/api/generator/status/{task_id}")
async def get_generation_status(task_id: str):
    """Get the status of a model generation task."""
    # Implementation details...

@app.get("/api/generator/templates")
async def get_templates():
    """Get available templates."""
    # Implementation details...

@app.websocket("/api/generator/ws/{task_id}")
async def generator_websocket(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time generation updates."""
    # Implementation details...
```

### Benchmark API

The Benchmark API provides endpoints for running benchmarks and retrieving performance metrics:

```python
@app.post("/api/benchmark/run")
async def run_benchmark(request: BenchmarkRunRequest, background_tasks: BackgroundTasks):
    """Run a benchmark for a specific model."""
    # Implementation details...

@app.get("/api/benchmark/status/{run_id}")
async def get_benchmark_status(run_id: str):
    """Get the status of a benchmark run."""
    # Implementation details...

@app.get("/api/benchmark/results/{run_id}")
async def get_benchmark_results(run_id: str):
    """Get the results of a completed benchmark run."""
    # Implementation details...

@app.websocket("/api/benchmark/ws/{run_id}")
async def benchmark_websocket(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time benchmark updates."""
    # Implementation details...
```

## Client Implementation

To ensure consistent interaction with the APIs, we provide both synchronous and asynchronous client implementations:

### Synchronous Client

```python
client = ApiClient(base_url="http://localhost:8000")
response = client.run_test("bert-base-uncased")
run_id = response["run_id"]
results = client.monitor_test(run_id)
```

### Asynchronous Client

```python
client = AsyncApiClient(base_url="http://localhost:8000")
response = await client.run_test("bert-base-uncased")
run_id = response["run_id"]
results = await client.monitor_test_ws(run_id)
```

## Implementation Status

| Component | API Server | Client | Integration Tests |
|-----------|------------|--------|-------------------|
| Test Suite | In Progress | Completed | In Progress |
| Generator Suite | Planned | Planned | Planned |
| Benchmark Suite | Completed | Planned | Planned |
| Unified API | Planned | Planned | Planned |

## Benefits of FastAPI Integration

The FastAPI integration provides several key benefits:

1. **Reduced code debt**: By standardizing API interfaces, we reduce code duplication and inconsistencies
2. **Improved user experience**: Consistent interfaces make it easier for users to work with our components
3. **Better testability**: Standardized APIs simplify integration testing
4. **Enhanced monitoring**: Real-time progress tracking via WebSockets
5. **Easier integration**: Components can be used together or separately through well-defined APIs
6. **Cleaner architecture**: Clear separation between core logic and API interfaces

## Integration Example: End-to-End Workflow

To demonstrate the power of our API integration, here's an example of an end-to-end workflow:

```python
# Generate a model implementation
generator_client = ApiClient(base_url="http://localhost:8001")
gen_response = generator_client.generate_model("bert-base-uncased")
model_path = generator_client.monitor_generation(gen_response["task_id"])["file_path"]

# Run tests on the generated model
test_client = ApiClient(base_url="http://localhost:8002")
test_response = test_client.run_test(model_path)
test_results = test_client.monitor_test(test_response["run_id"])

# If tests pass, run benchmarks
if test_results["status"] == "passed":
    benchmark_client = ApiClient(base_url="http://localhost:8003")
    bench_response = benchmark_client.run_benchmark(model_path)
    bench_results = benchmark_client.monitor_benchmark(bench_response["run_id"])
    
    # Print benchmark results
    print(f"Model: {model_path}")
    print(f"Latency: {bench_results['latency_ms']} ms")
    print(f"Throughput: {bench_results['throughput']} items/sec")
```

## Next Steps

1. Complete the Test Suite API implementation
2. Implement the Generator API following the same patterns
3. Ensure consistent patterns across all three component APIs
4. Create the unified API server
5. Implement comprehensive integration tests
6. Create user documentation

## Conclusion

Our FastAPI integration provides a solid foundation for future development, making our codebase more maintainable, testable, and user-friendly. By following consistent patterns across all components, we can significantly reduce code debt while improving functionality.