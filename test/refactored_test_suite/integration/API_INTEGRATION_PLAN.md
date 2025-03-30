# API Integration Plan

## Overview

This document outlines the plan for integrating FastAPI interfaces across the refactored components of the IPFS Accelerate project. By implementing consistent API patterns, we can reduce code debt and provide a more cohesive user experience.

## Goals

1. Create unified API interfaces across all refactored components
2. Implement consistent patterns for synchronous and asynchronous operations
3. Support real-time monitoring via WebSockets
4. Enable integration testing between components
5. Reduce code duplication across the codebase
6. Provide a clear entry point for users and external systems

## Architecture

### Component APIs

Each refactored component will expose a FastAPI interface with consistent patterns:

#### Test Suite API (`/api/test/*`)

- Primary responsibility: Running tests, validating models, and reporting results
- Key endpoints:
  - `POST /api/test/run`: Run tests for a specific model
  - `GET /api/test/status/{run_id}`: Get test status
  - `GET /api/test/results/{run_id}`: Get test results
  - `GET /api/test/models`: List available test models
  - `GET /api/test/hardware`: List available hardware platforms for testing
  - `WebSocket /api/test/ws/{run_id}`: Real-time test updates

#### Generator API (`/api/generator/*`)

- Primary responsibility: Generating model implementations and templates
- Key endpoints:
  - `POST /api/generator/model`: Generate model implementation
  - `GET /api/generator/status/{task_id}`: Get generation status
  - `GET /api/generator/result/{task_id}`: Get generation result
  - `GET /api/generator/templates`: List available templates
  - `GET /api/generator/hardware`: List available hardware platforms
  - `WebSocket /api/generator/ws/{task_id}`: Real-time generation updates

#### Benchmark API (`/api/benchmark/*`)

- Primary responsibility: Running benchmarks and reporting performance metrics
- Key endpoints:
  - `POST /api/benchmark/run`: Run benchmarks for a specific model
  - `GET /api/benchmark/status/{run_id}`: Get benchmark status
  - `GET /api/benchmark/results/{run_id}`: Get benchmark results
  - `GET /api/benchmark/models`: List available benchmark models
  - `GET /api/benchmark/hardware`: List available hardware platforms
  - `WebSocket /api/benchmark/ws/{run_id}`: Real-time benchmark updates

### Unified API Server

A unified API server will serve as the main entry point, combining all component APIs:

```
                    ┌───────────────────┐
                    │  Unified API      │
                    │  FastAPI Server   │
                    └───────────────────┘
                              │
                              ▼
         ┌──────────────────────────────────────┐
         │                                      │
┌────────▼─────────┐  ┌──────────▼───────┐  ┌───▼────────────┐
│  Test Suite API  │  │ Generator API   │  │ Benchmark API  │
└──────────────────┘  └─────────────────┘  └────────────────┘
         │                   │                     │
         ▼                   ▼                     ▼
┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐
│ refactored_test_ │  │ refactored_gen_ │  │ refactored_ben_│
│ suite            │  │ erator_suite    │  │ chmark_suite   │
└──────────────────┘  └─────────────────┘  └────────────────┘
```

## Implementation Status

### Phase 1: API Interface Definition (COMPLETED ✅)

- [x] Create consistent API models across components
- [x] Implement test_api_integration.py as prototype for Test Suite API
- [x] Review and align with existing benchmark_api_server.py patterns
- [x] Create generator_api_server.py with similar patterns
- [x] Document API interface specifications with OpenAPI

### Phase 2: Core Component Integration (IN PROGRESS 🔄)

- [x] Implement API client for consistent interaction patterns
- [x] Implement synchronous and asynchronous client options
- [x] Create integration tests for API components
- [x] Refactor test suite internal components to use FastAPI interfaces
- [x] Complete database integration for persistent storage of results
  - [x] Generator API DuckDB integration (August 14, 2025)
  - [x] Test Suite API DuckDB integration (in progress)

### Phase 3: Unified API Server (IN PROGRESS 🔄)

- [x] Create unified API server with gateway pattern
- [x] Implement request forwarding to component APIs
- [x] Support WebSocket forwarding for real-time updates
- [ ] Implement authentication and authorization
- [ ] Add cross-component workflow support

### Phase 4: Testing and Documentation (IN PROGRESS 🔄)

- [x] Create integration workflow example script
- [x] Create comprehensive FastAPI integration guide
- [ ] Expand integration tests for all API endpoints
- [ ] Complete interactive API documentation with Swagger UI
- [ ] Finalize user documentation

## File Implementations

### Core API Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Test Suite API | `/test/refactored_test_suite/integration/test_api_integration.py` | ✅ Complete | FastAPI implementation for Test Suite |
| Generator API | `/test/refactored_generator_suite/generator_api_server.py` | ✅ Complete | FastAPI implementation for Generator Suite |
| Generator DB Integration | `/test/refactored_generator_suite/database/db_integration.py` | ✅ Complete | DuckDB integration for Generator API |
| Generator DB Endpoints | `/test/refactored_generator_suite/database/api_endpoints.py` | ✅ Complete | FastAPI endpoints for Generator database |
| Benchmark API | `/test/refactored_benchmark_suite/benchmark_api_server.py` | ✅ Complete | FastAPI implementation for Benchmark Suite |
| Unified API Server | `/test/unified_api_server.py` | ✅ Complete | Gateway implementation for unified API access |

### API Clients and Integration

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| API Client | `/test/refactored_test_suite/api/api_client.py` | ✅ Complete | Synchronous and asynchronous client implementation |
| Integration Workflow | `/test/integration_workflow_example.py` | ✅ Complete | End-to-end workflow demonstration |
| API Documentation | `/test/FASTAPI_INTEGRATION_GUIDE.md` | ✅ Complete | Comprehensive integration guide |

## Key Features Implemented

### Consistent API Patterns

The API implementations follow consistent patterns across all components:

- RESTful endpoints with consistent naming
- Long-running operations handled as background tasks
- Real-time progress tracking via WebSockets
- Comprehensive error handling
- Unified response models
- Batch operation support

### WebSocket Support

All component APIs support WebSocket connections for real-time progress updates:

```python
@app.websocket("/api/component/ws/{operation_id}")
async def websocket_endpoint(websocket: WebSocket, operation_id: str):
    await manager.handle_websocket_connection(websocket, operation_id)
```

### Unified Gateway

The unified API server provides a gateway to all component APIs:

- Request forwarding to appropriate backend services
- WebSocket connection forwarding
- Single entry point for all API interactions
- Consistent error handling
- Combined API documentation

## Next Steps

1. Complete the refactoring of internal components to fully utilize the FastAPI interfaces
2. Implement additional integration tests to verify cross-component functionality
3. Add authentication and authorization to secure API endpoints
4. Complete the unified dashboard for monitoring all operations
5. Finalize documentation and provide examples for common workflows

## Common API Patterns

### Request/Response Models

All APIs will follow consistent patterns for request and response models:

```python
# Request models include the operation parameters
class OperationRequest(BaseModel):
    param1: str
    param2: List[str]
    param3: Optional[int] = None

# Response models include operation ID, status, and relevant data
class OperationResponse(BaseModel):
    operation_id: str
    status: str
    message: str
    started_at: str
    data: Optional[Dict[str, Any]] = None
```

### Background Tasks

Long-running operations will be handled as background tasks:

```python
@app.post("/api/component/operation")
async def start_operation(
    request: OperationRequest,
    background_tasks: BackgroundTasks
):
    operation_id = generate_id()
    background_tasks.add_task(
        perform_operation_in_background,
        operation_id,
        request.param1,
        request.param2
    )
    return OperationResponse(
        operation_id=operation_id,
        status="initializing",
        message="Operation started",
        started_at=datetime.now().isoformat()
    )
```

### Real-time Updates via WebSockets

All long-running operations will support real-time updates:

```python
@app.websocket("/api/component/ws/{operation_id}")
async def websocket_updates(websocket: WebSocket, operation_id: str):
    await websocket.accept()
    
    # Register connection
    # Send updates during operation
    # Handle disconnection
```

## Integration Testing

The integration tests will verify:

1. Component APIs work correctly in isolation
2. Components can communicate with each other through APIs
3. End-to-end workflows involving multiple components

Example test workflows:

1. Generate model → Test model → Benchmark model
2. Run multiple tests in parallel → Aggregate results
3. Monitor real-time progress of long-running operations

## Database Integration

All APIs will use a consistent database approach:

1. DuckDB for storage of results and performance metrics
2. SQLite for operational data and user preferences
3. Integration with existing benchmark database

## Next Steps

1. ✅ Complete Test Suite API DuckDB integration for advanced querying (August 15, 2025)
2. Add more advanced features like test batching and conditional execution
3. Complete integration of the Generator API with the Unified API Server
4. Create CI/CD pipelines to test API functionality
5. Finalize the authentication and authorization layer for the Unified API Server

## Conclusion

By implementing consistent API patterns across all components, we will significantly reduce code debt and provide a more cohesive user experience. The FastAPI interfaces will make it easier to integrate components and expose functionality to external systems.