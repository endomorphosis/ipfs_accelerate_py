# AnyIO Migration Guide

This document tracks the migration to anyio for cross-platform async compatibility.

## Why AnyIO?

- **Backend-agnostic**: Works with anyio, trio, and curio
- **Better structured concurrency**: Task groups for safer concurrent execution
- **Cross-platform**: Better support for different event loop implementations
- **Modern async patterns**: Cleaner API for common patterns

## AnyIO Quick Reference

- Entry point: `anyio.run(...)`
- Sleep: `anyio.sleep(...)`
- Concurrency: `anyio.create_task_group()`
- Timeouts: `anyio.fail_after(...)` / `anyio.move_on_after(...)`
- Sync primitives: `anyio.Event()`, `anyio.Lock()`
- Thread offload: `anyio.to_thread.run_sync(...)`
- Cancellation: `anyio.get_cancelled_exc_class()`
- Timeout exception: `TimeoutError` (built-in)

## Task Groups Example

```python
async with anyio.create_task_group() as tg:
    tg.start_soon(func1)
    tg.start_soon(func2)
# Results collected via shared data structures or returned differently
```

## Queue Migration

```python
send_stream, receive_stream = anyio.create_memory_object_stream(max_buffer_size=128)
await send_stream.send(item)
item = await receive_stream.receive()
```

## Migration Progress

### Phase 1: Core Infrastructure
- [x] Add anyio to requirements.txt
- [x] ipfs_accelerate_py.py (main framework file)
- [x] examples/example.py (basic example)
- [x] main.py (FastAPI server)
- [ ] ipfs_accelerate_py/worker/worker.py

### Phase 2: API Backends
- [ ] ipfs_accelerate_py/api_backends/apis.py
- [ ] ipfs_accelerate_py/api_backends/vllm.py
- [ ] ipfs_accelerate_py/api_backends/groq.py

### Phase 3: Worker Skillsets
- [ ] ipfs_accelerate_py/worker/skillset/*.py files

### Phase 4: MCP Integration
- [ ] mcp/server.py
- [ ] mcp/tools/*.py
- [ ] ipfs_mcp/ai_model_server.py

### Phase 5: Test Files
- [ ] test_*.py files
- [ ] test/**/*.py files

### Phase 6: Examples
- [ ] examples/*.py

## Notes

- AnyIO is compatible across async backends
- Migration can be incremental alongside other async backends
- Focus on hot paths first (worker, API backends)
- Tests should verify behavior remains unchanged

## Completed Migrations

### 1. ipfs_accelerate_py.py
- Updated imports to `anyio`
- Replaced queues with `anyio.create_memory_object_stream(...)`
- Replaced sleeps with `anyio.sleep(...)`
- Replaced coroutine checks with `inspect.iscoroutinefunction()`
- Replaced executor usage with `anyio.to_thread.run_sync(...)`
- Replaced loop management with `anyio.from_thread.run(...)`

### 2. examples/example.py
- Updated imports to `anyio`
- Updated entry point to `anyio.run(...)`

### 3. main.py
- Updated imports to `anyio`
- Updated sleeps to `anyio.sleep(...)`
- Replaced task creation with task groups
- Updated lifespan context manager to use AnyIO task groups

## Testing

A comprehensive test suite has been created in `test_anyio_migration.py` that verifies:
- Basic sleep functionality
- Memory object streams (Queue replacement)
- Task groups (create_task/gather replacement)
- Thread execution (to_thread)
- Timeout handling
- Event synchronization
- Lock primitives

All tests pass successfully.
