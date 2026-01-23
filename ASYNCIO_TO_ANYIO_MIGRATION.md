# AsyncIO to AnyIO Migration Guide

This document tracks the migration from asyncio to anyio for cross-platform async compatibility.

## Why AnyIO?

- **Backend-agnostic**: Works with asyncio, trio, and curio
- **Better structured concurrency**: Task groups for safer concurrent execution
- **Cross-platform**: Better support for different event loop implementations
- **Modern async patterns**: Cleaner API for common patterns

## Migration Mapping

| AsyncIO Pattern | AnyIO Equivalent |
|----------------|------------------|
| `asyncio.run()` | `anyio.run()` |
| `asyncio.sleep()` | `anyio.sleep()` |
| `asyncio.Queue()` | `anyio.create_memory_object_stream()` |
| `asyncio.create_task()` | Use task groups |
| `asyncio.gather()` | Use task groups |
| `asyncio.wait_for()` | `anyio.fail_after()` or `anyio.move_on_after()` |
| `asyncio.Event()` | `anyio.Event()` |
| `asyncio.Lock()` | `anyio.Lock()` |
| `asyncio.to_thread()` | `anyio.to_thread.run_sync()` |
| `asyncio.get_event_loop()` | Not needed (AnyIO manages this) |
| `asyncio.TimeoutError` | `TimeoutError` (built-in) |
| `asyncio.CancelledError` | `anyio.get_cancelled_exc_class()` |

## Task Groups vs create_task/gather

**Before (asyncio):**
```python
task1 = asyncio.create_task(func1())
task2 = asyncio.create_task(func2())
results = await asyncio.gather(task1, task2)
```

**After (anyio):**
```python
async with anyio.create_task_group() as tg:
    tg.start_soon(func1)
    tg.start_soon(func2)
# Results collected via shared data structures or returned differently
```

## Queue Migration

**Before (asyncio):**
```python
queue = asyncio.Queue(maxsize=128)
await queue.put(item)
item = await queue.get()
```

**After (anyio):**
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

- AnyIO is fully compatible with existing asyncio code during migration
- Can migrate incrementally - asyncio and anyio can coexist
- Focus on hot paths first (worker, API backends)
- Tests should verify behavior remains unchanged

## Completed Migrations

### 1. ipfs_accelerate_py.py
- Replaced `import asyncio` with `import anyio`
- Replaced `asyncio.Queue(128)` with `anyio.create_memory_object_stream(max_buffer_size=128)`
- Replaced `asyncio.sleep()` with `anyio.sleep()`
- Replaced `asyncio.iscoroutinefunction()` with `inspect.iscoroutinefunction()`
- Replaced `loop.run_in_executor()` with `anyio.to_thread.run_sync()`
- Replaced event loop management with `anyio.from_thread.run()` for sync to async bridge

### 2. examples/example.py
- Replaced `import asyncio` with `import anyio`
- Replaced `asyncio.run()` with `anyio.run()`

### 3. main.py
- Replaced `import asyncio` with `import anyio`
- Replaced `asyncio.sleep()` with `anyio.sleep()`
- Replaced `asyncio.create_task()` with task groups
- Updated lifespan context manager to use anyio task groups

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
