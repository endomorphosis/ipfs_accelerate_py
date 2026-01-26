# AnyIO Migration

## Quick Start

This project is being migrated to `anyio` for better cross-platform async compatibility.

### What Changed?

```python
import anyio

async def main():
    await anyio.sleep(1)
    send, recv = anyio.create_memory_object_stream()
    async with anyio.create_task_group() as tg:
        tg.start_soon(my_func)

anyio.run(main)
```

### Installation

```bash
pip install anyio>=4.0.0
```

Already included in `requirements.txt`.

### Running Tests

```bash
# Test the anyio migration
python3 test_anyio_migration.py

# Run existing tests (should still work)
python3 -m pytest
```

### Using the Migration Script

```bash
# Dry run (preview changes)
python3 migrate_to_anyio.py path/to/file.py

# Apply changes (creates backups)
python3 migrate_to_anyio.py path/to/file.py --apply

# Process entire directory
python3 migrate_to_anyio.py ipfs_accelerate_py/ --apply
```

## Migration Status

### ✅ Completed
- Core framework (`ipfs_accelerate_py.py`)
- FastAPI server (`main.py`)
- Basic example (`examples/example.py`)
- Test suite (`test_anyio_migration.py`)
- Migration tooling (`migrate_to_anyio.py`)

### 🔄 In Progress
- Worker module (`ipfs_accelerate_py/worker/worker.py`)
- API backends (`ipfs_accelerate_py/api_backends/`)
- Additional examples

### 📋 Remaining
- MCP integration files
- Worker skillsets (~100+ files)
- Test files (~50+ files)
- Tool files

## Key Changes

### 1. Imports
```python
import anyio
```

### 2. Running Async Code
```python
anyio.run(main)
```

### 3. Sleeping
```python
await anyio.sleep(1)
```

### 4. Queues
```python
send_stream, receive_stream = anyio.create_memory_object_stream(max_buffer_size=128)
await send_stream.send(item)
item = await receive_stream.receive()
```

### 5. Task Management
```python
async with anyio.create_task_group() as tg:
    tg.start_soon(func1)
    tg.start_soon(func2)
```

### 6. Timeouts
```python
try:
    with anyio.fail_after(5.0):
        await func()
except TimeoutError:
    pass
```

### 7. Thread Execution
```python
result = await anyio.to_thread.run_sync(sync_func, arg)
```

### 8. Events & Locks
```python
event = anyio.Event()
lock = anyio.Lock()
```

## Benefits

1. **Cross-Platform**: Works with Trio and Curio backends
2. **Better Structure**: Task groups provide cleaner concurrent code
3. **Safer Cancellation**: Automatic cleanup on errors
4. **Modern Patterns**: Follows current async best practices
5. **No Event Loop Management**: Simpler code without manual loop handling

## Documentation

- **Migration Guide**: See the migration guide and progress notes
- **Summary**: See the migration summary
- **Test Suite**: `test_anyio_migration.py`
- **Migration Tool**: `migrate_to_anyio.py`

## Compatibility

- **Python**: 3.8+ required
- **Backward Compatible**: Can use other async backends during migration
- **No Breaking Changes**: External APIs remain unchanged
- **Incremental**: Migrate file by file as needed

## Getting Help

1. Check the migration guide and progress notes
2. Run the test suite: `python3 test_anyio_migration.py`
3. Use the migration tool: `python3 migrate_to_anyio.py --help`
4. Review completed migrations in git history

## Contributing

When adding new async code:
1. Use `anyio`
2. Use task groups instead of `create_task`/`gather`
3. Use memory streams instead of `Queue`
4. Use `anyio.to_thread.run_sync()` for sync code
5. Run `test_anyio_migration.py` to verify

## References

- [AnyIO Documentation](https://anyio.readthedocs.io/)
- [AnyIO GitHub](https://github.com/agronholm/anyio)
- [Structured Concurrency](https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/)
