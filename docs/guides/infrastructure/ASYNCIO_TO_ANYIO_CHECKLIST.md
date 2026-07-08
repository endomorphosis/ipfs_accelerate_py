# AnyIO Migration Checklist

Use this checklist when migrating a file to AnyIO.

## Before Starting

- [ ] Read the migration docs and current progress notes
- [ ] Ensure `anyio>=4.0.0` is installed
- [ ] Run `test_anyio_migration.py` to verify anyio works
- [ ] Create a backup of the file or commit to git

## Step 1: Import Changes

- [ ] Add `import anyio` where needed
- [ ] Add `import inspect` if using `iscoroutinefunction`
- [ ] Remove or comment out unused legacy async imports

## Step 2: Basic Replacements

- [ ] Use `anyio.run(func)` at sync entry points
- [ ] Use `anyio.sleep(n)`
- [ ] Use `anyio.Event()` and `anyio.Lock()`
- [ ] Use `inspect.iscoroutinefunction()` for coroutine checks

## Step 3: Queue Migration

For each queue usage:

- [ ] Identify all `Queue` creations
- [ ] Replace queue creation with:
  ```python
  send_stream, receive_stream = anyio.create_memory_object_stream(max_buffer_size=n)
  ```
- [ ] Replace `await queue.put(item)` with `await send_stream.send(item)`
- [ ] Replace `item = await queue.get()` with `item = await receive_stream.receive()`
- [ ] Update any queue size checks
- [ ] Store both streams in a dict if needed: `{"send": send_stream, "receive": receive_stream}`

## Step 4: Task Management

For each `create_task` or `gather`:

- [ ] Identify all task creation points
- [ ] Replace with task group pattern:
  ```python
  async with anyio.create_task_group() as tg:
      tg.start_soon(func1)
      tg.start_soon(func2)
  ```
- [ ] Move result collection outside task group if needed
- [ ] Handle exceptions appropriately (task groups cancel all on first exception)

## Step 5: Timeout Handling

For each timed operation:

- [ ] Replace timeouts with:
  ```python
  with anyio.fail_after(n):
      await func()
  ```
- [ ] Or use `move_on_after(n)` if you want to continue on timeout
- [ ] Use `TimeoutError` (built-in)

## Step 6: Thread Execution

For each executor usage:

- [ ] Replace `loop.run_in_executor(None, func, *args)` with:
  ```python
  await anyio.to_thread.run_sync(func, *args)
  ```
- [ ] Remove event loop references
- [ ] Update any thread pool configuration

## Step 7: Event Loop Management

- [ ] Remove manual loop setup and teardown
- [ ] Remove `run_until_complete()` usage
- [ ] Replace sync-to-async bridges with `anyio.from_thread.run()`

## Step 8: Less Common Patterns

- [ ] Replace ad-hoc wait patterns with task groups
- [ ] Use AnyIO subprocess APIs where applicable
- [ ] Use `anyio.get_cancelled_exc_class()` for cancellation checks
- [ ] Review shield-like semantics and adjust patterns

## Step 9: Testing

- [ ] Run syntax check: `python3 -m py_compile filename.py`
- [ ] Run file's unit tests if they exist
- [ ] Test the functionality manually
- [ ] Run the full test suite
- [ ] Verify no legacy async references remain

## Step 10: Documentation

- [ ] Update docstrings if they mention legacy async APIs
- [ ] Update inline comments
- [ ] Add a note about anyio migration if significant
- [ ] Update migration progress notes if applicable

## Common Pitfalls

### ❌ Don't Do This:
```python
# Calling anyio.run inside an async context
async def main():
  anyio.run(some_func)  # WRONG

# Mixing AnyIO primitives with non-AnyIO sync loops
# (avoid manual loop management)
```

### ✅ Do This Instead:
```python
# Just await directly
async def main():
    await some_func()  # CORRECT

# Use anyio primitives consistently
send, recv = anyio.create_memory_object_stream()
event = anyio.Event()  # CONSISTENT

# Let anyio handle the loop
# No manual loop management needed
```

## Quick Reference

- Run: `anyio.run(f)`
- Sleep: `await anyio.sleep(1)`
- Queue: `anyio.create_memory_object_stream(10)`
- Task: task group with `tg.start_soon(f)`
- Timeout: `with anyio.fail_after(5): await f()`
- Thread: `anyio.to_thread.run_sync(f)`
- Event: `anyio.Event()`
- Lock: `anyio.Lock()`

## After Migration

- [ ] Remove `.backup` files if everything works
- [ ] Commit changes with descriptive message
- [ ] Update migration progress notes
- [ ] Share learnings with team if you found any edge cases

## Getting Help

If stuck:
1. Review migration docs for patterns
2. Look at already-migrated files (ipfs_accelerate_py.py, main.py)
3. Run `test_anyio_migration.py` to see examples
4. Use migration tooling for automated suggestions
5. Read AnyIO docs: https://anyio.readthedocs.io/

---

**Note**: This checklist is not exhaustive. Some files may have unique patterns that need custom solutions. When in doubt, preserve the original behavior.
