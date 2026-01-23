# AsyncIO to AnyIO Migration Checklist

Use this checklist when migrating a file from asyncio to anyio.

## Before Starting

- [ ] Read `ASYNCIO_TO_ANYIO_README.md` and `ASYNCIO_TO_ANYIO_MIGRATION.md`
- [ ] Ensure `anyio>=4.0.0` is installed
- [ ] Run `test_anyio_migration.py` to verify anyio works
- [ ] Create a backup of the file or commit to git

## Step 1: Import Changes

- [ ] Replace `import asyncio` with `import anyio`
- [ ] Add `import inspect` if using `iscoroutinefunction`
- [ ] Remove or comment out unused asyncio imports

## Step 2: Basic Replacements

- [ ] Replace `asyncio.run(func())` with `anyio.run(func)`
- [ ] Replace `asyncio.sleep(n)` with `anyio.sleep(n)`
- [ ] Replace `asyncio.Event()` with `anyio.Event()`
- [ ] Replace `asyncio.Lock()` with `anyio.Lock()`
- [ ] Replace `asyncio.iscoroutinefunction()` with `inspect.iscoroutinefunction()`

## Step 3: Queue Migration

For each `asyncio.Queue`:

- [ ] Identify all `Queue` creations
- [ ] Replace `queue = asyncio.Queue(n)` with:
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

For each `wait_for`:

- [ ] Replace `asyncio.wait_for(func(), timeout=n)` with:
  ```python
  with anyio.fail_after(n):
      await func()
  ```
- [ ] Or use `move_on_after(n)` if you want to continue on timeout
- [ ] Replace `asyncio.TimeoutError` with `TimeoutError` (built-in)

## Step 6: Thread Execution

For each executor usage:

- [ ] Replace `loop.run_in_executor(None, func, *args)` with:
  ```python
  await anyio.to_thread.run_sync(func, *args)
  ```
- [ ] Remove event loop references
- [ ] Update any thread pool configuration

## Step 7: Event Loop Management

- [ ] Remove `asyncio.get_event_loop()`
- [ ] Remove `asyncio.new_event_loop()`
- [ ] Remove `asyncio.set_event_loop()`
- [ ] Remove `loop.run_until_complete()`
- [ ] Replace sync-to-async bridges with `anyio.from_thread.run()`

## Step 8: Less Common Patterns

- [ ] Replace `asyncio.wait()` with task groups
- [ ] Replace `asyncio.as_completed()` with appropriate pattern
- [ ] Replace `asyncio.create_subprocess_*` with anyio subprocess
- [ ] Update any `asyncio.CancelledError` to `anyio.get_cancelled_exc_class()`
- [ ] Check for `asyncio.shield()` usage (may need different pattern)

## Step 9: Testing

- [ ] Run syntax check: `python3 -m py_compile filename.py`
- [ ] Run file's unit tests if they exist
- [ ] Test the functionality manually
- [ ] Run the full test suite
- [ ] Verify no asyncio references remain: `grep -n "asyncio" filename.py`

## Step 10: Documentation

- [ ] Update docstrings if they mention asyncio
- [ ] Update inline comments
- [ ] Add a note about anyio migration if significant
- [ ] Update `ASYNCIO_TO_ANYIO_MIGRATION.md` progress section

## Common Pitfalls

### ❌ Don't Do This:
```python
# Using asyncio.run inside anyio context
async def main():
    asyncio.run(some_func())  # WRONG

# Mixing asyncio and anyio primitives
queue = asyncio.Queue()
event = anyio.Event()  # INCONSISTENT

# Manual event loop in anyio
loop = asyncio.get_event_loop()  # UNNECESSARY
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

| Pattern | Before (asyncio) | After (anyio) |
|---------|------------------|---------------|
| Run | `asyncio.run(f())` | `anyio.run(f)` |
| Sleep | `await asyncio.sleep(1)` | `await anyio.sleep(1)` |
| Queue | `asyncio.Queue(10)` | `anyio.create_memory_object_stream(10)` |
| Task | `asyncio.create_task(f())` | Task group with `tg.start_soon(f)` |
| Gather | `asyncio.gather(a, b)` | Task group with multiple `start_soon` |
| Timeout | `asyncio.wait_for(f(), 5)` | `with anyio.fail_after(5): await f()` |
| Thread | `loop.run_in_executor(None, f)` | `anyio.to_thread.run_sync(f)` |
| Event | `asyncio.Event()` | `anyio.Event()` |
| Lock | `asyncio.Lock()` | `anyio.Lock()` |

## After Migration

- [ ] Remove `.backup` files if everything works
- [ ] Commit changes with descriptive message
- [ ] Update the migration progress in `ASYNCIO_TO_ANYIO_MIGRATION.md`
- [ ] Share learnings with team if you found any edge cases

## Getting Help

If stuck:
1. Check `ASYNCIO_TO_ANYIO_MIGRATION.md` for patterns
2. Look at already-migrated files (ipfs_accelerate_py.py, main.py)
3. Run `test_anyio_migration.py` to see examples
4. Use `migrate_to_anyio.py` for automated suggestions
5. Read AnyIO docs: https://anyio.readthedocs.io/

---

**Note**: This checklist is not exhaustive. Some files may have unique patterns that need custom solutions. When in doubt, preserve the original behavior.
