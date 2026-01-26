#!/usr/bin/env python3
"""
Test script to verify the asyncio to anyio migration works correctly.
"""

import anyio
import sys

async def test_basic_sleep():
    """Test basic sleep functionality."""
    print("Testing anyio.sleep...")
    await anyio.sleep(0.1)
    print("✓ anyio.sleep works")

async def test_memory_streams():
    """Test memory object streams (replacement for AnyioQueue)."""
    print("Testing anyio memory object streams...")
    send_stream, receive_stream = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Send some items
    await send_stream.send("item1")
    await send_stream.send("item2")
    await send_stream.send("item3")
    
    # Receive them back
    item1 = await receive_stream.receive()
    item2 = await receive_stream.receive()
    item3 = await receive_stream.receive()
    
    assert item1 == "item1"
    assert item2 == "item2"
    assert item3 == "item3"
    print("✓ anyio memory object streams work")

async def test_task_groups():
    """Test task groups (replacement for create_task/gather)."""
    print("Testing anyio task groups...")
    results = []
    
    async def task1():
        await anyio.sleep(0.1)
        results.append("task1")
    
    async def task2():
        await anyio.sleep(0.1)
        results.append("task2")
    
    async with anyio.create_task_group() as tg:
        tg.start_soon(task1)
        tg.start_soon(task2)
    
    assert "task1" in results
    assert "task2" in results
    print("✓ anyio task groups work")

async def test_to_thread():
    """Test running sync functions in threads."""
    print("Testing anyio.to_thread.run_sync...")
    
    def sync_function(x):
        return x * 2
    
    result = await anyio.to_thread.run_sync(sync_function, 21)
    assert result == 42
    print("✓ anyio.to_thread.run_sync works")

async def test_timeout():
    """Test timeout functionality."""
    print("Testing anyio timeout...")
    
    try:
        with anyio.fail_after(0.1):
            await anyio.sleep(1.0)
        print("✗ Timeout did not work")
        return False
    except TimeoutError:
        print("✓ anyio timeout works")
        return True

async def test_event():
    """Test anyio Event."""
    print("Testing anyio.Event...")
    event = anyio.Event()
    
    async def setter():
        await anyio.sleep(0.1)
        event.set()
    
    async def waiter():
        await event.wait()
        return "done"
    
    async with anyio.create_task_group() as tg:
        tg.start_soon(setter)
        result = await waiter()
    
    assert result == "done"
    print("✓ anyio.Event works")

async def test_lock():
    """Test anyio Lock."""
    print("Testing anyio.Lock...")
    lock = anyio.Lock()
    shared_resource = []
    
    async def task_with_lock(value):
        async with lock:
            shared_resource.append(value)
            await anyio.sleep(0.01)
    
    async with anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(task_with_lock, i)
    
    assert len(shared_resource) == 5
    print("✓ anyio.Lock works")

async def run_all_tests():
    """Run all tests."""
    print("\n=== Running AnyIO Migration Tests ===\n")
    
    try:
        await test_basic_sleep()
        await test_memory_streams()
        await test_task_groups()
        await test_to_thread()
        await test_timeout()
        await test_event()
        await test_lock()
        
        print("\n=== All Tests Passed ✓ ===\n")
        return 0
    except Exception as e:
        print(f"\n=== Test Failed ✗ ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(anyio.run(run_all_tests))
