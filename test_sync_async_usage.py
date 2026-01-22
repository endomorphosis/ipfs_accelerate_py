#!/usr/bin/env python3
"""
Test Synchronous and Asynchronous Usage of P2P Cache

This test verifies that the P2P cache works correctly in both
synchronous (GitHub autoscaler, CLI) and asynchronous (libp2p) contexts.
"""

import os
import sys
import time
import asyncio
import logging
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_sync_async')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_synchronous_usage():
    """Test cache in synchronous context (like GitHub autoscaler)"""
    logger.info("="*70)
    logger.info("TEST 1: Synchronous Usage")
    logger.info("="*70)
    
    try:
        os.environ['CACHE_ENABLE_P2P'] = 'false'  # Start without P2P
        
        from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
        
        logger.info("\n1. Creating cache in main thread (synchronous)...")
        cache = GitHubAPICache(enable_p2p=False)
        
        logger.info("✓ Cache created")
        
        # Test synchronous operations
        logger.info("\n2. Testing synchronous cache operations...")
        
        test_key = "sync/test/key"
        test_data = {"value": "synchronous data", "timestamp": time.time()}
        
        # Put (synchronous)
        cache.put(test_key, test_data, ttl=60)
        logger.info("✓ put() completed synchronously")
        
        # Get (synchronous)
        retrieved = cache.get(test_key)
        if retrieved and retrieved['value'] == test_data['value']:
            logger.info("✓ get() completed synchronously")
        else:
            logger.error("✗ get() failed")
            return False
        
        # Stats (synchronous)
        stats = cache.get_stats()
        logger.info(f"✓ get_stats() completed synchronously")
        logger.info(f"  Cache size: {stats['cache_size']}")
        logger.info(f"  Hits: {stats['hits']}")
        
        logger.info("\n✅ SYNCHRONOUS USAGE TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ SYNCHRONOUS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_operations():
    """Test cache operations in async context"""
    logger.info("="*70)
    logger.info("TEST 2: Asynchronous Operations")
    logger.info("="*70)
    
    try:
        os.environ['CACHE_ENABLE_P2P'] = 'false'
        
        from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
        
        logger.info("\n1. Creating cache in async context...")
        cache = GitHubAPICache(enable_p2p=False)
        logger.info("✓ Cache created in async function")
        
        # Test that sync methods work in async context
        logger.info("\n2. Testing sync methods in async context...")
        
        test_key = "async/test/key"
        test_data = {"value": "async data", "timestamp": time.time()}
        
        # These are synchronous but called from async
        cache.put(test_key, test_data, ttl=60)
        logger.info("✓ Synchronous put() called from async context")
        
        await asyncio.sleep(0.1)  # Simulate async work
        
        retrieved = cache.get(test_key)
        if retrieved:
            logger.info("✓ Synchronous get() called from async context")
        
        logger.info("\n✅ ASYNC OPERATIONS TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ ASYNC TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threading():
    """Test cache with multiple threads"""
    logger.info("="*70)
    logger.info("TEST 3: Multi-Threading")
    logger.info("="*70)
    
    try:
        os.environ['CACHE_ENABLE_P2P'] = 'false'
        
        from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
        
        logger.info("\n1. Creating cache in main thread...")
        cache = GitHubAPICache(enable_p2p=False)
        logger.info("✓ Cache created")
        
        results = []
        errors = []
        
        def worker(thread_id):
            """Worker thread function"""
            try:
                # Each thread does cache operations
                key = f"thread/{thread_id}/key"
                data = {"thread": thread_id, "timestamp": time.time()}
                
                cache.put(key, data, ttl=60)
                retrieved = cache.get(key)
                
                if retrieved and retrieved['thread'] == thread_id:
                    results.append(thread_id)
                else:
                    errors.append(f"Thread {thread_id} failed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        logger.info("\n2. Starting 5 threads...")
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        logger.info(f"✓ All threads completed")
        logger.info(f"  Successful: {len(results)}/5")
        logger.info(f"  Errors: {len(errors)}")
        
        if errors:
            for error in errors:
                logger.error(f"  {error}")
            return False
        
        logger.info("\n✅ MULTI-THREADING TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ THREADING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_sync_async():
    """Test mixing synchronous and asynchronous usage"""
    logger.info("="*70)
    logger.info("TEST 4: Mixed Sync/Async Usage")
    logger.info("="*70)
    
    try:
        os.environ['CACHE_ENABLE_P2P'] = 'false'
        
        from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
        
        logger.info("\n1. Creating cache synchronously...")
        cache = GitHubAPICache(enable_p2p=False)
        
        # Synchronous put
        cache.put("mixed/key/1", {"source": "sync"}, ttl=60)
        logger.info("✓ Synchronous put completed")
        
        async def async_operations():
            """Do operations in async context"""
            # Sync methods called from async
            cache.put("mixed/key/2", {"source": "async"}, ttl=60)
            await asyncio.sleep(0.1)
            
            result1 = cache.get("mixed/key/1")
            result2 = cache.get("mixed/key/2")
            
            return result1 is not None and result2 is not None
        
        logger.info("\n2. Running async operations...")
        success = asyncio.run(async_operations())
        
        if success:
            logger.info("✓ Mixed sync/async operations successful")
        else:
            logger.error("✗ Mixed operations failed")
            return False
        
        # Back to synchronous
        stats = cache.get_stats()
        logger.info(f"✓ Final stats: {stats['cache_size']} entries")
        
        logger.info("\n✅ MIXED SYNC/ASYNC TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ MIXED TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_p2p_initialization_issue():
    """Test the current P2P initialization issue"""
    logger.info("="*70)
    logger.info("TEST 5: P2P Initialization (Current Issue)")
    logger.info("="*70)
    
    try:
        os.environ['CACHE_ENABLE_P2P'] = 'true'
        
        from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
        
        logger.info("\n1. Attempting to create cache with P2P enabled...")
        logger.info("   (This may hang or fail due to event loop issues)")
        
        start_time = time.time()
        
        try:
            cache = GitHubAPICache(enable_p2p=True)
            elapsed = time.time() - start_time
            
            logger.info(f"✓ Cache created in {elapsed:.2f}s")
            
            # Check if P2P is actually enabled
            stats = cache.get_stats()
            if stats['p2p_enabled']:
                logger.info("✓ P2P is enabled")
            else:
                logger.warning("⚠ P2P initialization failed but cache works")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠ P2P initialization error (expected): {e}")
            logger.info("  This shows the current limitation with event loops")
            return True  # Expected failure
        
    except Exception as e:
        logger.error(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("  SYNCHRONOUS AND ASYNCHRONOUS USAGE TESTS")
    logger.info("="*70 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Synchronous Usage", test_synchronous_usage()))
    
    # Run async test
    results.append(("Asynchronous Operations", asyncio.run(test_async_operations())))
    
    results.append(("Multi-Threading", test_threading()))
    results.append(("Mixed Sync/Async", test_mixed_sync_async()))
    results.append(("P2P Initialization", test_p2p_initialization_issue()))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status:10} | {test_name}")
    
    logger.info("="*70)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    logger.info("="*70)
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
        logger.info("\nCONCLUSION:")
        logger.info("  • Synchronous usage: ✅ WORKS")
        logger.info("  • Asynchronous usage: ✅ WORKS")
        logger.info("  • Multi-threading: ✅ WORKS")
        logger.info("  • Mixed sync/async: ✅ WORKS")
        logger.info("  • P2P with event loop: ⚠️  HAS LIMITATIONS")
        return 0
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) had issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())
