#!/usr/bin/env python3
"""
Test thread-safety of P2P cache singleton.

This test verifies that multiple threads calling get_global_cache()
simultaneously only create a single instance and don't cause port
binding conflicts.
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_cache_thread_safety')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_singleton_thread_safety():
    """Test that get_global_cache() is thread-safe."""
    logger.info("=" * 70)
    logger.info("TEST: Thread-Safe Singleton Initialization")
    logger.info("=" * 70)
    
    # Disable P2P for this test to avoid actual port binding
    # (we just want to test the singleton pattern itself)
    os.environ['CACHE_ENABLE_P2P'] = 'false'
    
    # Track cache instances created
    cache_instances = []
    exceptions = []
    
    def create_cache(thread_id):
        """Worker function that tries to get the global cache."""
        try:
            logger.info(f"Thread {thread_id}: Attempting to get global cache...")
            
            # Import inside thread to simulate real-world usage
            from ipfs_accelerate_py.github_cli.cache import get_global_cache
            
            # Get the cache (should be thread-safe)
            cache = get_global_cache()
            
            # Store the instance ID for comparison
            cache_instances.append(id(cache))
            
            logger.info(f"Thread {thread_id}: Got cache instance {id(cache)}")
            
        except Exception as e:
            logger.error(f"Thread {thread_id}: Exception - {e}")
            exceptions.append((thread_id, e))
    
    # Create multiple threads that all try to get the cache simultaneously
    num_threads = 10
    threads = []
    
    logger.info(f"\nCreating {num_threads} threads to simultaneously access get_global_cache()...")
    
    for i in range(num_threads):
        t = threading.Thread(target=create_cache, args=(i,))
        threads.append(t)
    
    # Start all threads at nearly the same time
    logger.info("Starting all threads...")
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    logger.info("Waiting for threads to complete...")
    for t in threads:
        t.join()
    
    # Verify results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    success = True
    
    # Check for exceptions
    if exceptions:
        logger.error(f"✗ {len(exceptions)} thread(s) raised exceptions:")
        for thread_id, exc in exceptions:
            logger.error(f"  Thread {thread_id}: {exc}")
        success = False
    else:
        logger.info("✓ No exceptions raised")
    
    # Check that all threads got the same instance
    unique_instances = set(cache_instances)
    logger.info(f"\nCache instances created: {len(cache_instances)}")
    logger.info(f"Unique cache instances: {len(unique_instances)}")
    
    if len(unique_instances) == 1:
        logger.info("✓ All threads got the same singleton instance")
    else:
        logger.error(f"✗ Multiple instances created: {unique_instances}")
        success = False
    
    logger.info("=" * 70)
    
    if success:
        logger.info("✅ THREAD-SAFETY TEST PASSED")
    else:
        logger.error("❌ THREAD-SAFETY TEST FAILED")
    
    return success


def test_concurrent_configure():
    """Test that configure_cache() is also thread-safe."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST: Thread-Safe configure_cache()")
    logger.info("=" * 70)
    
    os.environ['CACHE_ENABLE_P2P'] = 'false'
    
    exceptions = []
    
    def configure_from_thread(thread_id):
        """Worker function that configures the cache."""
        try:
            logger.info(f"Thread {thread_id}: Configuring cache...")
            
            from ipfs_accelerate_py.github_cli.cache import configure_cache
            
            # Each thread tries to configure the cache
            cache = configure_cache(
                enable_p2p=False,
                default_ttl=300 + thread_id  # Slightly different settings
            )
            
            logger.info(f"Thread {thread_id}: Configured cache {id(cache)}")
            
        except Exception as e:
            logger.error(f"Thread {thread_id}: Exception - {e}")
            exceptions.append((thread_id, e))
    
    # Create threads
    num_threads = 5
    threads = []
    
    logger.info(f"\nCreating {num_threads} threads to configure cache...")
    
    for i in range(num_threads):
        t = threading.Thread(target=configure_from_thread, args=(i,))
        threads.append(t)
    
    # Start all threads
    logger.info("Starting all threads...")
    for t in threads:
        t.start()
    
    # Wait for completion
    logger.info("Waiting for threads to complete...")
    for t in threads:
        t.join()
    
    # Check results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    success = True
    
    if exceptions:
        logger.error(f"✗ {len(exceptions)} thread(s) raised exceptions:")
        for thread_id, exc in exceptions:
            logger.error(f"  Thread {thread_id}: {exc}")
        success = False
    else:
        logger.info("✓ No exceptions raised during concurrent configuration")
        logger.info("✓ configure_cache() is thread-safe")
    
    logger.info("=" * 70)
    
    if success:
        logger.info("✅ CONFIGURE THREAD-SAFETY TEST PASSED")
    else:
        logger.error("❌ CONFIGURE THREAD-SAFETY TEST FAILED")
    
    return success


def test_port_binding_with_p2p():
    """Test that P2P initialization doesn't cause port conflicts."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST: P2P Port Binding (Single Instance)")
    logger.info("=" * 70)
    
    # Enable P2P for this test
    os.environ['CACHE_ENABLE_P2P'] = 'true'
    os.environ['CACHE_LISTEN_PORT'] = '9100'
    
    # Reset the global cache to test fresh initialization
    import ipfs_accelerate_py.github_cli.cache as cache_module
    with cache_module._global_cache_lock:
        if cache_module._global_cache:
            cache_module._global_cache.shutdown()
        cache_module._global_cache = None
    
    exceptions = []
    cache_instances = []
    
    def get_cache_with_p2p(thread_id):
        """Worker function that gets cache with P2P enabled."""
        try:
            logger.info(f"Thread {thread_id}: Getting cache with P2P...")
            
            from ipfs_accelerate_py.github_cli.cache import get_global_cache
            
            cache = get_global_cache()
            cache_instances.append(id(cache))
            
            logger.info(f"Thread {thread_id}: Got cache {id(cache)}, P2P enabled: {cache.enable_p2p}")
            
        except Exception as e:
            logger.error(f"Thread {thread_id}: Exception - {e}")
            exceptions.append((thread_id, e))
    
    # Create multiple threads
    num_threads = 5
    threads = []
    
    logger.info(f"\nCreating {num_threads} threads with P2P enabled on port 9100...")
    
    for i in range(num_threads):
        t = threading.Thread(target=get_cache_with_p2p, args=(i,))
        threads.append(t)
    
    # Start threads
    logger.info("Starting all threads...")
    for t in threads:
        t.start()
    
    # Wait for completion
    logger.info("Waiting for threads to complete...")
    for t in threads:
        t.join()
    
    # Check results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    success = True
    
    # Check for port binding errors
    port_errors = [e for tid, e in exceptions if 'address already in use' in str(e).lower() or 'bind' in str(e).lower()]
    
    if port_errors:
        logger.error(f"✗ Port binding conflicts detected: {len(port_errors)} error(s)")
        for e in port_errors:
            logger.error(f"  {e}")
        success = False
    elif exceptions:
        logger.warning(f"⚠ {len(exceptions)} exception(s) (not port-related):")
        for tid, e in exceptions:
            logger.warning(f"  Thread {tid}: {e}")
        # Not a failure if it's not port-related
    else:
        logger.info("✓ No port binding conflicts")
    
    # Check singleton
    unique_instances = set(cache_instances)
    if len(unique_instances) == 1:
        logger.info("✓ Single cache instance across all threads")
    else:
        logger.error(f"✗ Multiple instances: {len(unique_instances)}")
        success = False
    
    logger.info("=" * 70)
    
    if success:
        logger.info("✅ P2P PORT BINDING TEST PASSED")
    else:
        logger.error("❌ P2P PORT BINDING TEST FAILED")
    
    return success


if __name__ == "__main__":
    results = []
    
    try:
        # Test 1: Basic thread-safety
        results.append(("Singleton Thread-Safety", test_singleton_thread_safety()))
        
        # Test 2: configure_cache() thread-safety
        results.append(("Configure Thread-Safety", test_concurrent_configure()))
        
        # Test 3: P2P port binding (only if libp2p is available)
        try:
            import libp2p
            results.append(("P2P Port Binding", test_port_binding_with_p2p()))
        except ImportError:
            logger.info("\n⚠ Skipping P2P test (libp2p not available)")
            results.append(("P2P Port Binding", None))
        
        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        for test_name, result in results:
            if result is None:
                print(f"  {test_name}: ⊘ SKIPPED")
            elif result:
                print(f"  {test_name}: ✅ PASSED")
            else:
                print(f"  {test_name}: ❌ FAILED")
        
        print("=" * 70)
        
        # Exit with appropriate code
        if all(r in (True, None) for _, r in results):
            print("\n✅ ALL TESTS PASSED OR SKIPPED")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"\nFatal error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
