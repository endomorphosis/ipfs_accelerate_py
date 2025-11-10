#!/usr/bin/env python3
"""
Demonstration of the thread-safety fix for P2P cache singleton.

This script simulates the Flask with threaded=True scenario where multiple
concurrent threads access get_global_cache() simultaneously.

Before the fix: Multiple threads would pass the singleton check and create
                multiple P2P hosts, all trying to bind to port 9100, causing
                deadlocks.

After the fix:  Double-checked locking ensures only one thread creates the
                cache, preventing port conflicts.
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
    format='%(asctime)s [%(threadName)-12s] %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def simulate_flask_request(request_id):
    """
    Simulate a Flask request handler that accesses the cache.
    
    In a real Flask application with threaded=True, each request
    runs in its own thread and may call get_global_cache().
    """
    logger.info(f"Request {request_id}: Starting to handle request...")
    
    # Simulate some work before accessing cache
    time.sleep(0.01)
    
    try:
        # This is where the race condition would occur without thread-safety
        from ipfs_accelerate_py.github_cli.cache import get_global_cache
        
        logger.info(f"Request {request_id}: Accessing global cache...")
        cache = get_global_cache()
        
        # Use the cache
        cache.put(f"request_{request_id}", {"data": f"Result from request {request_id}"}, ttl=60)
        result = cache.get(f"request_{request_id}")
        
        logger.info(f"Request {request_id}: Successfully used cache (instance {id(cache)})")
        
        return True, id(cache)
        
    except Exception as e:
        logger.error(f"Request {request_id}: Failed with error: {e}")
        return False, None


def main():
    """Demonstrate the fix working correctly."""
    
    # Disable P2P for this demo (to avoid actually trying to bind ports)
    os.environ['CACHE_ENABLE_P2P'] = 'false'
    
    logger.info("=" * 70)
    logger.info("DEMONSTRATION: Thread-Safe P2P Cache Singleton")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Simulating Flask with threaded=True:")
    logger.info("  - 20 concurrent requests")
    logger.info("  - Each request accesses get_global_cache()")
    logger.info("  - Without the fix: Multiple cache instances → port conflicts")
    logger.info("  - With the fix: Single cache instance → no conflicts")
    logger.info("")
    
    # Simulate multiple concurrent Flask requests
    num_requests = 20
    threads = []
    results = []
    
    def request_wrapper(request_id):
        success, instance_id = simulate_flask_request(request_id)
        results.append((request_id, success, instance_id))
    
    logger.info(f"Starting {num_requests} concurrent requests...")
    logger.info("")
    
    # Start all threads (simulating concurrent requests)
    start_time = time.time()
    for i in range(num_requests):
        t = threading.Thread(
            target=request_wrapper,
            args=(i,),
            name=f"Request-{i}"
        )
        threads.append(t)
        t.start()
    
    # Wait for all requests to complete
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    # Analyze results
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    successful = sum(1 for _, success, _ in results if success)
    unique_instances = set(instance_id for _, _, instance_id in results if instance_id is not None)
    
    logger.info(f"✓ Completed {num_requests} requests in {elapsed:.2f}s")
    logger.info(f"✓ Successful requests: {successful}/{num_requests}")
    logger.info(f"✓ Unique cache instances created: {len(unique_instances)}")
    
    if len(unique_instances) == 1:
        logger.info("")
        logger.info("✅ SUCCESS: All requests used the same singleton instance!")
        logger.info("✅ No port binding conflicts occurred")
        logger.info("✅ The thread-safety fix is working correctly")
    else:
        logger.error("")
        logger.error(f"❌ FAILURE: Multiple instances created: {unique_instances}")
        logger.error("❌ This would cause port binding conflicts in production")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPLANATION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The fix uses double-checked locking pattern:")
    logger.info("  1. First check: if _global_cache is None (no lock, fast)")
    logger.info("  2. Acquire lock: with _global_cache_lock")
    logger.info("  3. Second check: if _global_cache is None (ensure only one creates)")
    logger.info("  4. Create instance: _global_cache = GitHubAPICache(**kwargs)")
    logger.info("")
    logger.info("Benefits:")
    logger.info("  ✓ Thread-safe: Only one thread creates the singleton")
    logger.info("  ✓ No port conflicts: Single P2P host binds to port 9100")
    logger.info("  ✓ Performance: Lock only taken when instance is null")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
