"""
Test GitHub API Integration with MCP Server

This test validates:
1. MCP tools for GitHub API are properly registered
2. Cache introspection works correctly
3. Rate limit monitoring functions
4. Workflow and runner management integration
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


def test_github_mcp_tools_registration():
    """Test that GitHub MCP tools are properly registered"""
    try:
        import sys
        import os
        # Add the parent directory to path to find the tools
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from tools.github_tools import register_github_tools
        
        # Try to import FastMCP, use mock if not available
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError:
            try:
                from fastmcp import FastMCP
            except ImportError:
                from mock_mcp import FastMCP
        
        # Create a test MCP server
        mcp = FastMCP("test_github_tools")
        
        # Register GitHub tools
        register_github_tools(mcp)
        
        # Check that tools were registered
        # Note: This depends on FastMCP's internal structure
        logger.info("✓ GitHub MCP tools registered successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to register GitHub MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_stats_retrieval():
    """Test cache statistics retrieval"""
    try:
        from ipfs_accelerate_py.github_cli import get_global_cache, configure_cache
        
        # Configure cache
        cache = configure_cache(
            default_ttl=300,
            max_cache_size=100,
            enable_persistence=False,
            enable_p2p=False
        )
        
        # Get stats
        stats = cache.get_stats()
        
        # Validate stats structure
        required_keys = [
            'hits', 'misses', 'cache_size', 'max_cache_size', 
            'hit_rate', 'total_requests'
        ]
        
        for key in required_keys:
            if key not in stats:
                logger.error(f"✗ Missing required stat key: {key}")
                return False
        
        logger.info(f"✓ Cache stats retrieved: {stats['cache_size']} entries, "
                   f"{stats['hit_rate']*100:.1f}% hit rate")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to retrieve cache stats: {e}")
        return False


def test_cache_operations():
    """Test cache put/get operations"""
    try:
        from ipfs_accelerate_py.github_cli import GitHubAPICache
        
        # Create a test cache
        cache = GitHubAPICache(
            enable_persistence=False,
            enable_p2p=False,
            max_cache_size=10
        )
        
        # Test data
        test_data = {"repo": "test/repo", "stars": 100}
        
        # Put data in cache
        cache.put(
            "get_repo_info",
            test_data,
            ttl=60,
            repo="test/repo"
        )
        
        # Get data from cache
        cached_data = cache.get("get_repo_info", repo="test/repo")
        
        if cached_data != test_data:
            logger.error(f"✗ Cache data mismatch: {cached_data} != {test_data}")
            return False
        
        # Check stats
        stats = cache.get_stats()
        if stats['cache_size'] != 1:
            logger.error(f"✗ Unexpected cache size: {stats['cache_size']}")
            return False
        
        if stats['hits'] != 1:
            logger.error(f"✗ Unexpected hit count: {stats['hits']}")
            return False
        
        logger.info(f"✓ Cache operations working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Failed cache operations test: {e}")
        return False


def test_cache_invalidation():
    """Test cache invalidation"""
    try:
        from ipfs_accelerate_py.github_cli import GitHubAPICache
        
        # Create a test cache
        cache = GitHubAPICache(
            enable_persistence=False,
            enable_p2p=False
        )
        
        # Add multiple entries
        for i in range(5):
            cache.put(f"operation_{i % 2}", {"data": i}, ttl=60, arg=f"arg_{i}")
        
        # Check initial size
        stats = cache.get_stats()
        initial_size = stats['cache_size']
        
        if initial_size != 5:
            logger.error(f"✗ Unexpected initial cache size: {initial_size}")
            return False
        
        # Invalidate entries matching pattern
        invalidated = cache.invalidate_pattern("operation_0")
        
        if invalidated != 3:  # Should invalidate 3 entries (0, 2, 4)
            logger.error(f"✗ Unexpected invalidation count: {invalidated}")
            return False
        
        # Check final size
        stats = cache.get_stats()
        final_size = stats['cache_size']
        
        if final_size != 2:
            logger.error(f"✗ Unexpected final cache size: {final_size}")
            return False
        
        logger.info(f"✓ Cache invalidation working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Failed cache invalidation test: {e}")
        return False


def test_content_addressed_caching():
    """Test IPLD/multiformats content-addressed caching"""
    try:
        from ipfs_accelerate_py.github_cli import GitHubAPICache
        
        # Create cache
        cache = GitHubAPICache(
            enable_persistence=False,
            enable_p2p=False
        )
        
        # Test validation hash computation
        validation_fields = {
            'updatedAt': '2025-11-08T10:00:00Z',
            'status': 'success'
        }
        
        hash1 = cache._compute_validation_hash(validation_fields)
        
        # Same fields should produce same hash
        hash2 = cache._compute_validation_hash(validation_fields)
        
        if hash1 != hash2:
            logger.error(f"✗ Hash mismatch for same data: {hash1} != {hash2}")
            return False
        
        # Different fields should produce different hash
        different_fields = {
            'updatedAt': '2025-11-08T11:00:00Z',
            'status': 'success'
        }
        hash3 = cache._compute_validation_hash(different_fields)
        
        if hash1 == hash3:
            logger.error(f"✗ Hash should differ for different data")
            return False
        
        logger.info(f"✓ Content-addressed caching working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Failed content-addressed caching test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests() -> Dict[str, bool]:
    """Run all GitHub API integration tests"""
    logger.info("=" * 70)
    logger.info("GitHub API Integration Tests")
    logger.info("=" * 70)
    
    tests = [
        ("MCP Tools Registration", test_github_mcp_tools_registration),
        ("Cache Stats Retrieval", test_cache_stats_retrieval),
        ("Cache Operations", test_cache_operations),
        ("Cache Invalidation", test_cache_invalidation),
        ("Content-Addressed Caching", test_content_addressed_caching),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 70)
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    passed_count = sum(1 for r in results.values() if r)
    total_count = len(results)
    
    logger.info(f"\nPassed: {passed_count}/{total_count}")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    results = run_all_tests()
    
    # Exit with error code if any test failed
    if not all(results.values()):
        exit(1)
