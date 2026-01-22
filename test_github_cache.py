#!/usr/bin/env python3
"""
Test GitHub API caching functionality.

This script demonstrates the cache improvements for GitHub CLI wrapper.
"""

import time
import sys
from pathlib import Path

# Ensure the repo root's parent is on sys.path so `ipfs_accelerate_py` resolves
# to the repo-root package (and not the top-level `ipfs_accelerate_py.py` file).
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root.parent))

from ipfs_accelerate_py.ipfs_accelerate_py.github_cli import (  # type: ignore
    GitHubCLI,
    configure_cache,
)


def test_cache_performance():
    """Test the performance improvement from caching."""
    
    # Configure cache with custom settings
    cache = configure_cache(
        default_ttl=300,  # 5 minutes
        max_cache_size=500,
        enable_persistence=True
    )
    
    print("=" * 60)
    print("GitHub API Cache Performance Test")
    print("=" * 60)
    
    # Initialize GitHub CLI with caching enabled
    gh = GitHubCLI(enable_cache=True)
    
    # Test 1: List repositories (first call - miss)
    print("\n1. First call to list_repos (cache miss):")
    start = time.time()
    repos = gh.list_repos(owner="endomorphosis", limit=10)
    first_call_time = time.time() - start
    print(f"   Time: {first_call_time:.3f}s")
    print(f"   Repos found: {len(repos)}")
    
    # Test 2: Same call (cache hit)
    print("\n2. Second call to list_repos (cache hit):")
    start = time.time()
    repos_cached = gh.list_repos(owner="endomorphosis", limit=10)
    cached_call_time = time.time() - start
    print(f"   Time: {cached_call_time:.3f}s")
    print(f"   Repos found: {len(repos_cached)}")
    print(f"   Speed improvement: {first_call_time/cached_call_time:.1f}x faster")
    
    # Test 3: Cache statistics
    print("\n3. Cache Statistics:")
    stats = cache.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Cache size: {stats['cache_size']}/{stats['max_cache_size']}")
    
    # Test 4: Different parameters (cache miss)
    print("\n4. Different parameters (cache miss):")
    start = time.time()
    repos_limit_5 = gh.list_repos(owner="endomorphosis", limit=5)
    diff_params_time = time.time() - start
    print(f"   Time: {diff_params_time:.3f}s")
    print(f"   Repos found: {len(repos_limit_5)}")
    
    # Test 5: Cache invalidation
    print("\n5. Cache invalidation test:")
    print("   Invalidating 'list_repos' cache entries...")
    invalidated = cache.invalidate_pattern("list_repos")
    print(f"   Invalidated {invalidated} entries")
    
    # Test 6: After invalidation (cache miss)
    print("\n6. After invalidation (cache miss):")
    start = time.time()
    repos_after_clear = gh.list_repos(owner="endomorphosis", limit=10)
    after_clear_time = time.time() - start
    print(f"   Time: {after_clear_time:.3f}s")
    print(f"   Repos found: {len(repos_after_clear)}")
    
    # Final statistics
    print("\n7. Final Cache Statistics:")
    stats = cache.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Evictions: {stats['evictions']}")
    print(f"   Expirations: {stats['expirations']}")
    
    print("\n" + "=" * 60)
    print("Cache Benefits:")
    print("=" * 60)
    if cached_call_time > 0:
        print(f"✓ Cached requests are {first_call_time/cached_call_time:.1f}x faster")
    print(f"✓ Reduced API calls by {stats['hit_rate']:.1%}")
    print(f"✓ Cache persists across sessions")
    print(f"✓ Automatic expiration prevents stale data")
    print("=" * 60)


def test_cache_with_workflows():
    """Test caching with workflow operations."""
    from ipfs_accelerate_py.ipfs_accelerate_py.github_cli import WorkflowQueue
    
    print("\n" + "=" * 60)
    print("Workflow Queue Cache Test")
    print("=" * 60)
    
    # Get global cache for statistics
    from ipfs_accelerate_py.ipfs_accelerate_py.github_cli import get_global_cache

    cache = get_global_cache()
    
    # Initialize workflow queue
    wq = WorkflowQueue()
    
    # Test workflow runs (with caching)
    print("\n1. Fetching workflow runs (cache miss):")
    start = time.time()
    runs = wq.list_workflow_runs("endomorphosis/ipfs_accelerate_py", status="in_progress", limit=5)
    first_time = time.time() - start
    print(f"   Time: {first_time:.3f}s")
    print(f"   Runs found: {len(runs)}")
    
    # Second call (cache hit)
    print("\n2. Fetching same workflow runs (cache hit):")
    start = time.time()
    runs_cached = wq.list_workflow_runs("endomorphosis/ipfs_accelerate_py", status="in_progress", limit=5)
    cached_time = time.time() - start
    print(f"   Time: {cached_time:.3f}s")
    print(f"   Runs found: {len(runs_cached)}")
    if cached_time > 0:
        print(f"   Speed improvement: {first_time/cached_time:.1f}x faster")
    
    # Cache statistics
    print("\n3. Cache Statistics:")
    stats = cache.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_cache_performance()
        test_cache_with_workflows()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
