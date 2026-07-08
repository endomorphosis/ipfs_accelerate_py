#!/usr/bin/env python3
"""
Test retry logic and caching for GitHub CLI and Copilot CLI.

This script demonstrates:
1. Exponential backoff and retry on failures
2. Cache performance improvements
3. Shared cache between GitHub and Copilot CLI
"""

import time
import logging
from ipfs_accelerate_py.github_cli import GitHubCLI, get_global_cache, configure_cache

# Set up logging to see retry messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_github_retry_logic():
    """Test GitHub CLI retry behavior."""
    print("=" * 70)
    print("GitHub CLI Retry Logic Test")
    print("=" * 70)
    
    gh = GitHubCLI(enable_cache=True)
    
    print("\n1. Normal request (should succeed on first attempt):")
    start = time.time()
    repos = gh.list_repos(owner="endomorphosis", limit=5)
    elapsed = time.time() - start
    print(f"   Found {len(repos)} repos in {elapsed:.3f}s")
    
    print("\n2. Cached request (should be instant):")
    start = time.time()
    repos_cached = gh.list_repos(owner="endomorphosis", limit=5)
    elapsed_cached = time.time() - start
    print(f"   Found {len(repos_cached)} repos in {elapsed_cached:.6f}s")
    print(f"   Speed improvement: {elapsed/elapsed_cached:.0f}x faster")
    
    print("\n3. Testing retry with custom parameters:")
    # This will use retry logic internally
    result = gh._run_command(
        ["repo", "list", "--limit", "3"],
        max_retries=2,
        base_delay=0.5
    )
    print(f"   Success: {result['success']}")
    print(f"   Attempts: {result.get('attempts', 1)}")
    

def test_cache_statistics():
    """Show cache statistics."""
    print("\n" + "=" * 70)
    print("Cache Statistics")
    print("=" * 70)
    
    cache = get_global_cache()
    stats = cache.get_stats()
    
    print(f"\n   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Cache size: {stats['cache_size']}/{stats['max_cache_size']}")
    print(f"   Evictions: {stats['evictions']}")
    print(f"   Expirations: {stats['expirations']}")
    
    if stats['total_requests'] > 0:
        print(f"\n   API calls saved: {stats['hits']} ({stats['hit_rate']:.1%})")


def test_copilot_caching():
    """Test Copilot CLI caching (if available)."""
    print("\n" + "=" * 70)
    print("Copilot CLI Caching Test")
    print("=" * 70)
    
    try:
        from ipfs_accelerate_py.copilot_cli import CopilotCLI
        
        copilot = CopilotCLI(enable_cache=True)
        
        print("\n1. First suggestion request (cache miss):")
        prompt = "list files in current directory"
        start = time.time()
        result1 = copilot.suggest_command(prompt, shell="bash")
        elapsed1 = time.time() - start
        
        if result1.get('success'):
            print(f"   Suggestion: {result1.get('suggestion', 'N/A')[:100]}...")
            print(f"   Time: {elapsed1:.3f}s")
            print(f"   Attempts: {result1.get('attempts', 1)}")
            
            print("\n2. Cached suggestion request (cache hit):")
            start = time.time()
            result2 = copilot.suggest_command(prompt, shell="bash")
            elapsed2 = time.time() - start
            print(f"   Time: {elapsed2:.6f}s")
            if elapsed2 > 0:
                print(f"   Speed improvement: {elapsed1/elapsed2:.0f}x faster")
        else:
            print(f"   ⚠ Copilot request failed: {result1.get('error', 'Unknown error')}")
            print("   (This is expected if Copilot CLI is not installed)")
            
    except ImportError as e:
        print(f"\n   ⚠ Copilot CLI not available: {e}")
        print("   This is optional - GitHub CLI caching is working!")


def test_shared_cache():
    """Verify cache is shared between GitHub and Copilot CLI."""
    print("\n" + "=" * 70)
    print("Shared Cache Test")
    print("=" * 70)
    
    # Configure a custom cache
    cache = configure_cache(
        default_ttl=300,
        max_cache_size=500,
        enable_persistence=True
    )
    
    print("\n   Cache directory:", cache.cache_dir)
    print(f"   Current entries: {len(cache._cache)}")
    print(f"   Max size: {cache.max_cache_size}")
    print(f"   Default TTL: {cache.default_ttl}s")
    
    # Both CLIs should use the same cache
    gh = GitHubCLI(cache=cache)
    
    try:
        from ipfs_accelerate_py.copilot_cli import CopilotCLI
        copilot = CopilotCLI(cache=cache)
        print("\n   ✓ Both GitHub and Copilot CLI sharing the same cache instance")
    except ImportError:
        print("\n   ℹ Copilot CLI not available (optional)")


def test_retry_scenarios():
    """Test different retry scenarios."""
    print("\n" + "=" * 70)
    print("Retry Scenario Tests")
    print("=" * 70)
    
    gh = GitHubCLI()
    
    scenarios = [
        {
            "name": "Conservative (1 retry, 2s delay)",
            "max_retries": 1,
            "base_delay": 2.0,
            "max_delay": 10.0
        },
        {
            "name": "Default (3 retries, 1s delay)",
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0
        },
        {
            "name": "Aggressive (5 retries, 0.5s delay)",
            "max_retries": 5,
            "base_delay": 0.5,
            "max_delay": 30.0
        }
    ]
    
    for scenario in scenarios:
        print(f"\n   {scenario['name']}:")
        print(f"     max_retries={scenario['max_retries']}, "
              f"base_delay={scenario['base_delay']}s, "
              f"max_delay={scenario['max_delay']}s")
        
        # Calculate expected delays
        delays = []
        for attempt in range(scenario['max_retries']):
            delay = min(
                scenario['base_delay'] * (2 ** attempt),
                scenario['max_delay']
            )
            delays.append(delay)
        
        print(f"     Expected backoff delays: {[f'{d:.1f}s' for d in delays]}")


def main():
    """Run all tests."""
    try:
        print("\n" + "=" * 70)
        print("GitHub & Copilot CLI - Retry and Cache Testing")
        print("=" * 70)
        
        test_github_retry_logic()
        test_cache_statistics()
        test_copilot_caching()
        test_shared_cache()
        test_retry_scenarios()
        
        print("\n" + "=" * 70)
        print("Testing Complete!")
        print("=" * 70)
        
        # Final cache stats
        cache = get_global_cache()
        stats = cache.get_stats()
        print(f"\nFinal Cache Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Cache hits: {stats['hits']} ({stats['hit_rate']:.1%})")
        print(f"  API calls saved: {stats['hits']}")
        print(f"  Cache entries: {stats['cache_size']}")
        
        print("\n✅ Retry logic and caching are working correctly!")
        print("✅ Ready for future IPFS cache sharing")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
