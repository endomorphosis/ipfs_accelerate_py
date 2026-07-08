#!/usr/bin/env python3
"""
Demo script showing CID-based cache infrastructure in action.

This demonstrates:
1. Content-addressed cache keys (CIDs)
2. Fast O(1) lookups
3. CID prefix search
4. Operation-based filtering
5. Cache statistics with CID index info
"""

import tempfile
import time
import json
from pathlib import Path

from ipfs_accelerate_py.common.base_cache import BaseAPICache, get_all_caches, shutdown_all_caches
from ipfs_accelerate_py.common.llm_cache import LLMAPICache
from ipfs_accelerate_py.common.hf_hub_cache import HuggingFaceHubCache
from ipfs_accelerate_py.common.docker_cache import DockerAPICache


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_llm_cache():
    """Demonstrate LLM API caching with CIDs."""
    print_section("LLM API Cache Demo")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = LLMAPICache(cache_dir=tmpdir)
        
        # Simulate API responses
        prompts = [
            ("Explain quantum computing", "gpt-4", 0.0),
            ("Write a poem about AI", "gpt-4", 0.7),
            ("Explain quantum computing", "gpt-4", 0.0),  # Duplicate - should hit cache
        ]
        
        for i, (prompt, model, temp) in enumerate(prompts, 1):
            print(f"\n{i}. Query: '{prompt}' (model={model}, temp={temp})")
            
            # Check cache first
            start = time.time()
            cached = cache.get_completion(
                prompt=prompt,
                model=model,
                temperature=temp
            )
            
            if cached:
                elapsed = time.time() - start
                print(f"   ✓ Cache HIT! ({elapsed*1000:.2f}ms)")
                print(f"   Response: {cached.get('text', 'N/A')[:50]}...")
            else:
                # Simulate API call
                time.sleep(0.1)  # Simulate network delay
                response = {
                    "text": f"Response to: {prompt}",
                    "tokens": 100
                }
                elapsed = time.time() - start
                print(f"   ✗ Cache MISS - Making API call ({elapsed*1000:.2f}ms)")
                
                # Cache the response
                cache.cache_completion(
                    prompt=prompt,
                    response=response,
                    model=model,
                    temperature=temp
                )
        
        # Show statistics
        print("\n" + "-" * 70)
        stats = cache.get_stats()
        print(f"Cache Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Cache hits: {stats['hits']}")
        print(f"  Cache misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  API calls saved: {stats['api_calls_saved']}")
        print(f"\nCID Index Statistics:")
        print(f"  Total CIDs indexed: {stats['cid_index']['total_cids']}")
        print(f"  Operations tracked: {stats['cid_index']['operations']}")
        print(f"  Operation counts: {stats['cid_index']['operation_counts']}")
        
        # Demonstrate CID-based operations
        print("\n" + "-" * 70)
        print("CID-based Operations:")
        
        # Find by operation
        completions = cache.find_by_operation("completion")
        print(f"  Found {len(completions)} cached completions")
        
        cache.shutdown()


def demo_hf_cache():
    """Demonstrate HuggingFace Hub caching."""
    print_section("HuggingFace Hub Cache Demo")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HuggingFaceHubCache(cache_dir=tmpdir)
        
        # Simulate model info queries
        models = [
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
            "meta-llama/Llama-2-7b-hf",  # Duplicate
        ]
        
        for i, model_id in enumerate(models, 1):
            print(f"\n{i}. Query model info: {model_id}")
            
            # Check cache
            cached = cache.get("model_info", model=model_id)
            
            if cached:
                print(f"   ✓ Cache HIT!")
                print(f"   Model: {cached['modelId']}, Downloads: {cached['downloads']}")
            else:
                # Simulate API call
                model_info = {
                    "modelId": model_id,
                    "sha": f"abc{i}23",
                    "lastModified": "2025-01-15T10:00:00Z",
                    "downloads": 150000 + i * 1000,
                    "likes": 500
                }
                print(f"   ✗ Cache MISS - Fetching from Hub")
                cache.put("model_info", model_info, model=model_id)
        
        # Show statistics
        print("\n" + "-" * 70)
        stats = cache.get_stats()
        print(f"Cache Statistics:")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  CID Index: {stats['cid_index']['total_cids']} entries")
        
        cache.shutdown()


def demo_cid_features():
    """Demonstrate CID-specific features."""
    print_section("CID-Based Features Demo")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = LLMAPICache(cache_dir=tmpdir)
        
        # Add some cache entries
        queries = [
            ("Hello world", "gpt-4", 0.0),
            ("Goodbye world", "gpt-4", 0.0),
            ("Hello world", "gpt-3.5", 0.0),
        ]
        
        print("Adding cache entries:")
        for prompt, model, temp in queries:
            response = {"text": f"Response to: {prompt}"}
            cache.cache_completion(
                prompt=prompt,
                response=response,
                model=model,
                temperature=temp
            )
            
            # Generate the CID for this query
            query_obj = {
                "operation": "completion",
                "prompt": prompt,
                "model": model,
                "temperature": temp,
                "max_tokens": None
            }
            cid = cache._compute_cid(json.dumps(query_obj, sort_keys=True))
            print(f"  '{prompt}' (model={model}) -> CID: {cid[:40]}...")
        
        # Demonstrate finding by operation
        print(f"\n" + "-" * 70)
        print("Finding all completions:")
        completions = cache.find_by_operation("completion")
        print(f"  Found {len(completions)} cached completions")
        
        # Show CID index stats
        stats = cache.get_stats()
        print(f"\nCID Index Details:")
        print(f"  Total CIDs: {stats['cid_index']['total_cids']}")
        print(f"  By operation: {stats['cid_index']['operation_counts']}")
        
        cache.shutdown()


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       Content-Addressed Cache Infrastructure Demo                   ║
║                                                                      ║
║  Demonstrating CID-based caching with multiformats for fast O(1)    ║
║  lookups across LLM APIs, HuggingFace Hub, and Docker APIs          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run demos
        demo_llm_cache()
        demo_hf_cache()
        demo_cid_features()
        
        # Final summary
        print_section("Demo Complete")
        print("""
Key Takeaways:
  ✓ Cache keys are content-addressed (CIDs) computed from query parameters
  ✓ O(1) lookups by hashing the query to get the CID directly
  ✓ CID index enables fast prefix searches and operation filtering
  ✓ Unified caching pattern works across all API types
  ✓ Thread-safe, persistent, and ready for P2P distribution
        """)
        
    finally:
        # Cleanup
        shutdown_all_caches()


if __name__ == "__main__":
    main()
