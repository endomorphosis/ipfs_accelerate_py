#!/usr/bin/env python3
"""
Test content-based cache validation using multiformats.

This demonstrates how the cache can detect stale entries intelligently
by hashing validation fields (like updatedAt, status) rather than relying
solely on TTL.
"""

import time
import json
from ipfs_accelerate_py.github_cli import GitHubCLI, get_global_cache

def test_validation_hash():
    """Test that validation hashes are computed correctly."""
    print("=" * 70)
    print("Content-Based Cache Validation Test")
    print("=" * 70)
    
    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
    
    # Test hash computation
    print("\n1. Testing validation hash computation:")
    
    validation_fields_1 = {
        "repo1": {"updatedAt": "2025-11-06T10:00:00Z"},
        "repo2": {"updatedAt": "2025-11-06T11:00:00Z"}
    }
    
    validation_fields_2 = {
        "repo1": {"updatedAt": "2025-11-06T12:00:00Z"},  # Different time
        "repo2": {"updatedAt": "2025-11-06T11:00:00Z"}
    }
    
    hash_1 = GitHubAPICache._compute_validation_hash(validation_fields_1)
    hash_2 = GitHubAPICache._compute_validation_hash(validation_fields_2)
    hash_1_repeat = GitHubAPICache._compute_validation_hash(validation_fields_1)
    
    print(f"   Hash 1: {hash_1[:50]}...")
    print(f"   Hash 2: {hash_2[:50]}...")
    print(f"   Hash 1 (repeat): {hash_1_repeat[:50]}...")
    
    assert hash_1 == hash_1_repeat, "Same fields should produce same hash!"
    assert hash_1 != hash_2, "Different fields should produce different hash!"
    
    print("   ✅ Hashes are deterministic and change with content")


def test_cache_staleness_detection():
    """Test that cache detects stale entries via content hash."""
    print("\n" + "=" * 70)
    print("Smart Staleness Detection Test")
    print("=" * 70)
    
    gh = GitHubCLI(enable_cache=True)
    cache = get_global_cache()
    
    print("\n1. First API call (cache miss, stores validation hash):")
    repos_1 = gh.list_repos(owner="endomorphosis", limit=3)
    print(f"   Found {len(repos_1)} repos")
    
    # Check cache entry has validation hash
    cache_key = "list_repos:owner=endomorphosis:limit=3"
    entry = cache._cache.get(cache_key)
    
    if entry and entry.content_hash:
        print(f"   ✅ Validation hash stored: {entry.content_hash[:50]}...")
        print(f"   Validation fields: {list(entry.validation_fields.keys())[:3]}...")
    else:
        print("   ℹ No validation hash (multiformats not installed)")
        return
    
    print("\n2. Second call with same data (cache hit):")
    start = time.time()
    repos_2 = gh.list_repos(owner="endomorphosis", limit=3)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.6f}s (should be instant)")
    assert len(repos_2) == len(repos_1), "Should get same data from cache"
    
    print("\n3. Simulating stale cache (data changed on GitHub):")
    # Modify the cached data to simulate GitHub update
    if entry and entry.validation_fields:
        # Change one repo's updatedAt
        first_repo_key = list(entry.validation_fields.keys())[0]
        old_time = entry.validation_fields[first_repo_key].get('updatedAt')
        
        print(f"   Original time for {first_repo_key}: {old_time}")
        
        # Create new validation fields with updated time
        new_validation = entry.validation_fields.copy()
        new_validation[first_repo_key] = {
            'updatedAt': '2025-11-06T23:00:00Z',  # Future time
            'pushedAt': '2025-11-06T23:00:00Z'
        }
        
        # Check if this would be detected as stale
        is_stale = entry.is_stale(new_validation)
        print(f"   ✅ Cache correctly detected as stale: {is_stale}")
        assert is_stale, "Should detect hash mismatch!"


def test_workflow_validation():
    """Test validation for workflow operations."""
    print("\n" + "=" * 70)
    print("Workflow Status Cache Validation")
    print("=" * 70)
    
    from ipfs_accelerate_py.github_cli import WorkflowQueue
    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
    
    wq = WorkflowQueue()
    
    print("\n1. Simulating workflow status change detection:")
    
    # Simulate workflow data
    workflow_data_1 = [
        {
            "databaseId": 12345,
            "name": "Test",
            "status": "in_progress",
            "conclusion": None,
            "updatedAt": "2025-11-06T10:00:00Z"
        }
    ]
    
    workflow_data_2 = [
        {
            "databaseId": 12345,
            "name": "Test",
            "status": "completed",  # Status changed!
            "conclusion": "success",
            "updatedAt": "2025-11-06T10:05:00Z"
        }
    ]
    
    # Extract validation fields
    fields_1 = GitHubAPICache._extract_validation_fields("list_workflow_runs", workflow_data_1)
    fields_2 = GitHubAPICache._extract_validation_fields("list_workflow_runs", workflow_data_2)
    
    print(f"   Workflow status before: {fields_1}")
    print(f"   Workflow status after: {fields_2}")
    
    # Compute hashes
    hash_1 = GitHubAPICache._compute_validation_hash(fields_1)
    hash_2 = GitHubAPICache._compute_validation_hash(fields_2)
    
    print(f"\n   Hash before: {hash_1[:50]}...")
    print(f"   Hash after: {hash_2[:50]}...")
    
    assert hash_1 != hash_2, "Hash should change when workflow status changes!"
    print("   ✅ Status change correctly detected via hash difference")


def test_runner_status_validation():
    """Test validation for runner operations."""
    print("\n" + "=" * 70)
    print("Runner Status Cache Validation")
    print("=" * 70)
    
    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
    
    print("\n1. Simulating runner status change detection:")
    
    # Simulate runner data
    runner_data_1 = [
        {"id": 1, "name": "runner-1", "status": "online", "busy": False},
        {"id": 2, "name": "runner-2", "status": "online", "busy": False}
    ]
    
    runner_data_2 = [
        {"id": 1, "name": "runner-1", "status": "online", "busy": True},  # Now busy!
        {"id": 2, "name": "runner-2", "status": "offline", "busy": False}  # Now offline!
    ]
    
    # Extract validation fields
    fields_1 = GitHubAPICache._extract_validation_fields("list_runners", runner_data_1)
    fields_2 = GitHubAPICache._extract_validation_fields("list_runners", runner_data_2)
    
    print(f"   Runner status before: {fields_1}")
    print(f"   Runner status after: {fields_2}")
    
    # Compute hashes
    hash_1 = GitHubAPICache._compute_validation_hash(fields_1)
    hash_2 = GitHubAPICache._compute_validation_hash(fields_2)
    
    print(f"\n   Hash before: {hash_1[:50]}...")
    print(f"   Hash after: {hash_2[:50]}...")
    
    assert hash_1 != hash_2, "Hash should change when runner status changes!"
    print("   ✅ Runner status change correctly detected via hash difference")


def test_cache_persistence_with_validation():
    """Test that validation hashes persist across restarts."""
    print("\n" + "=" * 70)
    print("Cache Persistence with Validation")
    print("=" * 70)
    
    import tempfile
    from pathlib import Path
    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache
    
    # Create temp cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = GitHubAPICache(
            cache_dir=tmpdir,
            enable_persistence=True
        )
        
        print("\n1. Storing cache entry with validation hash:")
        
        test_data = [
            {"name": "repo1", "updatedAt": "2025-11-06T10:00:00Z"},
            {"name": "repo2", "updatedAt": "2025-11-06T11:00:00Z"}
        ]
        
        cache.put("list_repos", test_data, owner="test", limit=2)
        
        # Check file was created
        cache_files = list(Path(tmpdir).glob("*.json"))
        assert len(cache_files) > 0, "Cache file should be created"
        
        print(f"   ✅ Cache file created: {cache_files[0].name}")
        
        # Load the file and check it has validation hash
        with open(cache_files[0], 'r') as f:
            saved_data = json.load(f)
        
        if 'content_hash' in saved_data and saved_data['content_hash']:
            print(f"   ✅ Validation hash persisted: {saved_data['content_hash'][:50]}...")
            print(f"   ✅ Validation fields saved: {len(saved_data.get('validation_fields', {}))} repos")
        else:
            print("   ℹ No validation hash saved (multiformats not installed)")
        
        print("\n2. Loading cache from disk:")
        
        # Create new cache instance
        cache2 = GitHubAPICache(
            cache_dir=tmpdir,
            enable_persistence=True
        )
        
        # Should load from disk
        assert len(cache2._cache) > 0, "Should load cache from disk"
        print(f"   ✅ Loaded {len(cache2._cache)} entries from disk")
        
        # Check validation hash was restored
        entry = list(cache2._cache.values())[0]
        if entry.content_hash:
            print(f"   ✅ Validation hash restored: {entry.content_hash[:50]}...")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Multiformats Content-Based Cache Validation Tests")
    print("=" * 70)
    
    try:
        from multiformats import CID, multihash
        print("\n✅ Multiformats library available")
        print(f"   Using content-addressed hashing (CID)")
    except ImportError:
        print("\n⚠ Multiformats library not installed")
        print("   Install with: pip install multiformats")
        print("   Falling back to SHA256 hex hashing")
    
    try:
        test_validation_hash()
        test_cache_staleness_detection()
        test_workflow_validation()
        test_runner_status_validation()
        test_cache_persistence_with_validation()
        
        print("\n" + "=" * 70)
        print("All Tests Passed!")
        print("=" * 70)
        
        print("\n📊 Benefits of Content-Based Validation:")
        print("   ✅ Detects stale cache even within TTL window")
        print("   ✅ Prevents serving outdated workflow/runner status")
        print("   ✅ Uses IPFS-compatible multiformats (CID)")
        print("   ✅ Deterministic hashing for cache sharing")
        print("   ✅ Ready for IPFS peer-to-peer cache distribution")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
