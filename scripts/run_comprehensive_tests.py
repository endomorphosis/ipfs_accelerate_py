"""
Comprehensive API Integration Test Runner
Runs all tests without requiring pytest
"""

import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from ipfs_accelerate_py.common.base_cache import BaseAPICache, CacheEntry
    from ipfs_accelerate_py.common.cid_index import CIDCacheIndex
    from ipfs_accelerate_py.common.llm_cache import LLMAPICache, get_global_llm_cache
    from ipfs_accelerate_py.common.hf_hub_cache import HuggingFaceHubCache, get_global_hf_hub_cache
    from ipfs_accelerate_py.common.docker_cache import DockerAPICache, get_global_docker_cache
    from ipfs_accelerate_py.common.kubernetes_cache import KubernetesAPICache, get_global_kubernetes_cache
    from ipfs_accelerate_py.common.huggingface_hugs_cache import HuggingFaceHugsCache, get_global_hugs_cache
    # IPFS fallback is optional
    try:
        from ipfs_accelerate_py.common.ipfs_kit_fallback import IPFSKitFallbackStore, get_global_ipfs_fallback
        IPFS_AVAILABLE = True
    except ImportError:
        IPFS_AVAILABLE = False
        IPFSKitFallbackStore = None
        get_global_ipfs_fallback = None
    
    CACHE_AVAILABLE = True
    # Use the private method for CID computation
    compute_cid = BaseAPICache._compute_cid
    # Use correct class name
    CIDIndex = CIDCacheIndex
except ImportError as e:
    print(f"ERROR: Cache modules not available: {e}")
    CACHE_AVAILABLE = False
    sys.exit(1)

import json
import time
import tempfile
import shutil


def run_test(test_name, test_func):
    """Run a single test and report results."""
    try:
        test_func()
        print(f"✅ PASS: {test_name}")
        return True
    except AssertionError as e:
        print(f"❌ FAIL: {test_name}")
        print(f"   {str(e)}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {test_name}")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return False


def test_cid_deterministic():
    """Test that same input produces same CID."""
    data1 = json.dumps({"key": "value", "number": 42}, sort_keys=True)
    data2 = json.dumps({"number": 42, "key": "value"}, sort_keys=True)
    
    cid1 = compute_cid(data1)
    cid2 = compute_cid(data2)
    
    assert cid1 == cid2, f"Same data should produce same CID: {cid1} != {cid2}"
    assert len(cid1) > 0, "CID should not be empty"


def test_cid_different_data():
    """Test that different input produces different CID."""
    data1 = json.dumps({"key": "value1"}, sort_keys=True)
    data2 = json.dumps({"key": "value2"}, sort_keys=True)
    
    cid1 = compute_cid(data1)
    cid2 = compute_cid(data2)
    
    assert cid1 != cid2, f"Different data should produce different CIDs: {cid1} == {cid2}"


def test_cid_index_operations():
    """Test CID index basic operations."""
    index = CIDIndex()
    
    # Add entries
    index.add("cid123", "operation1", {"param": "value1"})
    index.add("cid456", "operation1", {"param": "value2"})
    index.add("cid789", "operation2", {"param": "value3"})
    
    # Test get
    metadata = index.get("cid123")
    assert metadata is not None, "Should find metadata"
    assert metadata["operation"] == "operation1", f"Wrong operation: {metadata['operation']}"
    
    # Test operations
    ops = index.get_operations()
    assert "operation1" in ops, "Should have operation1"
    assert "operation2" in ops, "Should have operation2"
    
    # Test by operation
    cids = index.get_cids_by_operation("operation1")
    assert "cid123" in cids, "Should find cid123"
    assert "cid456" in cids, "Should find cid456"
    assert "cid789" not in cids, "Should not find cid789"


def test_llm_cache_completion():
    """Test LLM completion caching."""
    cache = LLMAPICache()
    
    # Cache a completion
    prompt = "What is Python?"
    response = {"text": "Python is a programming language", "usage": {"tokens": 10}}
    cache.cache_completion(prompt, response, model="gpt-4", temperature=0.0)
    
    # Retrieve from cache
    cached = cache.get_completion(prompt, model="gpt-4", temperature=0.0)
    assert cached is not None, "Should find cached response"
    assert cached["text"] == response["text"], f"Wrong response: {cached['text']}"


def test_llm_cache_chat_completion():
    """Test chat completion caching."""
    cache = LLMAPICache()
    
    messages = [{"role": "user", "content": "Hello"}]
    response = {"choices": [{"message": {"content": "Hi there!"}}]}
    
    cache.cache_chat_completion(messages, response, model="gpt-4", temperature=0.0)
    
    cached = cache.get_chat_completion(messages, model="gpt-4", temperature=0.0)
    assert cached is not None, "Should find cached chat"
    assert cached["choices"][0]["message"]["content"] == "Hi there!", "Wrong content"


def test_hf_hub_cache():
    """Test HuggingFace Hub cache."""
    cache = HuggingFaceHubCache()
    
    model_data = {
        "modelId": "bert-base-uncased",
        "author": "google",
        "downloads": 1000000
    }
    
    cache.put("model_info", model_data, model_id="bert-base-uncased")
    
    cached = cache.get("model_info", model_id="bert-base-uncased")
    assert cached is not None, "Should find cached model"
    assert cached["modelId"] == "bert-base-uncased", "Wrong model ID"


def test_docker_cache():
    """Test Docker API cache."""
    cache = DockerAPICache()
    
    image_data = {
        "Id": "sha256:abc123",
        "RepoTags": ["ubuntu:latest"],
        "Size": 72800000
    }
    
    cache.put("image_info", image_data, image_id="ubuntu:latest")
    
    cached = cache.get("image_info", image_id="ubuntu:latest")
    assert cached is not None, "Should find cached image"
    assert cached["Id"] == "sha256:abc123", "Wrong image ID"


def test_kubernetes_cache():
    """Test Kubernetes API cache."""
    cache = KubernetesAPICache()
    
    pod_data = {
        "metadata": {"name": "my-pod"},
        "status": {"phase": "Running"}
    }
    
    cache.put("pod_status", pod_data, pod_name="my-pod", namespace="default")
    
    cached = cache.get("pod_status", pod_name="my-pod", namespace="default")
    assert cached is not None, "Should find cached pod"
    assert cached["status"]["phase"] == "Running", "Wrong pod status"


def test_hugs_cache():
    """Test HuggingFace Hugs cache."""
    cache = HuggingFaceHugsCache()
    
    model_data = {
        "id": "bert-base-uncased",
        "downloads": 1000000,
        "likes": 500
    }
    
    cache.put("model_info", model_data, model_id="bert-base-uncased")
    
    cached = cache.get("model_info", model_id="bert-base-uncased")
    assert cached is not None, "Should find cached model"
    assert cached["id"] == "bert-base-uncased", "Wrong model ID"


def test_multiple_caches_coexist():
    """Test that multiple cache instances can coexist."""
    llm_cache = get_global_llm_cache()
    hf_cache = get_global_hf_hub_cache()
    docker_cache = get_global_docker_cache()
    k8s_cache = get_global_kubernetes_cache()
    hugs_cache = get_global_hugs_cache()
    
    # All should be different instances
    assert llm_cache is not None, "LLM cache should exist"
    assert hf_cache is not None, "HF cache should exist"
    assert docker_cache is not None, "Docker cache should exist"
    assert k8s_cache is not None, "K8s cache should exist"
    assert hugs_cache is not None, "Hugs cache should exist"
    
    # Should be able to use all simultaneously
    llm_cache.cache_completion("test", {"text": "response"}, "gpt-4", 0.0)
    hf_cache.put("model_info", {"id": "bert"}, model_id="bert")
    docker_cache.put("image_info", {"Id": "123"}, image_id="ubuntu")
    k8s_cache.put("pod_status", {"status": "running"}, pod_name="pod1", namespace="default")
    hugs_cache.put("model_info", {"id": "model"}, model_id="model1")
    
    # All should be retrievable
    assert llm_cache.get_completion("test", "gpt-4", 0.0) is not None, "Should retrieve LLM cache"
    assert hf_cache.get("model_info", model_id="bert") is not None, "Should retrieve HF cache"
    assert docker_cache.get("image_info", image_id="ubuntu") is not None, "Should retrieve Docker cache"
    assert k8s_cache.get("pod_status", pod_name="pod1", namespace="default") is not None, "Should retrieve K8s cache"
    assert hugs_cache.get("model_info", model_id="model1") is not None, "Should retrieve Hugs cache"


def test_cache_statistics():
    """Test statistics tracking across caches."""
    caches = [
        get_global_llm_cache(),
        get_global_hf_hub_cache(),
        get_global_docker_cache(),
        get_global_kubernetes_cache(),
        get_global_hugs_cache()
    ]
    
    for cache in caches:
        stats = cache.get_stats()
        assert "total_requests" in stats, "Should have total_requests"
        assert "cache_hits" in stats, "Should have cache_hits"
        assert "cache_misses" in stats, "Should have cache_misses"
        assert "hit_rate" in stats, "Should have hit_rate"


def test_llm_workflow():
    """Test complete LLM workflow."""
    cache = get_global_llm_cache()
    
    prompt = "What is the capital of France?"
    model = "gpt-4"
    temp = 0.0
    
    result1 = cache.get_completion(prompt, model, temp)
    # May or may not be None depending on previous runs
    
    # Cache the response
    response = {"text": "Paris", "usage": {"tokens": 2}}
    cache.cache_completion(prompt, response, model, temp)
    
    # Second request - should be cache hit
    result2 = cache.get_completion(prompt, model, temp)
    assert result2 is not None, "Second request should be cache hit"
    assert result2["text"] == response["text"], "Should return cached response"


def test_cache_performance():
    """Test cache lookup performance."""
    cache = LLMAPICache()
    
    # Add entries
    for i in range(100):
        cache.cache_completion(f"prompt{i}", {"text": f"response{i}"}, "gpt-4", 0.0)
    
    # Time lookups
    start = time.time()
    for _ in range(100):
        cache.get_completion("prompt50", "gpt-4", 0.0)
    elapsed = time.time() - start
    
    # Should be fast (< 100ms for 100 lookups)
    assert elapsed < 0.1, f"Cache lookups too slow: {elapsed}s for 100 lookups"


def main():
    """Run all tests."""
    print("=" * 70)
    print("COMPREHENSIVE API INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("CID Deterministic", test_cid_deterministic),
        ("CID Different Data", test_cid_different_data),
        ("CID Index Operations", test_cid_index_operations),
        ("LLM Cache Completion", test_llm_cache_completion),
        ("LLM Cache Chat Completion", test_llm_cache_chat_completion),
        ("HuggingFace Hub Cache", test_hf_hub_cache),
        ("Docker Cache", test_docker_cache),
        ("Kubernetes Cache", test_kubernetes_cache),
        ("HuggingFace Hugs Cache", test_hugs_cache),
        ("Multiple Caches Coexist", test_multiple_caches_coexist),
        ("Cache Statistics", test_cache_statistics),
        ("LLM Workflow", test_llm_workflow),
        ("Cache Performance", test_cache_performance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
