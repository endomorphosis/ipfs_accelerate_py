"""
Tests for Common Cache Infrastructure

Tests the base cache, LLM cache, HuggingFace Hub cache, and Docker cache.
"""

import pytest
import time
import tempfile
from pathlib import Path

from ipfs_accelerate_py.common.base_cache import (
    BaseAPICache,
    CacheEntry,
    get_all_caches,
    register_cache,
    shutdown_all_caches
)
from ipfs_accelerate_py.common.llm_cache import LLMAPICache, get_global_llm_cache
from ipfs_accelerate_py.common.hf_hub_cache import HuggingFaceHubCache, get_global_hf_hub_cache
from ipfs_accelerate_py.common.docker_cache import DockerAPICache, get_global_docker_cache


class TestCacheEntry:
    """Test CacheEntry data class."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            data={"key": "value"},
            timestamp=time.time(),
            ttl=300
        )
        
        assert entry.data == {"key": "value"}
        assert entry.ttl == 300
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        entry = CacheEntry(
            data={"key": "value"},
            timestamp=time.time() - 400,  # 400 seconds ago
            ttl=300  # 5 minute TTL
        )
        
        assert entry.is_expired()
    
    def test_cache_entry_not_expired(self):
        """Test cache entry not expired."""
        entry = CacheEntry(
            data={"key": "value"},
            timestamp=time.time() - 100,  # 100 seconds ago
            ttl=300  # 5 minute TTL
        )
        
        assert not entry.is_expired()


class SimpleCacheImplementation(BaseAPICache):
    """Simple cache implementation for testing."""
    
    def get_cache_namespace(self):
        return "test_cache"
    
    def extract_validation_fields(self, operation, data):
        if isinstance(data, dict) and "version" in data:
            return {"version": data["version"]}
        return None


class TestBaseAPICache:
    """Test BaseAPICache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(
                cache_dir=tmpdir,
                default_ttl=300,
                max_cache_size=100,
                enable_persistence=True
            )
            
            assert cache.cache_name == "test_cache"
            assert cache.default_ttl == 300
            assert cache.max_cache_size == 100
            assert cache.enable_persistence
    
    def test_cache_put_and_get(self):
        """Test putting and getting values from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(cache_dir=tmpdir)
            
            # Put value in cache
            cache.put("test_op", {"result": "success"}, param1="value1")
            
            # Get value from cache
            result = cache.get("test_op", param1="value1")
            
            assert result == {"result": "success"}
    
    def test_cache_miss(self):
        """Test cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(cache_dir=tmpdir)
            
            result = cache.get("nonexistent_op", param1="value1")
            
            assert result is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(cache_dir=tmpdir)
            
            # Put value with short TTL
            cache.put("test_op", {"result": "success"}, ttl=1, param1="value1")
            
            # Should get value immediately
            result = cache.get("test_op", param1="value1")
            assert result == {"result": "success"}
            
            # Wait for expiration
            time.sleep(2)
            
            # Should be None after expiration
            result = cache.get("test_op", param1="value1")
            assert result is None
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(cache_dir=tmpdir)
            
            # Put value in cache
            cache.put("test_op", {"result": "success"}, param1="value1")
            
            # Verify it's cached
            result = cache.get("test_op", param1="value1")
            assert result is not None
            
            # Invalidate
            invalidated = cache.invalidate("test_op", param1="value1")
            assert invalidated
            
            # Should be None after invalidation
            result = cache.get("test_op", param1="value1")
            assert result is None
    
    def test_cache_invalidate_pattern(self):
        """Test pattern-based cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(cache_dir=tmpdir)
            
            # Put multiple values
            cache.put("test_op", {"result": "1"}, param1="value1")
            cache.put("test_op", {"result": "2"}, param1="value2")
            cache.put("other_op", {"result": "3"}, param1="value1")
            
            # Invalidate pattern
            count = cache.invalidate_pattern("test_op")
            assert count == 2
            
            # test_op entries should be gone
            assert cache.get("test_op", param1="value1") is None
            assert cache.get("test_op", param1="value2") is None
            
            # other_op should still be there
            assert cache.get("other_op", param1="value1") is not None
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(cache_dir=tmpdir)
            
            # Perform some operations
            cache.put("test_op", {"result": "success"}, param1="value1")
            cache.get("test_op", param1="value1")  # Hit
            cache.get("nonexistent", param1="value2")  # Miss
            
            stats = cache.get_stats()
            
            assert stats["cache_name"] == "test_cache"
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 0.5
            assert stats["cache_size"] == 1
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(
                cache_dir=tmpdir,
                max_cache_size=3
            )
            
            # Fill cache
            cache.put("op1", {"result": "1"}, id=1)
            cache.put("op2", {"result": "2"}, id=2)
            cache.put("op3", {"result": "3"}, id=3)
            
            # Add one more, should evict oldest
            cache.put("op4", {"result": "4"}, id=4)
            
            stats = cache.get_stats()
            assert stats["cache_size"] == 3
            assert stats["evictions"] == 1


class TestLLMAPICache:
    """Test LLM API Cache."""
    
    def test_llm_cache_initialization(self):
        """Test LLM cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMAPICache(cache_dir=tmpdir)
            
            assert cache.get_cache_namespace() == "llm_api"
    
    def test_completion_caching(self):
        """Test caching completion responses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMAPICache(cache_dir=tmpdir)
            
            prompt = "Explain quantum computing"
            response = {
                "choices": [{"text": "Quantum computing is..."}],
                "usage": {"total_tokens": 100}
            }
            
            # Cache completion
            cache.cache_completion(
                prompt=prompt,
                response=response,
                model="gpt-4",
                temperature=0.0
            )
            
            # Retrieve from cache
            cached = cache.get_completion(
                prompt=prompt,
                model="gpt-4",
                temperature=0.0
            )
            
            assert cached == response
    
    def test_chat_completion_caching(self):
        """Test caching chat completion responses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMAPICache(cache_dir=tmpdir)
            
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
            response = {
                "choices": [{"message": {"content": "How can I help?"}}]
            }
            
            # Cache chat completion
            cache.cache_chat_completion(
                messages=messages,
                response=response,
                model="gpt-4",
                temperature=0.0
            )
            
            # Retrieve from cache
            cached = cache.get_chat_completion(
                messages=messages,
                model="gpt-4",
                temperature=0.0
            )
            
            assert cached == response
    
    def test_different_temperatures_different_cache(self):
        """Test that different temperatures use different cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMAPICache(cache_dir=tmpdir)
            
            prompt = "Write a poem"
            response1 = {"choices": [{"text": "Roses are red..."}]}
            response2 = {"choices": [{"text": "Violets are blue..."}]}
            
            # Cache with temp=0.0
            cache.cache_completion(
                prompt=prompt,
                response=response1,
                model="gpt-4",
                temperature=0.0
            )
            
            # Cache with temp=0.7
            cache.cache_completion(
                prompt=prompt,
                response=response2,
                model="gpt-4",
                temperature=0.7
            )
            
            # Should get different responses
            cached1 = cache.get_completion(prompt=prompt, model="gpt-4", temperature=0.0)
            cached2 = cache.get_completion(prompt=prompt, model="gpt-4", temperature=0.7)
            
            assert cached1 == response1
            assert cached2 == response2
    
    def test_global_llm_cache(self):
        """Test global LLM cache singleton."""
        cache1 = get_global_llm_cache()
        cache2 = get_global_llm_cache()
        
        assert cache1 is cache2


class TestHuggingFaceHubCache:
    """Test HuggingFace Hub Cache."""
    
    def test_hf_cache_initialization(self):
        """Test HF cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HuggingFaceHubCache(cache_dir=tmpdir)
            
            assert cache.get_cache_namespace() == "huggingface_hub"
    
    def test_model_info_caching(self):
        """Test caching model info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HuggingFaceHubCache(cache_dir=tmpdir)
            
            model_info = {
                "modelId": "meta-llama/Llama-2-7b-hf",
                "sha": "abc123",
                "lastModified": "2025-01-15T10:00:00Z",
                "downloads": 150000,
                "likes": 500
            }
            
            # Cache model info
            cache.put("model_info", model_info, model="meta-llama/Llama-2-7b-hf")
            
            # Retrieve from cache
            cached = cache.get("model_info", model="meta-llama/Llama-2-7b-hf")
            
            assert cached == model_info
    
    def test_validation_fields_extraction(self):
        """Test extraction of validation fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HuggingFaceHubCache(cache_dir=tmpdir)
            
            model_info = {
                "sha": "abc123",
                "lastModified": "2025-01-15T10:00:00Z",
                "downloads": 150000,
                "likes": 500
            }
            
            fields = cache.extract_validation_fields("model_info", model_info)
            
            assert fields is not None
            assert fields["sha"] == "abc123"
            assert fields["lastModified"] == "2025-01-15T10:00:00Z"


class TestDockerAPICache:
    """Test Docker API Cache."""
    
    def test_docker_cache_initialization(self):
        """Test Docker cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DockerAPICache(cache_dir=tmpdir)
            
            assert cache.get_cache_namespace() == "docker_api"
    
    def test_image_inspect_caching(self):
        """Test caching image inspection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DockerAPICache(cache_dir=tmpdir)
            
            image_data = {
                "Id": "sha256:abc123",
                "Created": "2025-01-15T10:00:00Z",
                "Size": 1000000000,
                "RepoTags": ["ubuntu:22.04"]
            }
            
            # Cache image data
            cache.put("image_inspect", image_data, image="ubuntu:22.04")
            
            # Retrieve from cache
            cached = cache.get("image_inspect", image="ubuntu:22.04")
            
            assert cached == image_data
    
    def test_container_short_ttl(self):
        """Test that container operations have short TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DockerAPICache(cache_dir=tmpdir)
            
            ttl = cache.get_default_ttl_for_operation("container_inspect")
            
            assert ttl == 30  # 30 seconds


class TestCacheRegistry:
    """Test global cache registry."""
    
    def test_register_and_get_cache(self):
        """Test registering and retrieving caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCacheImplementation(cache_dir=tmpdir)
            register_cache("test_cache", cache)
            
            from ipfs_accelerate_py.common.base_cache import get_cache
            retrieved = get_cache("test_cache")
            
            assert retrieved is cache
    
    def test_get_all_caches(self):
        """Test getting all registered caches."""
        # Clear registry first
        shutdown_all_caches()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = SimpleCacheImplementation(cache_dir=f"{tmpdir}/1")
            cache2 = SimpleCacheImplementation(cache_dir=f"{tmpdir}/2")
            
            register_cache("cache1", cache1)
            register_cache("cache2", cache2)
            
            all_caches = get_all_caches()
            
            assert "cache1" in all_caches
            assert "cache2" in all_caches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
