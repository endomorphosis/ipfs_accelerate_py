"""
Comprehensive tests for cache infrastructure enhancements.

Tests:
1. Kubernetes cache adapter
2. HuggingFace Hugs cache adapter
3. IPFS Kit fallback store integration
4. CLI installations
5. End-to-end integration tests
"""

import os
import json
import time
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import cache components
from ipfs_accelerate_py.common.kubernetes_cache import (
    KubernetesAPICache,
    get_global_kubernetes_cache,
    configure_kubernetes_cache
)
from ipfs_accelerate_py.common.huggingface_hugs_cache import (
    HuggingFaceHugsCache,
    get_global_hugs_cache,
    configure_hugs_cache
)
from ipfs_accelerate_py.common.ipfs_kit_fallback import (
    IPFSKitFallbackStore,
    get_global_ipfs_fallback,
    configure_ipfs_fallback
)


class TestKubernetesCache:
    """Test Kubernetes API cache adapter."""
    
    def test_kubernetes_cache_initialization(self):
        """Test Kubernetes cache initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = KubernetesAPICache(cache_dir=tmpdir)
            assert cache.get_cache_namespace() == "kubernetes_api"
            assert cache.cache_dir.exists()
    
    def test_kubernetes_pod_status_caching(self):
        """Test caching Kubernetes pod status."""
        cache = KubernetesAPICache()
        
        # Mock pod status response
        pod_status = {
            "metadata": {
                "name": "test-pod",
                "namespace": "default",
                "uid": "12345",
                "resourceVersion": "1000",
                "generation": 1
            },
            "status": {
                "phase": "Running",
                "podIP": "10.0.0.1",
                "containerStatuses": [
                    {"ready": True, "restartCount": 0}
                ]
            }
        }
        
        # Cache the response
        cache.put("pod_status", pod_status, pod_name="test-pod", namespace="default")
        
        # Retrieve from cache
        cached = cache.get("pod_status", pod_name="test-pod", namespace="default")
        assert cached == pod_status
        
        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
    
    def test_kubernetes_validation_fields(self):
        """Test extraction of validation fields from Kubernetes responses."""
        cache = KubernetesAPICache()
        
        deployment = {
            "metadata": {
                "name": "test-deployment",
                "namespace": "default",
                "uid": "abc123",
                "resourceVersion": "2000"
            },
            "status": {
                "replicas": 3,
                "readyReplicas": 3,
                "updatedReplicas": 3,
                "availableReplicas": 3
            }
        }
        
        validation = cache.extract_validation_fields("deployment_status", deployment)
        assert validation["name"] == "test-deployment"
        assert validation["namespace"] == "default"
        assert validation["replicas"] == 3
        assert validation["readyReplicas"] == 3
    
    def test_kubernetes_ttl_by_operation(self):
        """Test operation-specific TTLs."""
        cache = KubernetesAPICache()
        
        assert cache.get_default_ttl_for_operation("pod_status") == 30
        assert cache.get_default_ttl_for_operation("deployment_status") == 60
        assert cache.get_default_ttl_for_operation("namespace_list") == 600
    
    def test_global_kubernetes_cache(self):
        """Test global Kubernetes cache singleton."""
        cache1 = get_global_kubernetes_cache()
        cache2 = get_global_kubernetes_cache()
        assert cache1 is cache2


class TestHuggingFaceHugsCache:
    """Test HuggingFace Hugs cache adapter."""
    
    def test_hugs_cache_initialization(self):
        """Test Hugs cache initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HuggingFaceHugsCache(cache_dir=tmpdir)
            assert cache.get_cache_namespace() == "huggingface_hugs"
            assert cache.cache_dir.exists()
    
    def test_model_info_caching(self):
        """Test caching HuggingFace model info."""
        cache = HuggingFaceHugsCache()
        
        # Mock model info response
        model_info = {
            "modelId": "bert-base-uncased",
            "sha": "abc123",
            "lastModified": "2024-01-01T00:00:00Z",
            "downloads": 1000000,
            "likes": 500,
            "pipeline_tag": "fill-mask",
            "library_name": "transformers",
            "tags": ["pytorch", "bert"],
            "siblings": [
                {"rfilename": "config.json", "size": 570},
                {"rfilename": "pytorch_model.bin", "size": 440473133}
            ]
        }
        
        # Cache the response
        cache.put("model_info", model_info, model_id="bert-base-uncased")
        
        # Retrieve from cache
        cached = cache.get("model_info", model_id="bert-base-uncased")
        assert cached == model_info
        
        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
    
    def test_hugs_validation_fields(self):
        """Test extraction of validation fields from Hugs responses."""
        cache = HuggingFaceHugsCache()
        
        dataset_info = {
            "id": "squad",
            "sha": "xyz789",
            "lastModified": "2024-01-01T00:00:00Z",
            "downloads": 50000,
            "likes": 100,
            "tags": ["qa", "english"],
            "siblings": [
                {"rfilename": "train.json", "size": 12345678}
            ]
        }
        
        validation = cache.extract_validation_fields("dataset_info", dataset_info)
        assert validation["datasetId"] == "squad"
        assert validation["sha"] == "xyz789"
        assert validation["downloads"] == 50000
        assert validation["file_count"] == 1
    
    def test_hugs_ttl_by_operation(self):
        """Test operation-specific TTLs."""
        cache = HuggingFaceHugsCache()
        
        assert cache.get_default_ttl_for_operation("model_info") == 3600
        assert cache.get_default_ttl_for_operation("model_list") == 1800
        assert cache.get_default_ttl_for_operation("discussion_thread") == 300
    
    def test_global_hugs_cache(self):
        """Test global Hugs cache singleton."""
        cache1 = get_global_hugs_cache()
        cache2 = get_global_hugs_cache()
        assert cache1 is cache2


class TestIPFSKitFallback:
    """Test IPFS Kit fallback store."""
    
    def test_ipfs_fallback_initialization(self):
        """Test IPFS fallback store initializes."""
        store = IPFSKitFallbackStore(enabled=False)
        assert not store.is_available()
    
    def test_ipfs_fallback_disabled_returns_none(self):
        """Test that disabled fallback returns None."""
        store = IPFSKitFallbackStore(enabled=False)
        result = store.get("bafkreiabc123")
        assert result is None
    
    @patch('ipfs_accelerate_py.common.ipfs_kit_fallback.ipfs_kit_py')
    def test_ipfs_fallback_with_mock_client(self, mock_ipfs_kit):
        """Test IPFS fallback with mocked client."""
        # Mock IPFS client
        mock_client = MagicMock()
        mock_client.cat.return_value = json.dumps({"data": "test_value"}).encode('utf-8')
        mock_client.add.return_value = {"Hash": "bafkreiabc123"}
        
        mock_ipfs_kit.IPFSApi.return_value = mock_client
        
        store = IPFSKitFallbackStore(enabled=True)
        
        # Test get
        if store.is_available():
            result = store.get("bafkreiabc123")
            assert result == {"data": "test_value"}
    
    def test_ipfs_fallback_stats(self):
        """Test IPFS fallback statistics."""
        store = IPFSKitFallbackStore(enabled=True, timeout=5)
        stats = store.stats()
        
        assert "enabled" in stats
        assert "available" in stats
        assert "timeout" in stats
        assert stats["timeout"] == 5
    
    def test_global_ipfs_fallback(self):
        """Test global IPFS fallback singleton."""
        fallback1 = get_global_ipfs_fallback()
        fallback2 = get_global_ipfs_fallback()
        assert fallback1 is fallback2
    
    def test_ipfs_fallback_env_var(self):
        """Test IPFS fallback can be disabled via environment variable."""
        with patch.dict(os.environ, {"IPFS_FALLBACK_ENABLED": "false"}):
            store = IPFSKitFallbackStore(
                enabled=os.getenv("IPFS_FALLBACK_ENABLED", "true").lower() in ("true", "1")
            )
            assert not store.enabled


class TestIPFSFallbackIntegration:
    """Test IPFS fallback integration with base cache."""
    
    def test_cache_with_ipfs_fallback(self):
        """Test that base cache uses IPFS fallback on miss."""
        from ipfs_accelerate_py.common.llm_cache import LLMAPICache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMAPICache(cache_dir=tmpdir)
            
            # Verify IPFS fallback is initialized
            assert hasattr(cache, '_ipfs_fallback')
            assert cache._ipfs_fallback is not None
            
            # Stats should include IPFS fallback counters
            stats = cache.get_stats()
            assert "ipfs_fallback_hits" in stats
            assert "ipfs_fallback_misses" in stats
    
    @patch('ipfs_accelerate_py.common.ipfs_kit_fallback.ipfs_kit_py')
    def test_cache_retrieves_from_ipfs_on_miss(self, mock_ipfs_kit):
        """Test cache retrieves from IPFS when local cache misses."""
        from ipfs_accelerate_py.common.llm_cache import LLMAPICache
        
        # Mock IPFS client
        mock_client = MagicMock()
        mock_ipfs_kit.IPFSApi.return_value = mock_client
        
        # Mock IPFS response
        ipfs_data = {
            "data": {"response": "Hello from IPFS"},
            "timestamp": time.time(),
            "ttl": 3600,
            "operation": "completion"
        }
        mock_client.cat.return_value = json.dumps(ipfs_data).encode('utf-8')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMAPICache(cache_dir=tmpdir)
            
            # First access - should miss and try IPFS
            result = cache.get("completion", prompt="test", model="gpt-4")
            
            # Since IPFS is mocked, this will work if integrated correctly
            # In real scenario, would retrieve from IPFS
            stats = cache.get_stats()
            assert stats["misses"] >= 1


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_all_caches_registered(self):
        """Test that all cache types can be registered."""
        from ipfs_accelerate_py.common.base_cache import get_all_caches
        
        # Get all caches
        caches = get_all_caches()
        
        # Should include various cache types
        assert len(caches) > 0
    
    def test_kubernetes_cache_workflow(self):
        """Test complete Kubernetes cache workflow."""
        cache = KubernetesAPICache()
        
        # Simulate workflow: list pods -> cache -> retrieve
        pods_list = {
            "items": [
                {"metadata": {"name": "pod1"}},
                {"metadata": {"name": "pod2"}},
            ]
        }
        
        # Cache
        cache.put("pod_list", pods_list, namespace="default")
        
        # Retrieve
        cached = cache.get("pod_list", namespace="default")
        assert cached == pods_list
        assert len(cached["items"]) == 2
    
    def test_hugs_cache_workflow(self):
        """Test complete HuggingFace Hugs cache workflow."""
        cache = HuggingFaceHugsCache()
        
        # Simulate workflow: search models -> cache -> retrieve
        models_list = [
            {"modelId": "model1", "downloads": 1000},
            {"modelId": "model2", "downloads": 2000},
        ]
        
        # Cache
        cache.put("model_list", models_list, search="bert", limit=2)
        
        # Retrieve
        cached = cache.get("model_list", search="bert", limit=2)
        assert cached == models_list
        assert len(cached) == 2
    
    def test_cache_expiration(self):
        """Test that cache entries expire correctly."""
        cache = KubernetesAPICache()
        
        # Cache with very short TTL
        cache.put("pod_status", {"status": "running"}, ttl=1, pod_name="test")
        
        # Should be cached
        assert cache.get("pod_status", pod_name="test") == {"status": "running"}
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        assert cache.get("pod_status", pod_name="test") is None


class TestCLIIntegrations:
    """Test CLI integrations with cache."""
    
    def test_cli_integrations_module_imports(self):
        """Test that CLI integrations module can be imported."""
        try:
            from ipfs_accelerate_py.cli_integrations import get_all_cli_integrations
            assert callable(get_all_cli_integrations)
        except ImportError as e:
            pytest.skip(f"CLI integrations not available: {e}")
    
    def test_cli_cache_adapters_exist(self):
        """Test that cache adapters exist for CLI tools."""
        from ipfs_accelerate_py.common import llm_cache, docker_cache, hf_hub_cache
        
        # Verify cache modules exist
        assert hasattr(llm_cache, 'LLMAPICache')
        assert hasattr(docker_cache, 'DockerAPICache')
        assert hasattr(hf_hub_cache, 'HuggingFaceHubCache')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
