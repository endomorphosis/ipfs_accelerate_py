#!/usr/bin/env python3
"""
Test script for Embeddings Router integration

This script tests the basic functionality of the embeddings_router without
requiring actual API keys or making external requests.
"""

import sys
import os

# Add current directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from ipfs_accelerate_py import (
            embed_texts,
            embed_text,
            get_embeddings_provider,
            register_embeddings_provider,
            RouterDeps,
            get_default_router_deps,
            embeddings_router_available
        )
        from ipfs_accelerate_py.embeddings_router import EmbeddingsProvider
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_router_deps():
    """Test RouterDeps functionality."""
    print("\nTesting RouterDeps...")
    try:
        from ipfs_accelerate_py.router_deps import RouterDeps, get_default_router_deps
        
        # Test creating a deps instance
        deps = RouterDeps()
        
        # Test caching
        deps.set_cached("test_key", "test_value")
        cached = deps.get_cached("test_key")
        assert cached == "test_value", "Cache get/set failed"
        
        # Test get_or_create
        result = deps.get_or_create("new_key", lambda: "created_value")
        assert result == "created_value", "get_or_create failed"
        
        # Test default deps singleton
        default_deps1 = get_default_router_deps()
        default_deps2 = get_default_router_deps()
        assert default_deps1 is default_deps2, "Default deps should be singleton"
        
        print("✓ RouterDeps tests passed")
        return True
    except Exception as e:
        print(f"✗ RouterDeps test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_registry():
    """Test provider registration."""
    print("\nTesting provider registry...")
    try:
        from ipfs_accelerate_py.embeddings_router import register_embeddings_provider, get_embeddings_provider
        
        # Create a test provider
        class TestEmbeddingsProvider:
            def embed_texts(self, texts, *, model_name=None, device=None, **kwargs):
                # Return mock embeddings (5-dimensional for simplicity)
                return [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in texts]
        
        # Register the provider
        register_embeddings_provider("test_embeddings", lambda: TestEmbeddingsProvider())
        
        # Get the provider
        provider = get_embeddings_provider("test_embeddings")
        assert provider is not None, "Provider should be registered"
        
        # Test embedding generation
        result = provider.embed_texts(["test text"])
        assert len(result) == 1, "Should generate one embedding"
        assert len(result[0]) == 5, "Embedding should have 5 dimensions"
        assert result[0] == [1.0, 2.0, 3.0, 4.0, 5.0], "Embedding values should match"
        
        print("✓ Provider registry tests passed")
        return True
    except Exception as e:
        print(f"✗ Provider registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_discovery():
    """Test built-in provider discovery."""
    print("\nTesting provider discovery...")
    try:
        from ipfs_accelerate_py.embeddings_router import _builtin_provider_by_name
        from ipfs_accelerate_py.router_deps import get_default_router_deps
        
        deps = get_default_router_deps()
        
        providers_to_check = [
            'openrouter',
            'gemini_cli',
            'huggingface'
        ]
        
        available_count = 0
        for name in providers_to_check:
            try:
                provider = _builtin_provider_by_name(name, deps=deps)
                if provider:
                    available_count += 1
                    print(f"  ✓ {name} available")
                else:
                    print(f"  ✗ {name} not configured")
            except Exception as e:
                print(f"  ✗ {name} error: {str(e)[:50]}")
        
        print(f"✓ Provider discovery completed ({available_count}/{len(providers_to_check)} available)")
        return True
    except Exception as e:
        print(f"✗ Provider discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_caching():
    """Test response caching."""
    print("\nTesting response caching...")
    try:
        from ipfs_accelerate_py.embeddings_router import _response_cache_key
        
        # Test cache key generation
        key1 = _response_cache_key(
            provider="test",
            model_name="test-model",
            device="cpu",
            text="test text",
            kwargs={"param": "value"}
        )
        
        # Same parameters should generate same key
        key2 = _response_cache_key(
            provider="test",
            model_name="test-model",
            device="cpu",
            text="test text",
            kwargs={"param": "value"}
        )
        
        assert key1 == key2, "Cache keys should match for same parameters"
        
        # Different parameters should generate different key
        key3 = _response_cache_key(
            provider="test",
            model_name="test-model",
            device="cpu",
            text="different text",
            kwargs={"param": "value"}
        )
        
        assert key1 != key3, "Cache keys should differ for different parameters"
        
        print("✓ Caching tests passed")
        return True
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embed_texts_with_custom_provider():
    """Test embed_texts with a custom provider."""
    print("\nTesting embed_texts with custom provider...")
    try:
        from ipfs_accelerate_py import embed_texts, register_embeddings_provider
        
        # Register a mock provider
        class MockProvider:
            def embed_texts(self, texts, *, model_name=None, device=None, **kwargs):
                # Return fixed embeddings for testing
                return [[float(i) for i in range(10)] for _ in texts]
        
        register_embeddings_provider("mock_embed", lambda: MockProvider())
        
        # Test embedding generation
        texts = ["Test text 1", "Test text 2"]
        result = embed_texts(texts, provider="mock_embed")
        
        assert len(result) == 2, "Should generate 2 embeddings"
        assert len(result[0]) == 10, "Each embedding should have 10 dimensions"
        assert result[0] == [float(i) for i in range(10)], "Embedding values should match"
        
        print("✓ embed_texts with custom provider passed")
        return True
    except Exception as e:
        print(f"✗ embed_texts test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embed_text_single():
    """Test embed_text for single text."""
    print("\nTesting embed_text for single text...")
    try:
        from ipfs_accelerate_py import embed_text, register_embeddings_provider
        
        # Register a mock provider
        class MockProvider:
            def embed_texts(self, texts, *, model_name=None, device=None, **kwargs):
                # Return fixed embeddings
                return [[1.0, 2.0, 3.0] for _ in texts]
        
        register_embeddings_provider("mock_single", lambda: MockProvider())
        
        # Test single text embedding
        text = "Single test text"
        result = embed_text(text, provider="mock_single")
        
        assert len(result) == 3, "Embedding should have 3 dimensions"
        assert result == [1.0, 2.0, 3.0], "Embedding values should match"
        
        print("✓ embed_text for single text passed")
        return True
    except Exception as e:
        print(f"✗ embed_text test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Embeddings Router Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_router_deps,
        test_provider_registry,
        test_provider_discovery,
        test_caching,
        test_embed_texts_with_custom_provider,
        test_embed_text_single
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
