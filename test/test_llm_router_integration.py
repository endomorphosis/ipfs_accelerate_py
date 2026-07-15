#!/usr/bin/env python3
"""
Test script for LLM Router integration

This script tests the basic functionality of the llm_router without
requiring actual API keys or making external requests.
"""

import sys
import os

# Only manipulate path when run directly (not via pytest)
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all imports work correctly."""
    from ipfs_accelerate_py import (
        generate_text,
        get_llm_provider,
        register_llm_provider,
        RouterDeps,
        get_default_router_deps,
        llm_router_available
    )
    from ipfs_accelerate_py.llm_router import LLMProvider
    assert llm_router_available


def test_router_deps():
    """Test RouterDeps functionality."""
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


def test_provider_registry():
    """Test provider registration."""
    from ipfs_accelerate_py.llm_router import register_llm_provider, get_llm_provider
    
    # Create a test provider
    class TestProvider:
        def generate(self, prompt, *, model_name=None, **kwargs):
            return f"Test response to: {prompt}"
    
    # Register the provider
    register_llm_provider("test_provider", lambda: TestProvider())
    
    # Get the provider
    provider = get_llm_provider("test_provider")
    assert provider is not None, "Provider should be registered"
    
    # Test generation
    result = provider.generate("test prompt")
    assert "Test response" in result, "Provider should generate response"


def test_provider_discovery():
    """Test built-in provider discovery."""
    from ipfs_accelerate_py.llm_router import _builtin_provider_by_name
    from ipfs_accelerate_py.router_deps import get_default_router_deps
    
    deps = get_default_router_deps()
    
    providers_to_check = [
        'codex_cli',
        'copilot_cli',
        'gemini_cli',
        'claude_code'
    ]
    
    # At least check that the function doesn't crash
    for name in providers_to_check:
        try:
            provider = _builtin_provider_by_name(name, deps=deps)
            # Provider may or may not be available depending on environment
        except Exception:
            pass  # Expected if dependencies not installed


def test_caching():
    """Test response caching."""
    from ipfs_accelerate_py.llm_router import _response_cache_key
    
    # Test cache key generation
    key1 = _response_cache_key(
        provider="test",
        model_name="gpt-4",
        prompt="test prompt",
        kwargs={"temperature": 0.7}
    )
    
    # Same parameters should generate same key
    key2 = _response_cache_key(
        provider="test",
        model_name="gpt-4",
        prompt="test prompt",
        kwargs={"temperature": 0.7}
    )
    
    assert key1 == key2, "Cache keys should match for same parameters"
    
    # Different parameters should generate different key
    key3 = _response_cache_key(
        provider="test",
        model_name="gpt-4",
        prompt="different prompt",
        kwargs={"temperature": 0.7}
    )
    
    assert key1 != key3, "Cache keys should differ for different parameters"


def test_generate_text_with_custom_provider():
    """Test generate_text with a custom provider."""
    from ipfs_accelerate_py import generate_text, register_llm_provider
    
    # Register a mock provider
    class MockProvider:
        def generate(self, prompt, *, model_name=None, **kwargs):
            return f"Mock response: {prompt[:20]}..."
    
    register_llm_provider("mock", lambda: MockProvider())
    
    # Test generation
    result = generate_text("Test prompt for mock provider", provider="mock")
    assert "Mock response" in result, "Should get mock response"


def test_xai_backend_import():
    """Test that the xAI backend can be imported."""
    from ipfs_accelerate_py.api_backends.xai import xai, ALL_MODELS, CHAT_MODELS
    assert "grok-3" in CHAT_MODELS, "grok-3 should be in CHAT_MODELS"
    assert "grok-2-1212" in CHAT_MODELS, "grok-2-1212 should be in CHAT_MODELS"
    assert len(ALL_MODELS) >= len(CHAT_MODELS)


def test_xai_backend_init_no_key():
    """Test that xAI client initialises without error even when no key is set."""
    import os
    from ipfs_accelerate_py.api_backends.xai import xai as xai_cls

    # Temporarily unset API keys so we can test key-less init.
    saved = {k: os.environ.pop(k, None) for k in ("XAI_API_KEY", "ipfs_accelerate_py_XAI_API_KEY")}
    try:
        client = xai_cls(resources={}, metadata={})
        assert client.api_key is None
        assert client.base_url == "https://api.x.ai/v1"
        assert client.default_model == "grok-3"
        assert client.list_models()  # should return a non-empty list
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_xai_backend_init_with_metadata():
    """Test xAI client initialisation with metadata overrides."""
    from ipfs_accelerate_py.api_backends.xai import xai as xai_cls

    client = xai_cls(
        resources={},
        metadata={
            "api_key": "test-key-123",
            "model": "grok-2-1212",
            "base_url": "https://custom.api.example/v1",
            "max_retries": 2,
            "timeout": 30.0,
        },
    )
    assert client.api_key == "test-key-123"
    assert client.default_model == "grok-2-1212"
    assert client.base_url == "https://custom.api.example/v1"
    assert client.max_retries == 2
    assert client.timeout == 30.0


def test_xai_llm_router_provider_names():
    """Verify xAI provider is reachable under all expected aliases."""
    from ipfs_accelerate_py.llm_router import _get_xai_provider
    import os

    saved = {k: os.environ.pop(k, None) for k in ("XAI_API_KEY", "ipfs_accelerate_py_XAI_API_KEY")}
    try:
        # Without a key the factory should return None.
        provider = _get_xai_provider()
        assert provider is None, "xAI provider should be None without API key"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    # With a key the factory should return a provider.
    os.environ["XAI_API_KEY"] = "dummy-key"
    try:
        provider = _get_xai_provider()
        assert provider is not None, "xAI provider should be available with API key"
        assert hasattr(provider, "generate"), "xAI provider must have generate method"
        assert hasattr(provider, "chat_completions"), "xAI provider must have chat_completions method"
    finally:
        os.environ.pop("XAI_API_KEY", None)


def test_xai_builtin_provider_by_name():
    """Verify _builtin_provider_by_name resolves xAI under all aliases."""
    from ipfs_accelerate_py.llm_router import _builtin_provider_by_name
    import os

    os.environ["XAI_API_KEY"] = "dummy-key"
    try:
        for alias in ("xai", "grok", "xai_grok"):
            provider = _builtin_provider_by_name(alias)
            assert provider is not None, f"xAI provider missing for alias '{alias}'"
    finally:
        os.environ.pop("XAI_API_KEY", None)


def test_meta_ai_backend_import():
    """Test that the Meta AI backend can be imported."""
    from ipfs_accelerate_py.api_backends.meta_ai import meta_ai, ALL_MODELS, CHAT_MODELS
    assert "meta-llama/Llama-3.3-70B-Instruct" in CHAT_MODELS
    assert "meta-spark/Spark-1.1" in CHAT_MODELS, "Meta Spark 1.1 should be listed"
    assert len(ALL_MODELS) >= len(CHAT_MODELS)


def test_meta_ai_backend_init_no_key():
    """Test that Meta AI client initialises without error when no key is set."""
    import os
    from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_ai_cls

    saved = {k: os.environ.pop(k, None) for k in ("META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY")}
    try:
        client = meta_ai_cls(resources={}, metadata={})
        assert client.api_key is None
        assert client.base_url == "https://api.llamameta.net/v1"
        assert "Llama" in client.default_model
        assert client.list_models()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_meta_ai_backend_init_with_metadata():
    """Test Meta AI client initialisation with metadata overrides."""
    from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_ai_cls

    client = meta_ai_cls(
        resources={},
        metadata={
            "api_key": "meta-test-key",
            "model": "meta-spark/Spark-1.1",
            "base_url": "https://custom.meta.example/v1",
            "max_retries": 1,
            "timeout": 45.0,
        },
    )
    assert client.api_key == "meta-test-key"
    assert client.default_model == "meta-spark/Spark-1.1"
    assert client.base_url == "https://custom.meta.example/v1"
    assert client.max_retries == 1
    assert client.timeout == 45.0


def test_meta_ai_llm_router_provider():
    """Verify Meta AI provider behaviour with/without API key."""
    from ipfs_accelerate_py.llm_router import _get_meta_ai_provider
    import os

    saved = {k: os.environ.pop(k, None) for k in ("META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY")}
    try:
        provider = _get_meta_ai_provider()
        assert provider is None, "Meta AI provider should be None without API key"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    os.environ["META_AI_API_KEY"] = "dummy-meta-key"
    try:
        provider = _get_meta_ai_provider()
        assert provider is not None, "Meta AI provider should be available with API key"
        assert hasattr(provider, "generate")
        assert hasattr(provider, "chat_completions")
    finally:
        os.environ.pop("META_AI_API_KEY", None)


def test_meta_ai_builtin_provider_by_name():
    """Verify _builtin_provider_by_name resolves Meta AI under all aliases."""
    from ipfs_accelerate_py.llm_router import _builtin_provider_by_name
    import os

    os.environ["META_AI_API_KEY"] = "dummy-meta-key"
    try:
        for alias in ("meta_ai", "meta-ai", "meta_llama", "meta", "meta_spark", "spark"):
            provider = _builtin_provider_by_name(alias)
            assert provider is not None, f"Meta AI provider missing for alias '{alias}'"
    finally:
        os.environ.pop("META_AI_API_KEY", None)


def test_xai_and_meta_ai_in_auto_discovery():
    """Verify that xAI and Meta AI are tried during provider auto-discovery."""
    import os
    from ipfs_accelerate_py.llm_router import _builtin_provider_by_name

    # xAI should be discoverable when a key is present.
    os.environ["XAI_API_KEY"] = "dummy-key"
    try:
        provider = _builtin_provider_by_name("xai")
        assert provider is not None, "xAI provider should be found via auto-discovery"
    finally:
        os.environ.pop("XAI_API_KEY", None)

    # Meta AI should be discoverable when a key is present.
    os.environ["META_AI_API_KEY"] = "dummy-meta-key"
    try:
        provider = _builtin_provider_by_name("meta_ai")
        assert provider is not None, "Meta AI provider should be found via auto-discovery"
    finally:
        os.environ.pop("META_AI_API_KEY", None)


def test_cache_key_includes_xai_and_meta_ai():
    """Verify that changing xAI/Meta AI env vars causes a different cache key."""
    import os
    from ipfs_accelerate_py.llm_router import _provider_cache_key

    # Baseline: no keys set
    for k in ("XAI_API_KEY", "ipfs_accelerate_py_XAI_API_KEY", "META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY"):
        os.environ.pop(k, None)
    base_key = _provider_cache_key()

    # Setting xAI key should change the cache key
    os.environ["XAI_API_KEY"] = "xai-test"
    try:
        xai_key = _provider_cache_key()
        assert xai_key != base_key, "xAI API key should affect the provider cache key"
    finally:
        os.environ.pop("XAI_API_KEY", None)

    # Setting Meta AI key should change the cache key
    os.environ["META_AI_API_KEY"] = "meta-test"
    try:
        meta_key = _provider_cache_key()
        assert meta_key != base_key, "Meta AI API key should affect the provider cache key"
    finally:
        os.environ.pop("META_AI_API_KEY", None)


# Standalone execution for manual testing
if __name__ == "__main__":
    print("=" * 60)
    print("LLM Router Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("RouterDeps", test_router_deps),
        ("Provider Registry", test_provider_registry),
        ("Provider Discovery", test_provider_discovery),
        ("Caching", test_caching),
        ("Generate Text", test_generate_text_with_custom_provider),
        ("xAI Backend Import", test_xai_backend_import),
        ("xAI Backend Init (no key)", test_xai_backend_init_no_key),
        ("xAI Backend Init (metadata)", test_xai_backend_init_with_metadata),
        ("xAI Router Provider", test_xai_llm_router_provider_names),
        ("xAI Builtin Provider Names", test_xai_builtin_provider_by_name),
        ("Meta AI Backend Import", test_meta_ai_backend_import),
        ("Meta AI Backend Init (no key)", test_meta_ai_backend_init_no_key),
        ("Meta AI Backend Init (metadata)", test_meta_ai_backend_init_with_metadata),
        ("Meta AI Router Provider", test_meta_ai_llm_router_provider),
        ("Meta AI Builtin Provider Names", test_meta_ai_builtin_provider_by_name),
        ("Auto-discovery includes new providers", test_xai_and_meta_ai_in_auto_discovery),
        ("Cache key includes new providers", test_cache_key_includes_xai_and_meta_ai),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            test_func()
            print(f"✓ {name} tests passed")
            passed += 1
        except Exception as e:
            print(f"✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {failed} tests failed")
        sys.exit(1)
