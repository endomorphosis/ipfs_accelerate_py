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
        ("Generate Text", test_generate_text_with_custom_provider)
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
