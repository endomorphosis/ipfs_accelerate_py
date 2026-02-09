#!/usr/bin/env python3
"""
Test script for LLM Router integration

This script tests the basic functionality of the llm_router without
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
            generate_text,
            get_llm_provider,
            register_llm_provider,
            RouterDeps,
            get_default_router_deps,
            llm_router_available
        )
        from ipfs_accelerate_py.llm_router import LLMProvider
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
        from ipfs_accelerate_py.llm_router import _builtin_provider_by_name
        from ipfs_accelerate_py.router_deps import get_default_router_deps
        
        deps = get_default_router_deps()
        
        providers_to_check = [
            'codex_cli',
            'copilot_cli',
            'gemini_cli',
            'claude_code'
        ]
        
        available_count = 0
        for name in providers_to_check:
            try:
                provider = _builtin_provider_by_name(name, deps)
                if provider:
                    available_count += 1
                    print(f"  ✓ {name} available")
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
        
        print("✓ Caching tests passed")
        return True
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_text_with_custom_provider():
    """Test generate_text with a custom provider."""
    print("\nTesting generate_text with custom provider...")
    try:
        from ipfs_accelerate_py import generate_text, register_llm_provider
        
        # Register a mock provider
        class MockProvider:
            def generate(self, prompt, *, model_name=None, **kwargs):
                return f"Mock response: {prompt[:20]}..."
        
        register_llm_provider("mock", lambda: MockProvider())
        
        # Test generation
        result = generate_text("Test prompt for mock provider", provider="mock")
        assert "Mock response" in result, "Should get mock response"
        
        print("✓ generate_text with custom provider passed")
        return True
    except Exception as e:
        print(f"✗ generate_text test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Router Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_router_deps,
        test_provider_registry,
        test_provider_discovery,
        test_caching,
        test_generate_text_with_custom_provider
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
