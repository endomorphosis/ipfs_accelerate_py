#!/usr/bin/env python3
"""
Example: Using the LLM Router

This example demonstrates the new llm_router functionality that integrates
multiple LLM providers (OpenRouter, Codex CLI, Copilot CLI/SDK, Gemini, Claude)
with the existing endpoint multiplexing capabilities.

The router automatically selects the best available provider based on:
- Environment configuration
- Available CLI tools
- Backend manager endpoints
- Fallback to local HuggingFace models
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
except ImportError as e:
    logger.error(f"Failed to import llm_router: {e}")
    logger.error("Make sure ipfs_accelerate_py is properly installed")
    exit(1)


def example_basic_usage():
    """Example 1: Basic text generation with automatic provider selection."""
    print("\n=== Example 1: Basic Text Generation ===")
    
    prompt = "Explain what IPFS is in one sentence."
    
    try:
        result = generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {e}")


def example_specific_provider():
    """Example 2: Using a specific provider."""
    print("\n=== Example 2: Specific Provider ===")
    
    # Try OpenRouter if API key is configured
    if os.getenv("OPENROUTER_API_KEY"):
        prompt = "Write a Python function to calculate fibonacci numbers."
        try:
            result = generate_text(
                prompt,
                provider="openrouter",
                model_name="openai/gpt-4o-mini",
                max_tokens=256,
                temperature=0.7
            )
            print(f"Provider: openrouter")
            print(f"Prompt: {prompt}")
            print(f"Response: {result[:200]}...")
        except Exception as e:
            print(f"OpenRouter error: {e}")
    else:
        print("OPENROUTER_API_KEY not set, skipping OpenRouter example")


def example_with_caching():
    """Example 3: Demonstrating response caching."""
    print("\n=== Example 3: Response Caching ===")
    
    prompt = "What is 2 + 2?"
    
    # Enable response cache
    os.environ["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "1"
    
    try:
        import time
        
        # First call (cache miss)
        start = time.time()
        result1 = generate_text(prompt, provider="local_hf")
        time1 = time.time() - start
        
        # Second call (cache hit)
        start = time.time()
        result2 = generate_text(prompt, provider="local_hf")
        time2 = time.time() - start
        
        print(f"First call: {time1:.3f}s")
        print(f"Second call (cached): {time2:.3f}s")
        print(f"Speedup: {time1/time2:.1f}x")
        print(f"Results match: {result1 == result2}")
        
    except Exception as e:
        print(f"Caching example error: {e}")


def example_custom_provider():
    """Example 4: Registering a custom provider."""
    print("\n=== Example 4: Custom Provider ===")
    
    # Define a custom provider
    class EchoProvider:
        """A simple provider that echoes the prompt."""
        def generate(self, prompt: str, *, model_name=None, **kwargs):
            return f"Echo: {prompt}"
    
    # Register the provider
    register_llm_provider("echo", lambda: EchoProvider())
    
    # Use the custom provider
    try:
        result = generate_text(
            "Hello, custom provider!",
            provider="echo"
        )
        print(f"Custom provider result: {result}")
    except Exception as e:
        print(f"Error: {e}")


def example_dependency_injection():
    """Example 5: Using dependency injection for shared resources."""
    print("\n=== Example 5: Dependency Injection ===")
    
    # Create a RouterDeps instance to share resources
    deps = RouterDeps()
    
    # You can inject pre-configured components
    # deps.backend_manager = my_backend_manager
    # deps.remote_cache = my_remote_cache
    
    try:
        # Use the deps instance for multiple requests
        prompt1 = "First request"
        result1 = generate_text(prompt1, deps=deps)
        
        prompt2 = "Second request"
        result2 = generate_text(prompt2, deps=deps)
        
        print(f"Both requests used the same RouterDeps instance")
        print(f"Cached items: {len(deps.router_cache)}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_backend_manager_integration():
    """Example 6: Using backend manager for distributed inference."""
    print("\n=== Example 6: Backend Manager Integration ===")
    
    # Enable backend manager provider
    os.environ["IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER"] = "1"
    
    print("Note: Backend manager provider requires backends with callable 'instance' attribute.")
    print("This is a best-effort provider for future backend integration.")
    
    try:
        # Get the backend manager provider
        provider = get_llm_provider("backend_manager")
        
        if provider:
            print("✓ Backend manager provider is available")
            print("  (Actual execution depends on backend implementation)")
        else:
            print("✗ Backend manager not available")
            
    except Exception as e:
        print(f"Backend manager error: {e}")


def example_list_available_providers():
    """Example 7: List all available providers."""
    print("\n=== Example 7: Available Providers ===")
    
    providers_to_check = [
        "openrouter",
        "codex_cli",
        "copilot_cli",
        "copilot_sdk",
        "gemini_cli",
        "claude_code",
        "backend_manager",
        "local_hf"
    ]
    
    print("Checking available providers:")
    for provider_name in providers_to_check:
        try:
            provider = get_llm_provider(provider_name)
            status = "✓ Available" if provider else "✗ Not available"
        except Exception as e:
            status = f"✗ Error: {str(e)[:50]}"
        
        print(f"  {provider_name:20} {status}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LLM Router Examples")
    print("=" * 60)
    
    if not llm_router_available:
        print("ERROR: llm_router is not available")
        print("Make sure the module is properly installed")
        return
    
    print(f"\nLLM Router is available!")
    
    # Run examples
    example_list_available_providers()
    example_basic_usage()
    example_specific_provider()
    example_with_caching()
    example_custom_provider()
    example_dependency_injection()
    example_backend_manager_integration()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
