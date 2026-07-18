#!/usr/bin/env python3
"""
Example: Using the Multimodal Router

This example demonstrates the multimodal_router functionality that integrates
multiple vision/language providers (OpenRouter, OpenAI GPT-4V, HuggingFace LLaVA)
with the existing endpoint multiplexing capabilities.

The router automatically selects the best available provider based on:
- Environment configuration
- Available API keys
- Backend manager endpoints
- Fallback to local HuggingFace models (LLaVA, InstructBLIP, etc.)
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ipfs_accelerate_py import (
        generate_multimodal,
        get_multimodal_provider,
        register_multimodal_provider,
        RouterDeps,
        get_default_router_deps,
        multimodal_router_available,
    )
    from ipfs_accelerate_py.multimodal_router import MultimodalProvider
except ImportError as e:
    logger.error(f"Failed to import multimodal_router: {e}")
    logger.error("Make sure ipfs_accelerate_py is properly installed")
    exit(1)


def example_basic_usage():
    """Example 1: Basic text-only generation (no image)."""
    print("\n=== Example 1: Text-Only Generation ===")

    prompt = "Describe what a photo of a sunset over the ocean would look like."

    try:
        result = generate_multimodal(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {result[:300]}")
    except Exception as e:
        print(f"Error: {e}")


def example_with_image_url():
    """Example 2: Multimodal inference with an image URL."""
    print("\n=== Example 2: Image URL ===")

    prompt = "What is in this image?"
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        print("Note: No API key configured. This example requires OPENAI_API_KEY or OPENROUTER_API_KEY.")
        print("Falling back to text-only mode for demonstration.")
        image_url = None

    try:
        result = generate_multimodal(prompt, image=image_url, max_tokens=128)
        print(f"Prompt: {prompt}")
        if image_url:
            print(f"Image: {image_url}")
        print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {e}")


def example_specific_provider():
    """Example 3: Using a specific provider."""
    print("\n=== Example 3: Specific Provider (OpenRouter) ===")

    if os.getenv("OPENROUTER_API_KEY"):
        prompt = "Describe this image in one sentence."
        try:
            result = generate_multimodal(
                prompt,
                provider="openrouter",
                model_name="openai/gpt-4o",
                max_tokens=64,
            )
            print(f"Provider: openrouter")
            print(f"Prompt: {prompt}")
            print(f"Response: {result}")
        except Exception as e:
            print(f"OpenRouter error: {e}")
    else:
        print("OPENROUTER_API_KEY not set, skipping OpenRouter example")


def example_with_image_bytes():
    """Example 4: Multimodal inference with raw image bytes."""
    print("\n=== Example 4: Image Bytes ===")

    # Create a minimal valid 1x1 PNG in memory
    try:
        import struct
        import zlib

        def make_minimal_png() -> bytes:
            def chunk(name: bytes, data: bytes) -> bytes:
                c = struct.pack(">I", len(data)) + name + data
                c += struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
                return c

            header = b"\x89PNG\r\n\x1a\n"
            ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            raw_data = b"\x00\xFF\x00\x00"  # filter=0, R=255, G=0, B=0
            idat = chunk(b"IDAT", zlib.compress(raw_data))
            iend = chunk(b"IEND", b"")
            return header + ihdr + idat + iend

        image_bytes = make_minimal_png()
        result = generate_multimodal(
            "What color is this image?",
            image=image_bytes,
            max_tokens=32,
        )
        print(f"Image: {len(image_bytes)} bytes (1x1 red PNG)")
        print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {e}")


def example_custom_provider():
    """Example 5: Registering a custom provider."""
    print("\n=== Example 5: Custom Provider ===")

    class EchoMultimodalProvider:
        """A simple provider that echoes the prompt with image info."""
        def generate(self, prompt: str, *, image=None, model_name=None, **kwargs):
            has_image = image is not None
            return f"Echo (image={'yes' if has_image else 'no'}): {prompt}"

    register_multimodal_provider("echo", lambda: EchoMultimodalProvider())

    try:
        result = generate_multimodal(
            "Hello, custom provider!",
            provider="echo",
        )
        print(f"Custom provider result: {result}")

        result_with_image = generate_multimodal(
            "What do you see?",
            image=b"fake_image_bytes",
            provider="echo",
        )
        print(f"With image: {result_with_image}")
    except Exception as e:
        print(f"Error: {e}")


def example_dependency_injection():
    """Example 6: Using dependency injection for shared resources."""
    print("\n=== Example 6: Dependency Injection ===")

    deps = RouterDeps()

    try:
        result1 = generate_multimodal("First multimodal request", deps=deps)
        result2 = generate_multimodal("Second multimodal request", deps=deps)
        print("Both requests used the same RouterDeps instance")
        print(f"Cached items: {len(deps.router_cache)}")
    except Exception as e:
        print(f"Error: {e}")


def example_list_available_providers():
    """Example 7: List all available providers."""
    print("\n=== Example 7: Available Providers ===")

    providers_to_check = [
        "openrouter",
        "openai",
        "huggingface",
        "backend_manager",
    ]

    print("Checking available providers:")
    for provider_name in providers_to_check:
        try:
            provider = get_multimodal_provider(provider_name)
            status = "✓ Available" if provider else "✗ Not available"
        except Exception as e:
            status = f"✗ Error: {str(e)[:60]}"
        print(f"  {provider_name:20} {status}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Multimodal Router Examples")
    print("=" * 60)

    if not multimodal_router_available:
        print("ERROR: multimodal_router is not available")
        print("Make sure the module is properly installed")
        return

    print("\nMultimodal Router is available!")

    example_list_available_providers()
    example_basic_usage()
    example_with_image_url()
    example_specific_provider()
    example_with_image_bytes()
    example_custom_provider()
    example_dependency_injection()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
