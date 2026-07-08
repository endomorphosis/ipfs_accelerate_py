#!/usr/bin/env python3
"""
Example: Using the Embeddings Router

This example demonstrates the new embeddings_router functionality that integrates
multiple embeddings providers (OpenRouter, Gemini CLI, HuggingFace, Backend Manager)
with the existing endpoint multiplexing capabilities.

The router automatically selects the best available provider based on:
- Environment configuration
- Available CLI tools/SDKs
- Backend manager endpoints
- Fallback to local HuggingFace models
"""

import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
except ImportError as e:
    logger.error(f"Failed to import embeddings_router: {e}")
    logger.error("Make sure ipfs_accelerate_py is properly installed")
    exit(1)


def example_basic_usage():
    """Example 1: Basic embeddings generation with automatic provider selection."""
    print("\n=== Example 1: Basic Embeddings Generation ===")
    
    texts = ["Hello world", "IPFS accelerates machine learning"]
    
    try:
        embeddings = embed_texts(texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
        print(f"First embedding (first 5 dims): {embeddings[0][:5] if embeddings else 'N/A'}")
    except Exception as e:
        print(f"Error: {e}")


def example_single_text():
    """Example 2: Generate embedding for a single text."""
    print("\n=== Example 2: Single Text Embedding ===")
    
    text = "Distributed machine learning with IPFS"
    
    try:
        embedding = embed_text(text)
        print(f"Text: {text}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"Embedding (first 10 dims): {embedding[:10]}")
    except Exception as e:
        print(f"Error: {e}")


def example_specific_provider():
    """Example 3: Using a specific provider."""
    print("\n=== Example 3: Specific Provider ===")
    
    # Try OpenRouter if API key is configured
    if os.getenv("OPENROUTER_API_KEY"):
        texts = ["Machine learning", "Deep learning", "Neural networks"]
        try:
            embeddings = embed_texts(
                texts,
                provider="openrouter",
                model_name="text-embedding-3-small"
            )
            print(f"Provider: openrouter")
            print(f"Generated {len(embeddings)} embeddings")
            print(f"Embedding dimensions: {len(embeddings[0])}")
        except Exception as e:
            print(f"OpenRouter error: {e}")
    else:
        print("OPENROUTER_API_KEY not set, skipping OpenRouter example")


def example_with_caching():
    """Example 4: Demonstrating response caching."""
    print("\n=== Example 4: Response Caching ===")
    
    text = "Cached embedding example"
    
    # Enable response cache
    os.environ["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "1"
    
    try:
        import time
        
        # First call (cache miss)
        start = time.time()
        embedding1 = embed_text(text)
        time1 = time.time() - start
        
        # Second call (cache hit)
        start = time.time()
        embedding2 = embed_text(text)
        time2 = time.time() - start
        
        print(f"First call: {time1:.3f}s")
        print(f"Second call (cached): {time2:.3f}s")
        if time1 > 0 and time2 > 0:
            print(f"Speedup: {time1/time2:.1f}x")
        print(f"Embeddings match: {np.allclose(embedding1, embedding2) if embedding1 and embedding2 else 'N/A'}")
        
    except Exception as e:
        print(f"Caching example error: {e}")


def example_custom_provider():
    """Example 5: Registering a custom provider."""
    print("\n=== Example 5: Custom Provider ===")
    
    # Define a custom provider
    class MockEmbeddingsProvider:
        """A simple provider that returns random embeddings."""
        def embed_texts(self, texts, *, model_name=None, device=None, **kwargs):
            import numpy as np
            # Generate random 384-dimensional embeddings
            embeddings = [np.random.randn(384).tolist() for _ in texts]
            return embeddings
    
    # Register the provider
    register_embeddings_provider("mock", lambda: MockEmbeddingsProvider())
    
    # Use the custom provider
    try:
        texts = ["Text 1", "Text 2"]
        embeddings = embed_texts(
            texts,
            provider="mock"
        )
        print(f"Custom provider generated {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {len(embeddings[0])}")
    except Exception as e:
        print(f"Error: {e}")


def example_dependency_injection():
    """Example 6: Using dependency injection for shared resources."""
    print("\n=== Example 6: Dependency Injection ===")
    
    # Create a RouterDeps instance to share resources
    deps = RouterDeps()
    
    # You can inject pre-configured components
    # deps.backend_manager = my_backend_manager
    # deps.remote_cache = my_remote_cache
    
    try:
        # Use the deps instance for multiple requests
        texts1 = ["First batch", "of texts"]
        embeddings1 = embed_texts(texts1, deps=deps)
        
        texts2 = ["Second batch", "of texts"]
        embeddings2 = embed_texts(texts2, deps=deps)
        
        print(f"Both requests used the same RouterDeps instance")
        print(f"Cached items: {len(deps.router_cache)}")
        print(f"Generated {len(embeddings1)} + {len(embeddings2)} embeddings")
        
    except Exception as e:
        print(f"Error: {e}")


def example_backend_manager_integration():
    """Example 7: Using backend manager for distributed inference."""
    print("\n=== Example 7: Backend Manager Integration ===")
    
    # Enable backend manager provider
    os.environ["IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER"] = "1"
    
    try:
        # Get the backend manager provider
        provider = get_embeddings_provider("backend_manager")
        
        if provider:
            texts = ["Test distributed", "embeddings inference"]
            embeddings = embed_texts(
                texts,
                provider="backend_manager"
            )
            print(f"Backend manager result: {len(embeddings)} embeddings")
            print(f"Embedding dimensions: {len(embeddings[0])}")
        else:
            print("Backend manager not available")
            
    except Exception as e:
        print(f"Backend manager error: {e}")


def example_list_available_providers():
    """Example 8: List all available providers."""
    print("\n=== Example 8: Available Providers ===")
    
    providers_to_check = [
        "openrouter",
        "gemini_cli",
        "huggingface",
        "backend_manager"
    ]
    
    print("Checking available providers:")
    for provider_name in providers_to_check:
        try:
            provider = get_embeddings_provider(provider_name)
            status = "✓ Available" if provider else "✗ Not available"
        except Exception as e:
            status = f"✗ Error: {str(e)[:50]}"
        
        print(f"  {provider_name:20} {status}")


def example_similarity_search():
    """Example 9: Using embeddings for similarity search."""
    print("\n=== Example 9: Similarity Search ===")
    
    # Generate embeddings for a corpus
    corpus = [
        "Machine learning with IPFS",
        "Distributed computing networks",
        "Neural network architectures",
        "Blockchain technology basics"
    ]
    
    query = "AI and distributed systems"
    
    try:
        # Generate embeddings
        corpus_embeddings = embed_texts(corpus)
        query_embedding = embed_text(query)
        
        # Calculate cosine similarity
        similarities = []
        for emb in corpus_embeddings:
            dot_product = sum(a * b for a, b in zip(query_embedding, emb))
            norm_q = sum(x ** 2 for x in query_embedding) ** 0.5
            norm_e = sum(x ** 2 for x in emb) ** 0.5
            similarity = dot_product / (norm_q * norm_e) if norm_q and norm_e else 0
            similarities.append(similarity)
        
        # Find most similar
        max_idx = similarities.index(max(similarities))
        print(f"Query: {query}")
        print(f"Most similar: {corpus[max_idx]}")
        print(f"Similarity: {similarities[max_idx]:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Embeddings Router Examples")
    print("=" * 60)
    
    if not embeddings_router_available:
        print("ERROR: embeddings_router is not available")
        print("Make sure the module is properly installed")
        return
    
    print(f"\nEmbeddings Router is available!")
    
    # Run examples
    example_list_available_providers()
    example_basic_usage()
    example_single_text()
    example_specific_provider()
    example_with_caching()
    example_custom_provider()
    example_dependency_injection()
    example_backend_manager_integration()
    example_similarity_search()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
