# Integrating Semantic Similarity Caching with the Gemini API

This guide describes how to integrate the semantic similarity caching implementation with the existing Gemini API client in the IPFS Accelerate Python Framework.

## Integration Steps

1. Import the required components
2. Create a wrapper for the existing GeminiClient
3. Hook into the cache for API calls
4. Integrate with the embedding functionality
5. Update configurations and metrics

## Step 1: Import Components

First, add the necessary imports to the Gemini client module:

```python
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import OrderedDict
import threading
import hashlib
import logging
import time
import os

# Import the semantic cache implementation
from ..utils.semantic_cache import SemanticCache
```

## Step 2: Add Embedding Functionality

Add embedding functionality to the GeminiClient class:

```python
class GeminiClient:
    # ... existing code ...
    
    def create_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Create embeddings for the given text using Google's embedding model.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        # Use Gemini's embedding API
        try:
            model = self.embedding_model or "models/embedding-001"
            
            # Handle both single texts and batches
            if isinstance(text, list):
                result = self.embeddings_client.batch_embed_content(
                    model=model,
                    content=text
                )
                return np.array([item.embedding for item in result.embeddings])
            else:
                result = self.embeddings_client.embed_content(
                    model=model,
                    content=text
                )
                return np.array(result.embedding)
                
        except Exception as e:
            self.logger.warning(f"Error creating embeddings: {e}")
            # Fallback to hash-based pseudo-embeddings if the API fails
            return self._create_fallback_embedding(text)
    
    def _create_fallback_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """Create fallback embedding when the API is unavailable."""
        if isinstance(text, list):
            return np.array([self._text_to_pseudo_embedding(t) for t in text])
        return self._text_to_pseudo_embedding(text)
    
    def _text_to_pseudo_embedding(self, text: str) -> np.ndarray:
        """Convert text to a pseudo-embedding using hash."""
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Create a pseudo-embedding from the hash (128 dimensions)
        pseudo_embedding = np.array(
            [(hash_val >> (i * 8)) & 0xFF for i in range(16)], 
            dtype=np.float32
        )
        # Normalize and repeat to get higher dimensionality
        pseudo_embedding = pseudo_embedding / np.linalg.norm(pseudo_embedding)
        return np.tile(pseudo_embedding, 8)  # 128 dimensions
```

## Step 3: Add Semantic Cache to GeminiClient

Update the GeminiClient class to include the semantic cache:

```python
class GeminiClient:
    # ... existing code ...
    
    def __init__(self, api_key=None, **kwargs):
        # ... existing initialization ...
        
        # Cache configuration
        self.use_semantic_cache = kwargs.get('use_semantic_cache', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.85)
        self.cache_size = kwargs.get('cache_size', 1000)
        self.cache_ttl = kwargs.get('cache_ttl', 3600)  # 1 hour
        
        # Initialize semantic cache
        if self.use_semantic_cache:
            self.semantic_cache = SemanticCache(
                embedding_model=self,  # Self as embedding model
                similarity_threshold=self.similarity_threshold,
                max_cache_size=self.cache_size,
                ttl=self.cache_ttl
            )
            self.logger.info(f"Initialized semantic cache with threshold {self.similarity_threshold}")
        else:
            self.semantic_cache = None
            
        # Cache statistics
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_similarity": 0.0,
        }
        self.stats_lock = threading.Lock()
```

## Step 4: Modify the generate_content method

Update the generate_content method to use the semantic cache:

```python
async def generate_content(self, prompt, temperature=0.7, max_tokens=None, **kwargs):
    """Generate content using Gemini API with semantic caching."""
    # Update request stats
    with self.stats_lock:
        self.cache_stats["total_requests"] += 1
    
    # Skip cache for non-deterministic generations or disabled cache
    if not self.use_semantic_cache or temperature > 0.0:
        return await self._generate_content_direct(prompt, temperature, max_tokens, **kwargs)
    
    # Convert prompt to string for caching
    if isinstance(prompt, list):
        cache_key = str(prompt)
    else:
        cache_key = prompt
        
    # Include important kwargs in the cache key
    cache_metadata = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        **{k: v for k, v in kwargs.items() if k in ['stream', 'n']}
    }
    
    # Try to get response from cache
    cached_response, similarity, _ = self.semantic_cache.get(cache_key, metadata=cache_metadata)
    
    # Update similarity stats
    with self.stats_lock:
        # Running average of similarity scores
        self.cache_stats["avg_similarity"] = (
            (self.cache_stats["avg_similarity"] * (self.cache_stats["total_requests"] - 1) + similarity) / 
            self.cache_stats["total_requests"]
        )
    
    if cached_response is not None:
        # Cache hit
        with self.stats_lock:
            self.cache_stats["cache_hits"] += 1
        self.logger.debug(f"Cache hit with similarity {similarity:.4f}")
        return cached_response
    
    # Cache miss - call the API
    with self.stats_lock:
        self.cache_stats["cache_misses"] += 1
    
    self.logger.debug(f"Cache miss (best similarity: {similarity:.4f})")
    response = await self._generate_content_direct(prompt, temperature, max_tokens, **kwargs)
    
    # Store in cache if it's a deterministic generation
    if temperature == 0.0:
        self.semantic_cache.put(cache_key, response, metadata=cache_metadata)
    
    return response

async def _generate_content_direct(self, prompt, temperature=0.7, max_tokens=None, **kwargs):
    """Direct API call without caching - original implementation."""
    # Existing implementation here
    # ...
```

## Step 5: Add Cache Control Methods

Add methods to control the cache:

```python
def get_cache_stats(self):
    """Get statistics about the cache usage."""
    with self.stats_lock:
        stats_copy = self.cache_stats.copy()
    
    # Add cache internal stats if available
    if self.semantic_cache:
        cache_stats = self.semantic_cache.get_stats()
        return {**stats_copy, **cache_stats}
    return stats_copy

def clear_cache(self):
    """Clear the semantic cache."""
    if self.semantic_cache:
        self.semantic_cache.clear()
        self.logger.info("Semantic cache cleared")

def set_cache_enabled(self, enabled):
    """Enable or disable the semantic cache."""
    self.use_semantic_cache = enabled
    self.logger.info(f"Semantic cache {'enabled' if enabled else 'disabled'}")
```

## Step 6: Add Cache Configuration to API

Update the API configuration to include cache settings:

```python
def create_gemini_client(api_key=None, **kwargs):
    """
    Create a Gemini client with the given API key and configuration.
    
    Args:
        api_key: Gemini API key (optional if using environment variable)
        **kwargs: Additional configuration options
            - use_semantic_cache: Whether to use semantic caching (default: True)
            - similarity_threshold: Threshold for semantic similarity (default: 0.85)
            - cache_size: Maximum cache size (default: 1000)
            - cache_ttl: Time-to-live for cache entries in seconds (default: 3600)
    
    Returns:
        GeminiClient instance
    """
    return GeminiClient(api_key, **kwargs)
```

## Performance Considerations

1. **Memory Usage**: The semantic cache stores embeddings which can consume significant memory for large caches
2. **Computation Overhead**: Computing embeddings adds some overhead to API calls
3. **API Costs**: Using the embedding API incurs additional API costs
4. **Thread Safety**: The implementation is thread-safe for concurrent usage

## Configuration Recommendations

Different use cases may require different cache configurations:

| Use Case | Similarity Threshold | Cache Size | TTL |
|----------|----------------------|------------|-----|
| High Accuracy | 0.90 - 0.95 | 500 | 1 hour |
| Balanced | 0.85 - 0.90 | 1000 | 6 hours |
| High Hit Rate | 0.80 - 0.85 | 2000 | 24 hours |
| Development | 0.75 - 0.80 | 100 | 1 hour |

## Implementation Status

This implementation enhances the Gemini API client with semantic caching capabilities, allowing for more efficient handling of similar queries while maintaining the existing API interface.