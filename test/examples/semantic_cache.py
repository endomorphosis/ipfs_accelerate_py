import os
import sys
import time
import torch
import numpy as np
import heapq
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from collections import OrderedDict
import threading
import hashlib
import logging
import asyncio

# Add parent directory to path
sys.path.insert()))))))))0, os.path.dirname()))))))))os.path.dirname()))))))))os.path.abspath()))))))))__file__))))
sys.path.insert()))))))))0, os.path.dirname()))))))))os.path.dirname()))))))))os.path.dirname()))))))))os.path.abspath()))))))))__file__)))))

logging.basicConfig()))))))))level=logging.INFO)
logger = logging.getLogger()))))))))__name__)

class SemanticCache:
    """
    Cache implementation that uses semantic similarity between queries to determine cache hits.
    
    Instead of exact string matching, this cache computes embeddings for queries and uses
    cosine similarity to find semantically similar cached entries.
    """
    
    def __init__()))))))))
    self,
    embedding_model: Optional[]],,Any] = None,
    similarity_threshold: float = 0.85,
    max_cache_size: int = 1000,
    ttl: int = 3600,  # Time-to-live in seconds
    use_lru: bool = True
    ):
        """
        Initialize the semantic cache.
        
        Args:
            embedding_model: Model used to generate embeddings for queries
            similarity_threshold: Minimum cosine similarity score to consider a cache hit
            max_cache_size: Maximum number of entries in the cache
            ttl: Time-to-live for cache entries in seconds
            use_lru: Whether to use LRU eviction policy
            """
            self.embedding_model = embedding_model
            self.similarity_threshold = similarity_threshold
            self.max_cache_size = max_cache_size
            self.ttl = ttl
            self.use_lru = use_lru
        
        # Main cache storage: {}}}cache_key: ()))))))))embedding, response, timestamp, metadata)}
            self.cache: Dict[]],,str, Tuple[]],,torch.Tensor, Any, float, Dict]] = OrderedDict())))))))))
            ,
        # Lock for thread safety
            self.lock = threading.RLock())))))))))
        
            logger.info()))))))))f"Initialized semantic cache with threshold {}}}similarity_threshold}")
    
    def _generate_embedding()))))))))self, query: str) -> torch.Tensor:
        """
        Generate an embedding for the given query.
        
        Args:
            query: Input query text
            
        Returns:
            Embedding vector for the query
            """
        if self.embedding_model is None:
            # Fallback to hash-based representation if no embedding model
            hash_val = int()))))))))hashlib.md5()))))))))query.encode())))))))))).hexdigest()))))))))), 16)
            # Create a pseudo-embedding from the hash
            pseudo_embedding = torch.tensor())))))))):
                []],,()))))))))hash_val >> ()))))))))8 * i)) & 0xFF for i in range()))))))))16)],:,
                dtype=torch.float32
                )
            # Normalize the pseudo-embedding
            return pseudo_embedding / torch.norm()))))))))pseudo_embedding, p=2)
        
        # Use the actual embedding model
        with torch.no_grad()))))))))):
            embedding = self.embedding_model.encode()))))))))query)
            if isinstance()))))))))embedding, np.ndarray):
                embedding = torch.tensor()))))))))embedding, dtype=torch.float32)
            # Normalize the embedding
            return embedding / torch.norm()))))))))embedding, p=2)
    
    def _compute_similarity()))))))))self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score between the embeddings
            """
            return torch.nn.functional.cosine_similarity()))))))))
            emb1.unsqueeze()))))))))0), emb2.unsqueeze()))))))))0)
            ).item())))))))))
    
            def _find_most_similar()))))))))self, query_embedding: torch.Tensor) -> Tuple[]],,Optional[]],,str], float]:,
            """
            Find the most similar cached entry::::: to the given query embedding.
        
        Args:
            query_embedding: Query embedding to compare against cache
            
        Returns:
            Tuple of ()))))))))cache_key, similarity_score) for the most similar entry:::::
                """
                most_similar_key = None
                highest_similarity = -1.0
        
        for key, ()))))))))cached_embedding, _, timestamp, _) in self.cache.items()))))))))):
            # Skip expired entries
            if time.time()))))))))) - timestamp > self.ttl:
            continue
                
            similarity = self._compute_similarity()))))))))query_embedding, cached_embedding)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_key = key
        
            return most_similar_key, highest_similarity
    
    def _clean_expired_entries()))))))))self) -> None:
        """Remove expired entries from the cache."""
        with self.lock:
            current_time = time.time())))))))))
            keys_to_remove = []],,
            key for key, ()))))))))_, _, timestamp, _) in self.cache.items())))))))))
            if current_time - timestamp > self.ttl
            ]
            :
            for key in keys_to_remove:
                del self.cache[]],,key]
                
                logger.debug()))))))))f"Cleaned {}}}len()))))))))keys_to_remove)} expired entries from cache")
    
    def _make_space_if_needed()))))))))self) -> None:
        """Remove entries if cache exceeds maximum size.""":
        if len()))))))))self.cache) < self.max_cache_size:
            return
            
        with self.lock:
            # If using LRU, the first item in OrderedDict is the least recently used
            if self.use_lru and isinstance()))))))))self.cache, OrderedDict) and self.cache:
                self.cache.popitem()))))))))last=False)
                logger.debug()))))))))"Removed least recently used cache entry:::::")
            else:
                # Otherwise remove random entry:::::
                if self.cache:
                    key = next()))))))))iter()))))))))self.cache.keys())))))))))))
                    del self.cache[]],,key]
                    logger.debug()))))))))"Removed random cache entry:::::")
    
    def get()))))))))self, query: str, metadata: Optional[]],,Dict] = None) -> Tuple[]],,Optional[]],,Any], float, Optional[]],,Dict]]:
        """
        Get a cached response for a query if a similar one exists.
        :
        Args:
            query: Query to look up in the cache
            metadata: Optional metadata for the query ()))))))))used for filtering)
            
        Returns:
            Tuple of ()))))))))cached_response, similarity_score, cache_metadata)
            """
        # Periodically clean expired entries
            if time.time()))))))))) % 10 < 0.1:  # Clean roughly every 10 seconds
            self._clean_expired_entries())))))))))
            
        # Generate embedding for the query
            query_embedding = self._generate_embedding()))))))))query)
        
        with self.lock:
            # Find the most similar cached entry:::::
            most_similar_key, similarity = self._find_most_similar()))))))))query_embedding)
            
            if most_similar_key is not None and similarity >= self.similarity_threshold:
                # Cache hit
                cached_embedding, response, timestamp, cached_metadata = self.cache[]],,most_similar_key]
                
                # Update position in OrderedDict if using LRU:
                if self.use_lru and isinstance()))))))))self.cache, OrderedDict):
                    self.cache.move_to_end()))))))))most_similar_key)
                
                    logger.debug()))))))))f"Cache hit with similarity {}}}similarity:.4f}")
                return response, similarity, cached_metadata
                
        # Cache miss
                logger.debug()))))))))f"Cache miss ()))))))))best similarity: {}}}similarity:.4f})")
            return None, similarity, None
    
    def put()))))))))self, query: str, response: Any, metadata: Optional[]],,Dict] = None) -> None:
        """
        Add a query-response pair to the cache.
        
        Args:
            query: Query string
            response: Response to cache
            metadata: Optional metadata to store with the cache entry:::::
                """
                self._make_space_if_needed())))))))))
        
                query_embedding = self._generate_embedding()))))))))query)
                current_time = time.time())))))))))
        
        with self.lock:
            # Generate a unique cache key
            cache_key = f"{}}}hash()))))))))query)}_{}}}hash()))))))))str()))))))))response))}"
            
            # Store the entry::::: in the cache
            self.cache[]],,cache_key] = ()))))))))
            query_embedding,
            response,
            current_time,
            metadata or {}}}}
            )
            
            # Move to end if using LRU: to mark as most recently used
            if self.use_lru and isinstance()))))))))self.cache, OrderedDict):
                self.cache.move_to_end()))))))))cache_key)
                
                logger.debug()))))))))f"Added new entry::::: to cache ()))))))))size: {}}}len()))))))))self.cache)})")
    
    def clear()))))))))self) -> None:
        """Clear all entries from the cache."""
        with self.lock:
            self.cache.clear())))))))))
            logger.info()))))))))"Cache cleared")
    
    def get_stats()))))))))self) -> Dict[]],,str, Any]:
        """Get statistics about the cache."""
        with self.lock:
            current_time = time.time())))))))))
            active_entries = sum()))))))))
            1 for _, _, timestamp, _ in self.cache.values())))))))))
            if current_time - timestamp <= self.ttl
            )
            
            return {}}}:
                "total_entries": len()))))))))self.cache),
                "active_entries": active_entries,
                "expired_entries": len()))))))))self.cache) - active_entries,
                "max_size": self.max_cache_size,
                "similarity_threshold": self.similarity_threshold,
                "ttl": self.ttl,
                }


class SemanticCacheGeminiClient:
    """
    A wrapper around the Gemini API client that adds semantic caching capabilities.
    """
    
    def __init__()))))))))
    self,
    base_client: Any,
    embedding_model: Optional[]],,Any] = None,
    similarity_threshold: float = 0.85,
    max_cache_size: int = 1000,
    ttl: int = 3600,
    cache_enabled: bool = True
    ):
        """
        Initialize the semantic cache wrapper for Gemini client.
        
        Args:
            base_client: The base Gemini client to wrap
            embedding_model: Model used to generate embeddings for queries
            similarity_threshold: Minimum similarity threshold for cache hits
            max_cache_size: Maximum cache size
            ttl: Time-to-live for cache entries
            cache_enabled: Whether caching is enabled
            """
            self.base_client = base_client
            self.cache_enabled = cache_enabled
        
        # Initialize semantic cache
            self.cache = SemanticCache()))))))))
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size,
            ttl=ttl
            )
        
        # Statistics
            self.stats = {}}}
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_similarity": 0.0,
            }
            self.stats_lock = threading.Lock())))))))))
    
            async def generate_content()))))))))self,
            prompt: Union[]],,str, List],
            temperature: float = 0.7,
            max_tokens: Optional[]],,int] = None,
                         **kwargs) -> Any:
                             """
                             Generate content with semantic caching.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the base client
            
        Returns:
            Generated content response
            """
        # Update request stats
        with self.stats_lock:
            self.stats[]],,"total_requests"] += 1
        
        # Skip cache for non-deterministic generations
        if not self.cache_enabled or temperature > 0.0:
            logger.debug()))))))))"Bypassing cache due to non-zero temperature or disabled cache")
            return await self.base_client.generate_content()))))))))prompt, temperature, max_tokens, **kwargs)
        
        # Convert prompt to string for caching
        if isinstance()))))))))prompt, list):
            cache_key = str()))))))))prompt)
        else:
            cache_key = prompt
            
        # Include important kwargs in the cache key
            cache_metadata = {}}}
            "temperature": temperature,
            "max_tokens": max_tokens,
            **{}}}k: v for k, v in kwargs.items()))))))))) if k in []],,'stream', 'n']}
            }
        
        # Try to get response from cache
            cached_response, similarity, _ = self.cache.get()))))))))cache_key, metadata=cache_metadata)
        
        # Update similarity stats:
        with self.stats_lock:
            # Running average of similarity scores
            self.stats[]],,"avg_similarity"] = ()))))))))
            ()))))))))self.stats[]],,"avg_similarity"] * ()))))))))self.stats[]],,"total_requests"] - 1) + similarity) /
            self.stats[]],,"total_requests"]
            )
        
        if cached_response is not None:
            # Cache hit
            with self.stats_lock:
                self.stats[]],,"cache_hits"] += 1
                logger.info()))))))))f"Cache hit with similarity {}}}similarity:.4f}")
            return cached_response
        
        # Cache miss - call the base client
        with self.stats_lock:
            self.stats[]],,"cache_misses"] += 1
        
            logger.info()))))))))f"Cache miss ()))))))))best similarity: {}}}similarity:.4f})")
            response = await self.base_client.generate_content()))))))))prompt, temperature, max_tokens, **kwargs)
        
        # Store in cache if it's a deterministic generation:
        if temperature == 0.0:
            self.cache.put()))))))))cache_key, response, metadata=cache_metadata)
        
            return response
    
            async def generate_content_stream()))))))))self,
            prompt: Union[]],,str, List],
            temperature: float = 0.7,
            max_tokens: Optional[]],,int] = None,
                                **kwargs) -> Any:
                                    """
                                    Generate streaming content ()))))))))bypasses cache).
        
        Args:
            prompt: The prompt to send to the model
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the base client
            
        Returns:
            Streaming response
            """
        # Streaming always bypasses cache
        with self.stats_lock:
            self.stats[]],,"total_requests"] += 1
            self.stats[]],,"cache_misses"] += 1
            
            return await self.base_client.generate_content_stream()))))))))prompt, temperature, max_tokens, **kwargs)
    
    # Pass through all other methods to the base client
    def __getattr__()))))))))self, name):
            return getattr()))))))))self.base_client, name)
    
    def get_cache_stats()))))))))self) -> Dict[]],,str, Any]:
        """Get statistics about the cache usage."""
        with self.stats_lock:
            stats_copy = self.stats.copy())))))))))
        
        # Add cache internal stats
            cache_stats = self.cache.get_stats())))))))))
        return {}}}**stats_copy, **cache_stats}
    
    def clear_cache()))))))))self) -> None:
        """Clear the cache."""
        self.cache.clear())))))))))
        
    def set_cache_enabled()))))))))self, enabled: bool) -> None:
        """Enable or disable the cache."""
        self.cache_enabled = enabled
        logger.info()))))))))f"Cache {}}}'enabled' if enabled else 'disabled'}")


# Example usage:
async def example_usage()))))))))):
    """
    Example of how to use the semantic cache with the Gemini API.
    """
    # Import your actual Gemini client here
    try::::::
        from api_backends.gemini import GeminiClient
    except ImportError:
        # Mock client for demonstration
        class MockGeminiClient:
            async def generate_content()))))))))self, prompt, temperature=0.7, max_tokens=None, **kwargs):
                print()))))))))f"[]],,API CALL] Generating content for: {}}}prompt}")
                await asyncio.sleep()))))))))1)  # Simulate API delay
            return f"Response for: {}}}prompt}"
                
            async def generate_content_stream()))))))))self, prompt, temperature=0.7, max_tokens=None, **kwargs):
                print()))))))))f"[]],,API CALL] Streaming content for: {}}}prompt}")
                await asyncio.sleep()))))))))1)  # Simulate API delay
            return iter()))))))))[]],,f"Streaming response for: {}}}prompt}"])
        
            GeminiClient = MockGeminiClient
    
    # Create base client
            base_client = GeminiClient())))))))))
    
    # Create semantic cache wrapper
            cached_client = SemanticCacheGeminiClient()))))))))
            base_client=base_client,
            similarity_threshold=0.85,
            max_cache_size=100,
            ttl=3600
            )
    
    # Example prompts with semantic similarity
            prompts = []],,
            "What is the capital of France?",
            "Could you tell me the capital city of France?",  # Semantically similar
            "What's the capital of France?",  # Semantically similar
            "What is the population of Paris?",  # Different question
        "What is the capital of Italy?",  # Different country:::::
            "What's France's capital city?",  # Very similar to earlier prompts
            "Paris is the capital of which country:::::?",  # Related but different structure
            "Tell me about the capital of France",  # Request for more information
            ]
    
    for prompt in prompts:
        print()))))))))f"\nProcessing: {}}}prompt}")
        response = await cached_client.generate_content()))))))))prompt, temperature=0.0)
        print()))))))))f"Response: {}}}response}")
    
    # Print cache stats
        print()))))))))"\nCache Statistics:")
    for key, value in cached_client.get_cache_stats()))))))))).items()))))))))):
        print()))))))))f"  {}}}key}: {}}}value}")

if __name__ == "__main__":
    asyncio.run()))))))))example_usage()))))))))))