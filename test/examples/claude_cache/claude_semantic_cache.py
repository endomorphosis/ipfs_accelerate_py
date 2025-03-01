import os
import sys
import time
import numpy as np
import threading
import hashlib
import logging
import asyncio
import json
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from collections import OrderedDict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeSemanticCache:
    """
    Cache implementation that uses semantic similarity between queries to determine cache hits.
    Specifically designed for the Claude API, supporting Messages API format.
    """
    
    def __init__(
        self, 
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
        ttl: int = 3600,  # Time-to-live in seconds
        use_lru: bool = True,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the semantic cache.
        
        Args:
            embedding_model: Model used to generate embeddings for queries
            similarity_threshold: Minimum cosine similarity score to consider a cache hit
            max_cache_size: Maximum number of entries in the cache
            ttl: Time-to-live for cache entries in seconds
            use_lru: Whether to use LRU eviction policy
            normalize_embeddings: Whether to normalize embeddings before comparison
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl = ttl
        self.use_lru = use_lru
        self.normalize_embeddings = normalize_embeddings
        
        # Main cache storage: {cache_key: (embedding, response, timestamp, metadata)}
        self.cache = OrderedDict()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized Claude semantic cache with threshold {similarity_threshold}")
    
    def _generate_embedding(self, messages: List[Dict]) -> np.ndarray:
        """
        Generate an embedding for the given messages.
        
        Args:
            messages: List of message dictionaries in Claude format
            
        Returns:
            Embedding vector for the query
        """
        # Convert messages to a string representation
        message_str = self._messages_to_string(messages)
        
        if self.embedding_model is None:
            # Fallback to hash-based representation if no embedding model
            hash_val = int(hashlib.md5(message_str.encode()).hexdigest(), 16)
            # Create a pseudo-embedding from the hash
            pseudo_embedding = np.array(
                [(hash_val >> (8 * i)) & 0xFF for i in range(16)], 
                dtype=np.float32
            )
            # Normalize the pseudo-embedding
            if self.normalize_embeddings:
                pseudo_embedding = pseudo_embedding / np.linalg.norm(pseudo_embedding)
            return pseudo_embedding
        
        # Use the actual embedding model
        try:
            if hasattr(self.embedding_model, 'embed_query'):
                # SentenceTransformers style
                embedding = self.embedding_model.embed_query(message_str)
            elif hasattr(self.embedding_model, 'encode'):
                # Generic encode method
                embedding = self.embedding_model.encode(message_str)
            elif hasattr(self.embedding_model, 'get_embedding'):
                # Claude-specific embedding method
                embedding = self.embedding_model.get_embedding(message_str)
            else:
                # Call the model as a function
                embedding = self.embedding_model(message_str)
                
            # Convert to numpy array if not already
            if not isinstance(embedding, np.ndarray):
                if TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().cpu().numpy()
                else:
                    embedding = np.array(embedding)
                    
            # Normalize the embedding
            if self.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding
            
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}")
            # Fallback to hash
            return self._generate_embedding(None)
    
    def _messages_to_string(self, messages: List[Dict]) -> str:
        """
        Convert a list of Claude message dictionaries to a single string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            String representation of the messages
        """
        if not messages:
            return ""
            
        # Extract the content from each message
        message_texts = []
        for message in messages:
            role = message.get('role', '')
            
            # Handle different message content formats
            content = message.get('content', '')
            if isinstance(content, list):
                # Handle content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = ' '.join(text_parts)
            
            message_texts.append(f"{role}: {content}")
            
        return "\n".join(message_texts)
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score between the embeddings
        """
        if TORCH_AVAILABLE:
            # Use torch for potentially faster computation
            t_emb1 = torch.tensor(emb1, dtype=torch.float32)
            t_emb2 = torch.tensor(emb2, dtype=torch.float32)
            return torch.nn.functional.cosine_similarity(
                t_emb1.unsqueeze(0), t_emb2.unsqueeze(0)
            ).item()
        else:
            # Numpy implementation
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    def _find_most_similar(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the most similar cached entry to the given query embedding.
        
        Args:
            query_embedding: Query embedding to compare against cache
            
        Returns:
            Tuple of (cache_key, similarity_score) for the most similar entry
        """
        most_similar_key = None
        highest_similarity = -1.0
        
        for key, (cached_embedding, _, timestamp, _) in self.cache.items():
            # Skip expired entries
            if time.time() - timestamp > self.ttl:
                continue
                
            similarity = self._compute_similarity(query_embedding, cached_embedding)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_key = key
        
        return most_similar_key, highest_similarity
    
    def _clean_expired_entries(self) -> None:
        """Remove expired entries from the cache."""
        with self.lock:
            current_time = time.time()
            keys_to_remove = [
                key for key, (_, _, timestamp, _) in self.cache.items() 
                if current_time - timestamp > self.ttl
            ]
            
            for key in keys_to_remove:
                del self.cache[key]
                
            logger.debug(f"Cleaned {len(keys_to_remove)} expired entries from cache")
    
    def _make_space_if_needed(self) -> None:
        """Remove entries if cache exceeds maximum size."""
        if len(self.cache) < self.max_cache_size:
            return
            
        with self.lock:
            # If using LRU, the first item in OrderedDict is the least recently used
            if self.use_lru and isinstance(self.cache, OrderedDict) and self.cache:
                self.cache.popitem(last=False)
                logger.debug("Removed least recently used cache entry")
            else:
                # Otherwise remove random entry
                if self.cache:
                    key = next(iter(self.cache.keys()))
                    del self.cache[key]
                    logger.debug("Removed random cache entry")
    
    def get(self, messages: List[Dict], metadata: Optional[Dict] = None) -> Tuple[Optional[Any], float, Optional[Dict]]:
        """
        Get a cached response for a query if a similar one exists.
        
        Args:
            messages: List of message dictionaries to look up in the cache
            metadata: Optional metadata for the query (used for filtering)
            
        Returns:
            Tuple of (cached_response, similarity_score, cache_metadata)
        """
        # Periodically clean expired entries
        if time.time() % 10 < 0.1:  # Clean roughly every 10 seconds
            self._clean_expired_entries()
            
        # Generate embedding for the query
        query_embedding = self._generate_embedding(messages)
        
        with self.lock:
            # Find the most similar cached entry
            most_similar_key, similarity = self._find_most_similar(query_embedding)
            
            if most_similar_key is not None and similarity >= self.similarity_threshold:
                # Cache hit
                cached_embedding, response, timestamp, cached_metadata = self.cache[most_similar_key]
                
                # Update position in OrderedDict if using LRU
                if self.use_lru and isinstance(self.cache, OrderedDict):
                    self.cache.move_to_end(most_similar_key)
                
                logger.debug(f"Cache hit with similarity {similarity:.4f}")
                return response, similarity, cached_metadata
                
        # Cache miss
        logger.debug(f"Cache miss (best similarity: {similarity:.4f})")
        return None, similarity, None
    
    def put(self, messages: List[Dict], response: Any, metadata: Optional[Dict] = None) -> None:
        """
        Add a query-response pair to the cache.
        
        Args:
            messages: List of message dictionaries
            response: Response to cache
            metadata: Optional metadata to store with the cache entry
        """
        self._make_space_if_needed()
        
        query_embedding = self._generate_embedding(messages)
        current_time = time.time()
        
        with self.lock:
            # Generate a unique cache key
            message_str = self._messages_to_string(messages)
            response_str = str(response)
            cache_key = f"{hash(message_str)}_{hash(response_str)}"
            
            # Store the entry in the cache
            self.cache[cache_key] = (
                query_embedding, 
                response, 
                current_time,
                metadata or {}
            )
            
            # Move to end if using LRU to mark as most recently used
            if self.use_lru and isinstance(self.cache, OrderedDict):
                self.cache.move_to_end(cache_key)
                
            logger.debug(f"Added new entry to cache (size: {len(self.cache)})")
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        with self.lock:
            current_time = time.time()
            active_entries = sum(
                1 for _, _, timestamp, _ in self.cache.values() 
                if current_time - timestamp <= self.ttl
            )
            
            return {
                "total_entries": len(self.cache),
                "active_entries": active_entries,
                "expired_entries": len(self.cache) - active_entries,
                "max_size": self.max_cache_size,
                "similarity_threshold": self.similarity_threshold,
                "ttl": self.ttl,
            }


class SemanticCacheClaudeClient:
    """
    A wrapper around the Claude API client that adds semantic caching capabilities.
    Supports the Anthropic Messages API format for Claude.
    """
    
    def __init__(
        self, 
        base_client: Any,
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
        ttl: int = 3600,
        cache_enabled: bool = True,
        embedding_dimensions: int = 1536  # Claude embeddings are 1536-dimensional
    ):
        """
        Initialize the semantic cache wrapper for Claude client.
        
        Args:
            base_client: The base Claude client to wrap
            embedding_model: Model used to generate embeddings for queries
            similarity_threshold: Minimum similarity threshold for cache hits
            max_cache_size: Maximum cache size
            ttl: Time-to-live for cache entries
            cache_enabled: Whether caching is enabled
            embedding_dimensions: Dimensions for embeddings when using fallback
        """
        self.base_client = base_client
        self.cache_enabled = cache_enabled
        self.embedding_dimensions = embedding_dimensions
        
        # Initialize semantic cache
        self.cache = ClaudeSemanticCache(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size,
            ttl=ttl
        )
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_similarity": 0.0,
            "token_savings": 0,
        }
        self.stats_lock = threading.Lock()
    
    async def chat(self, 
                  messages: List[Dict],
                  model: str = "claude-3-opus-20240229",
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  **kwargs) -> Dict:
        """
        Generate a chat response with semantic caching.
        
        Args:
            messages: List of message dictionaries in Claude format
            model: Claude model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional arguments for the base client
            
        Returns:
            Generated content response
        """
        # Update request stats
        with self.stats_lock:
            self.stats["total_requests"] += 1
        
        # Skip cache for non-deterministic generations
        if not self.cache_enabled or temperature > 0.0:
            logger.debug("Bypassing cache due to non-zero temperature or disabled cache")
            return await self._chat_direct(messages, model, max_tokens, temperature, **kwargs)
            
        # Include important kwargs in the cache metadata
        cache_metadata = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **{k: v for k, v in kwargs.items() if k in ['stream', 'stop_sequences', 'top_p', 'top_k']}
        }
        
        # Try to get response from cache
        cached_response, similarity, _ = self.cache.get(messages, metadata=cache_metadata)
        
        # Update similarity stats
        with self.stats_lock:
            # Running average of similarity scores
            self.stats["avg_similarity"] = (
                (self.stats["avg_similarity"] * (self.stats["total_requests"] - 1) + similarity) / 
                self.stats["total_requests"]
            )
        
        if cached_response is not None:
            # Cache hit
            with self.stats_lock:
                self.stats["cache_hits"] += 1
                # Estimate token savings (very rough estimate)
                prompt_tokens = int(len(self.cache._messages_to_string(messages).split()) * 0.75)
                completion_tokens = int(len(str(cached_response).split()) * 0.75)
                self.stats["token_savings"] += prompt_tokens + completion_tokens
                
            logger.info(f"Cache hit with similarity {similarity:.4f}")
            return cached_response
        
        # Cache miss - call the base client
        with self.stats_lock:
            self.stats["cache_misses"] += 1
        
        logger.info(f"Cache miss (best similarity: {similarity:.4f})")
        response = await self._chat_direct(messages, model, max_tokens, temperature, **kwargs)
        
        # Store in cache if it's a deterministic generation
        if temperature == 0.0:
            self.cache.put(messages, response, metadata=cache_metadata)
        
        return response
    
    async def _chat_direct(self,
                         messages: List[Dict],
                         model: str = "claude-3-opus-20240229",
                         max_tokens: int = 1024,
                         temperature: float = 0.7,
                         **kwargs) -> Dict:
        """
        Direct call to base client's chat method without caching.
        
        Args:
            messages: List of message dictionaries in Claude format
            model: Claude model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional arguments for the base client
            
        Returns:
            Generated content response
        """
        # Handle different client interfaces - try both anthropic.Anthropic and custom clients
        if hasattr(self.base_client, 'messages'):
            # Anthropic Python SDK
            response = await self.base_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response
        else:
            # Custom client implementation
            if hasattr(self.base_client, 'chat'):
                # Call chat method
                return await self.base_client.chat(
                    messages, model, max_tokens, temperature, **kwargs
                )
            elif hasattr(self.base_client, 'create_chat_completion'):
                # OpenAI-style interface
                return await self.base_client.create_chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            else:
                # Call the client directly
                return await self.base_client(
                    messages=messages,
                    model=model, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
    
    async def stream_chat(self,
                        messages: List[Dict],
                        model: str = "claude-3-opus-20240229",
                        max_tokens: int = 1024,
                        temperature: float = 0.7,
                        **kwargs) -> Any:
        """
        Generate streaming chat response (always bypasses cache).
        
        Args:
            messages: List of message dictionaries in Claude format
            model: Claude model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional arguments for the base client
            
        Returns:
            Streaming response
        """
        # Streaming always bypasses cache
        with self.stats_lock:
            self.stats["total_requests"] += 1
            self.stats["cache_misses"] += 1
            
        # Handle different client interfaces
        if hasattr(self.base_client, 'messages'):
            # Anthropic Python SDK
            kwargs['stream'] = True
            return await self.base_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        else:
            # Custom client implementation
            if hasattr(self.base_client, 'stream_chat'):
                return await self.base_client.stream_chat(
                    messages, model, max_tokens, temperature, **kwargs
                )
            elif hasattr(self.base_client, 'create_chat_completion'):
                # OpenAI-style interface with streaming
                kwargs['stream'] = True
                return await self.base_client.create_chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            else:
                # Assume streaming is handled by kwargs
                kwargs['stream'] = True
                return await self.base_client(
                    messages=messages,
                    model=model, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
    
    def get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get embeddings for text using Claude's embedding API if available.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not hasattr(self.base_client, 'embeddings'):
            # Fallback to hash-based pseudo embeddings
            return self._create_fallback_embedding(text)
            
        try:
            if isinstance(text, list):
                # Batch embedding
                response = self.base_client.embeddings.create(
                    model="claude-3-sonnet-20240229-v1:0",
                    input=text
                )
                return np.array([item.embedding for item in response.embeddings])
            else:
                # Single text embedding
                response = self.base_client.embeddings.create(
                    model="claude-3-sonnet-20240229-v1:0",
                    input=[text]
                )
                return np.array(response.embeddings[0].embedding)
        except Exception as e:
            logger.warning(f"Error generating Claude embedding: {e}")
            return self._create_fallback_embedding(text)
    
    def _create_fallback_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """Create fallback embedding when the API is unavailable."""
        if isinstance(text, list):
            return np.array([self._text_to_pseudo_embedding(t) for t in text])
        return self._text_to_pseudo_embedding(text)
    
    def _text_to_pseudo_embedding(self, text: str) -> np.ndarray:
        """Convert text to a pseudo-embedding using hash."""
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Create a pseudo-embedding from the hash (16 dimensions)
        pseudo_embedding = np.array(
            [(hash_val >> (i * 8)) & 0xFF for i in range(16)], 
            dtype=np.float32
        )
        # Normalize and repeat to get higher dimensionality
        pseudo_embedding = pseudo_embedding / np.linalg.norm(pseudo_embedding)
        # Repeat to match desired dimensions
        repeat_factor = self.embedding_dimensions // 16
        return np.tile(pseudo_embedding, repeat_factor + 1)[:self.embedding_dimensions]
    
    # Pass through all other methods to the base client
    def __getattr__(self, name):
        return getattr(self.base_client, name)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage."""
        with self.stats_lock:
            stats_copy = self.stats.copy()
        
        # Add cache internal stats
        cache_stats = self.cache.get_stats()
        return {**stats_copy, **cache_stats}
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable the cache."""
        self.cache_enabled = enabled
        logger.info(f"Cache {'enabled' if enabled else 'disabled'}")


# Example usage
async def example_usage():
    """
    Example of how to use the semantic cache with the Claude API.
    """
    # Import the Anthropic client
    try:
        import anthropic
        client = anthropic.Anthropic()
    except (ImportError, Exception) as e:
        # Mock client for demonstration
        class MockClaudeClient:
            async def chat(self, messages, model="claude-3-opus-20240229", max_tokens=1024, temperature=0.7, **kwargs):
                print(f"[API CALL] Generating content for: {messages[-1]['content']}")
                await asyncio.sleep(1)  # Simulate API delay
                return {
                    "id": "msg_" + hashlib.md5(str(messages).encode()).hexdigest()[:10],
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Response for: {messages[-1]['content']}"}],
                    "model": model,
                    "stop_reason": "end_turn"
                }
                
            async def stream_chat(self, messages, model="claude-3-opus-20240229", max_tokens=1024, temperature=0.7, **kwargs):
                print(f"[API CALL] Streaming content for: {messages[-1]['content']}")
                await asyncio.sleep(1)  # Simulate API delay
                yield {
                    "type": "content_block_delta",
                    "delta": {"type": "text", "text": f"Streaming response for: {messages[-1]['content']}"},
                    "index": 0
                }
        
        client = MockClaudeClient()
    
    # Create semantic cache wrapper
    cached_client = SemanticCacheClaudeClient(
        base_client=client,
        similarity_threshold=0.85,
        max_cache_size=100,
        ttl=3600
    )
    
    # Example prompts with semantic similarity
    example_messages = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "Could you tell me the capital city of France?"}],  # Semantically similar
        [{"role": "user", "content": "What's the capital of France?"}],  # Semantically similar
        [{"role": "user", "content": "What is the population of Paris?"}],  # Different question
        [{"role": "user", "content": "What is the capital of Italy?"}],  # Different country
        [{"role": "user", "content": "What's France's capital city?"}],  # Very similar to earlier prompts
        [{"role": "user", "content": "Paris is the capital of which country?"}],  # Related but different structure
        [{"role": "user", "content": "Tell me about the capital of France"}],  # Request for more information
    ]
    
    print("Testing Claude semantic cache with various queries...\n")
    
    for messages in example_messages:
        print(f"\nProcessing: {messages[-1]['content']}")
        response = await cached_client.chat(messages, temperature=0.0)
        
        # Extract the response text
        if isinstance(response, dict):
            content = response.get('content', '')
            if isinstance(content, list):
                response_text = ' '.join([
                    block.get('text', '') if isinstance(block, dict) else block
                    for block in content
                ])
            else:
                response_text = content
        else:
            response_text = str(response)
            
        print(f"Response: {response_text}")
    
    # Print cache stats
    print("\nCache Statistics:")
    for key, value in cached_client.get_cache_stats().items():
        print(f"  {key}: {value}")

    # Show token savings
    print(f"\nEstimated token savings: {cached_client.stats['token_savings']:.0f} tokens")
    print(f"API cost savings: ${cached_client.stats['token_savings'] * 0.00001:.4f} (based on $0.01/1K tokens)")
    print(f"Cache hit rate: {cached_client.stats['cache_hits'] / cached_client.stats['total_requests']:.1%}")

if __name__ == "__main__":
    asyncio.run(example_usage())