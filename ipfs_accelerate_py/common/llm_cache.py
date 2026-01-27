"""
LLM API Cache

Cache adapter for Language Model API responses (OpenAI, Claude, Gemini, Groq, etc.).
Caches completions, chat responses, and embeddings to reduce API costs and latency.

Uses content-addressed identifiers (CID) with multiformats for cache keys,
enabling fast lookups by hashing the query parameters.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from .base_cache import BaseAPICache

logger = logging.getLogger(__name__)


class LLMAPICache(BaseAPICache):
    """
    Cache for LLM API responses.
    
    Caches:
    - Completions (text generation)
    - Chat completions (conversation)
    - Embeddings (vector representations)
    
    Uses semantic hashing to detect similar prompts and return cached responses.
    """
    
    # Default TTLs for different operations (in seconds)
    DEFAULT_TTLS = {
        "completion": 3600,  # 1 hour (deterministic responses can be cached longer)
        "chat_completion": 1800,  # 30 minutes
        "embedding": 86400,  # 24 hours (embeddings are deterministic)
        "model_list": 3600,  # 1 hour (model lists change infrequently)
    }
    
    def get_cache_namespace(self) -> str:
        """Get cache namespace."""
        return "llm_api"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from LLM API response.
        
        For LLM APIs, we primarily rely on prompt hashing rather than response validation,
        since the same prompt should yield the same (or similar) response.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Validation fields dictionary or None
        """
        # For LLM APIs, we don't validate based on response content
        # Instead, cache key includes prompt hash, so we get exact matches
        # Return None to fall back to TTL-based expiration
        return None
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        """Get operation-specific TTL."""
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)
    
    def make_cache_key_for_prompt(
        self,
        operation: str,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Create a content-addressed cache key (CID) for a prompt-based operation.
        
        Args:
            operation: Operation name (e.g., 'completion', 'chat_completion')
            prompt: The prompt text
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            CID string for the cache entry
        """
        # Build the complete query object
        query = {
            "operation": operation,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add any other relevant kwargs
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                query[key] = kwargs[key]
        
        # Use parent's CID computation
        query_json = json.dumps(query, sort_keys=True)
        return self._compute_cid(query_json)
    
    def make_cache_key_for_messages(
        self,
        operation: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Create a content-addressed cache key (CID) for a messages-based operation.
        
        Args:
            operation: Operation name (e.g., 'chat_completion')
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            CID string for the cache entry
        """
        # Build the complete query object
        query = {
            "operation": operation,
            "messages": messages,  # Keep full messages for accurate CID
            "model": model,
            "temperature": temperature
        }
        
        # Add any other relevant kwargs
        for key in ["top_p", "max_tokens", "frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                query[key] = kwargs[key]
        
        # Use parent's CID computation
        query_json = json.dumps(query, sort_keys=True)
        return self._compute_cid(query_json)
    
    def cache_completion(
        self,
        prompt: str,
        response: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Cache a completion response.
        
        Args:
            prompt: The prompt text
            response: The API response
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            ttl: Custom TTL (optional)
            **kwargs: Additional parameters
        """
        cache_key = self.make_cache_key_for_prompt(
            "completion", prompt, model, temperature, max_tokens, **kwargs
        )
        
        # Use the cache key directly since we already computed it
        with self._lock:
            # Extract validation fields and create entry
            validation_fields = self.extract_validation_fields("completion", response)
            content_hash = None
            if validation_fields:
                content_hash = self._compute_validation_hash(validation_fields)
            
            from .base_cache import CacheEntry
            import time
            
            entry = CacheEntry(
                data=response,
                timestamp=time.time(),
                ttl=ttl or self.get_default_ttl_for_operation("completion"),
                content_hash=content_hash,
                validation_fields=validation_fields,
                metadata={"model": model, "temperature": temperature}
            )
            
            # Evict if full
            if len(self._cache) >= self.max_cache_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
            
            self._cache[cache_key] = entry
        
        if self.enable_persistence:
            self._save_to_disk(cache_key, entry)
    
    def get_completion(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Get a cached completion response.
        
        Args:
            prompt: The prompt text
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Cached response or None
        """
        cache_key = self.make_cache_key_for_prompt(
            "completion", prompt, model, temperature, max_tokens, **kwargs
        )
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                if entry.is_expired():
                    del self._cache[cache_key]
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    self._stats["api_calls_made"] += 1
                    return None
                
                self._stats["hits"] += 1
                self._stats["api_calls_saved"] += 1
                return entry.data
            
            self._stats["misses"] += 1
            self._stats["api_calls_made"] += 1
            return None
    
    def cache_chat_completion(
        self,
        messages: List[Dict[str, str]],
        response: Any,
        model: str,
        temperature: float = 0.0,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """Cache a chat completion response."""
        cache_key = self.make_cache_key_for_messages(
            "chat_completion", messages, model, temperature, **kwargs
        )
        
        with self._lock:
            validation_fields = self.extract_validation_fields("chat_completion", response)
            content_hash = None
            if validation_fields:
                content_hash = self._compute_validation_hash(validation_fields)
            
            from .base_cache import CacheEntry
            import time
            
            entry = CacheEntry(
                data=response,
                timestamp=time.time(),
                ttl=ttl or self.get_default_ttl_for_operation("chat_completion"),
                content_hash=content_hash,
                validation_fields=validation_fields,
                metadata={"model": model, "temperature": temperature}
            )
            
            if len(self._cache) >= self.max_cache_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
            
            self._cache[cache_key] = entry
        
        if self.enable_persistence:
            self._save_to_disk(cache_key, entry)
    
    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        **kwargs
    ) -> Optional[Any]:
        """Get a cached chat completion response."""
        cache_key = self.make_cache_key_for_messages(
            "chat_completion", messages, model, temperature, **kwargs
        )
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                if entry.is_expired():
                    del self._cache[cache_key]
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    self._stats["api_calls_made"] += 1
                    return None
                
                self._stats["hits"] += 1
                self._stats["api_calls_saved"] += 1
                return entry.data
            
            self._stats["misses"] += 1
            self._stats["api_calls_made"] += 1
            return None


# Global LLM cache instance
_global_llm_cache: Optional[LLMAPICache] = None
_llm_cache_lock = threading.Lock()


def get_global_llm_cache() -> LLMAPICache:
    """Get or create the global LLM API cache instance."""
    global _global_llm_cache
    
    with _llm_cache_lock:
        if _global_llm_cache is None:
            _global_llm_cache = LLMAPICache()
            from .base_cache import register_cache
            register_cache("llm_api", _global_llm_cache)
        
        return _global_llm_cache


def configure_llm_cache(**kwargs) -> LLMAPICache:
    """
    Configure the global LLM cache.
    
    Args:
        **kwargs: Arguments to pass to LLMAPICache constructor
        
    Returns:
        Configured LLM cache instance
    """
    global _global_llm_cache
    
    with _llm_cache_lock:
        if _global_llm_cache is not None:
            _global_llm_cache.shutdown()
        
        _global_llm_cache = LLMAPICache(**kwargs)
        from .base_cache import register_cache
        register_cache("llm_api", _global_llm_cache)
        
        return _global_llm_cache


import threading
