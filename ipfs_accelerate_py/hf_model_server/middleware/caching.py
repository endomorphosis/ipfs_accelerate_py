"""
Response caching middleware.
"""

import anyio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


class ResponseCache:
    """LRU cache for response caching with TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = anyio.Lock()
        self._hits = 0
        self._misses = 0
    
    def _generate_cache_key(self, model_id: str, request: Dict[str, Any]) -> str:
        """Generate deterministic cache key."""
        # Sort dict keys for consistency
        sorted_request = json.dumps(request, sort_keys=True)
        key_str = f"{model_id}:{sorted_request}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds
    
    async def get_or_compute(
        self,
        model_id: str,
        request: Dict[str, Any],
        compute_fn: Callable
    ) -> Any:
        """
        Get cached response or compute new one.
        
        Args:
            model_id: Model identifier
            request: Request parameters
            compute_fn: Function to compute result if cache miss
            
        Returns:
            Cached or computed result
        """
        cache_key = self._generate_cache_key(model_id, request)
        
        # Check cache
        async with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if not self._is_expired(timestamp):
                    # Move to end (LRU)
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    logger.debug(f"Cache hit for {model_id}")
                    return result
                else:
                    # Expired, remove
                    del self._cache[cache_key]
                    logger.debug(f"Cache expired for {model_id}")
        
        # Cache miss, compute
        self._misses += 1
        logger.debug(f"Cache miss for {model_id}")
        result = await compute_fn()
        
        # Store in cache
        async with self._lock:
            self._cache[cache_key] = (result, time.time())
            self._cache.move_to_end(cache_key)
            
            # Evict if over limit
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
        
        return result
    
    async def invalidate(self, model_id: str, request: Optional[Dict[str, Any]] = None):
        """
        Invalidate cache entries.
        
        Args:
            model_id: Model identifier
            request: Optional specific request to invalidate
        """
        async with self._lock:
            if request:
                # Invalidate specific entry
                cache_key = self._generate_cache_key(model_id, request)
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.info(f"Invalidated cache entry for {model_id}")
            else:
                # Invalidate all entries for model
                to_remove = [
                    key for key in self._cache.keys()
                    if key.startswith(hashlib.sha256(f"{model_id}:".encode()).hexdigest()[:16])
                ]
                for key in to_remove:
                    del self._cache[key]
                logger.info(f"Invalidated {len(to_remove)} cache entries for {model_id}")
    
    async def clear(self):
        """Clear entire cache."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared response cache ({count} entries)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }
