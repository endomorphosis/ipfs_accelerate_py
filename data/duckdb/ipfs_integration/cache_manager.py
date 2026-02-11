"""
IPFS-based Cache Manager

This module provides intelligent caching for benchmark queries and results
using IPFS content addressing and local caching strategies.
"""

import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from .ipfs_config import IPFSConfig, get_ipfs_config

logger = logging.getLogger(__name__)


class IPFSCacheManager:
    """
    IPFS-based cache manager for benchmark data.
    
    This class provides intelligent caching with:
    - Content-addressable storage via IPFS
    - Local cache with TTL
    - Prefetching for common queries
    - Cache invalidation strategies
    """
    
    def __init__(self, config: Optional[IPFSConfig] = None):
        """Initialize cache manager.
        
        Args:
            config: IPFS configuration
        """
        self.config = config or get_ipfs_config()
        self.is_enabled = self.config.enable_cache
        self.cache_ttl = self.config.cache_ttl_seconds
        self.prefetch_enabled = self.config.prefetch_enabled
        
        self.cache_dir = Path(self.config.local_cache_dir) / "query_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache metadata
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        
        if self.is_enabled:
            logger.info(f"Cache manager enabled (TTL: {self.cache_ttl}s)")
        else:
            logger.info("Cache manager disabled")
    
    def is_available(self) -> bool:
        """Check if caching is available.
        
        Returns:
            True if caching is enabled
        """
        return self.is_enabled
    
    def _get_cache_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for a query.
        
        Args:
            query: Query string
            params: Query parameters
            
        Returns:
            Cache key (hash)
        """
        cache_input = f"{query}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(cache_input.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cached file.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid.
        
        Args:
            cache_key: Cache key
            
        Returns:
            True if cache is valid
        """
        if cache_key not in self.cache_metadata:
            return False
        
        metadata = self.cache_metadata[cache_key]
        cached_time = datetime.fromisoformat(metadata['timestamp'])
        age = (datetime.utcnow() - cached_time).total_seconds()
        
        return age < self.cache_ttl
    
    def get(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached query result.
        
        Args:
            query: Query string
            params: Query parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self.is_available():
            return None
        
        cache_key = self._get_cache_key(query, params)
        
        if not self._is_cache_valid(cache_key):
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            logger.debug(f"Cache hit: {cache_key[:12]}...")
            return cached_data
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, query: str, result: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> bool:
        """Cache query result.
        
        Args:
            query: Query string
            result: Query result to cache
            params: Query parameters
            
        Returns:
            True if cached successfully
        """
        if not self.is_available():
            return False
        
        cache_key = self._get_cache_key(query, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            self.cache_metadata[cache_key] = {
                'query': query,
                'params': params,
                'timestamp': datetime.utcnow().isoformat(),
                'size': cache_path.stat().st_size
            }
            
            logger.debug(f"Cached result: {cache_key[:12]}...")
            return True
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
            return False
    
    def invalidate(self, query: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> int:
        """Invalidate cache entries.
        
        Args:
            query: Specific query to invalidate (None = all)
            params: Query parameters (only used if query is specified)
            
        Returns:
            Number of entries invalidated
        """
        if not self.is_available():
            return 0
        
        if query is None:
            # Invalidate all
            count = 0
            for cache_key in list(self.cache_metadata.keys()):
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                del self.cache_metadata[cache_key]
                count += 1
            logger.info(f"Invalidated {count} cache entries")
            return count
        else:
            # Invalidate specific query
            cache_key = self._get_cache_key(query, params)
            if cache_key in self.cache_metadata:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                del self.cache_metadata[cache_key]
                logger.debug(f"Invalidated cache: {cache_key[:12]}...")
                return 1
            return 0
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        if not self.is_available():
            return 0
        
        count = 0
        for cache_key in list(self.cache_metadata.keys()):
            if not self._is_cache_valid(cache_key):
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                del self.cache_metadata[cache_key]
                count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")
        
        return count
    
    def prefetch(self, queries: list[tuple[str, Optional[Dict[str, Any]]]], executor: Callable) -> int:
        """Prefetch results for common queries.
        
        Args:
            queries: List of (query, params) tuples to prefetch
            executor: Function to execute queries
            
        Returns:
            Number of queries prefetched
        """
        if not self.is_available() or not self.prefetch_enabled:
            return 0
        
        prefetched = 0
        for query, params in queries:
            cache_key = self._get_cache_key(query, params)
            if not self._is_cache_valid(cache_key):
                try:
                    result = executor(query, params)
                    self.set(query, result, params)
                    prefetched += 1
                except Exception as e:
                    logger.error(f"Error prefetching query: {e}")
        
        if prefetched > 0:
            logger.info(f"Prefetched {prefetched} queries")
        
        return prefetched
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        total_size = sum(
            self._get_cache_path(k).stat().st_size
            for k in self.cache_metadata.keys()
            if self._get_cache_path(k).exists()
        )
        
        valid_count = sum(1 for k in self.cache_metadata.keys() if self._is_cache_valid(k))
        
        return {
            'enabled': self.is_enabled,
            'total_entries': len(self.cache_metadata),
            'valid_entries': valid_count,
            'expired_entries': len(self.cache_metadata) - valid_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'ttl_seconds': self.cache_ttl,
            'prefetch_enabled': self.prefetch_enabled
        }
