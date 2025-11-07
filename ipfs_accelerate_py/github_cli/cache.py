"""
GitHub API Response Cache

This module provides caching capabilities for GitHub API responses to reduce
the number of API calls and avoid rate limiting.

Uses content-addressed hashing (multiformats) to intelligently detect stale cache.
"""

import json
import logging
import os
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock

# Try to import multiformats for content-addressed caching
try:
    from multiformats import CID, multihash
    HAVE_MULTIFORMATS = True
except ImportError:
    HAVE_MULTIFORMATS = False
    CID = None
    multihash = None

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached API response with content-based validation."""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    content_hash: Optional[str] = None  # Multihash of validation fields
    validation_fields: Optional[Dict[str, Any]] = None  # Fields used for hash
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl
    
    def is_stale(self, current_validation_fields: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if cache is stale by comparing validation fields.
        
        Args:
            current_validation_fields: Current values of validation fields
            
        Returns:
            True if cache is stale (hash mismatch), False if still valid
        """
        # If no validation fields, fall back to TTL-based expiration
        if not self.content_hash or not current_validation_fields:
            return self.is_expired()
        
        # Compute hash of current validation fields
        current_hash = GitHubAPICache._compute_validation_hash(current_validation_fields)
        
        # Cache is stale if hash changed
        return current_hash != self.content_hash


class GitHubAPICache:
    """
    Cache for GitHub API responses with TTL and persistence.
    
    Features:
    - In-memory caching with TTL
    - Optional disk persistence
    - Thread-safe operations
    - Automatic expiration
    - Cache statistics
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_ttl: int = 300,  # 5 minutes
        max_cache_size: int = 1000,
        enable_persistence: bool = True
    ):
        """
        Initialize the GitHub API cache.
        
        Args:
            cache_dir: Directory for persistent cache (default: ~/.cache/github_cli)
            default_ttl: Default time-to-live for cache entries in seconds
            max_cache_size: Maximum number of entries to keep in memory
            enable_persistence: Whether to persist cache to disk
        """
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self.enable_persistence = enable_persistence
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "github_cli"
        
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "expirations": 0,
            "evictions": 0
        }
        
        # Load persistent cache if enabled
        if self.enable_persistence:
            self._load_from_disk()
    
    def _make_cache_key(self, operation: str, *args, **kwargs) -> str:
        """
        Create a cache key from operation and parameters.
        
        Args:
            operation: Operation name (e.g., 'list_repos', 'workflow_runs')
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_parts = [operation] + list(map(str, args)) + [f"{k}={v}" for k, v in sorted_kwargs]
        return ":".join(key_parts)
    
    @staticmethod
    def _compute_validation_hash(validation_fields: Dict[str, Any]) -> str:
        """
        Compute content-addressed hash of validation fields using multiformats.
        
        Args:
            validation_fields: Fields to hash (e.g., {'updatedAt': '2025-11-06T10:00:00Z'})
            
        Returns:
            CID string if multiformats available, otherwise SHA256 hex
        """
        # Sort fields for deterministic hashing
        sorted_fields = json.dumps(validation_fields, sort_keys=True)
        
        if HAVE_MULTIFORMATS:
            # Use multiformats for content-addressed hashing
            content_bytes = sorted_fields.encode('utf-8')
            hasher = hashlib.sha256()
            hasher.update(content_bytes)
            digest = hasher.digest()
            
            # Wrap in multihash
            mh = multihash.wrap(digest, 'sha2-256')
            # Create CID
            cid = CID('base32', 1, 'raw', mh)
            return str(cid)
        else:
            # Fallback to simple SHA256 hex
            hasher = hashlib.sha256()
            hasher.update(sorted_fields.encode('utf-8'))
            return hasher.hexdigest()
    
    @staticmethod
    def _extract_validation_fields(operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from API response based on operation type.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Dictionary of fields to use for validation hashing
        """
        if not data:
            return None
        
        validation = {}
        
        # Repository operations - use updatedAt/pushedAt
        if operation.startswith('list_repos') or operation == 'get_repo_info':
            if isinstance(data, list):
                # For list operations, hash all repo update times
                for repo in data:
                    if isinstance(repo, dict):
                        repo_id = repo.get('name') or repo.get('url', '')
                        validation[repo_id] = {
                            'updatedAt': repo.get('updatedAt'),
                            'pushedAt': repo.get('pushedAt')
                        }
            elif isinstance(data, dict):
                # For single repo
                validation['updatedAt'] = data.get('updatedAt')
                validation['pushedAt'] = data.get('pushedAt')
        
        # Workflow operations - use updatedAt/status/conclusion
        elif 'workflow' in operation:
            if isinstance(data, list):
                for workflow in data:
                    if isinstance(workflow, dict):
                        wf_id = str(workflow.get('databaseId', workflow.get('id', '')))
                        validation[wf_id] = {
                            'status': workflow.get('status'),
                            'conclusion': workflow.get('conclusion'),
                            'updatedAt': workflow.get('updatedAt')
                        }
            elif isinstance(data, dict):
                validation['status'] = data.get('status')
                validation['conclusion'] = data.get('conclusion')
                validation['updatedAt'] = data.get('updatedAt')
        
        # Runner operations - use status/busy
        elif 'runner' in operation:
            if isinstance(data, list):
                for runner in data:
                    if isinstance(runner, dict):
                        runner_id = str(runner.get('id', runner.get('name', '')))
                        validation[runner_id] = {
                            'status': runner.get('status'),
                            'busy': runner.get('busy')
                        }
            elif isinstance(data, dict):
                validation['status'] = data.get('status')
                validation['busy'] = data.get('busy')
        
        # Copilot operations - hash the prompt for deterministic results
        elif operation.startswith('copilot_'):
            # Copilot responses should be stable for same prompts
            # No validation needed - rely on TTL
            return None
        
        return validation if validation else None
    
    def get(
        self,
        operation: str,
        *args,
        validation_fields: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Get a cached response with optional validation field checking.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            validation_fields: Current validation fields to check staleness
            **kwargs: Keyword arguments
            
        Returns:
            Cached data or None if not found/expired/stale
        """
        cache_key = self._make_cache_key(operation, *args, **kwargs)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            # Check TTL-based expiration
            if entry.is_expired():
                logger.debug(f"Cache entry expired for {cache_key}")
                del self._cache[cache_key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None
            
            # Check content-based staleness
            if validation_fields and entry.is_stale(validation_fields):
                logger.debug(f"Cache entry stale (hash mismatch) for {cache_key}")
                del self._cache[cache_key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            logger.debug(f"Cache hit for {cache_key}")
            return entry.data
    
    def put(
        self,
        operation: str,
        data: Any,
        ttl: Optional[int] = None,
        *args,
        **kwargs
    ) -> None:
        """
        Store a response in the cache with content-based validation.
        
        Args:
            operation: Operation name
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if None)
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._make_cache_key(operation, *args, **kwargs)
        ttl = ttl if ttl is not None else self.default_ttl
        
        # Extract validation fields and compute hash
        validation_fields = self._extract_validation_fields(operation, data)
        content_hash = None
        if validation_fields:
            content_hash = self._compute_validation_hash(validation_fields)
            logger.debug(f"Computed validation hash for {cache_key}: {content_hash[:16]}...")
        
        with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self.max_cache_size:
                self._evict_oldest()
            
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=ttl,
                content_hash=content_hash,
                validation_fields=validation_fields
            )
            
            self._cache[cache_key] = entry
            logger.debug(f"Cached {cache_key} with TTL {ttl}s")
            
            # Persist to disk if enabled
            if self.enable_persistence:
                self._save_to_disk(cache_key, entry)
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return
        
        # Find oldest entry
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def invalidate(self, operation: str, *args, **kwargs) -> None:
        """
        Invalidate a specific cache entry.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._make_cache_key(operation, *args, **kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"Invalidated cache entry: {cache_key}")
                
                # Remove from disk if persistence enabled
                if self.enable_persistence:
                    cache_file = self.cache_dir / f"{self._sanitize_filename(cache_key)}.json"
                    if cache_file.exists():
                        cache_file.unlink()
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., 'list_repos' will invalidate all list_repos calls)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
            
            for key in keys_to_delete:
                del self._cache[key]
                
                # Remove from disk if persistence enabled
                if self.enable_persistence:
                    cache_file = self.cache_dir / f"{self._sanitize_filename(key)}.json"
                    if cache_file.exists():
                        cache_file.unlink()
            
            logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching '{pattern}'")
            return len(keys_to_delete)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "expirations": 0,
                "evictions": 0
            }
            
            # Clear disk cache if persistence enabled
            if self.enable_persistence and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
            
            logger.info("Cleared all cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size
            }
    
    def _sanitize_filename(self, key: str) -> str:
        """Sanitize a cache key for use as a filename."""
        # Replace invalid filename characters with underscores
        return key.replace("/", "_").replace(":", "_").replace("*", "_")
    
    def _save_to_disk(self, cache_key: str, entry: CacheEntry) -> None:
        """Save a cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{self._sanitize_filename(cache_key)}.json"
            cache_data = {
                "data": entry.data,
                "timestamp": entry.timestamp,
                "ttl": entry.ttl,
                "content_hash": entry.content_hash,
                "validation_fields": entry.validation_fields
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logger.debug(f"Saved cache entry to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache entries from disk."""
        if not self.cache_dir.exists():
            return
        
        loaded_count = 0
        expired_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    entry = CacheEntry(
                        data=cache_data["data"],
                        timestamp=cache_data["timestamp"],
                        ttl=cache_data["ttl"],
                        content_hash=cache_data.get("content_hash"),
                        validation_fields=cache_data.get("validation_fields")
                    )
                    
                    # Only load non-expired entries
                    if not entry.is_expired():
                        # Reconstruct cache key from filename
                        cache_key = cache_file.stem
                        self._cache[cache_key] = entry
                        loaded_count += 1
                    else:
                        # Remove expired cache file
                        cache_file.unlink()
                        expired_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
            
            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} cache entries from disk ({expired_count} expired)")
        
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")


# Global cache instance (can be configured at module level)
_global_cache: Optional[GitHubAPICache] = None


def get_global_cache(**kwargs) -> GitHubAPICache:
    """
    Get or create the global GitHub API cache instance.
    
    Args:
        **kwargs: Arguments to pass to GitHubAPICache constructor
        
    Returns:
        Global GitHubAPICache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = GitHubAPICache(**kwargs)
    
    return _global_cache


def configure_cache(**kwargs) -> GitHubAPICache:
    """
    Configure the global cache with custom settings.
    
    Args:
        **kwargs: Arguments to pass to GitHubAPICache constructor
        
    Returns:
        Configured GitHubAPICache instance
    """
    global _global_cache
    _global_cache = GitHubAPICache(**kwargs)
    return _global_cache
