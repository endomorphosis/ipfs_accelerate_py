"""
Base Cache Infrastructure

This module provides a common caching infrastructure that can be shared across
all API calls amenable to caching. It extracts the common patterns from the
GitHub CLI cache and makes them reusable for other APIs.

Features:
- TTL-based expiration
- Content-addressed validation (using multiformats)
- Thread-safe operations
- Optional disk persistence
- P2P cache sharing via libp2p
- Encryption support
- Cache statistics and monitoring
"""

import json
import logging
import os
import time
import hashlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from threading import Lock

# Try to import cryptography for message encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False
    Fernet = None
    PBKDF2HMAC = None
    hashes = None
    default_backend = None

# Try to import multiformats for content-addressed caching
try:
    from multiformats import CID
    from multiformats import multihash as multiformats_multihash
    HAVE_MULTIFORMATS = True
except ImportError:
    HAVE_MULTIFORMATS = False
    CID = None
    multiformats_multihash = None

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached API response with content-based validation."""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    content_hash: Optional[str] = None  # Multihash of validation fields
    validation_fields: Optional[Dict[str, Any]] = None  # Fields used for hash
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata (e.g., response headers)
    
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
        current_hash = BaseAPICache._compute_validation_hash(current_validation_fields)
        
        # Cache is stale if hash changed
        return current_hash != self.content_hash


class BaseAPICache(ABC):
    """
    Base class for API response caching with TTL and persistence.
    
    Features:
    - In-memory caching with TTL
    - Optional disk persistence
    - Thread-safe operations
    - Automatic expiration
    - Cache statistics
    - Content-addressed validation
    - P2P cache sharing (when enabled)
    
    Subclasses should implement:
    - get_cache_namespace(): Return unique namespace for the API
    - extract_validation_fields(): Extract fields for content validation
    - get_default_ttl_for_operation(): Get operation-specific TTL
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_ttl: int = 300,  # 5 minutes
        max_cache_size: int = 1000,
        enable_persistence: bool = True,
        enable_p2p: bool = False,
        cache_name: Optional[str] = None
    ):
        """
        Initialize the API cache.
        
        Args:
            cache_dir: Directory for persistent cache (default: ~/.cache/<cache_name>)
            default_ttl: Default time-to-live for cache entries in seconds
            max_cache_size: Maximum number of entries to keep in memory
            enable_persistence: Whether to persist cache to disk
            enable_p2p: Whether to enable P2P cache sharing
            cache_name: Name for this cache (used in directory path)
        """
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self.enable_persistence = enable_persistence
        self.enable_p2p = enable_p2p
        self.cache_name = cache_name or self.get_cache_namespace()
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            env_dir = os.environ.get("IPFS_ACCELERATE_CACHE_DIR")
            if env_dir:
                self.cache_dir = Path(env_dir) / self.cache_name
            else:
                self.cache_dir = Path.home() / ".cache" / self.cache_name
        
        if self.enable_persistence:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Verify directory is writable
                write_probe = self.cache_dir / ".write_test"
                try:
                    with open(write_probe, "w") as f:
                        f.write("ok")
                    write_probe.unlink(missing_ok=True)
                except Exception as e:
                    raise OSError(f"Cache directory not writable: {e}")
            except OSError as e:
                # Fall back to /tmp
                fallback = Path("/tmp") / f"ipfs_accelerate_{self.cache_name}_cache"
                logger.warning(f"⚠ Cache dir not writable ({self.cache_dir}): {e} - falling back to {fallback}")
                self.cache_dir = fallback
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        
        # CID index for fast lookups
        from .cid_index import get_global_cid_index
        self._cid_index = get_global_cid_index()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "expirations": 0,
            "evictions": 0,
            "api_calls_saved": 0,
            "api_calls_made": 0,
            "peer_hits": 0
        }
        
        # Load persistent cache if enabled
        if self.enable_persistence:
            self._load_from_disk()
        
        logger.info(f"✓ {self.cache_name} cache initialized (persistence={enable_persistence}, p2p={enable_p2p})")
    
    @abstractmethod
    def get_cache_namespace(self) -> str:
        """
        Get the cache namespace for this API.
        
        Returns:
            Unique namespace string (e.g., 'github_api', 'huggingface_hub', 'openai')
        """
        pass
    
    @abstractmethod
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from API response based on operation type.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Dictionary of fields to use for validation hashing, or None if not applicable
        """
        pass
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        """
        Get the default TTL for a specific operation.
        
        Subclasses can override this to provide operation-specific TTLs.
        
        Args:
            operation: Operation name
            
        Returns:
            TTL in seconds
        """
        return self.default_ttl
    
    def make_cache_key(self, operation: str, *args, **kwargs) -> str:
        """
        Create a content-addressed cache key (CID) from operation and parameters.
        
        Uses multiformats CID to create a unique, content-addressed identifier
        for the cache entry. This allows fast lookups by hashing the query.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            CID string (if multiformats available) or fallback hash string
        """
        # Create a deterministic representation of the query
        query = {
            "operation": operation,
            "args": list(args),
            "kwargs": dict(sorted(kwargs.items()))
        }
        
        # Serialize to JSON for consistent hashing
        query_json = json.dumps(query, sort_keys=True)
        
        # Create content-addressed identifier (CID)
        return self._compute_cid(query_json)
    
    @staticmethod
    def _compute_cid(content: str) -> str:
        """
        Compute content-addressed identifier (CID) using multiformats.
        
        Args:
            content: String content to hash
            
        Returns:
            CID string if multiformats available, otherwise SHA256 hex
        """
        if HAVE_MULTIFORMATS:
            # Use multiformats for content-addressed hashing
            content_bytes = content.encode('utf-8')
            hasher = hashlib.sha256()
            hasher.update(content_bytes)
            digest = hasher.digest()
            
            # Wrap in multihash
            mh = multiformats_multihash.wrap(digest, 'sha2-256')
            # Create CID (base32, version 1, raw codec)
            cid = CID('base32', 1, 'raw', mh)
            return str(cid)
        else:
            # Fallback to simple SHA256 hex (prefixed to indicate it's a hash)
            hasher = hashlib.sha256()
            hasher.update(content.encode('utf-8'))
            return f"sha256:{hasher.hexdigest()}"
    
    @staticmethod
    def _compute_validation_hash(validation_fields: Dict[str, Any]) -> str:
        """
        Compute content-addressed hash of validation fields using multiformats.
        
        Args:
            validation_fields: Fields to hash
            
        Returns:
            CID string if multiformats available, otherwise SHA256 hex
        """
        # Sort fields for deterministic hashing
        sorted_fields = json.dumps(validation_fields, sort_keys=True)
        return BaseAPICache._compute_cid(sorted_fields)
    
    def get(
        self,
        operation: str,
        *args,
        use_cache: bool = True,
        **kwargs
    ) -> Optional[Any]:
        """
        Get a cached value.
        
        Args:
            operation: Operation name
            *args: Positional arguments for cache key
            use_cache: Whether to use cache (default: True)
            **kwargs: Keyword arguments for cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not use_cache:
            with self._lock:
                self._stats["misses"] += 1
                self._stats["api_calls_made"] += 1
            return None
        
        cache_key = self.make_cache_key(operation, *args, **kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check if entry is expired
                if entry.is_expired():
                    del self._cache[cache_key]
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    self._stats["api_calls_made"] += 1
                    return None
                
                # Entry is valid
                self._stats["hits"] += 1
                self._stats["api_calls_saved"] += 1
                return entry.data
            
            # Cache miss
            self._stats["misses"] += 1
            self._stats["api_calls_made"] += 1
            return None
    
    def put(
        self,
        operation: str,
        value: Any,
        *args,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Store a value in the cache.
        
        Args:
            operation: Operation name
            value: Value to cache
            *args: Positional arguments for cache key
            ttl: Time-to-live in seconds (default: operation-specific or default_ttl)
            metadata: Additional metadata to store with the entry
            **kwargs: Keyword arguments for cache key
        """
        if ttl is None:
            ttl = self.get_default_ttl_for_operation(operation)
        
        cache_key = self.make_cache_key(operation, *args, **kwargs)
        
        # Extract validation fields for content-addressed caching
        validation_fields = self.extract_validation_fields(operation, value)
        content_hash = None
        if validation_fields:
            content_hash = self._compute_validation_hash(validation_fields)
        
        entry = CacheEntry(
            data=value,
            timestamp=time.time(),
            ttl=ttl,
            content_hash=content_hash,
            validation_fields=validation_fields,
            metadata=metadata
        )
        
        with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self.max_cache_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
                # Remove from CID index
                self._cid_index.remove(oldest_key)
            
            self._cache[cache_key] = entry
            
            # Add to CID index
            self._cid_index.add(
                cache_key,
                operation,
                metadata={
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                    "cache_name": self.cache_name
                }
            )
        
        # Save to disk if persistence is enabled
        if self.enable_persistence:
            self._save_to_disk(cache_key, entry)
    
    def invalidate(self, operation: str, *args, **kwargs) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            operation: Operation name
            *args: Positional arguments for cache key
            **kwargs: Keyword arguments for cache key
            
        Returns:
            True if entry was found and invalidated, False otherwise
        """
        cache_key = self.make_cache_key(operation, *args, **kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                
                # Remove from CID index
                self._cid_index.remove(cache_key)
                
                # Delete from disk if persistence is enabled
                if self.enable_persistence:
                    cache_file = self.cache_dir / f"{cache_key}.json"
                    cache_file.unlink(missing_ok=True)
                
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., 'list_repos' matches all list_repos calls)
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            
            for key in keys_to_delete:
                del self._cache[key]
                count += 1
                
                # Remove from CID index
                self._cid_index.remove(key)
                
                # Delete from disk if persistence is enabled
                if self.enable_persistence:
                    cache_file = self.cache_dir / f"{key}.json"
                    cache_file.unlink(missing_ok=True)
        
        return count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            # Clear CID index for this cache's entries
            for key in self._cache.keys():
                self._cid_index.remove(key)
            
            self._cache.clear()
            
            # Clear disk cache if persistence is enabled
            if self.enable_persistence:
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            # Get CID index stats
            cid_stats = self._cid_index.get_stats()
            
            return {
                "cache_name": self.cache_name,
                "total_requests": total_requests,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "expirations": self._stats["expirations"],
                "evictions": self._stats["evictions"],
                "api_calls_saved": self._stats["api_calls_saved"],
                "api_calls_made": self._stats["api_calls_made"],
                "peer_hits": self._stats["peer_hits"],
                "enable_persistence": self.enable_persistence,
                "enable_p2p": self.enable_p2p,
                "cid_index": cid_stats
            }
    
    def _save_to_disk(self, cache_key: str, entry: CacheEntry) -> None:
        """Save a cache entry to disk."""
        if not self.enable_persistence:
            return
        
        try:
            # Use CID as filename (safe for filesystems)
            # Replace : with _ for Windows compatibility
            safe_key = cache_key.replace(":", "_")
            cache_file = self.cache_dir / f"{safe_key}.json"
            
            data = {
                "data": entry.data,
                "timestamp": entry.timestamp,
                "ttl": entry.ttl,
                "content_hash": entry.content_hash,
                "validation_fields": entry.validation_fields,
                "metadata": entry.metadata,
                "cid": cache_key  # Store the original CID
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache entries from disk."""
        if not self.enable_persistence or not self.cache_dir.exists():
            return
        
        loaded = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                entry = CacheEntry(
                    data=data["data"],
                    timestamp=data["timestamp"],
                    ttl=data["ttl"],
                    content_hash=data.get("content_hash"),
                    validation_fields=data.get("validation_fields"),
                    metadata=data.get("metadata")
                )
                
                # Only load non-expired entries
                if not entry.is_expired():
                    cache_key = cache_file.stem
                    self._cache[cache_key] = entry
                    loaded += 1
                    
                    # Add to CID index
                    # Extract operation from filename if possible (format: cid)
                    # Since we're using CIDs, we need to check if metadata exists
                    try:
                        if "operation" in data:
                            self._cid_index.add(
                                cache_key,
                                data["operation"],
                                metadata={
                                    "timestamp": entry.timestamp,
                                    "ttl": entry.ttl,
                                    "cache_name": self.cache_name
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Could not add to CID index during load: {e}")
                else:
                    # Delete expired cache file
                    cache_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to load cache entry from {cache_file}: {e}")
        
        if loaded > 0:
            logger.info(f"✓ Loaded {loaded} cache entries from disk")
    
    def shutdown(self) -> None:
        """Gracefully shutdown and save cache."""
        if self.enable_persistence:
            try:
                with self._lock:
                    for cache_key, entry in self._cache.items():
                        self._save_to_disk(cache_key, entry)
                logger.info(f"✓ Saved {len(self._cache)} cache entries to disk")
            except Exception as e:
                logger.warning(f"Failed to save cache during shutdown: {e}")
    
    def find_by_cid_prefix(self, cid_prefix: str, max_results: int = 100) -> List[Any]:
        """
        Find cache entries by CID prefix.
        
        Useful for exploring related cache entries or debugging.
        
        Args:
            cid_prefix: CID prefix to search for
            max_results: Maximum number of results
            
        Returns:
            List of cached values matching the prefix
        """
        matching_cids = self._cid_index.find_by_prefix(cid_prefix, max_results)
        
        results = []
        with self._lock:
            for cid in matching_cids:
                if cid in self._cache:
                    entry = self._cache[cid]
                    if not entry.is_expired():
                        results.append(entry.data)
        
        return results
    
    def find_by_operation(self, operation: str) -> List[Any]:
        """
        Find all cached values for a given operation.
        
        Args:
            operation: Operation name
            
        Returns:
            List of cached values for this operation
        """
        matching_cids = self._cid_index.find_by_operation(operation)
        
        results = []
        with self._lock:
            for cid in matching_cids:
                if cid in self._cache:
                    entry = self._cache[cid]
                    if not entry.is_expired():
                        results.append(entry.data)
        
        return results


# Global cache registry for managing multiple cache instances
_cache_registry: Dict[str, BaseAPICache] = {}
_registry_lock = Lock()


def register_cache(cache_name: str, cache: BaseAPICache) -> None:
    """
    Register a cache instance globally.
    
    Args:
        cache_name: Unique name for the cache
        cache: Cache instance to register
    """
    with _registry_lock:
        _cache_registry[cache_name] = cache
        logger.info(f"✓ Registered cache: {cache_name}")


def get_cache(cache_name: str) -> Optional[BaseAPICache]:
    """
    Get a registered cache instance.
    
    Args:
        cache_name: Name of the cache to retrieve
        
    Returns:
        Cache instance or None if not found
    """
    with _registry_lock:
        return _cache_registry.get(cache_name)


def get_all_caches() -> Dict[str, BaseAPICache]:
    """
    Get all registered cache instances.
    
    Returns:
        Dictionary of cache_name -> cache instance
    """
    with _registry_lock:
        return dict(_cache_registry)


def shutdown_all_caches() -> None:
    """Shutdown all registered caches."""
    with _registry_lock:
        for cache_name, cache in _cache_registry.items():
            try:
                cache.shutdown()
                logger.info(f"✓ Shutdown cache: {cache_name}")
            except Exception as e:
                logger.warning(f"Failed to shutdown cache {cache_name}: {e}")
        _cache_registry.clear()
