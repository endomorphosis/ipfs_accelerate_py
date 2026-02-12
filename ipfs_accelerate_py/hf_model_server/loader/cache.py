"""
LRU cache for loaded models with IPFS backend support.
"""

import anyio
from collections import OrderedDict
from typing import Dict, Optional, Any
import logging
import os

from .types import LoadedModel, ModelStatus

logger = logging.getLogger(__name__)


class ModelCache:
    """LRU cache for loaded models with memory management and IPFS backend support."""
    
    def __init__(self, max_size: int = 10, max_memory_mb: float = 16384, enable_ipfs: bool = None):
        """
        Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache
            max_memory_mb: Maximum memory usage in MB
            enable_ipfs: Enable IPFS backend for model storage (auto-detected if None)
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = anyio.Lock()
        self._hits = 0
        self._misses = 0
        
        # IPFS backend integration
        self.enable_ipfs = enable_ipfs if enable_ipfs is not None else \
                          os.getenv("ENABLE_IPFS_MODEL_CACHE", "").lower() in ("1", "true", "yes")
        self._ipfs_backend = None
        if self.enable_ipfs:
            self._init_ipfs_backend()
    
    def _init_ipfs_backend(self):
        """Initialize IPFS backend for model storage."""
        try:
            from ... import ipfs_backend_router
            self._ipfs_backend = ipfs_backend_router
            logger.info("✓ IPFS backend initialized for model cache")
        except Exception as e:
            logger.warning(f"Failed to initialize IPFS backend: {e}")
            self._ipfs_backend = None
    
    async def get(self, model_id: str) -> Optional[LoadedModel]:
        """
        Get model from cache.
        
        Args:
            model_id: Model identifier
            
        Returns:
            LoadedModel if found, None otherwise
        """
        async with self._lock:
            if model_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_id)
                model = self._cache[model_id]
                model.mark_used()
                self._hits += 1
                logger.debug(f"Cache hit for model: {model_id}")
                return model
            else:
                self._misses += 1
                logger.debug(f"Cache miss for model: {model_id}")
                return None
    
    async def put(self, model_id: str, model: LoadedModel):
        """
        Put model into cache, evicting if necessary.
        
        Args:
            model_id: Model identifier
            model: Loaded model instance
        """
        async with self._lock:
            # Remove if already exists
            if model_id in self._cache:
                del self._cache[model_id]
            
            # Add to cache
            self._cache[model_id] = model
            self._cache.move_to_end(model_id)
            
            # Evict if over limits
            await self._evict_if_needed()
            
            logger.info(f"Cached model: {model_id} (cache size: {len(self._cache)})")
    
    async def remove(self, model_id: str) -> Optional[LoadedModel]:
        """
        Remove model from cache.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Removed model if found, None otherwise
        """
        async with self._lock:
            if model_id in self._cache:
                model = self._cache.pop(model_id)
                logger.info(f"Removed model from cache: {model_id}")
                return model
            return None
    
    async def _evict_if_needed(self):
        """Evict models if cache is over limits."""
        # Evict by count
        while len(self._cache) > self.max_size:
            model_id, model = self._cache.popitem(last=False)
            logger.info(f"Evicted model (count limit): {model_id}")
        
        # Evict by memory
        while self._get_total_memory() > self.max_memory_mb and len(self._cache) > 0:
            model_id, model = self._cache.popitem(last=False)
            logger.info(f"Evicted model (memory limit): {model_id}")
    
    def _get_total_memory(self) -> float:
        """Get total memory usage of cached models."""
        return sum(model.memory_mb for model in self._cache.values())
    
    async def clear(self):
        """Clear all models from cache."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared cache ({count} models)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_mb": self._get_total_memory(),
            "max_memory_mb": self.max_memory_mb,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ipfs_enabled": self.enable_ipfs,
            "ipfs_available": self._ipfs_backend is not None,
        }
    
    async def store_model_to_ipfs(self, model_id: str, model_path: str) -> Optional[str]:
        """
        Store model weights to IPFS and return CID.
        
        Args:
            model_id: Model identifier
            model_path: Path to model weights
            
        Returns:
            CID if successful, None otherwise
        """
        if not self._ipfs_backend:
            logger.debug("IPFS backend not available for model storage")
            return None
        
        try:
            cid = await anyio.to_thread.run_sync(
                self._ipfs_backend.add_path,
                model_path,
                recursive=True,
                pin=True
            )
            logger.info(f"Stored model {model_id} to IPFS: {cid}")
            return cid
        except Exception as e:
            logger.error(f"Failed to store model {model_id} to IPFS: {e}")
            return None
    
    async def retrieve_model_from_ipfs(self, cid: str, output_path: str) -> bool:
        """
        Retrieve model weights from IPFS by CID.
        
        Args:
            cid: Content identifier
            output_path: Where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ipfs_backend:
            logger.debug("IPFS backend not available for model retrieval")
            return False
        
        try:
            await anyio.to_thread.run_sync(
                self._ipfs_backend.get_to_path,
                cid,
                output_path=output_path
            )
            logger.info(f"Retrieved model from IPFS: {cid} -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to retrieve model from IPFS {cid}: {e}")
            return False
    
    def __len__(self) -> int:
        """Get number of cached models."""
        return len(self._cache)
    
    def __contains__(self, model_id: str) -> bool:
        """Check if model is in cache."""
        return model_id in self._cache
