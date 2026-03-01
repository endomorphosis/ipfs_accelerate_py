"""Result cache primitive for MCP++ runtime integration."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policy types."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return self.age_seconds > self.ttl

    def access(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract cache backend contract."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        raise NotImplementedError

    @abstractmethod
    async def put(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend using OrderedDict."""

    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._total_bytes = 0

    async def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
                entry.access()
            return entry

    async def put(self, entry: CacheEntry) -> None:
        with self._lock:
            if entry.key in self._cache:
                old_entry = self._cache[entry.key]
                self._total_bytes -= old_entry.size_bytes
                del self._cache[entry.key]

            self._cache[entry.key] = entry
            self._total_bytes += entry.size_bytes
            await self._evict_if_needed()

    async def delete(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry is None:
                return False
            self._total_bytes -= entry.size_bytes
            return True

    async def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._total_bytes = 0
            return count

    async def _evict_if_needed(self) -> None:
        while (len(self._cache) > self.max_size) or (self._total_bytes > self.max_memory_bytes):
            if not self._cache:
                break
            _, entry = self._cache.popitem(last=False)
            self._total_bytes -= entry.size_bytes

    async def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_bytes": self._total_bytes,
                "max_bytes": self.max_memory_bytes,
                "utilization": (len(self._cache) / self.max_size) if self.max_size > 0 else 0,
            }


class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend using pickle."""

    def __init__(self, cache_dir: Path, max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Path] = {}
        self._lock = RLock()

    async def _load_index(self) -> None:
        with self._lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, "rb") as f:
                        entry = pickle.load(f)
                    self._index[entry.key] = cache_file
                except Exception as exc:
                    logger.warning("Skipping unreadable cache file %s: %s", cache_file, exc)

    def _get_cache_path(self, key: str) -> Path:
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.cache"

    async def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            cache_path = self._index.get(key)
            if cache_path is None or not cache_path.exists():
                return None

            try:
                with open(cache_path, "rb") as f:
                    entry = pickle.load(f)
                entry.access()
                with open(cache_path, "wb") as f:
                    pickle.dump(entry, f)
                return entry
            except Exception as exc:
                logger.error("Error loading cache entry %s: %s", key, exc)
                return None

    async def put(self, entry: CacheEntry) -> None:
        with self._lock:
            cache_path = self._get_cache_path(entry.key)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(entry, f)
                self._index[entry.key] = cache_path
                await self._evict_if_needed()
            except Exception as exc:
                logger.error("Error saving cache entry %s: %s", entry.key, exc)

    async def delete(self, key: str) -> bool:
        with self._lock:
            cache_path = self._index.pop(key, None)
            if cache_path and cache_path.exists():
                try:
                    cache_path.unlink()
                    return True
                except Exception as exc:
                    logger.error("Error deleting cache entry %s: %s", key, exc)
            return False

    async def clear(self) -> int:
        with self._lock:
            count = 0
            for cache_path in list(self._index.values()):
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        count += 1
                    except Exception as exc:
                        logger.error("Error deleting cache file %s: %s", cache_path, exc)
            self._index.clear()
            return count

    async def _evict_if_needed(self) -> None:
        with self._lock:
            if len(self._index) <= self.max_size:
                return

            entries = []
            for key, path in self._index.items():
                if path.exists():
                    entries.append((key, path.stat().st_mtime))
            entries.sort(key=lambda x: x[1])

            to_remove = len(self._index) - self.max_size
            for key, _ in entries[:to_remove]:
                await self.delete(key)

    async def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_size = sum(path.stat().st_size for path in self._index.values() if path.exists())
            return {
                "backend": "disk",
                "size": len(self._index),
                "max_size": self.max_size,
                "total_bytes": total_size,
                "cache_dir": str(self.cache_dir),
            }


class ResultCache:
    """Result caching with TTL, invalidation, and backend abstraction."""

    def __init__(
        self,
        backend: CacheBackend,
        default_ttl: float = 3600.0,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        self.backend = backend
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = RLock()

    def _compute_key(self, task_id: str, inputs: Optional[Dict[str, Any]] = None) -> str:
        if inputs:
            inputs_str = json.dumps(inputs, sort_keys=True)
            key_str = f"{task_id}:{inputs_str}"
        else:
            key_str = task_id
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024

    async def get(self, task_id: str, inputs: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        key = self._compute_key(task_id, inputs)
        entry = await self.backend.get(key)

        if entry is None:
            with self._lock:
                self.misses += 1
            return None

        if entry.is_expired:
            await self.backend.delete(key)
            with self._lock:
                self.misses += 1
            return None

        with self._lock:
            self.hits += 1
        return entry.value

    async def put(
        self,
        task_id: str,
        value: Any,
        ttl: Optional[float] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = self._compute_key(task_id, inputs)
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            ttl=(self.default_ttl if ttl is None else ttl),
            size_bytes=self._estimate_size(value),
        )
        await self.backend.put(entry)

    async def invalidate(self, task_id: str, inputs: Optional[Dict[str, Any]] = None) -> bool:
        key = self._compute_key(task_id, inputs)
        return await self.backend.delete(key)

    async def clear(self) -> int:
        return await self.backend.clear()

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self.hits + self.misses
            return (self.hits / total) if total else 0.0

    async def get_stats(self) -> Dict[str, Any]:
        backend_stats = await self.backend.get_stats()
        with self._lock:
            return {
                **backend_stats,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hit_rate,
                "evictions": self.evictions,
                "default_ttl": self.default_ttl,
                "eviction_policy": self.eviction_policy.value,
            }


__all__ = [
    "EvictionPolicy",
    "CacheEntry",
    "CacheBackend",
    "MemoryCacheBackend",
    "DiskCacheBackend",
    "ResultCache",
]
