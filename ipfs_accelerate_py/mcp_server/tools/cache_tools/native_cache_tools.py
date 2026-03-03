"""Native cache tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_cache_manager_class() -> Any:
    """Resolve CacheManager from source package with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.caching.cache_manager import CacheManager  # type: ignore

        return CacheManager
    except Exception:
        return None


class _FallbackCacheManager:
    """Dependency-light in-memory fallback for cache operations."""

    def __init__(self) -> None:
        self.storage: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_operations": 0,
        }

    def get(self, key: str, namespace: str = "default") -> Dict[str, Any]:
        cache_key = f"{namespace}:{key}"
        now = datetime.now()
        self.stats["total_operations"] += 1
        if cache_key in self.storage:
            meta = self.metadata.get(cache_key, {})
            expires_at = meta.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) < now:
                self.storage.pop(cache_key, None)
                self.metadata.pop(cache_key, None)
                self.stats["misses"] += 1
                self.stats["evictions"] += 1
                return {"success": True, "key": key, "value": None, "hit": False, "reason": "expired"}
            self.stats["hits"] += 1
            meta["last_accessed"] = now.isoformat()
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            return {"success": True, "key": key, "value": self.storage[cache_key], "hit": True, "metadata": meta}
        self.stats["misses"] += 1
        return {"success": True, "key": key, "value": None, "hit": False, "reason": "not_found"}

    def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "default") -> Dict[str, Any]:
        cache_key = f"{namespace}:{key}"
        now = datetime.now()
        self.stats["total_operations"] += 1
        expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl else None
        self.storage[cache_key] = value
        self.metadata[cache_key] = {
            "created_at": now.isoformat(),
            "expires_at": expires_at,
            "ttl": ttl,
            "namespace": namespace,
            "access_count": 0,
            "size_bytes": len(str(value).encode("utf-8")),
        }
        return {"success": True, "key": key, "stored": True, "expires_at": expires_at, "namespace": namespace}

    def delete(self, key: str, namespace: str = "default") -> Dict[str, Any]:
        cache_key = f"{namespace}:{key}"
        self.stats["total_operations"] += 1
        deleted = cache_key in self.storage
        self.storage.pop(cache_key, None)
        self.metadata.pop(cache_key, None)
        return {"success": True, "key": key, "deleted": deleted, **({"reason": "not_found"} if not deleted else {})}

    def clear(self, namespace: str = "default") -> Dict[str, Any]:
        if namespace == "all":
            keys = list(self.storage.keys())
        else:
            keys = [k for k in self.storage.keys() if k.startswith(f"{namespace}:")]
        for key in keys:
            self.storage.pop(key, None)
            self.metadata.pop(key, None)
        return {"success": True, "namespace": namespace, "keys_cleared": len(keys)}

    def _namespace_stats(self) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for cache_key, meta in self.metadata.items():
            ns = cache_key.split(":", 1)[0] if ":" in cache_key else "default"
            data = grouped.setdefault(ns, {"total_keys": 0, "total_size_bytes": 0})
            data["total_keys"] += 1
            data["total_size_bytes"] += int(meta.get("size_bytes", 0))
        return grouped

    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        total_requests = int(self.stats["hits"]) + int(self.stats["misses"])
        hit_rate = (float(self.stats["hits"]) / float(total_requests) * 100.0) if total_requests else 0.0
        total_size_bytes = sum(int(self.metadata.get(k, {}).get("size_bytes", 0)) for k in self.storage.keys())
        ns_stats = self._namespace_stats()
        payload: Dict[str, Any] = {
            "success": True,
            "global_stats": {
                **self.stats,
                "hit_rate_percent": round(hit_rate, 2),
                "total_keys": len(self.storage),
                "total_size_bytes": total_size_bytes,
                "total_size_mb": round(float(total_size_bytes) / (1024.0 * 1024.0), 2),
                "active_namespaces": len(ns_stats),
            },
            "namespace_stats": ns_stats,
            "timestamp": datetime.now().isoformat(),
        }
        if namespace:
            payload["filtered_namespace"] = namespace
            payload["namespace_data"] = ns_stats.get(namespace, {})
        return payload

    def list_keys(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        records = []
        for cache_key, value in self.storage.items():
            ns, key = cache_key.split(":", 1) if ":" in cache_key else ("default", cache_key)
            if namespace and ns != namespace:
                continue
            records.append({"key": key, "namespace": ns, "value": value, "metadata": self.metadata.get(cache_key, {})})
        return {"success": True, "keys": records, "count": len(records)}

    def optimize(
        self,
        strategy: str = "lru",
        max_size_mb: Optional[int] = None,
        max_age_hours: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = max_size_mb, max_age_hours
        return {
            "success": True,
            "strategy": strategy,
            "keys_evicted": 0,
            "evicted_keys": [],
            "namespace": namespace,
            "timestamp": datetime.now().isoformat(),
        }

    def cache_embeddings(
        self,
        text: str,
        embeddings: list[float],
        model: str = "default",
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        cache_key = f"{model}:{str(abs(hash(text)))[:16]}"
        return self.set(
            key=cache_key,
            value={"text": text, "embeddings": embeddings, "model": model},
            ttl=ttl or 86400,
            namespace="embeddings",
        )

    def get_cached_embeddings(self, text: str, model: str = "default") -> Dict[str, Any]:
        cache_key = f"{model}:{str(abs(hash(text)))[:16]}"
        result = self.get(key=cache_key, namespace="embeddings")
        if result.get("hit"):
            value = result.get("value", {})
            return {
                "success": True,
                "cache_hit": True,
                "embeddings": value.get("embeddings"),
                "model": model,
            }
        return {
            "success": True,
            "cache_hit": False,
            "reason": result.get("reason", "not_found"),
            "model": model,
        }


_CACHE_MANAGER: Optional[Any] = None


def _get_cache_manager() -> Any:
    global _CACHE_MANAGER
    if _CACHE_MANAGER is None:
        cache_manager_class = _load_cache_manager_class()
        if cache_manager_class is None:
            logger.warning("Source CacheManager import unavailable, using fallback manager")
            _CACHE_MANAGER = _FallbackCacheManager()
        else:
            _CACHE_MANAGER = cache_manager_class()
    return _CACHE_MANAGER


async def cache_get(key: str, namespace: str = "default") -> Dict[str, Any]:
    """Get value from cache namespace."""
    result = _get_cache_manager().get(str(key), str(namespace))
    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("success", True) else "error")
    return payload


async def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Set cache value in namespace with optional TTL."""
    result = _get_cache_manager().set(str(key), value, ttl, str(namespace))
    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("success", True) else "error")
    return payload


async def cache_delete(key: str, namespace: str = "default") -> Dict[str, Any]:
    """Delete cache key from namespace."""
    result = _get_cache_manager().delete(str(key), str(namespace))
    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("success", True) else "error")
    return payload


async def cache_clear(namespace: str = "default") -> Dict[str, Any]:
    """Clear cache namespace."""
    result = _get_cache_manager().clear(str(namespace))
    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("success", True) else "error")
    return payload


async def cache_stats(namespace: Optional[str] = None) -> Dict[str, Any]:
    """Return cache statistics."""
    result = _get_cache_manager().get_stats(str(namespace) if namespace else None)
    payload = dict(result or {})
    if isinstance(payload.get("global_stats"), dict):
        payload.setdefault("stats", payload["global_stats"])
    payload.setdefault("status", "success" if payload.get("success", True) else "error")
    return payload


async def manage_cache(
    operation: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[Any] = None,
    ttl: Optional[int] = None,
    namespace: str = "default",
    action: Optional[str] = None,
    cache_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Manage cache operations through a single compatibility entrypoint."""
    op = str(operation or action or "").strip().lower()
    if cache_type and namespace == "default":
        namespace = str(cache_type)

    if not op:
        return {
            "success": False,
            "error": "Operation is required",
            "valid_operations": ["get", "set", "delete", "clear", "stats", "list"],
        }

    manager = _get_cache_manager()
    if op == "get":
        if not key:
            return {"success": False, "error": "Key required"}
        return {**manager.get(str(key), str(namespace)), "operation": op}
    if op == "set":
        if not key or value is None:
            return {"success": False, "error": "Key/value required"}
        return {**manager.set(str(key), value, ttl, str(namespace)), "operation": op}
    if op == "delete":
        if not key:
            return {"success": False, "error": "Key required"}
        return {**manager.delete(str(key), str(namespace)), "operation": op}
    if op == "clear":
        return {**manager.clear(str(namespace)), "operation": op}
    if op == "stats":
        stats = manager.get_stats(str(namespace) if namespace != "default" else None)
        return {
            **stats,
            "operation": op,
            "status": "success",
            "cache_stats": stats.get("global_stats", {}),
            "stats": stats.get("global_stats", {}),
            "namespaces": stats.get("namespace_stats", {}),
        }
    if op == "list":
        return {**manager.list_keys(str(namespace) if namespace != "default" else None), "operation": op}

    return {
        "success": False,
        "error": f"Unknown operation: {op}",
        "valid_operations": ["get", "set", "delete", "clear", "stats", "list"],
    }


async def optimize_cache(
    cache_type: Optional[str] = None,
    strategy: str = "lru",
    max_size_mb: Optional[int] = None,
    max_age_hours: Optional[int] = None,
) -> Dict[str, Any]:
    """Optimize cache entries using a source-compatible optimization contract."""
    manager = _get_cache_manager()
    namespace = str(cache_type) if cache_type and str(cache_type) != "default" else None

    result = manager.optimize(
        strategy=str(strategy or "lru"),
        max_size_mb=max_size_mb,
        max_age_hours=max_age_hours,
        namespace=namespace,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("success", True) else "error")
    payload.setdefault("optimization_strategy", str(strategy or "lru"))
    if max_size_mb is not None:
        payload["max_size_mb"] = max_size_mb
    if max_age_hours is not None:
        payload["max_age_hours"] = max_age_hours
    return payload


async def cache_embeddings(
    text: str,
    embeddings: list[float] | str,
    model: str = "default",
    ttl: Optional[int] = None,
) -> Dict[str, Any]:
    """Cache embeddings for text with source-compatible inputs and envelope."""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return {"success": False, "status": "error", "error": "text is required"}

    parsed_embeddings: list[float]
    if isinstance(embeddings, str):
        try:
            decoded = json.loads(embeddings)
        except Exception as exc:
            return {
                "success": False,
                "status": "error",
                "error": f"invalid embeddings JSON: {exc}",
            }
        if not isinstance(decoded, list):
            return {
                "success": False,
                "status": "error",
                "error": "embeddings must decode to a list",
            }
        parsed_embeddings = decoded
    else:
        parsed_embeddings = embeddings

    result = _get_cache_manager().cache_embeddings(
        text=normalized_text,
        embeddings=parsed_embeddings,
        model=str(model or "default"),
        ttl=ttl,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("success", True) else "error")
    payload.setdefault("cache_operation", "set")
    return payload


async def get_cached_embeddings(text: str, model: str = "default") -> Dict[str, Any]:
    """Return cached embeddings using source-compatible hit/miss envelope."""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return {"success": False, "status": "error", "error": "text is required"}

    result = _get_cache_manager().get_cached_embeddings(
        text=normalized_text,
        model=str(model or "default"),
    )
    payload = dict(result or {})
    payload.setdefault("success", True)
    if payload.get("success") is False:
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "found" if payload.get("cache_hit") else "not_found")
    return payload


def register_native_cache_tools(manager: Any) -> None:
    """Register native cache tools in unified hierarchical manager."""
    manager.register_tool(
        category="cache_tools",
        name="cache_get",
        func=cache_get,
        description="Get cache value from namespace.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "namespace": {"type": "string"},
            },
            "required": ["key"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="cache_set",
        func=cache_set,
        description="Set cache value in namespace with optional TTL.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {},
                "ttl": {"type": ["integer", "null"]},
                "namespace": {"type": "string"},
            },
            "required": ["key", "value"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="cache_delete",
        func=cache_delete,
        description="Delete cache key from namespace.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "namespace": {"type": "string"},
            },
            "required": ["key"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="cache_clear",
        func=cache_clear,
        description="Clear cache namespace.",
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="cache_stats",
        func=cache_stats,
        description="Get cache statistics.",
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="manage_cache",
        func=manage_cache,
        description="Manage cache operations through compatibility wrapper.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": ["string", "null"]},
                "action": {"type": ["string", "null"]},
                "key": {"type": ["string", "null"]},
                "value": {},
                "ttl": {"type": ["integer", "null"]},
                "namespace": {"type": "string"},
                "cache_type": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="optimize_cache",
        func=optimize_cache,
        description="Optimize cache according to strategy and optional size/age constraints.",
        input_schema={
            "type": "object",
            "properties": {
                "cache_type": {"type": ["string", "null"]},
                "strategy": {"type": "string", "default": "lru"},
                "max_size_mb": {"type": ["integer", "null"]},
                "max_age_hours": {"type": ["integer", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="cache_embeddings",
        func=cache_embeddings,
        description="Cache embeddings for text input.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "embeddings": {"oneOf": [{"type": "array", "items": {"type": "number"}}, {"type": "string"}]},
                "model": {"type": "string", "default": "default"},
                "ttl": {"type": ["integer", "null"]},
            },
            "required": ["text", "embeddings"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )

    manager.register_tool(
        category="cache_tools",
        name="get_cached_embeddings",
        func=get_cached_embeddings,
        description="Retrieve cached embeddings for text input.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "model": {"type": "string", "default": "default"},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache"],
    )
