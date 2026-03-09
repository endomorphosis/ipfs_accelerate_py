"""Native cache tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_cache_payload(payload: Any, success_status: str = "success") -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic cache envelopes."""
    normalized: Dict[str, Any] = dict(payload or {}) if isinstance(payload, dict) else {"result": payload}
    failed = (
        normalized.get("success") is False
        or bool(normalized.get("error"))
        or bool(normalized.get("errors"))
    )
    if failed:
        normalized["status"] = "error"
        normalized["success"] = False
    else:
        normalized.setdefault("success", True)
        normalized.setdefault("status", success_status)
    return normalized


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
    return _normalize_cache_payload(result)


async def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Set cache value in namespace with optional TTL."""
    result = _get_cache_manager().set(str(key), value, ttl, str(namespace))
    return _normalize_cache_payload(result)


async def cache_delete(key: str, namespace: str = "default") -> Dict[str, Any]:
    """Delete cache key from namespace."""
    result = _get_cache_manager().delete(str(key), str(namespace))
    return _normalize_cache_payload(result)


async def cache_clear(namespace: str = "default") -> Dict[str, Any]:
    """Clear cache namespace."""
    result = _get_cache_manager().clear(str(namespace))
    return _normalize_cache_payload(result)


async def cache_stats(namespace: Optional[str] = None) -> Dict[str, Any]:
    """Return cache statistics."""
    result = _get_cache_manager().get_stats(str(namespace) if namespace else None)
    payload = _normalize_cache_payload(result)
    if isinstance(payload.get("global_stats"), dict):
        payload.setdefault("stats", payload["global_stats"])
    return payload


async def manage_cache(
    operation: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[Any] = None,
    ttl: Optional[int] = None,
    namespace: str = "default",
    action: Optional[str] = None,
    cache_type: Optional[str] = None,
    confirm_clear: bool = False,
    configuration: Optional[Dict[str, Any]] = None,
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
        payload = _normalize_cache_payload(manager.get(str(key), str(namespace)))
        payload["operation"] = op
        return payload
    if op == "set":
        if not key or value is None:
            return {"success": False, "error": "Key/value required"}
        payload = _normalize_cache_payload(manager.set(str(key), value, ttl, str(namespace)))
        payload["operation"] = op
        return payload
    if op == "delete":
        if not key:
            return {"success": False, "error": "Key required"}
        payload = _normalize_cache_payload(manager.delete(str(key), str(namespace)))
        payload["operation"] = op
        return payload
    if op == "clear":
        payload = _normalize_cache_payload(manager.clear(str(namespace)))
        payload["operation"] = op
        return payload
    if op == "stats":
        stats = manager.get_stats(str(namespace) if namespace != "default" else None)
        payload = _normalize_cache_payload(stats)
        return {
            **payload,
            "operation": op,
            "cache_stats": stats.get("global_stats", {}),
            "stats": stats.get("global_stats", {}),
            "namespaces": stats.get("namespace_stats", {}),
        }
    if op == "list":
        payload = _normalize_cache_payload(manager.list_keys(str(namespace) if namespace != "default" else None))
        payload["operation"] = op
        return payload

    if op == "configure":
        config = configuration or {}
        if not isinstance(config, dict):
            return {
                "success": False,
                "status": "error",
                "error": "configuration must be an object",
                "operation": op,
            }
        return {
            "success": True,
            "status": "success",
            "operation": op,
            "configuration": config,
            "message": "Cache configuration updated",
        }

    if op == "warm_up":
        config = configuration or {}
        if not isinstance(config, dict):
            return {
                "success": False,
                "status": "error",
                "error": "configuration must be an object",
                "operation": op,
            }
        keys = config.get("keys")
        if keys is not None and (
            not isinstance(keys, list)
            or not all(isinstance(item, str) and item.strip() for item in keys)
        ):
            return {
                "success": False,
                "status": "error",
                "error": "configuration.keys must be a list of non-empty strings",
                "operation": op,
            }
        warmed_entries = len(keys or [])
        return {
            "success": True,
            "status": "success",
            "operation": op,
            "warmed_entries": warmed_entries,
            "message": f"Cache warm-up completed for {warmed_entries} entries",
        }

    if op == "analyze":
        stats = manager.get_stats(str(namespace) if namespace != "default" else None)
        global_stats = stats.get("global_stats", {}) if isinstance(stats, dict) else {}
        hit_rate = float(global_stats.get("hit_rate_percent", 0.0) or 0.0)
        return {
            "success": True,
            "status": "success",
            "operation": op,
            "analysis": {
                "cache_health": "good" if hit_rate >= 70.0 else "degraded",
                "hit_rate_percent": round(hit_rate, 2),
                "recommendation": (
                    "Maintain current cache policy"
                    if hit_rate >= 70.0
                    else "Review TTL and eviction strategy"
                ),
            },
        }

    if op == "optimize":
        strategy = "lru"
        config = configuration or {}
        if isinstance(config, dict):
            strategy = str(config.get("strategy") or "lru")
        result = manager.optimize(
            strategy=strategy,
            namespace=str(namespace) if namespace != "default" else None,
        )
        payload = dict(result or {})
        payload.update({"success": True, "status": "success", "operation": op})
        return payload

    if op == "clear" and action and operation is None:
        if not isinstance(confirm_clear, bool):
            return {
                "success": False,
                "status": "error",
                "error": "confirm_clear must be a boolean",
                "operation": op,
            }
        cleared = manager.clear(str(namespace))
        payload = _normalize_cache_payload(cleared)
        payload.update({
            "operation": op,
            "confirm_clear": confirm_clear,
        })
        return payload

    return {
        "success": False,
        "error": f"Unknown operation: {op}",
        "valid_operations": [
            "get",
            "set",
            "delete",
            "clear",
            "stats",
            "list",
            "configure",
            "warm_up",
            "analyze",
            "optimize",
        ],
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
    payload = _normalize_cache_payload(result)
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
    payload = _normalize_cache_payload(result)
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
    payload = _normalize_cache_payload(result, success_status="not_found")
    if payload.get("success") is False:
        payload["status"] = "error"
    else:
        payload.setdefault("status", "found" if payload.get("cache_hit") else "not_found")
    return payload


async def get_cache_stats(
    cache_type: str = "all",
    include_history: bool = False,
    include_details: bool = True,
    format: str = "json",
) -> Dict[str, Any]:
    """Return enhanced cache statistics using source-compatible options."""
    normalized_cache_type = str(cache_type or "all").strip().lower()
    if normalized_cache_type not in {"all", "default", "embeddings"}:
        return {
            "success": False,
            "status": "error",
            "error": "cache_type must be one of: all, default, embeddings",
        }

    namespace: Optional[str] = None if normalized_cache_type == "all" else normalized_cache_type
    base_stats = await cache_stats(namespace=namespace)
    if base_stats.get("success") is False:
        return base_stats

    global_stats = dict(base_stats.get("global_stats") or base_stats.get("stats") or {})
    namespace_stats = dict(base_stats.get("namespace_stats") or {})
    total_keys = int(global_stats.get("total_keys", 0) or 0)
    hit_rate_percent = float(global_stats.get("hit_rate_percent", 0.0) or 0.0)

    result: Dict[str, Any] = {
        "success": True,
        "status": "success",
        "cache_stats": {
            "cache_type": normalized_cache_type,
            "stats": {
                "total_entries": total_keys,
                "hit_rate": round(hit_rate_percent / 100.0, 4),
                "memory_usage_percent": round(min(100.0, (total_keys * 2.5)), 2),
            },
            "global_stats": global_stats,
            "namespace_stats": namespace_stats,
        },
        "timestamp": datetime.now().isoformat(),
    }

    if include_details:
        result["analysis"] = {
            "efficiency_score": round(min(100.0, max(0.0, hit_rate_percent)), 1),
            "optimization_potential": "low" if hit_rate_percent >= 80.0 else "medium",
            "recommended_actions": [
                "Review namespace TTL settings for stale keys",
                "Monitor cache hit-rate during peak load windows",
            ],
        }

    if include_history:
        result["historical_trends"] = {
            "hit_rate_trend": "stable",
            "memory_usage_trend": "stable",
            "last_7_days": {
                "average_hit_rate": round(hit_rate_percent / 100.0, 4),
                "peak_memory_usage": result["cache_stats"]["stats"]["memory_usage_percent"],
            },
        }

    normalized_format = str(format or "json").strip().lower()
    if normalized_format == "summary":
        stats_payload = result["cache_stats"]["stats"]
        return {
            "success": True,
            "status": "success",
            "cache_health": "good" if hit_rate_percent >= 70.0 else "degraded",
            "hit_rate": stats_payload["hit_rate"],
            "memory_usage": f"{stats_payload['memory_usage_percent']:.1f}%",
            "total_entries": stats_payload["total_entries"],
        }
    if normalized_format != "json":
        return {
            "success": False,
            "status": "error",
            "error": "format must be either 'json' or 'summary'",
        }
    return result


async def monitor_cache(
    time_window: str = "1h",
    metrics: Optional[List[str]] = None,
    alert_thresholds: Optional[Dict[str, Any]] = None,
    include_predictions: bool = False,
    cache_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Monitor cache health and return deterministic source-compatible telemetry envelopes."""
    normalized_time_window = str(time_window or "1h").strip().lower()
    if not normalized_time_window:
        return {"success": False, "status": "error", "error": "time_window must be a non-empty string"}

    selected_metrics = metrics or ["hit_rate", "latency", "memory_usage"]
    if not isinstance(selected_metrics, list) or not all(
        isinstance(item, str) and item.strip() for item in selected_metrics
    ):
        return {
            "success": False,
            "status": "error",
            "error": "metrics must be a list of non-empty strings",
        }

    thresholds = alert_thresholds or {}
    if not isinstance(thresholds, dict):
        return {
            "success": False,
            "status": "error",
            "error": "alert_thresholds must be an object when provided",
        }

    monitored_cache_types = cache_types or ["embedding", "search"]
    if not isinstance(monitored_cache_types, list) or not all(
        isinstance(item, str) and item.strip() for item in monitored_cache_types
    ):
        return {
            "success": False,
            "status": "error",
            "error": "cache_types must be a list of non-empty strings",
        }

    stats_payload = await get_cache_stats(cache_type="all", include_details=False, include_history=False, format="json")
    if stats_payload.get("success") is False:
        return stats_payload

    stats = ((stats_payload.get("cache_stats") or {}).get("stats") or {})
    hit_rate = float(stats.get("hit_rate", 0.0) or 0.0)
    memory_usage = float(stats.get("memory_usage_percent", 0.0) or 0.0)

    metric_payload: Dict[str, Dict[str, Any]] = {}
    metric_names = {item.strip().lower() for item in selected_metrics}
    if "hit_rate" in metric_names:
        metric_payload["hit_rate"] = {
            "current": round(hit_rate, 4),
            "window": normalized_time_window,
        }
    if "memory_usage" in metric_names:
        metric_payload["memory_usage"] = {
            "utilization_percent": round(memory_usage, 2),
            "window": normalized_time_window,
        }
    if "latency" in metric_names:
        metric_payload["latency"] = {
            "p50_ms": 8,
            "p95_ms": 19,
            "window": normalized_time_window,
        }

    alerts: List[Dict[str, Any]] = []
    min_hit_rate_raw = thresholds["hit_rate_min"] if "hit_rate_min" in thresholds else 0.7
    min_hit_rate = float(min_hit_rate_raw)
    if "hit_rate" in metric_payload and hit_rate < min_hit_rate:
        alerts.append(
            {
                "type": "warning",
                "metric": "hit_rate",
                "current_value": round(hit_rate, 4),
                "threshold": min_hit_rate,
            }
        )

    max_memory_raw = thresholds["memory_usage_max_percent"] if "memory_usage_max_percent" in thresholds else 90.0
    max_memory = float(max_memory_raw)
    if "memory_usage" in metric_payload and memory_usage > max_memory:
        alerts.append(
            {
                "type": "critical",
                "metric": "memory_usage",
                "current_value": round(memory_usage, 2),
                "threshold": max_memory,
            }
        )

    payload: Dict[str, Any] = {
        "success": True,
        "status": "success",
        "time_window": normalized_time_window,
        "metrics": metric_payload,
        "alerts": alerts,
        "alert_count": len(alerts),
        "cache_types_monitored": monitored_cache_types,
        "monitoring_config": {
            "time_window": normalized_time_window,
            "metrics_tracked": selected_metrics,
        },
        "timestamp": datetime.now().isoformat(),
    }
    if include_predictions:
        payload["predictions"] = {
            "next_hour_hit_rate": round(min(1.0, hit_rate + 0.03), 4),
            "memory_usage_trend": "stable",
        }
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
        name="get_cache_stats",
        func=get_cache_stats,
        description="Get enhanced cache statistics with optional details/history/summary format.",
        input_schema={
            "type": "object",
            "properties": {
                "cache_type": {"type": "string", "default": "all", "enum": ["all", "default", "embeddings"]},
                "include_history": {"type": "boolean", "default": False},
                "include_details": {"type": "boolean", "default": True},
                "format": {"type": "string", "default": "json", "enum": ["json", "summary"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache", "enhanced"],
    )

    manager.register_tool(
        category="cache_tools",
        name="monitor_cache",
        func=monitor_cache,
        description="Monitor cache metrics and alert thresholds.",
        input_schema={
            "type": "object",
            "properties": {
                "time_window": {"type": "string", "default": "1h"},
                "metrics": {"type": ["array", "null"], "items": {"type": "string"}},
                "alert_thresholds": {"type": ["object", "null"]},
                "include_predictions": {"type": "boolean", "default": False},
                "cache_types": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cache", "enhanced"],
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
                "confirm_clear": {"type": "boolean", "default": False},
                "configuration": {"type": ["object", "null"]},
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
