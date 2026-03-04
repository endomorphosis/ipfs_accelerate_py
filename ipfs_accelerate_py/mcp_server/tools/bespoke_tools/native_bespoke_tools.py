"""Native bespoke-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_bespoke_tools_api() -> Dict[str, Any]:
    """Resolve source bespoke-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.bespoke_tools import (  # type: ignore
            cache_stats as _cache_stats,
            system_health as _system_health,
        )

        return {
            "system_health": _system_health,
            "cache_stats": _cache_stats,
        }
    except Exception:
        logger.warning(
            "Source bespoke_tools import unavailable, using fallback bespoke functions"
        )

        async def _system_health_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "health_score": 100.0,
                "fallback": True,
            }

        async def _cache_stats_fallback(namespace: Optional[str] = None) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "namespace": namespace,
                "global_stats": {"hit_rate": 100.0},
                "fallback": True,
            }

        return {
            "system_health": _system_health_fallback,
            "cache_stats": _cache_stats_fallback,
        }


_API = _load_bespoke_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    return {"result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def system_health() -> Dict[str, Any]:
    """Return system health metrics for MCP runtime smoke workflows."""
    try:
        result = _API["system_health"]()
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        return envelope
    except Exception as exc:
        return _error_result(str(exc))


async def cache_stats(namespace: Optional[str] = None) -> Dict[str, Any]:
    """Return cache statistics for optional namespace scope."""
    if namespace is not None and (not isinstance(namespace, str) or not namespace.strip()):
        return _error_result("namespace must be null or a non-empty string", namespace=namespace)

    clean_namespace = namespace.strip() if isinstance(namespace, str) else None

    try:
        result = _API["cache_stats"](namespace=clean_namespace)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("namespace", clean_namespace)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), namespace=clean_namespace)


def register_native_bespoke_tools(manager: Any) -> None:
    """Register native bespoke-tools category tools in unified manager."""
    manager.register_tool(
        category="bespoke_tools",
        name="system_health",
        func=system_health,
        description="Get high-level system health metrics for MCP runtime components.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )

    manager.register_tool(
        category="bespoke_tools",
        name="cache_stats",
        func=cache_stats,
        description="Get cache statistics and performance metrics by namespace.",
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {"type": ["string", "null"], "minLength": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )
