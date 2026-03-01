"""Native p2p-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_p2p_tools_api() -> Dict[str, Any]:
    """Resolve source p2p-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.p2p_tools.p2p_tools import (  # type: ignore
            p2p_cache_get as _p2p_cache_get,
            p2p_cache_set as _p2p_cache_set,
            p2p_service_status as _p2p_service_status,
            p2p_task_get as _p2p_task_get,
            p2p_task_submit as _p2p_task_submit,
        )

        return {
            "p2p_service_status": _p2p_service_status,
            "p2p_cache_get": _p2p_cache_get,
            "p2p_cache_set": _p2p_cache_set,
            "p2p_task_submit": _p2p_task_submit,
            "p2p_task_get": _p2p_task_get,
        }
    except Exception:
        logger.warning("Source p2p_tools import unavailable, using fallback p2p-tools functions")

        def _status_fallback(include_peers: bool = True, peers_limit: int = 50) -> Dict[str, Any]:
            _ = include_peers, peers_limit
            return {"ok": True, "service": {}, "peers": []}

        def _cache_get_fallback(key: str) -> Dict[str, Any]:
            return {"ok": True, "key": str(key), "hit": False, "value": None}

        def _cache_set_fallback(key: str, value: Any, ttl_s: Optional[float] = None) -> Dict[str, Any]:
            _ = value, ttl_s
            return {"ok": True, "key": str(key)}

        def _task_submit_fallback(task_type: str, payload: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
            _ = task_type, payload, model_name
            return {"ok": True, "task_id": "fallback-task-id"}

        def _task_get_fallback(task_id: str) -> Dict[str, Any]:
            return {"ok": False, "error": "task_not_found", "task_id": str(task_id)}

        return {
            "p2p_service_status": _status_fallback,
            "p2p_cache_get": _cache_get_fallback,
            "p2p_cache_set": _cache_set_fallback,
            "p2p_task_submit": _task_submit_fallback,
            "p2p_task_get": _task_get_fallback,
        }


_API = _load_p2p_tools_api()


async def p2p_service_status(
    include_peers: bool = True,
    peers_limit: int = 50,
) -> Dict[str, Any]:
    """Return local P2P service status and peers."""
    result = _API["p2p_service_status"](include_peers=include_peers, peers_limit=peers_limit)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_cache_get(key: str) -> Dict[str, Any]:
    """Get a value from local P2P shared cache."""
    result = _API["p2p_cache_get"](key=key)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_cache_set(
    key: str,
    value: Any,
    ttl_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Set a value in local P2P shared cache."""
    result = _API["p2p_cache_set"](key=key, value=value, ttl_s=ttl_s)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_task_submit(
    task_type: str,
    payload: Dict[str, Any],
    model_name: str = "",
) -> Dict[str, Any]:
    """Submit a task to local P2P task queue."""
    result = _API["p2p_task_submit"](task_type=task_type, payload=payload, model_name=model_name)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_task_get(task_id: str) -> Dict[str, Any]:
    """Get task status from local P2P task queue."""
    result = _API["p2p_task_get"](task_id=task_id)
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_p2p_tools_category(manager: Any) -> None:
    """Register native p2p-tools category tools in unified hierarchical manager."""
    manager.register_tool(
        category="p2p_tools",
        name="p2p_service_status",
        func=p2p_service_status,
        description="Get local P2P service status and peer list.",
        input_schema={
            "type": "object",
            "properties": {
                "include_peers": {"type": "boolean"},
                "peers_limit": {"type": "integer"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_cache_get",
        func=p2p_cache_get,
        description="Get a value from local P2P cache.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_cache_set",
        func=p2p_cache_set,
        description="Set a value in local P2P cache.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {},
                "ttl_s": {"type": ["number", "null"]},
            },
            "required": ["key", "value"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_task_submit",
        func=p2p_task_submit,
        description="Submit a task to the local P2P task queue.",
        input_schema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "payload": {"type": "object"},
                "model_name": {"type": "string"},
            },
            "required": ["task_type", "payload"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_task_get",
        func=p2p_task_get,
        description="Get task status from local P2P task queue.",
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )
