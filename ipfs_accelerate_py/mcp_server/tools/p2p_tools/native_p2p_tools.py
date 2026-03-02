"""Native p2p-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_p2p_tools_api() -> Dict[str, Any]:
    """Resolve source p2p-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.p2p_tools.p2p_tools import (  # type: ignore
            p2p_cache_delete as _p2p_cache_delete,
            p2p_cache_get as _p2p_cache_get,
            p2p_cache_has as _p2p_cache_has,
            p2p_cache_set as _p2p_cache_set,
            p2p_remote_cache_delete as _p2p_remote_cache_delete,
            p2p_remote_cache_get as _p2p_remote_cache_get,
            p2p_remote_cache_has as _p2p_remote_cache_has,
            p2p_remote_cache_set as _p2p_remote_cache_set,
            p2p_remote_call_tool as _p2p_remote_call_tool,
            p2p_remote_status as _p2p_remote_status,
            p2p_remote_submit_task as _p2p_remote_submit_task,
            p2p_service_status as _p2p_service_status,
            p2p_task_delete as _p2p_task_delete,
            p2p_task_get as _p2p_task_get,
            p2p_task_submit as _p2p_task_submit,
        )

        return {
            "p2p_service_status": _p2p_service_status,
            "p2p_cache_get": _p2p_cache_get,
            "p2p_cache_has": _p2p_cache_has,
            "p2p_cache_set": _p2p_cache_set,
            "p2p_cache_delete": _p2p_cache_delete,
            "p2p_task_submit": _p2p_task_submit,
            "p2p_task_get": _p2p_task_get,
            "p2p_task_delete": _p2p_task_delete,
            "p2p_remote_status": _p2p_remote_status,
            "p2p_remote_call_tool": _p2p_remote_call_tool,
            "p2p_remote_cache_get": _p2p_remote_cache_get,
            "p2p_remote_cache_set": _p2p_remote_cache_set,
            "p2p_remote_cache_has": _p2p_remote_cache_has,
            "p2p_remote_cache_delete": _p2p_remote_cache_delete,
            "p2p_remote_submit_task": _p2p_remote_submit_task,
        }
    except Exception:
        logger.warning("Source p2p_tools import unavailable, using fallback p2p-tools functions")

        def _status_fallback(include_peers: bool = True, peers_limit: int = 50) -> Dict[str, Any]:
            _ = include_peers, peers_limit
            return {"ok": True, "service": {}, "peers": []}

        def _cache_get_fallback(key: str) -> Dict[str, Any]:
            return {"ok": True, "key": str(key), "hit": False, "value": None}

        def _cache_has_fallback(key: str) -> Dict[str, Any]:
            return {"ok": True, "key": str(key), "hit": False}

        def _cache_set_fallback(key: str, value: Any, ttl_s: Optional[float] = None) -> Dict[str, Any]:
            _ = value, ttl_s
            return {"ok": True, "key": str(key)}

        def _cache_delete_fallback(key: str) -> Dict[str, Any]:
            return {"ok": True, "key": str(key), "deleted": False}

        def _task_submit_fallback(task_type: str, payload: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
            _ = task_type, payload, model_name
            return {"ok": True, "task_id": "fallback-task-id"}

        def _task_get_fallback(task_id: str) -> Dict[str, Any]:
            return {"ok": False, "error": "task_not_found", "task_id": str(task_id)}

        def _task_delete_fallback(task_id: str) -> Dict[str, Any]:
            return {"ok": True, "task_id": str(task_id), "deleted": False}

        async def _remote_status_fallback(**_kwargs: Any) -> Dict[str, Any]:
            return {"ok": False, "error": "remote_p2p_unavailable"}

        async def _remote_call_fallback(**_kwargs: Any) -> Dict[str, Any]:
            return {"ok": False, "error": "remote_p2p_unavailable"}

        async def _remote_cache_get_fallback(**_kwargs: Any) -> Dict[str, Any]:
            return {"ok": False, "error": "remote_p2p_unavailable"}

        async def _remote_cache_set_fallback(**_kwargs: Any) -> Dict[str, Any]:
            return {"ok": False, "error": "remote_p2p_unavailable"}

        async def _remote_cache_has_fallback(**_kwargs: Any) -> Dict[str, Any]:
            return {"ok": False, "error": "remote_p2p_unavailable"}

        async def _remote_cache_delete_fallback(**_kwargs: Any) -> Dict[str, Any]:
            return {"ok": False, "error": "remote_p2p_unavailable"}

        async def _remote_submit_fallback(**_kwargs: Any) -> Dict[str, Any]:
            return {"ok": False, "error": "remote_p2p_unavailable"}

        return {
            "p2p_service_status": _status_fallback,
            "p2p_cache_get": _cache_get_fallback,
            "p2p_cache_has": _cache_has_fallback,
            "p2p_cache_set": _cache_set_fallback,
            "p2p_cache_delete": _cache_delete_fallback,
            "p2p_task_submit": _task_submit_fallback,
            "p2p_task_get": _task_get_fallback,
            "p2p_task_delete": _task_delete_fallback,
            "p2p_remote_status": _remote_status_fallback,
            "p2p_remote_call_tool": _remote_call_fallback,
            "p2p_remote_cache_get": _remote_cache_get_fallback,
            "p2p_remote_cache_set": _remote_cache_set_fallback,
            "p2p_remote_cache_has": _remote_cache_has_fallback,
            "p2p_remote_cache_delete": _remote_cache_delete_fallback,
            "p2p_remote_submit_task": _remote_submit_fallback,
        }


_API = _load_p2p_tools_api()


async def p2p_service_status(include_peers: bool = True, peers_limit: int = 50) -> Dict[str, Any]:
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


async def p2p_cache_has(key: str) -> Dict[str, Any]:
    """Check if a key exists in local P2P shared cache."""
    result = _API["p2p_cache_has"](key=key)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_cache_set(key: str, value: Any, ttl_s: Optional[float] = None) -> Dict[str, Any]:
    """Set a value in local P2P shared cache."""
    result = _API["p2p_cache_set"](key=key, value=value, ttl_s=ttl_s)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_cache_delete(key: str) -> Dict[str, Any]:
    """Delete a key from local P2P shared cache."""
    result = _API["p2p_cache_delete"](key=key)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_task_submit(task_type: str, payload: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
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


async def p2p_task_delete(task_id: str) -> Dict[str, Any]:
    """Delete task from local P2P task queue."""
    result = _API["p2p_task_delete"](task_id=task_id)
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_remote_status(
    remote_multiaddr: str = "",
    peer_id: str = "",
    timeout_s: float = 10.0,
    detail: bool = False,
) -> Dict[str, Any]:
    """Get status from remote P2P peer."""
    result = _API["p2p_remote_status"](
        remote_multiaddr=remote_multiaddr,
        peer_id=peer_id,
        timeout_s=timeout_s,
        detail=detail,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_remote_call_tool(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Call MCP tool on remote P2P peer."""
    result = _API["p2p_remote_call_tool"](
        tool_name=tool_name,
        args=args,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_remote_cache_get(
    key: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Get cache value from remote peer."""
    result = _API["p2p_remote_cache_get"](
        key=key,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_remote_cache_set(
    key: str,
    value: Any,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Set cache value on remote peer."""
    result = _API["p2p_remote_cache_set"](
        key=key,
        value=value,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_remote_cache_has(
    key: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Check cache key existence on remote peer."""
    result = _API["p2p_remote_cache_has"](
        key=key,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_remote_cache_delete(
    key: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Delete cache key on remote peer."""
    result = _API["p2p_remote_cache_delete"](
        key=key,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def p2p_remote_submit_task(
    task_type: str,
    model_name: str,
    payload: Dict[str, Any],
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Submit task to remote peer queue."""
    result = _API["p2p_remote_submit_task"](
        task_type=task_type,
        model_name=model_name,
        payload=payload,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
    )
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
        name="p2p_cache_has",
        func=p2p_cache_has,
        description="Check if local P2P cache contains a key.",
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
        name="p2p_cache_delete",
        func=p2p_cache_delete,
        description="Delete a key from local P2P cache.",
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

    manager.register_tool(
        category="p2p_tools",
        name="p2p_task_delete",
        func=p2p_task_delete,
        description="Delete task from local P2P task queue.",
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_remote_status",
        func=p2p_remote_status,
        description="Get status from remote P2P peer.",
        input_schema={
            "type": "object",
            "properties": {
                "remote_multiaddr": {"type": "string"},
                "peer_id": {"type": "string"},
                "timeout_s": {"type": "number"},
                "detail": {"type": "boolean"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_remote_call_tool",
        func=p2p_remote_call_tool,
        description="Call MCP tool on remote P2P peer.",
        input_schema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "args": {"type": ["object", "null"]},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["tool_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_remote_cache_get",
        func=p2p_remote_cache_get,
        description="Get cache value from remote peer.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["key"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_remote_cache_set",
        func=p2p_remote_cache_set,
        description="Set cache value on remote peer.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["key", "value"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_remote_cache_has",
        func=p2p_remote_cache_has,
        description="Check cache key existence on remote peer.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["key"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_remote_cache_delete",
        func=p2p_remote_cache_delete,
        description="Delete cache key on remote peer.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["key"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )

    manager.register_tool(
        category="p2p_tools",
        name="p2p_remote_submit_task",
        func=p2p_remote_submit_task,
        description="Submit task to remote peer queue.",
        input_schema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "model_name": {"type": "string"},
                "payload": {"type": "object"},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
            },
            "required": ["task_type", "model_name", "payload"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools"],
    )
