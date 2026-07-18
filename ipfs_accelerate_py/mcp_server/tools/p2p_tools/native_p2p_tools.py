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


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads into deterministic dictionary envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        if envelope.get("success") is False or envelope.get("error") or envelope.get("ok") is False:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _validate_remote_cache_args(
    *,
    key: Any,
    remote_multiaddr: Any,
    remote_peer_id: Any,
    timeout_s: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str], str, str, float]:
    """Validate common remote cache wrapper arguments."""
    if not isinstance(key, str) or not key.strip():
        return _error_result("key must be a non-empty string", key=key), None, "", "", 0.0
    if not isinstance(remote_multiaddr, str):
        return _error_result("remote_multiaddr must be a string", remote_multiaddr=remote_multiaddr), None, "", "", 0.0
    if not isinstance(remote_peer_id, str):
        return _error_result("remote_peer_id must be a string", remote_peer_id=remote_peer_id), None, "", "", 0.0
    if not isinstance(timeout_s, (int, float)) or float(timeout_s) <= 0:
        return _error_result("timeout_s must be a number > 0", timeout_s=timeout_s), None, "", "", 0.0
    return None, key.strip(), remote_multiaddr.strip(), remote_peer_id.strip(), float(timeout_s)


async def p2p_service_status(include_peers: bool = True, peers_limit: int = 50) -> Dict[str, Any]:
    """Return local P2P service status and peers."""
    if not isinstance(include_peers, bool):
        return _error_result("include_peers must be a boolean", include_peers=include_peers)
    if not isinstance(peers_limit, int) or peers_limit <= 0:
        return _error_result("peers_limit must be a positive integer", peers_limit=peers_limit)
    try:
        result = _API["p2p_service_status"](include_peers=include_peers, peers_limit=peers_limit)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("ok", envelope.get("status") == "success")
        envelope.setdefault("service", {})
        envelope.setdefault("peers", [])
        envelope.setdefault("include_peers", include_peers)
        envelope.setdefault("peers_limit", peers_limit)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), include_peers=include_peers, peers_limit=peers_limit)


async def p2p_cache_get(key: str) -> Dict[str, Any]:
    """Get a value from local P2P shared cache."""
    if not isinstance(key, str) or not key.strip():
        return _error_result("key must be a non-empty string", key=key)
    key = key.strip()
    try:
        result = _API["p2p_cache_get"](key=key)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("hit", False)
            envelope.setdefault("value", None)
        envelope.setdefault("key", key)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=key)


async def p2p_cache_has(key: str) -> Dict[str, Any]:
    """Check if a key exists in local P2P shared cache."""
    if not isinstance(key, str) or not key.strip():
        return _error_result("key must be a non-empty string", key=key)
    key = key.strip()
    try:
        result = _API["p2p_cache_has"](key=key)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("key", key)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=key)


async def p2p_cache_set(key: str, value: Any, ttl_s: Optional[float] = None) -> Dict[str, Any]:
    """Set a value in local P2P shared cache."""
    if not isinstance(key, str) or not key.strip():
        return _error_result("key must be a non-empty string", key=key)
    if ttl_s is not None and (not isinstance(ttl_s, (int, float)) or float(ttl_s) <= 0):
        return _error_result("ttl_s must be a number > 0 when provided", key=key, ttl_s=ttl_s)
    key = key.strip()
    try:
        result = _API["p2p_cache_set"](key=key, value=value, ttl_s=float(ttl_s) if ttl_s is not None else None)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("key", key)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=key)


async def p2p_cache_delete(key: str) -> Dict[str, Any]:
    """Delete a key from local P2P shared cache."""
    if not isinstance(key, str) or not key.strip():
        return _error_result("key must be a non-empty string", key=key)
    key = key.strip()
    try:
        result = _API["p2p_cache_delete"](key=key)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("key", key)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=key)


async def p2p_task_submit(task_type: str, payload: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
    """Submit a task to local P2P task queue."""
    if not isinstance(task_type, str) or not task_type.strip():
        return _error_result("task_type must be a non-empty string", task_type=task_type)
    if not isinstance(payload, dict):
        return _error_result("payload must be an object", payload=payload)
    if not isinstance(model_name, str):
        return _error_result("model_name must be a string", model_name=model_name)
    task_type = task_type.strip()
    model_name = model_name.strip()
    try:
        result = _API["p2p_task_submit"](task_type=task_type, payload=payload, model_name=model_name)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("task_id", None)
        envelope.setdefault("task_type", task_type)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), task_type=task_type)


async def p2p_task_get(task_id: str) -> Dict[str, Any]:
    """Get task status from local P2P task queue."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result("task_id must be a non-empty string", task_id=task_id)
    task_id = task_id.strip()
    try:
        result = _API["p2p_task_get"](task_id=task_id)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("task_id", task_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), task_id=task_id)


async def p2p_task_delete(task_id: str) -> Dict[str, Any]:
    """Delete task from local P2P task queue."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result("task_id must be a non-empty string", task_id=task_id)
    task_id = task_id.strip()
    try:
        result = _API["p2p_task_delete"](task_id=task_id)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("task_id", task_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), task_id=task_id)


async def p2p_remote_status(
    remote_multiaddr: str = "",
    peer_id: str = "",
    timeout_s: float = 10.0,
    detail: bool = False,
) -> Dict[str, Any]:
    """Get status from remote P2P peer."""
    if not isinstance(remote_multiaddr, str):
        return _error_result("remote_multiaddr must be a string", remote_multiaddr=remote_multiaddr)
    if not isinstance(peer_id, str):
        return _error_result("peer_id must be a string", peer_id=peer_id)
    if not isinstance(timeout_s, (int, float)) or float(timeout_s) <= 0:
        return _error_result("timeout_s must be a number > 0", timeout_s=timeout_s)
    if not isinstance(detail, bool):
        return _error_result("detail must be a boolean", detail=detail)
    try:
        result = _API["p2p_remote_status"](
            remote_multiaddr=remote_multiaddr.strip(),
            peer_id=peer_id.strip(),
            timeout_s=float(timeout_s),
            detail=detail,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("remote_multiaddr", remote_multiaddr.strip())
            envelope.setdefault("peer_id", peer_id.strip())
        return envelope
    except Exception as exc:
        return _error_result(str(exc), peer_id=peer_id, remote_multiaddr=remote_multiaddr)


async def p2p_remote_call_tool(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Call MCP tool on remote P2P peer."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        return _error_result("tool_name must be a non-empty string", tool_name=tool_name)
    if args is not None and not isinstance(args, dict):
        return _error_result("args must be an object when provided", args=args, tool_name=tool_name)
    if not isinstance(remote_multiaddr, str):
        return _error_result("remote_multiaddr must be a string", remote_multiaddr=remote_multiaddr)
    if not isinstance(remote_peer_id, str):
        return _error_result("remote_peer_id must be a string", remote_peer_id=remote_peer_id)
    if not isinstance(timeout_s, (int, float)) or float(timeout_s) <= 0:
        return _error_result("timeout_s must be a number > 0", timeout_s=timeout_s)
    tool_name = tool_name.strip()
    try:
        result = _API["p2p_remote_call_tool"](
            tool_name=tool_name,
            args=args if isinstance(args, dict) else None,
            remote_multiaddr=remote_multiaddr.strip(),
            remote_peer_id=remote_peer_id.strip(),
            timeout_s=float(timeout_s),
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("args", args if isinstance(args, dict) else None)
            envelope.setdefault("remote_multiaddr", remote_multiaddr.strip())
            envelope.setdefault("remote_peer_id", remote_peer_id.strip())
        envelope.setdefault("tool_name", tool_name)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), tool_name=tool_name)


async def p2p_remote_cache_get(
    key: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Get cache value from remote peer."""
    error, normalized_key, normalized_multiaddr, normalized_peer_id, normalized_timeout = _validate_remote_cache_args(
        key=key,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if error is not None:
        return error
    try:
        result = _API["p2p_remote_cache_get"](
            key=normalized_key,
            remote_multiaddr=normalized_multiaddr,
            remote_peer_id=normalized_peer_id,
            timeout_s=normalized_timeout,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("hit", False)
            envelope.setdefault("value", None)
        envelope.setdefault("key", normalized_key)
        envelope.setdefault("remote_multiaddr", normalized_multiaddr)
        envelope.setdefault("remote_peer_id", normalized_peer_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=normalized_key)


async def p2p_remote_cache_set(
    key: str,
    value: Any,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Set cache value on remote peer."""
    error, normalized_key, normalized_multiaddr, normalized_peer_id, normalized_timeout = _validate_remote_cache_args(
        key=key,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if error is not None:
        return error
    try:
        result = _API["p2p_remote_cache_set"](
            key=normalized_key,
            value=value,
            remote_multiaddr=normalized_multiaddr,
            remote_peer_id=normalized_peer_id,
            timeout_s=normalized_timeout,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
        envelope.setdefault("key", normalized_key)
        envelope.setdefault("remote_multiaddr", normalized_multiaddr)
        envelope.setdefault("remote_peer_id", normalized_peer_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=normalized_key)


async def p2p_remote_cache_has(
    key: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Check cache key existence on remote peer."""
    error, normalized_key, normalized_multiaddr, normalized_peer_id, normalized_timeout = _validate_remote_cache_args(
        key=key,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if error is not None:
        return error
    try:
        result = _API["p2p_remote_cache_has"](
            key=normalized_key,
            remote_multiaddr=normalized_multiaddr,
            remote_peer_id=normalized_peer_id,
            timeout_s=normalized_timeout,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("hit", False)
        envelope.setdefault("key", normalized_key)
        envelope.setdefault("remote_multiaddr", normalized_multiaddr)
        envelope.setdefault("remote_peer_id", normalized_peer_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=normalized_key)


async def p2p_remote_cache_delete(
    key: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Delete cache key on remote peer."""
    error, normalized_key, normalized_multiaddr, normalized_peer_id, normalized_timeout = _validate_remote_cache_args(
        key=key,
        remote_multiaddr=remote_multiaddr,
        remote_peer_id=remote_peer_id,
        timeout_s=timeout_s,
    )
    if error is not None:
        return error
    try:
        result = _API["p2p_remote_cache_delete"](
            key=normalized_key,
            remote_multiaddr=normalized_multiaddr,
            remote_peer_id=normalized_peer_id,
            timeout_s=normalized_timeout,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("deleted", False)
        envelope.setdefault("key", normalized_key)
        envelope.setdefault("remote_multiaddr", normalized_multiaddr)
        envelope.setdefault("remote_peer_id", normalized_peer_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), key=normalized_key)


async def p2p_remote_submit_task(
    task_type: str,
    model_name: str,
    payload: Dict[str, Any],
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Submit task to remote peer queue."""
    if not isinstance(task_type, str) or not task_type.strip():
        return _error_result("task_type must be a non-empty string", task_type=task_type)
    if not isinstance(model_name, str):
        return _error_result("model_name must be a string", model_name=model_name)
    if not isinstance(payload, dict):
        return _error_result("payload must be an object", payload=payload)
    if not isinstance(remote_multiaddr, str):
        return _error_result("remote_multiaddr must be a string", remote_multiaddr=remote_multiaddr)
    if not isinstance(remote_peer_id, str):
        return _error_result("remote_peer_id must be a string", remote_peer_id=remote_peer_id)

    normalized_task_type = task_type.strip()
    normalized_model_name = model_name.strip()
    normalized_multiaddr = remote_multiaddr.strip()
    normalized_peer_id = remote_peer_id.strip()
    try:
        result = _API["p2p_remote_submit_task"](
            task_type=normalized_task_type,
            model_name=normalized_model_name,
            payload=payload,
            remote_multiaddr=normalized_multiaddr,
            remote_peer_id=normalized_peer_id,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("task", None)
        envelope.setdefault("task_type", normalized_task_type)
        envelope.setdefault("model_name", normalized_model_name)
        envelope.setdefault("payload", payload)
        envelope.setdefault("remote_multiaddr", normalized_multiaddr)
        envelope.setdefault("remote_peer_id", normalized_peer_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), task_type=normalized_task_type)


# ---------------------------------------------------------------------------
# P2P TaskQueue tools (migrated from legacy mcp/tools/p2p_taskqueue.py)
# ---------------------------------------------------------------------------


def _load_taskqueue_api() -> Dict[str, Any]:
    """Resolve legacy p2p_taskqueue tool implementations with fallback."""
    try:
        import ipfs_accelerate_py.mcp.tools.p2p_taskqueue as _tq_mod  # type: ignore

        return {"_module": _tq_mod}
    except Exception:
        return {}


_TASKQUEUE_API = _load_taskqueue_api()


def _get_tq_remote_queue(peer_id: str = "", multiaddr: str = "") -> Any:
    """Get a remote queue instance from the legacy module."""
    mod = _TASKQUEUE_API.get("_module")
    if mod is None:
        return None
    fn = getattr(mod, "_remote_queue", None)
    if fn is None:
        return None
    try:
        return fn(peer_id=peer_id, multiaddr=multiaddr)
    except Exception:
        return None


async def p2p_taskqueue_status(peer_id: str = "", multiaddr: str = "") -> Dict[str, Any]:
    """Get the status of the P2P task queue."""
    try:
        rq = _get_tq_remote_queue(peer_id=peer_id, multiaddr=multiaddr)
        if rq is not None and hasattr(rq, "status"):
            result = await rq.status()
            return _normalize_payload(result if isinstance(result, dict) else {"status_data": result})
        return _normalize_payload({"queue_available": False, "peer_id": peer_id or None})
    except Exception as exc:
        return _error_result(str(exc))


async def p2p_taskqueue_submit(
    task_type: str,
    payload: Optional[Dict[str, Any]] = None,
    model_name: str = "",
    peer_id: str = "",
    multiaddr: str = "",
) -> Dict[str, Any]:
    """Submit a task to the P2P task queue."""
    try:
        rq = _get_tq_remote_queue(peer_id=peer_id, multiaddr=multiaddr)
        if rq is not None and hasattr(rq, "submit"):
            result = await rq.submit(
                task_type=task_type, payload=payload or {}, model_name=model_name
            )
            return _normalize_payload(result if isinstance(result, dict) else {"task_id": result})
        return _normalize_payload({"submitted": False, "task_type": task_type, "queue_available": False})
    except Exception as exc:
        return _error_result(str(exc), task_type=task_type)


async def p2p_taskqueue_claim_next(
    task_type: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> Dict[str, Any]:
    """Claim the next available task from the P2P queue."""
    try:
        rq = _get_tq_remote_queue(peer_id=peer_id, multiaddr=multiaddr)
        if rq is not None and hasattr(rq, "claim_next"):
            result = await rq.claim_next(task_type=task_type)
            return _normalize_payload(result if isinstance(result, dict) else {"task": result})
        return _normalize_payload({"task": None, "queue_available": False})
    except Exception as exc:
        return _error_result(str(exc))


async def p2p_taskqueue_call_tool(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> Dict[str, Any]:
    """Call a tool on a remote P2P peer via the task queue."""
    try:
        rq = _get_tq_remote_queue(peer_id=peer_id, multiaddr=multiaddr)
        if rq is not None and hasattr(rq, "call_tool"):
            result = await rq.call_tool(tool_name=tool_name, arguments=arguments or {})
            return _normalize_payload(result if isinstance(result, dict) else {"result": result, "tool": tool_name})
        return _normalize_payload({"result": None, "tool": tool_name, "queue_available": False})
    except Exception as exc:
        return _error_result(str(exc), tool_name=tool_name)


async def p2p_taskqueue_list_tasks(
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> Dict[str, Any]:
    """List tasks in the P2P task queue."""
    try:
        rq = _get_tq_remote_queue(peer_id=peer_id, multiaddr=multiaddr)
        if rq is not None and hasattr(rq, "list_tasks"):
            result = await rq.list_tasks(status=status, task_type=task_type)
            tasks = result if isinstance(result, list) else result.get("tasks", []) if isinstance(result, dict) else []
            return _normalize_payload({"tasks": tasks, "count": len(tasks)})
        return _normalize_payload({"tasks": [], "count": 0, "queue_available": False})
    except Exception as exc:
        return _error_result(str(exc))


async def p2p_taskqueue_get_task(
    task_id: str,
    peer_id: str = "",
    multiaddr: str = "",
) -> Dict[str, Any]:
    """Get details of a specific task from the P2P queue."""
    try:
        rq = _get_tq_remote_queue(peer_id=peer_id, multiaddr=multiaddr)
        if rq is not None and hasattr(rq, "get_task"):
            result = await rq.get_task(task_id=task_id)
            return _normalize_payload(result if isinstance(result, dict) else {"task": result, "task_id": task_id})
        return _normalize_payload({"task": None, "task_id": task_id, "queue_available": False})
    except Exception as exc:
        return _error_result(str(exc), task_id=task_id)


async def p2p_taskqueue_complete_task(
    task_id: str,
    result: Optional[Any] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> Dict[str, Any]:
    """Mark a task as completed in the P2P queue."""
    try:
        rq = _get_tq_remote_queue(peer_id=peer_id, multiaddr=multiaddr)
        if rq is not None and hasattr(rq, "complete_task"):
            res = await rq.complete_task(task_id=task_id, result=result)
            return _normalize_payload(res if isinstance(res, dict) else {"task_id": task_id, "completed": True})
        return _normalize_payload({"task_id": task_id, "completed": False, "queue_available": False})
    except Exception as exc:
        return _error_result(str(exc), task_id=task_id)


async def p2p_taskqueue_list_peers() -> Dict[str, Any]:
    """List peers in the P2P task queue network."""
    try:
        mod = _TASKQUEUE_API.get("_module")
        if mod is not None and hasattr(mod, "list_peers"):
            # list_peers is a regular tool; create a minimal mock MCP object to run it.
            class _MockMCP:
                _peers: list = []

                def tool(self):
                    def _dec(fn):
                        self._peers.append(fn)
                        return fn
                    return _dec

            # The function is defined via @mcp.tool(); we cannot call it directly.
            # Fall through to empty fallback.
            pass
        return _normalize_payload({"peers": [], "count": 0, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc))


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
                "include_peers": {"type": "boolean", "default": True},
                "peers_limit": {"type": "integer", "minimum": 1, "default": 50},
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
            "properties": {"key": {"type": "string", "minLength": 1}},
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
            "properties": {"key": {"type": "string", "minLength": 1}},
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
                "key": {"type": "string", "minLength": 1},
                "value": {},
                "ttl_s": {"type": ["number", "null"], "exclusiveMinimum": 0},
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
            "properties": {"key": {"type": "string", "minLength": 1}},
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
                "task_type": {"type": "string", "minLength": 1},
                "payload": {"type": "object"},
                "model_name": {"type": "string", "default": ""},
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
            "properties": {"task_id": {"type": "string", "minLength": 1}},
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
            "properties": {"task_id": {"type": "string", "minLength": 1}},
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
                "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 10.0},
                "detail": {"type": "boolean", "default": False},
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
                "tool_name": {"type": "string", "minLength": 1},
                "args": {"type": ["object", "null"]},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 30.0},
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
                "key": {"type": "string", "minLength": 1},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 10.0},
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
                "key": {"type": "string", "minLength": 1},
                "value": {},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 10.0},
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
                "key": {"type": "string", "minLength": 1},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 10.0},
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
                "key": {"type": "string", "minLength": 1},
                "remote_multiaddr": {"type": "string"},
                "remote_peer_id": {"type": "string"},
                "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 10.0},
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
                "task_type": {"type": "string", "minLength": 1},
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

    # --- P2P TaskQueue tools (migrated from legacy mcp/tools/p2p_taskqueue.py) ---
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_status",
        func=p2p_taskqueue_status,
        description="Get the status of the P2P task queue.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string", "description": "Optional peer ID for remote queue."},
                "multiaddr": {"type": "string", "description": "Optional multiaddr for remote peer."},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_submit",
        func=p2p_taskqueue_submit,
        description="Submit a task to the P2P task queue.",
        input_schema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Task type identifier."},
                "payload": {"type": "object", "description": "Task payload data."},
                "model_name": {"type": "string", "description": "Optional model name.", "default": ""},
                "peer_id": {"type": "string", "description": "Optional remote peer ID."},
                "multiaddr": {"type": "string", "description": "Optional remote multiaddr."},
            },
            "required": ["task_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_claim_next",
        func=p2p_taskqueue_claim_next,
        description="Claim the next available task from the P2P queue.",
        input_schema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Optional task type filter."},
                "peer_id": {"type": "string", "description": "Optional remote peer ID."},
                "multiaddr": {"type": "string", "description": "Optional remote multiaddr."},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_call_tool",
        func=p2p_taskqueue_call_tool,
        description="Call a tool on a remote P2P peer via the task queue.",
        input_schema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Name of the tool to call."},
                "arguments": {"type": "object", "description": "Tool arguments."},
                "peer_id": {"type": "string", "description": "Optional remote peer ID."},
                "multiaddr": {"type": "string", "description": "Optional remote multiaddr."},
            },
            "required": ["tool_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_list_tasks",
        func=p2p_taskqueue_list_tasks,
        description="List tasks in the P2P task queue.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Optional status filter."},
                "task_type": {"type": "string", "description": "Optional task type filter."},
                "peer_id": {"type": "string", "description": "Optional remote peer ID."},
                "multiaddr": {"type": "string", "description": "Optional remote multiaddr."},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_get_task",
        func=p2p_taskqueue_get_task,
        description="Get details of a specific task from the P2P queue.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task identifier."},
                "peer_id": {"type": "string", "description": "Optional remote peer ID."},
                "multiaddr": {"type": "string", "description": "Optional remote multiaddr."},
            },
            "required": ["task_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_complete_task",
        func=p2p_taskqueue_complete_task,
        description="Mark a P2P queue task as completed.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task identifier to complete."},
                "result": {"description": "Optional result data for the task."},
                "peer_id": {"type": "string", "description": "Optional remote peer ID."},
                "multiaddr": {"type": "string", "description": "Optional remote multiaddr."},
            },
            "required": ["task_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
    manager.register_tool(
        category="p2p_tools",
        name="p2p_taskqueue_list_peers",
        func=p2p_taskqueue_list_peers,
        description="List peers in the P2P task queue network.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-tools", "taskqueue"],
    )
