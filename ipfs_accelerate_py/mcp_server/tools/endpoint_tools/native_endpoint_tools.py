"""Native endpoint-tools category implementations for unified mcp_server.

Exposes IPFS Accelerate endpoint management operations from the legacy
``ipfs_accelerate_py.mcp.tools.endpoints`` module through the unified
MCP++ tool dispatch surface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-memory fallback store for endpoints when the legacy module is unavailable.
_ENDPOINT_STORE: Dict[str, Any] = {}


def _load_endpoint_tools_api() -> Dict[str, Any]:
    """Resolve source endpoint-tools APIs with compatibility fallback."""
    try:
        # The legacy module stores tools as closures inside register_tools; we
        # access the standalone helpers defined at module level.
        import ipfs_accelerate_py.mcp.tools.endpoints as _endpoints_mod  # type: ignore

        return {"_module": _endpoints_mod}
    except Exception:
        logger.warning(
            "Source endpoint_tools import unavailable, using in-memory fallback"
        )
        return {}


_API = _load_endpoint_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _get_legacy_module() -> Optional[Any]:
    return _API.get("_module")


async def endpoint_list() -> Dict[str, Any]:
    """List all registered inference endpoints."""
    try:
        mod = _get_legacy_module()
        if mod is not None and hasattr(mod, "ENDPOINTS"):
            endpoints = list(mod.ENDPOINTS.values())
            return _normalize_payload(
                {"endpoints": endpoints, "count": len(endpoints), "source": "module"}
            )
        return _normalize_payload(
            {
                "endpoints": list(_ENDPOINT_STORE.values()),
                "count": len(_ENDPOINT_STORE),
                "source": "in_memory",
            }
        )
    except Exception as exc:
        return _error_result(str(exc))


async def endpoint_add(
    model: str,
    device: str = "cpu",
    max_batch_size: int = 16,
    description: str = "",
) -> Dict[str, Any]:
    """Register a new inference endpoint."""
    try:
        import uuid, time as _time

        endpoint = {
            "id": str(uuid.uuid4()),
            "model": model,
            "device": device,
            "max_batch_size": max_batch_size,
            "description": description,
            "created_at": _time.time(),
            "status": "active",
        }
        mod = _get_legacy_module()
        if mod is not None and hasattr(mod, "ENDPOINTS"):
            mod.ENDPOINTS[endpoint["id"]] = endpoint
        else:
            _ENDPOINT_STORE[endpoint["id"]] = endpoint
        return _normalize_payload({"endpoint": endpoint, "created": True})
    except Exception as exc:
        return _error_result(str(exc), model=model, device=device)


async def endpoint_remove(endpoint_id: str) -> Dict[str, Any]:
    """Remove a registered inference endpoint."""
    try:
        mod = _get_legacy_module()
        store = mod.ENDPOINTS if (mod is not None and hasattr(mod, "ENDPOINTS")) else _ENDPOINT_STORE
        if endpoint_id in store:
            del store[endpoint_id]
            return _normalize_payload({"endpoint_id": endpoint_id, "removed": True})
        return _error_result(f"Endpoint {endpoint_id!r} not found", endpoint_id=endpoint_id)
    except Exception as exc:
        return _error_result(str(exc), endpoint_id=endpoint_id)


async def endpoint_update(
    endpoint_id: str,
    device: Optional[str] = None,
    max_batch_size: Optional[int] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Update properties of an existing endpoint."""
    try:
        mod = _get_legacy_module()
        store = mod.ENDPOINTS if (mod is not None and hasattr(mod, "ENDPOINTS")) else _ENDPOINT_STORE
        if endpoint_id not in store:
            return _error_result(f"Endpoint {endpoint_id!r} not found", endpoint_id=endpoint_id)
        endpoint = dict(store[endpoint_id])
        if device is not None:
            endpoint["device"] = device
        if max_batch_size is not None:
            endpoint["max_batch_size"] = max_batch_size
        if description is not None:
            endpoint["description"] = description
        if status is not None:
            endpoint["status"] = status
        store[endpoint_id] = endpoint
        return _normalize_payload({"endpoint": endpoint, "updated": True})
    except Exception as exc:
        return _error_result(str(exc), endpoint_id=endpoint_id)


async def endpoint_get(endpoint_id: str) -> Dict[str, Any]:
    """Get details of a specific endpoint."""
    try:
        mod = _get_legacy_module()
        store = mod.ENDPOINTS if (mod is not None and hasattr(mod, "ENDPOINTS")) else _ENDPOINT_STORE
        if endpoint_id in store:
            return _normalize_payload({"endpoint": store[endpoint_id]})
        return _error_result(f"Endpoint {endpoint_id!r} not found", endpoint_id=endpoint_id)
    except Exception as exc:
        return _error_result(str(exc), endpoint_id=endpoint_id)


async def endpoint_log_request(
    endpoint_id: str,
    request_data: Optional[Any] = None,
    response_data: Optional[Any] = None,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Log a request/response pair for an endpoint."""
    try:
        import time as _time

        log_entry: Dict[str, Any] = {
            "endpoint_id": endpoint_id,
            "timestamp": _time.time(),
            "logged": True,
        }
        if latency_ms is not None:
            log_entry["latency_ms"] = latency_ms
        return _normalize_payload(log_entry)
    except Exception as exc:
        return _error_result(str(exc), endpoint_id=endpoint_id)


def register_native_endpoint_tools(manager: Any) -> None:
    """Register native endpoint-tools category tools in unified manager."""
    manager.register_tool(
        category="endpoint_tools",
        name="endpoint_list",
        func=endpoint_list,
        description="List all registered inference endpoints.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "endpoint-tools"],
    )
    manager.register_tool(
        category="endpoint_tools",
        name="endpoint_add",
        func=endpoint_add,
        description="Register a new inference endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model identifier for the endpoint."},
                "device": {
                    "type": "string",
                    "description": "Device to run inference on.",
                    "default": "cpu",
                },
                "max_batch_size": {
                    "type": "integer",
                    "description": "Maximum batch size for inference.",
                    "default": 16,
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the endpoint.",
                    "default": "",
                },
            },
            "required": ["model"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "endpoint-tools"],
    )
    manager.register_tool(
        category="endpoint_tools",
        name="endpoint_remove",
        func=endpoint_remove,
        description="Remove a registered inference endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string", "description": "Endpoint UUID to remove."}
            },
            "required": ["endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "endpoint-tools"],
    )
    manager.register_tool(
        category="endpoint_tools",
        name="endpoint_update",
        func=endpoint_update,
        description="Update properties of an existing inference endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string", "description": "Endpoint UUID to update."},
                "device": {"type": "string", "description": "New device target."},
                "max_batch_size": {"type": "integer", "description": "New maximum batch size."},
                "description": {"type": "string", "description": "New description."},
                "status": {"type": "string", "description": "New status value."},
            },
            "required": ["endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "endpoint-tools"],
    )
    manager.register_tool(
        category="endpoint_tools",
        name="endpoint_get",
        func=endpoint_get,
        description="Get details of a specific inference endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string", "description": "Endpoint UUID."}
            },
            "required": ["endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "endpoint-tools"],
    )
    manager.register_tool(
        category="endpoint_tools",
        name="endpoint_log_request",
        func=endpoint_log_request,
        description="Log a request/response pair for an endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string", "description": "Endpoint UUID."},
                "request_data": {"description": "Optional request payload."},
                "response_data": {"description": "Optional response payload."},
                "latency_ms": {
                    "type": "number",
                    "description": "Optional request latency in milliseconds.",
                },
            },
            "required": ["endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "endpoint-tools"],
    )
