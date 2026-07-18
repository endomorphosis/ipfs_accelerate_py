"""Native backend-management-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_backend_management_tools_api() -> Dict[str, Any]:
    """Resolve source backend-management-tools APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.mcp.tools.backend_management import (  # type: ignore
            list_inference_backends as _list_inference_backends,
            get_backend_status as _get_backend_status,
            select_backend_for_inference as _select_backend_for_inference,
            route_inference_request as _route_inference_request,
            get_supported_tasks as _get_supported_tasks,
        )

        return {
            "list_inference_backends": _list_inference_backends,
            "get_backend_status": _get_backend_status,
            "select_backend_for_inference": _select_backend_for_inference,
            "route_inference_request": _route_inference_request,
            "get_supported_tasks": _get_supported_tasks,
        }
    except Exception:
        logger.warning(
            "Source backend_management import unavailable, using fallback stubs"
        )

        def _list_fallback(
            backend_type: Optional[str] = None,
            status: Optional[str] = None,
            task: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {"status": "success", "backends": [], "count": 0}

        def _status_fallback() -> Dict[str, Any]:
            return {"status": "success", "backends": {}}

        def _select_fallback(
            model: str,
            task: str = "text-generation",
            preferred_backend: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "model": model,
                "task": task,
                "selected_backend": None,
            }

        def _route_fallback(
            model: str,
            inputs: Any,
            task: str = "text-generation",
            backend: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "model": model,
                "task": task,
                "result": None,
                "backend_available": False,
            }

        def _tasks_fallback() -> Dict[str, Any]:
            return {"status": "success", "tasks": []}

        return {
            "list_inference_backends": _list_fallback,
            "get_backend_status": _status_fallback,
            "select_backend_for_inference": _select_fallback,
            "route_inference_request": _route_fallback,
            "get_supported_tasks": _tasks_fallback,
        }


_API = _load_backend_management_tools_api()


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


async def backend_list(
    backend_type: Optional[str] = None,
    status: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """List registered inference backends with optional filtering."""
    try:
        result = _API["list_inference_backends"](
            backend_type=backend_type, status=status, task=task
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def backend_get_status() -> Dict[str, Any]:
    """Get status of all registered inference backends."""
    try:
        result = _API["get_backend_status"]()
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def backend_select_for_inference(
    model: str,
    task: str = "text-generation",
    preferred_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Select the best backend for a model and task."""
    try:
        result = _API["select_backend_for_inference"](
            model=model, task=task, preferred_backend=preferred_backend
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model=model, task=task)


async def backend_route_inference_request(
    model: str,
    inputs: Any,
    task: str = "text-generation",
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Route an inference request to the appropriate backend."""
    try:
        result = _API["route_inference_request"](
            model=model, inputs=inputs, task=task, backend=backend
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model=model, task=task)


async def backend_get_supported_tasks() -> Dict[str, Any]:
    """Get the list of inference tasks supported by registered backends."""
    try:
        result = _API["get_supported_tasks"]()
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


def register_native_backend_management_tools(manager: Any) -> None:
    """Register native backend-management-tools category tools in unified manager."""
    manager.register_tool(
        category="backend_management_tools",
        name="backend_list",
        func=backend_list,
        description="List registered inference backends with optional type/status/task filters.",
        input_schema={
            "type": "object",
            "properties": {
                "backend_type": {
                    "type": "string",
                    "description": "Filter by backend type (gpu, api, cli, p2p, websocket, mcp).",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status (healthy, degraded, unhealthy, offline).",
                },
                "task": {
                    "type": "string",
                    "description": "Filter by supported task (e.g., text-generation).",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "backend-management-tools"],
    )
    manager.register_tool(
        category="backend_management_tools",
        name="backend_get_status",
        func=backend_get_status,
        description="Get real-time status of all registered inference backends.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "backend-management-tools"],
    )
    manager.register_tool(
        category="backend_management_tools",
        name="backend_select_for_inference",
        func=backend_select_for_inference,
        description="Select the optimal backend for a model and task combination.",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model identifier."},
                "task": {
                    "type": "string",
                    "description": "Inference task type.",
                    "default": "text-generation",
                },
                "preferred_backend": {
                    "type": "string",
                    "description": "Preferred backend type if available.",
                },
            },
            "required": ["model"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "backend-management-tools"],
    )
    manager.register_tool(
        category="backend_management_tools",
        name="backend_route_inference_request",
        func=backend_route_inference_request,
        description="Route an inference request to the best available backend.",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model identifier."},
                "inputs": {"description": "Input data for inference."},
                "task": {
                    "type": "string",
                    "description": "Inference task type.",
                    "default": "text-generation",
                },
                "backend": {
                    "type": "string",
                    "description": "Optional specific backend to use.",
                },
            },
            "required": ["model", "inputs"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "backend-management-tools"],
    )
    manager.register_tool(
        category="backend_management_tools",
        name="backend_get_supported_tasks",
        func=backend_get_supported_tasks,
        description="Get the list of inference tasks supported across all registered backends.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "backend-management-tools"],
    )
