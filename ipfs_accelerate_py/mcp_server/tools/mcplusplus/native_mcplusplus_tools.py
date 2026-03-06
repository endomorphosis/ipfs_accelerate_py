"""Native mcplusplus category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _load_mcplusplus_api() -> Dict[str, Any]:
    """Resolve source mcplusplus APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.mcplusplus import (  # type: ignore
            PeerEngine,
            TaskQueueEngine,
            WorkflowEngine,
        )

        return {
            "TaskQueueEngine": TaskQueueEngine,
            "PeerEngine": PeerEngine,
            "WorkflowEngine": WorkflowEngine,
        }
    except Exception:
        logger.warning("Source mcplusplus import unavailable, using fallback mcplusplus functions")
        return {}


_API = _load_mcplusplus_api()


def _error_result(message: str, engine: str = "", method: str = "") -> Dict[str, Any]:
    return {
        "status": "error",
        "available": bool(_API),
        "engine": engine,
        "method": method,
        "error": message,
    }


async def _invoke_engine_method(engine_name: str, method_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Invoke an engine method with graceful fallback when unavailable."""
    cls = _API.get(engine_name)
    if cls is None:
        return {
            "status": "error",
            "available": False,
            "engine": engine_name,
            "method": method_name,
            "error": "engine_unavailable",
            "fallback": True,
        }

    try:
        engine = cls()
        method = getattr(engine, method_name, None)
        if not callable(method):
            return {
                "status": "error",
                "available": True,
                "engine": engine_name,
                "method": method_name,
                "error": "method_unavailable",
            }

        result = method(**kwargs)
        if hasattr(result, "__await__"):
            result = await result

        if isinstance(result, dict):
            return result
        return {
            "status": "success",
            "engine": engine_name,
            "method": method_name,
            "result": result,
        }
    except Exception as exc:
        return {
            "status": "error",
            "available": True,
            "engine": engine_name,
            "method": method_name,
            "error": str(exc),
        }


async def mcplusplus_engine_status() -> Dict[str, Any]:
    """Return availability and instantiation status for MCP++ engine shims."""
    if not _API:
        return {
            "status": "success",
            "engines": {},
            "available": False,
            "fallback": True,
        }

    engines: Dict[str, Dict[str, Any]] = {}
    for name in ["TaskQueueEngine", "PeerEngine", "WorkflowEngine"]:
        cls = _API.get(name)
        if cls is None:
            engines[name] = {"available": False}
            continue
        try:
            _ = cls()
            engines[name] = {"available": True, "instantiated": True}
        except Exception as exc:
            engines[name] = {"available": True, "instantiated": False, "error": str(exc)}

    return {
        "status": "success",
        "available": True,
        "engines": engines,
    }


async def mcplusplus_list_engines() -> Dict[str, List[str]]:
    """List MCP++ engine classes exposed through source compatibility shims."""
    if not _API:
        return {"status": "success", "engines": [], "fallback": True}

    return {
        "status": "success",
        "engines": sorted(list(_API.keys())),
    }


async def mcplusplus_taskqueue_get_status(
    task_id: str,
    include_logs: bool = False,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """Get task status via MCP++ TaskQueueEngine adapter."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result(
            "task_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="get_status",
        )
    if not isinstance(include_logs, bool):
        return _error_result(
            "include_logs must be a boolean",
            engine="TaskQueueEngine",
            method="get_status",
        )
    if not isinstance(include_metrics, bool):
        return _error_result(
            "include_metrics must be a boolean",
            engine="TaskQueueEngine",
            method="get_status",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "get_status",
        task_id=task_id.strip(),
        include_logs=include_logs,
        include_metrics=include_metrics,
    )


async def mcplusplus_taskqueue_submit(
    task_id: str,
    task_type: str,
    payload: Dict[str, Any],
    priority: float = 1.0,
    tags: List[str] | None = None,
    timeout: int | None = None,
) -> Dict[str, Any]:
    """Submit a task through MCP++ TaskQueueEngine adapter."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result(
            "task_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="submit",
        )
    if not isinstance(task_type, str) or not task_type.strip():
        return _error_result(
            "task_type must be a non-empty string",
            engine="TaskQueueEngine",
            method="submit",
        )
    if not isinstance(payload, dict):
        return _error_result(
            "payload must be an object",
            engine="TaskQueueEngine",
            method="submit",
        )
    if not isinstance(priority, (int, float)):
        return _error_result(
            "priority must be a number",
            engine="TaskQueueEngine",
            method="submit",
        )
    if float(priority) <= 0:
        return _error_result(
            "priority must be > 0",
            engine="TaskQueueEngine",
            method="submit",
        )
    if tags is not None and (
        not isinstance(tags, list)
        or any(not isinstance(item, str) or not item.strip() for item in tags)
    ):
        return _error_result(
            "tags must be an array of non-empty strings",
            engine="TaskQueueEngine",
            method="submit",
        )
    if timeout is not None and (not isinstance(timeout, int) or timeout < 1):
        return _error_result(
            "timeout must be an integer >= 1 when provided",
            engine="TaskQueueEngine",
            method="submit",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "submit",
        task_id=task_id.strip(),
        task_type=task_type.strip(),
        payload=dict(payload),
        priority=float(priority),
        tags=[str(item).strip() for item in (tags or [])],
        timeout=timeout,
    )


async def mcplusplus_workflow_get_status(
    workflow_id: str,
    include_steps: bool = True,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """Get workflow status via MCP++ WorkflowEngine adapter."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result(
            "workflow_id must be a non-empty string",
            engine="WorkflowEngine",
            method="get_status",
        )
    if not isinstance(include_steps, bool):
        return _error_result(
            "include_steps must be a boolean",
            engine="WorkflowEngine",
            method="get_status",
        )
    if not isinstance(include_metrics, bool):
        return _error_result(
            "include_metrics must be a boolean",
            engine="WorkflowEngine",
            method="get_status",
        )

    return await _invoke_engine_method(
        "WorkflowEngine",
        "get_status",
        workflow_id=workflow_id.strip(),
        include_steps=include_steps,
        include_metrics=include_metrics,
    )


async def mcplusplus_workflow_submit(
    workflow_id: str,
    name: str,
    steps: List[Dict[str, Any]],
    priority: float = 1.0,
    tags: List[str] | None = None,
) -> Dict[str, Any]:
    """Submit a workflow through MCP++ WorkflowEngine adapter."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result(
            "workflow_id must be a non-empty string",
            engine="WorkflowEngine",
            method="submit",
        )
    if not isinstance(name, str) or not name.strip():
        return _error_result(
            "name must be a non-empty string",
            engine="WorkflowEngine",
            method="submit",
        )
    if not isinstance(steps, list) or not steps or any(not isinstance(step, dict) for step in steps):
        return _error_result(
            "steps must be a non-empty array of objects",
            engine="WorkflowEngine",
            method="submit",
        )
    if not isinstance(priority, (int, float)):
        return _error_result(
            "priority must be a number",
            engine="WorkflowEngine",
            method="submit",
        )
    if float(priority) <= 0:
        return _error_result(
            "priority must be > 0",
            engine="WorkflowEngine",
            method="submit",
        )
    if tags is not None and (
        not isinstance(tags, list)
        or any(not isinstance(item, str) or not item.strip() for item in tags)
    ):
        return _error_result(
            "tags must be an array of non-empty strings",
            engine="WorkflowEngine",
            method="submit",
        )

    return await _invoke_engine_method(
        "WorkflowEngine",
        "submit",
        workflow_id=workflow_id.strip(),
        name=name.strip(),
        steps=list(steps),
        priority=float(priority),
        tags=[str(item).strip() for item in (tags or [])],
    )


async def mcplusplus_peer_list(
    status_filter: str = "",
    limit: int = 50,
) -> Dict[str, Any]:
    """List peers via MCP++ PeerEngine adapter."""
    if status_filter is not None and not isinstance(status_filter, str):
        return _error_result(
            "status_filter must be a string",
            engine="PeerEngine",
            method="list_peers",
        )
    if not isinstance(limit, int) or limit < 1:
        return _error_result(
            "limit must be an integer >= 1",
            engine="PeerEngine",
            method="list_peers",
        )

    normalized_status_filter = (status_filter or "").strip()

    return await _invoke_engine_method(
        "PeerEngine",
        "list_peers",
        status_filter=(normalized_status_filter or None),
        limit=limit,
    )


async def mcplusplus_peer_discover(
    capability_filter: List[str] | None = None,
    max_peers: int = 10,
    timeout: int = 30,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """Discover peers through MCP++ PeerEngine adapter."""
    if capability_filter is not None and (
        not isinstance(capability_filter, list)
        or any(not isinstance(item, str) or not item.strip() for item in capability_filter)
    ):
        return _error_result(
            "capability_filter must be an array of non-empty strings",
            engine="PeerEngine",
            method="discover",
        )
    if not isinstance(max_peers, int) or max_peers < 1:
        return _error_result(
            "max_peers must be an integer >= 1",
            engine="PeerEngine",
            method="discover",
        )
    if not isinstance(timeout, int) or timeout < 1:
        return _error_result(
            "timeout must be an integer >= 1",
            engine="PeerEngine",
            method="discover",
        )
    if not isinstance(include_metrics, bool):
        return _error_result(
            "include_metrics must be a boolean",
            engine="PeerEngine",
            method="discover",
        )

    return await _invoke_engine_method(
        "PeerEngine",
        "discover",
        capability_filter=[str(item).strip() for item in (capability_filter or [])] or None,
        max_peers=max_peers,
        timeout=timeout,
        include_metrics=include_metrics,
    )


def register_native_mcplusplus_tools(manager: Any) -> None:
    """Register native mcplusplus category tools in unified manager."""
    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_engine_status",
        func=mcplusplus_engine_status,
        description="Get availability status for MCP++ engine shim classes.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_list_engines",
        func=mcplusplus_list_engines,
        description="List MCP++ engine class names exported by source shim module.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_get_status",
        func=mcplusplus_taskqueue_get_status,
        description="Get task status through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "include_logs": {"type": "boolean", "default": False},
                "include_metrics": {"type": "boolean", "default": False},
            },
            "required": ["task_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_submit",
        func=mcplusplus_taskqueue_submit,
        description="Submit task through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "task_type": {"type": "string", "minLength": 1},
                "payload": {"type": "object"},
                "priority": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "timeout": {"type": "integer", "minimum": 1},
            },
            "required": ["task_id", "task_type", "payload"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_workflow_get_status",
        func=mcplusplus_workflow_get_status,
        description="Get workflow status through WorkflowEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
                "include_steps": {"type": "boolean", "default": True},
                "include_metrics": {"type": "boolean", "default": False},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_workflow_submit",
        func=mcplusplus_workflow_submit,
        description="Submit workflow through WorkflowEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "object"},
                },
                "priority": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": ["workflow_id", "name", "steps"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_peer_list",
        func=mcplusplus_peer_list,
        description="List peers through PeerEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "status_filter": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "default": 50},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_peer_discover",
        func=mcplusplus_peer_discover,
        description="Discover peers through PeerEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "capability_filter": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "max_peers": {"type": "integer", "minimum": 1, "default": 10},
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
                "include_metrics": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )
