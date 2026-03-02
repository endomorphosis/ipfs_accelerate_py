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
    return await _invoke_engine_method(
        "TaskQueueEngine",
        "get_status",
        task_id=task_id,
        include_logs=include_logs,
        include_metrics=include_metrics,
    )


async def mcplusplus_workflow_get_status(
    workflow_id: str,
    include_steps: bool = True,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """Get workflow status via MCP++ WorkflowEngine adapter."""
    return await _invoke_engine_method(
        "WorkflowEngine",
        "get_status",
        workflow_id=workflow_id,
        include_steps=include_steps,
        include_metrics=include_metrics,
    )


async def mcplusplus_peer_list(
    status_filter: str = "",
    limit: int = 50,
) -> Dict[str, Any]:
    """List peers via MCP++ PeerEngine adapter."""
    return await _invoke_engine_method(
        "PeerEngine",
        "list_peers",
        status_filter=(status_filter or None),
        limit=limit,
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
                "task_id": {"type": "string"},
                "include_logs": {"type": "boolean"},
                "include_metrics": {"type": "boolean"},
            },
            "required": ["task_id"],
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
                "workflow_id": {"type": "string"},
                "include_steps": {"type": "boolean"},
                "include_metrics": {"type": "boolean"},
            },
            "required": ["workflow_id"],
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
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )
