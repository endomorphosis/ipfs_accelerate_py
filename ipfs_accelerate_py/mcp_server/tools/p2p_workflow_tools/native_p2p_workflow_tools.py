"""Native P2P-workflow tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_p2p_workflow_api() -> Dict[str, Any]:
    """Resolve source P2P-workflow APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.p2p_workflow_tools.p2p_workflow_tools import (  # type: ignore
            get_assigned_workflows as _get_assigned_workflows,
            get_next_p2p_workflow as _get_next_p2p_workflow,
            get_p2p_scheduler_status as _get_p2p_scheduler_status,
            initialize_p2p_scheduler as _initialize_p2p_scheduler,
            schedule_p2p_workflow as _schedule_p2p_workflow,
        )

        return {
            "initialize_p2p_scheduler": _initialize_p2p_scheduler,
            "schedule_p2p_workflow": _schedule_p2p_workflow,
            "get_next_p2p_workflow": _get_next_p2p_workflow,
            "get_p2p_scheduler_status": _get_p2p_scheduler_status,
            "get_assigned_workflows": _get_assigned_workflows,
        }
    except Exception:
        logger.warning(
            "Source p2p_workflow_tools import unavailable, using fallback p2p-workflow functions"
        )

        async def _initialize_fallback(
            peer_id: Optional[str] = None,
            peers: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "message": "Fallback P2P scheduler initialized",
                "status": {
                    "peer_id": peer_id or "fallback-peer",
                    "peer_count": len(peers or []),
                },
            }

        async def _schedule_fallback(
            workflow_id: str,
            name: str,
            tags: List[str],
            priority: float = 1.0,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = name, tags, priority, metadata
            return {
                "success": True,
                "workflow_id": workflow_id,
                "result": {"success": True},
            }

        async def _next_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "message": "No workflows in queue",
                "workflow": None,
            }

        async def _status_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "status": {
                    "queue_size": 0,
                    "peer_count": 0,
                },
            }

        async def _assigned_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "assigned_workflows": [],
                "count": 0,
            }

        return {
            "initialize_p2p_scheduler": _initialize_fallback,
            "schedule_p2p_workflow": _schedule_fallback,
            "get_next_p2p_workflow": _next_fallback,
            "get_p2p_scheduler_status": _status_fallback,
            "get_assigned_workflows": _assigned_fallback,
        }


_API = _load_p2p_workflow_api()


async def initialize_p2p_scheduler(
    peer_id: Optional[str] = None,
    peers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Initialize or retrieve the P2P workflow scheduler state."""
    result = _API["initialize_p2p_scheduler"](peer_id=peer_id, peers=peers)
    if hasattr(result, "__await__"):
        return await result
    return result


async def schedule_p2p_workflow(
    workflow_id: str,
    name: str,
    tags: List[str],
    priority: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Schedule a workflow for P2P-aware execution."""
    result = _API["schedule_p2p_workflow"](
        workflow_id=workflow_id,
        name=name,
        tags=tags,
        priority=priority,
        metadata=metadata,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_next_p2p_workflow() -> Dict[str, Any]:
    """Get the next workflow from the scheduler queue."""
    result = _API["get_next_p2p_workflow"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_p2p_scheduler_status() -> Dict[str, Any]:
    """Get current P2P scheduler status."""
    result = _API["get_p2p_scheduler_status"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_assigned_workflows() -> Dict[str, Any]:
    """Get workflows currently assigned to this peer."""
    result = _API["get_assigned_workflows"]()
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_p2p_workflow_tools(manager: Any) -> None:
    """Register native P2P-workflow tools in unified hierarchical manager."""
    manager.register_tool(
        category="p2p_workflow_tools",
        name="initialize_p2p_scheduler",
        func=initialize_p2p_scheduler,
        description="Initialize the P2P workflow scheduler.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": ["string", "null"]},
                "peers": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-workflow"],
    )

    manager.register_tool(
        category="p2p_workflow_tools",
        name="schedule_p2p_workflow",
        func=schedule_p2p_workflow,
        description="Schedule a workflow for P2P execution.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "number"},
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["workflow_id", "name", "tags"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-workflow"],
    )

    manager.register_tool(
        category="p2p_workflow_tools",
        name="get_next_p2p_workflow",
        func=get_next_p2p_workflow,
        description="Get the next queued P2P workflow.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-workflow"],
    )

    manager.register_tool(
        category="p2p_workflow_tools",
        name="get_p2p_scheduler_status",
        func=get_p2p_scheduler_status,
        description="Get P2P scheduler runtime status.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-workflow"],
    )

    manager.register_tool(
        category="p2p_workflow_tools",
        name="get_assigned_workflows",
        func=get_assigned_workflows,
        description="Get workflows assigned to this peer.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "p2p-workflow"],
    )
