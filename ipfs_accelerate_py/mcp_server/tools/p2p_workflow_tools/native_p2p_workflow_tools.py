"""Native P2P-workflow tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "message": message}
    payload.update(extra)
    return payload


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


def _normalize_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Infer deterministic status from success/error fields when absent."""
    normalized = dict(payload or {})
    if "status" in normalized and not isinstance(normalized.get("status"), str):
        normalized.setdefault("scheduler_status", normalized.get("status"))
        normalized.pop("status", None)
    if "status" not in normalized:
        normalized["status"] = "error" if (normalized.get("error") or normalized.get("success") is False) else "success"
    return normalized


async def initialize_p2p_scheduler(
    peer_id: Optional[str] = None,
    peers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Initialize or retrieve the P2P workflow scheduler state."""
    if peer_id is not None and not str(peer_id).strip():
        return _error_result("peer_id must be a non-empty string when provided", peer_id=peer_id)
    if peers is not None and (not isinstance(peers, list) or not all(isinstance(item, str) and item.strip() for item in peers)):
        return _error_result("peers must be an array of non-empty strings when provided", peers=peers)

    try:
        result = _API["initialize_p2p_scheduler"](peer_id=peer_id, peers=peers)
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(f"initialize_p2p_scheduler failed: {exc}")

    normalized = _normalize_status(dict(payload or {}))
    if normalized.get("status") == "success":
        normalized.setdefault("success", True)
        normalized.setdefault("message", "P2P scheduler initialized")
        normalized.setdefault("scheduler_status", {})
        if isinstance(normalized.get("scheduler_status"), dict):
            normalized["scheduler_status"].setdefault("peer_id", peer_id)
            normalized["scheduler_status"].setdefault("peer_count", len(peers or []))
    return normalized


async def schedule_p2p_workflow(
    workflow_id: str,
    name: str,
    tags: List[str],
    priority: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Schedule a workflow for P2P-aware execution."""
    normalized_workflow_id = str(workflow_id or "").strip()
    normalized_name = str(name or "").strip()
    if not normalized_workflow_id:
        return _error_result("workflow_id is required", workflow_id=workflow_id)
    if not normalized_name:
        return _error_result("name is required", name=name)
    if not isinstance(tags, list) or not tags or not all(isinstance(item, str) and item.strip() for item in tags):
        return _error_result("tags must be a non-empty array of strings", tags=tags)
    try:
        normalized_priority = float(priority)
    except (TypeError, ValueError):
        return _error_result("priority must be numeric", priority=priority)
    if normalized_priority <= 0:
        return _error_result("priority must be a positive number", priority=priority)
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result("metadata must be an object when provided", metadata=metadata)

    try:
        result = _API["schedule_p2p_workflow"](
            workflow_id=normalized_workflow_id,
            name=normalized_name,
            tags=tags,
            priority=normalized_priority,
            metadata=metadata,
        )
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(
            f"schedule_p2p_workflow failed: {exc}",
            workflow_id=normalized_workflow_id,
        )

    normalized = _normalize_status(dict(payload or {}))
    if normalized.get("status") == "success":
        normalized.setdefault("success", True)
        normalized.setdefault("name", normalized_name)
        normalized.setdefault("tags", tags)
        normalized.setdefault("priority", normalized_priority)
        normalized.setdefault("metadata", metadata or {})
    normalized.setdefault("workflow_id", normalized_workflow_id)
    return normalized


async def get_next_p2p_workflow() -> Dict[str, Any]:
    """Get the next workflow from the scheduler queue."""
    try:
        result = _API["get_next_p2p_workflow"]()
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(f"get_next_p2p_workflow failed: {exc}")

    normalized = _normalize_status(dict(payload or {}))
    if normalized.get("status") == "success":
        normalized.setdefault("success", True)
        normalized.setdefault("workflow", None)
        normalized.setdefault("message", "No workflows in queue")
    return normalized


async def get_p2p_scheduler_status() -> Dict[str, Any]:
    """Get current P2P scheduler status."""
    try:
        result = _API["get_p2p_scheduler_status"]()
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(f"get_p2p_scheduler_status failed: {exc}")

    normalized = _normalize_status(dict(payload or {}))
    if normalized.get("status") == "success":
        normalized.setdefault("success", True)
        normalized.setdefault("scheduler_status", {})
        if isinstance(normalized.get("scheduler_status"), dict):
            normalized["scheduler_status"].setdefault("queue_size", 0)
            normalized["scheduler_status"].setdefault("peer_count", 0)
    return normalized


async def get_assigned_workflows() -> Dict[str, Any]:
    """Get workflows currently assigned to this peer."""
    try:
        result = _API["get_assigned_workflows"]()
        payload = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return _error_result(f"get_assigned_workflows failed: {exc}")

    normalized = _normalize_status(dict(payload or {}))
    if normalized.get("status") == "success":
        normalized.setdefault("success", True)
        normalized.setdefault("assigned_workflows", [])
        normalized.setdefault("count", len(normalized.get("assigned_workflows") or []))
    return normalized


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
                "peer_id": {"type": ["string", "null"], "minLength": 1},
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
                "workflow_id": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "tags": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "number", "minimum": 0.0000001, "default": 1.0},
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
