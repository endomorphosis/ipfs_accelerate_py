"""Native workflow-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_workflow_tools_api() -> Dict[str, Any]:
    """Resolve source workflow-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.workflow_tools.workflow_tools import (  # type: ignore
            add_p2p_peer as _add_p2p_peer,
            batch_process_datasets as _batch_process_datasets,
            calculate_peer_distance as _calculate_peer_distance,
            create_template as _create_template,
            create_workflow as _create_workflow,
            execute_workflow as _execute_workflow,
            get_assigned_workflows as _get_assigned_workflows,
            get_next_p2p_workflow as _get_next_p2p_workflow,
            get_p2p_scheduler_status as _get_p2p_scheduler_status,
            get_workflow_tags as _get_workflow_tags,
            get_workflow_status as _get_workflow_status,
            get_workflow_metrics as _get_workflow_metrics,
            initialize_p2p_scheduler as _initialize_p2p_scheduler,
            list_workflows as _list_workflows,
            list_templates as _list_templates,
            merge_merkle_clock as _merge_merkle_clock,
            pause_workflow as _pause_workflow,
            remove_p2p_peer as _remove_p2p_peer,
            resume_workflow as _resume_workflow,
            run_workflow as _run_workflow,
            schedule_p2p_workflow as _schedule_p2p_workflow,
            schedule_workflow as _schedule_workflow,
        )

        return {
            "execute_workflow": _execute_workflow,
            "batch_process_datasets": _batch_process_datasets,
            "create_template": _create_template,
            "create_workflow": _create_workflow,
            "schedule_workflow": _schedule_workflow,
            "get_workflow_status": _get_workflow_status,
            "get_assigned_workflows": _get_assigned_workflows,
            "get_workflow_tags": _get_workflow_tags,
            "resume_workflow": _resume_workflow,
            "get_workflow_metrics": _get_workflow_metrics,
            "initialize_p2p_scheduler": _initialize_p2p_scheduler,
            "pause_workflow": _pause_workflow,
            "list_workflows": _list_workflows,
            "list_templates": _list_templates,
            "run_workflow": _run_workflow,
            "schedule_p2p_workflow": _schedule_p2p_workflow,
            "get_next_p2p_workflow": _get_next_p2p_workflow,
            "add_p2p_peer": _add_p2p_peer,
            "remove_p2p_peer": _remove_p2p_peer,
            "get_p2p_scheduler_status": _get_p2p_scheduler_status,
            "calculate_peer_distance": _calculate_peer_distance,
            "merge_merkle_clock": _merge_merkle_clock,
        }
    except Exception:
        logger.warning("Source workflow_tools import unavailable, using fallback workflow-tools functions")

        async def _execute_fallback(
            workflow_definition: Optional[Dict[str, Any]] = None,
            workflow_id: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = workflow_definition, context
            return {
                "success": False,
                "status": "failed",
                "workflow_id": workflow_id,
                "error": "No steps defined in workflow",
            }

        async def _batch_fallback(
            datasets: List[Dict[str, Any]],
            processing_pipeline: List[str],
            batch_size: int = 10,
            parallel_workers: int = 3,
        ) -> Dict[str, Any]:
            _ = processing_pipeline, batch_size, parallel_workers
            return {
                "success": True,
                "batch_id": "fallback-batch",
                "total_datasets": len(datasets),
            }

        async def _create_workflow_fallback(
            workflow_id: Optional[str] = None,
            workflow_definition: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = workflow_definition
            return {
                "status": "created",
                "workflow_id": workflow_id or "fallback-workflow",
            }

        async def _create_template_fallback(template: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "success": True,
                "template_id": "fallback-template",
                "template_name": str((template or {}).get("name") or "fallback-template"),
            }

        async def _schedule_fallback(
            workflow_definition: Optional[Dict[str, Any]] = None,
            schedule_config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = workflow_definition, schedule_config
            return {
                "success": True,
                "schedule_id": "fallback-schedule",
                "status": "scheduled",
            }

        async def _status_fallback(
            workflow_id: Optional[str] = None,
            execution_id: Optional[str] = None,
            include_details: Optional[bool] = None,
        ) -> Dict[str, Any]:
            _ = execution_id, include_details
            return {
                "success": False,
                "status": "not_found",
                "workflow_id": workflow_id,
                "error": "Workflow not found",
            }

        async def _get_assigned_workflows_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "assigned_workflows": [],
                "count": 0,
            }

        async def _get_workflow_tags_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "tags": [],
                "descriptions": {},
            }

        async def _list_templates_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "templates": [],
                "total": 0,
            }

        async def _resume_workflow_fallback(workflow_id: str) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "running",
                "workflow_id": workflow_id,
            }

        async def _get_workflow_metrics_fallback(
            workflow_id: Optional[str] = None,
            include_performance: Optional[bool] = None,
            time_range: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "metrics": {
                    "workflow_id": workflow_id,
                    "status": None,
                    "time_range": time_range,
                    "has_performance": bool(include_performance),
                },
            }

        async def _pause_workflow_fallback(workflow_id: str) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "paused",
                "workflow_id": workflow_id,
            }

        async def _initialize_p2p_scheduler_fallback(
            peer_id: Optional[str] = None,
            peers: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            return {
                "success": False,
                "error": "P2P workflow scheduler not available",
                "peer_id": peer_id,
                "peers": peers or [],
            }

        async def _list_workflows_fallback(
            include_logs: Optional[bool] = None,
        ) -> Dict[str, Any]:
            _ = include_logs
            return {
                "status": "success",
                "workflows": [],
            }

        async def _run_workflow_fallback(workflow_id: str) -> Dict[str, Any]:
            return {
                "success": False,
                "status": "not_found",
                "workflow_id": workflow_id,
                "error": "Workflow not found",
            }

        async def _schedule_p2p_workflow_fallback(
            workflow_id: str,
            name: str,
            tags: List[str],
            priority: float = 1.0,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = name, tags, priority, metadata
            return {
                "success": False,
                "error": "P2P workflow scheduler not available",
                "workflow_id": workflow_id,
            }

        async def _get_next_p2p_workflow_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "message": "No workflows in queue",
                "workflow": None,
            }

        async def _add_p2p_peer_fallback(peer_id: str) -> Dict[str, Any]:
            return {
                "success": False,
                "error": "P2P workflow scheduler not available",
                "peer_id": peer_id,
            }

        async def _remove_p2p_peer_fallback(peer_id: str) -> Dict[str, Any]:
            return {
                "success": False,
                "error": "P2P workflow scheduler not available",
                "peer_id": peer_id,
            }

        async def _get_p2p_scheduler_status_fallback() -> Dict[str, Any]:
            return {
                "success": False,
                "error": "P2P workflow scheduler not available",
            }

        async def _calculate_peer_distance_fallback(hash1: str, hash2: str) -> Dict[str, Any]:
            return {
                "success": False,
                "error": "P2P workflow scheduler not available",
                "hash1": hash1,
                "hash2": hash2,
            }

        async def _merge_merkle_clock_fallback(
            other_peer_id: str,
            other_counter: int,
            other_parent_hash: Optional[str] = None,
            other_timestamp: Optional[float] = None,
        ) -> Dict[str, Any]:
            _ = other_parent_hash, other_timestamp
            return {
                "success": False,
                "error": "P2P workflow scheduler not available",
                "other_peer_id": other_peer_id,
                "other_counter": other_counter,
            }

        return {
            "execute_workflow": _execute_fallback,
            "batch_process_datasets": _batch_fallback,
            "create_template": _create_template_fallback,
            "create_workflow": _create_workflow_fallback,
            "schedule_workflow": _schedule_fallback,
            "get_workflow_status": _status_fallback,
            "get_assigned_workflows": _get_assigned_workflows_fallback,
            "get_workflow_tags": _get_workflow_tags_fallback,
            "resume_workflow": _resume_workflow_fallback,
            "get_workflow_metrics": _get_workflow_metrics_fallback,
            "initialize_p2p_scheduler": _initialize_p2p_scheduler_fallback,
            "pause_workflow": _pause_workflow_fallback,
            "list_workflows": _list_workflows_fallback,
            "list_templates": _list_templates_fallback,
            "run_workflow": _run_workflow_fallback,
            "schedule_p2p_workflow": _schedule_p2p_workflow_fallback,
            "get_next_p2p_workflow": _get_next_p2p_workflow_fallback,
            "add_p2p_peer": _add_p2p_peer_fallback,
            "remove_p2p_peer": _remove_p2p_peer_fallback,
            "get_p2p_scheduler_status": _get_p2p_scheduler_status_fallback,
            "calculate_peer_distance": _calculate_peer_distance_fallback,
            "merge_merkle_clock": _merge_merkle_clock_fallback,
        }


_API = _load_workflow_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize tool results to deterministic dictionary envelopes."""
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


async def execute_workflow(
    workflow_definition: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a workflow definition."""
    result = _API["execute_workflow"](
        workflow_definition=workflow_definition,
        workflow_id=workflow_id,
        context=context,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def batch_process_datasets(
    datasets: List[Dict[str, Any]],
    processing_pipeline: List[str],
    batch_size: int = 10,
    parallel_workers: int = 3,
) -> Dict[str, Any]:
    """Process multiple datasets through a workflow pipeline."""
    result = _API["batch_process_datasets"](
        datasets=datasets,
        processing_pipeline=processing_pipeline,
        batch_size=batch_size,
        parallel_workers=parallel_workers,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def create_workflow(
    workflow_id: Optional[str] = None,
    workflow_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or register a workflow definition."""
    result = _API["create_workflow"](
        workflow_id=workflow_id,
        workflow_definition=workflow_definition,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def create_template(template: Dict[str, Any]) -> Dict[str, Any]:
    """Create a reusable workflow template."""
    try:
        if not isinstance(template, dict) or not template:
            return _error_result(
                "template must be a non-empty object",
                template=template,
            )
        result = _API["create_template"](template=template)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("template", template)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), template=template)


async def schedule_workflow(
    workflow_definition: Optional[Dict[str, Any]] = None,
    schedule_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Schedule workflow execution for future or repeated runs."""
    result = _API["schedule_workflow"](
        workflow_definition=workflow_definition,
        schedule_config=schedule_config,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_workflow_status(
    workflow_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    include_details: Optional[bool] = None,
) -> Dict[str, Any]:
    """Get status for a workflow execution."""
    result = _API["get_workflow_status"](
        workflow_id=workflow_id,
        execution_id=execution_id,
        include_details=include_details,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_assigned_workflows() -> Dict[str, Any]:
    """Get workflows assigned to this peer/runtime."""
    result = _API["get_assigned_workflows"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_workflow_tags() -> Dict[str, Any]:
    """Get available workflow tags."""
    result = _API["get_workflow_tags"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def list_templates() -> Dict[str, Any]:
    """List available workflow templates."""
    result = _API["list_templates"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def resume_workflow(workflow_id: str) -> Dict[str, Any]:
    """Resume a paused workflow."""
    try:
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            return _error_result("workflow_id must be a non-empty string", workflow_id=workflow_id)
        workflow_id = workflow_id.strip()
        result = _API["resume_workflow"](workflow_id=workflow_id)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("workflow_id", workflow_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def get_workflow_metrics(
    workflow_id: Optional[str] = None,
    include_performance: Optional[bool] = None,
    time_range: Optional[str] = None,
) -> Dict[str, Any]:
    """Return workflow metrics summary."""
    result = _API["get_workflow_metrics"](
        workflow_id=workflow_id,
        include_performance=include_performance,
        time_range=time_range,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def initialize_p2p_scheduler(
    peer_id: Optional[str] = None,
    peers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Initialize the P2P workflow scheduler."""
    result = _API["initialize_p2p_scheduler"](
        peer_id=peer_id,
        peers=peers,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def pause_workflow(workflow_id: str) -> Dict[str, Any]:
    """Pause a running workflow."""
    try:
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            return _error_result("workflow_id must be a non-empty string", workflow_id=workflow_id)
        workflow_id = workflow_id.strip()
        result = _API["pause_workflow"](workflow_id=workflow_id)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("workflow_id", workflow_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def list_workflows(include_logs: Optional[bool] = None) -> Dict[str, Any]:
    """List available workflows."""
    result = _API["list_workflows"](include_logs=include_logs)
    if hasattr(result, "__await__"):
        return await result
    return result


async def run_workflow(workflow_id: str) -> Dict[str, Any]:
    """Run a registered workflow by ID."""
    try:
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            return _error_result("workflow_id must be a non-empty string", workflow_id=workflow_id)
        workflow_id = workflow_id.strip()
        result = _API["run_workflow"](workflow_id=workflow_id)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("workflow_id", workflow_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def schedule_p2p_workflow(
    workflow_id: str,
    name: str,
    tags: List[str],
    priority: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Schedule a workflow for P2P execution."""
    try:
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            return _error_result("workflow_id must be a non-empty string", workflow_id=workflow_id)
        if not isinstance(name, str) or not name.strip():
            return _error_result("name must be a non-empty string", name=name, workflow_id=workflow_id)
        if not isinstance(tags, list) or not tags or not all(
            isinstance(tag, str) and tag.strip() for tag in tags
        ):
            return _error_result(
                "tags must be a non-empty list of non-empty strings",
                tags=tags,
                workflow_id=workflow_id,
            )
        if not isinstance(priority, (int, float)) or priority < 0:
            return _error_result(
                "priority must be a number >= 0",
                priority=priority,
                workflow_id=workflow_id,
            )
        if metadata is not None and not isinstance(metadata, dict):
            return _error_result(
                "metadata must be an object when provided",
                metadata=metadata,
                workflow_id=workflow_id,
            )

        workflow_id = workflow_id.strip()
        name = name.strip()
        clean_tags = [tag.strip() for tag in tags]

        result = _API["schedule_p2p_workflow"](
            workflow_id=workflow_id,
            name=name,
            tags=clean_tags,
            priority=float(priority),
            metadata=metadata,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("workflow_id", workflow_id)
        envelope.setdefault("name", name)
        envelope.setdefault("tags", clean_tags)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id, name=name)


async def get_next_p2p_workflow() -> Dict[str, Any]:
    """Get the next queued P2P workflow, if any."""
    result = _API["get_next_p2p_workflow"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def add_p2p_peer(peer_id: str) -> Dict[str, Any]:
    """Add a peer to the P2P scheduler membership."""
    try:
        if not isinstance(peer_id, str) or not peer_id.strip():
            return _error_result("peer_id must be a non-empty string", peer_id=peer_id)
        peer_id = peer_id.strip()
        result = _API["add_p2p_peer"](peer_id=peer_id)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("peer_id", peer_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), peer_id=peer_id)


async def remove_p2p_peer(peer_id: str) -> Dict[str, Any]:
    """Remove a peer from P2P scheduler membership."""
    try:
        if not isinstance(peer_id, str) or not peer_id.strip():
            return _error_result("peer_id must be a non-empty string", peer_id=peer_id)
        peer_id = peer_id.strip()
        result = _API["remove_p2p_peer"](peer_id=peer_id)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("peer_id", peer_id)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), peer_id=peer_id)


async def get_p2p_scheduler_status() -> Dict[str, Any]:
    """Get status for the P2P scheduler."""
    result = _API["get_p2p_scheduler_status"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def calculate_peer_distance(hash1: str, hash2: str) -> Dict[str, Any]:
    """Calculate peer-distance/hamming-distance between two hashes."""
    try:
        if not isinstance(hash1, str) or not hash1.strip():
            return _error_result("hash1 must be a non-empty string", hash1=hash1, hash2=hash2)
        if not isinstance(hash2, str) or not hash2.strip():
            return _error_result("hash2 must be a non-empty string", hash1=hash1, hash2=hash2)
        hash1 = hash1.strip()
        hash2 = hash2.strip()
        result = _API["calculate_peer_distance"](hash1=hash1, hash2=hash2)
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("hash1", hash1)
        envelope.setdefault("hash2", hash2)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), hash1=hash1, hash2=hash2)


async def merge_merkle_clock(
    other_peer_id: str,
    other_counter: int,
    other_parent_hash: Optional[str] = None,
    other_timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Merge remote peer merkle-clock state into local scheduler state."""
    try:
        if not isinstance(other_peer_id, str) or not other_peer_id.strip():
            return _error_result(
                "other_peer_id must be a non-empty string",
                other_peer_id=other_peer_id,
                other_counter=other_counter,
            )
        if not isinstance(other_counter, int) or other_counter < 0:
            return _error_result(
                "other_counter must be an integer >= 0",
                other_peer_id=other_peer_id,
                other_counter=other_counter,
            )
        if other_parent_hash is not None and (
            not isinstance(other_parent_hash, str) or not other_parent_hash.strip()
        ):
            return _error_result(
                "other_parent_hash must be a non-empty string when provided",
                other_parent_hash=other_parent_hash,
                other_peer_id=other_peer_id,
            )
        if other_timestamp is not None and not isinstance(other_timestamp, (int, float)):
            return _error_result(
                "other_timestamp must be a number when provided",
                other_timestamp=other_timestamp,
                other_peer_id=other_peer_id,
            )

        other_peer_id = other_peer_id.strip()
        parent_hash = other_parent_hash.strip() if isinstance(other_parent_hash, str) else other_parent_hash

        result = _API["merge_merkle_clock"](
            other_peer_id=other_peer_id,
            other_counter=other_counter,
            other_parent_hash=parent_hash,
            other_timestamp=float(other_timestamp) if isinstance(other_timestamp, (int, float)) else None,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("success", True)
        envelope.setdefault("other_peer_id", other_peer_id)
        envelope.setdefault("other_counter", other_counter)
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            other_peer_id=other_peer_id,
            other_counter=other_counter,
        )


def register_native_workflow_tools_category(manager: Any) -> None:
    """Register native workflow-tools category tools in unified manager."""
    manager.register_tool(
        category="workflow_tools",
        name="execute_workflow",
        func=execute_workflow,
        description="Execute a multi-step workflow definition.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_definition": {"type": ["object", "null"]},
                "workflow_id": {"type": ["string", "null"]},
                "context": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="batch_process_datasets",
        func=batch_process_datasets,
        description="Batch process datasets through a workflow pipeline.",
        input_schema={
            "type": "object",
            "properties": {
                "datasets": {"type": "array", "items": {"type": "object"}},
                "processing_pipeline": {"type": "array", "items": {"type": "string"}},
                "batch_size": {"type": "integer"},
                "parallel_workers": {"type": "integer"},
            },
            "required": ["datasets", "processing_pipeline"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="create_workflow",
        func=create_workflow,
        description="Create or register a workflow definition.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": ["string", "null"]},
                "workflow_definition": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="create_template",
        func=create_template,
        description="Create a reusable workflow template.",
        input_schema={
            "type": "object",
            "properties": {
                "template": {"type": "object", "minProperties": 1},
            },
            "required": ["template"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="schedule_workflow",
        func=schedule_workflow,
        description="Schedule a workflow for future execution.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_definition": {"type": ["object", "null"]},
                "schedule_config": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="get_workflow_status",
        func=get_workflow_status,
        description="Get status and metadata for a workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": ["string", "null"]},
                "execution_id": {"type": ["string", "null"]},
                "include_details": {"type": ["boolean", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="get_assigned_workflows",
        func=get_assigned_workflows,
        description="Get list of workflows assigned to this peer/runtime.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="get_workflow_tags",
        func=get_workflow_tags,
        description="Get available workflow tags.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="list_templates",
        func=list_templates,
        description="List available workflow templates.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="resume_workflow",
        func=resume_workflow,
        description="Resume a paused workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="get_workflow_metrics",
        func=get_workflow_metrics,
        description="Return workflow metrics summary.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": ["string", "null"]},
                "include_performance": {"type": ["boolean", "null"]},
                "time_range": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="initialize_p2p_scheduler",
        func=initialize_p2p_scheduler,
        description="Initialize the P2P workflow scheduler.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": ["string", "null"]},
                "peers": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="pause_workflow",
        func=pause_workflow,
        description="Pause a running workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="list_workflows",
        func=list_workflows,
        description="List workflows.",
        input_schema={
            "type": "object",
            "properties": {
                "include_logs": {"type": ["boolean", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="run_workflow",
        func=run_workflow,
        description="Run a registered workflow by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="schedule_p2p_workflow",
        func=schedule_p2p_workflow,
        description="Schedule a workflow for P2P execution.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "priority": {"type": "number", "minimum": 0, "default": 1.0},
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["workflow_id", "name", "tags"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="get_next_p2p_workflow",
        func=get_next_p2p_workflow,
        description="Get the next queued P2P workflow.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="add_p2p_peer",
        func=add_p2p_peer,
        description="Add a peer to P2P scheduler membership.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string", "minLength": 1},
            },
            "required": ["peer_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="remove_p2p_peer",
        func=remove_p2p_peer,
        description="Remove a peer from P2P scheduler membership.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string", "minLength": 1},
            },
            "required": ["peer_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="get_p2p_scheduler_status",
        func=get_p2p_scheduler_status,
        description="Get current P2P scheduler status.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="calculate_peer_distance",
        func=calculate_peer_distance,
        description="Calculate hamming distance between two peer hashes.",
        input_schema={
            "type": "object",
            "properties": {
                "hash1": {"type": "string", "minLength": 1},
                "hash2": {"type": "string", "minLength": 1},
            },
            "required": ["hash1", "hash2"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )

    manager.register_tool(
        category="workflow_tools",
        name="merge_merkle_clock",
        func=merge_merkle_clock,
        description="Merge remote peer merkle-clock state.",
        input_schema={
            "type": "object",
            "properties": {
                "other_peer_id": {"type": "string", "minLength": 1},
                "other_counter": {"type": "integer", "minimum": 0},
                "other_parent_hash": {"type": ["string", "null"]},
                "other_timestamp": {"type": ["number", "null"]},
            },
            "required": ["other_peer_id", "other_counter"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "workflow-tools"],
    )
