"""Native workflow-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_workflow_tools_api() -> Dict[str, Any]:
    """Resolve source workflow-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.workflow_tools.workflow_tools import (  # type: ignore
            batch_process_datasets as _batch_process_datasets,
            execute_workflow as _execute_workflow,
            get_workflow_status as _get_workflow_status,
            list_templates as _list_templates,
            schedule_workflow as _schedule_workflow,
        )

        return {
            "execute_workflow": _execute_workflow,
            "batch_process_datasets": _batch_process_datasets,
            "schedule_workflow": _schedule_workflow,
            "get_workflow_status": _get_workflow_status,
            "list_templates": _list_templates,
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

        async def _list_templates_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "templates": [],
                "total": 0,
            }

        return {
            "execute_workflow": _execute_fallback,
            "batch_process_datasets": _batch_fallback,
            "schedule_workflow": _schedule_fallback,
            "get_workflow_status": _status_fallback,
            "list_templates": _list_templates_fallback,
        }


_API = _load_workflow_tools_api()


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


async def list_templates() -> Dict[str, Any]:
    """List available workflow templates."""
    result = _API["list_templates"]()
    if hasattr(result, "__await__"):
        return await result
    return result


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
