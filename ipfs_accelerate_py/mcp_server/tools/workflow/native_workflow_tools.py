"""Native workflow tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _get_workflow_manager() -> Optional[Any]:
    """Resolve workflow manager lazily to avoid hard dependency at import time."""
    try:
        from ipfs_accelerate_py.mcp.tools.workflows import get_workflow_manager

        return get_workflow_manager()
    except Exception:
        return None


def get_workflow_templates() -> Dict[str, Any]:
    """Return pre-built workflow templates for common AI pipelines.

    This native implementation avoids optional workflow-manager dependencies
    and provides a stable control-plane operation during Wave A migration.
    """
    templates = {
        "image_generation": {
            "name": "Image Generation Pipeline",
            "description": "LLM prompt enhancement -> image generation -> upscaling",
            "use_case": "Create high-quality images with enhanced prompts",
            "models": ["gpt-4", "stable-diffusion-xl", "real-esrgan"],
        },
        "video_generation": {
            "name": "Text-to-Video Pipeline",
            "description": "Enhanced prompt -> image -> animated video",
            "use_case": "Generate videos from text descriptions",
            "models": ["gpt-4", "stable-diffusion-xl", "animatediff"],
        },
        "safe_image": {
            "name": "Safe Image Generation",
            "description": "NSFW filter -> image generation -> quality validation",
            "use_case": "Generate safe, high-quality images with content filtering",
            "models": ["nsfw-text-classifier", "stable-diffusion", "image-quality-scorer"],
        },
        "multimodal": {
            "name": "Multimodal Content Pipeline",
            "description": "Text -> Image -> Audio -> Video generation",
            "use_case": "Create complete multimedia content from text",
            "models": ["gpt-4", "dalle-3", "tts-1", "video-composer"],
        },
    }

    return {
        "status": "success",
        "templates": templates,
        "total": len(templates),
    }


def list_workflows(status: str | None = None) -> Dict[str, Any]:
    """List workflows using the shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    workflows = manager.list_workflows(status=status)
    workflow_list = []
    for wf in workflows:
        progress = wf.get_progress()
        workflow_list.append(
            {
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "description": wf.description,
                "status": wf.status,
                "created_at": wf.created_at,
                "started_at": wf.started_at,
                "completed_at": wf.completed_at,
                "progress": progress,
                "task_count": len(wf.tasks),
                "error": wf.error,
            }
        )

    return {
        "status": "success",
        "workflows": workflow_list,
        "total": len(workflow_list),
    }


def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get detailed workflow information using shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    workflow = manager.get_workflow(workflow_id)
    if not workflow:
        return {
            "status": "error",
            "error": f"Workflow {workflow_id} not found",
        }

    progress = workflow.get_progress()
    tasks_data = []
    for task in workflow.tasks:
        tasks_data.append(
            {
                "task_id": task.task_id,
                "name": task.name,
                "type": task.type,
                "status": task.status,
                "config": task.config,
                "result": task.result,
                "error": task.error,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "dependencies": task.dependencies,
            }
        )

    return {
        "status": "success",
        "workflow": {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "error": workflow.error,
            "progress": progress,
            "tasks": tasks_data,
            "metadata": workflow.metadata,
        },
    }


def create_workflow(name: str, description: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a workflow using shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    workflow = manager.create_workflow(name, description, tasks)
    return {
        "status": "success",
        "workflow_id": workflow.workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "task_count": len(workflow.tasks),
        "created_at": workflow.created_at,
    }


def update_workflow(
    workflow_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Update a workflow using shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    workflow = manager.update_workflow(
        workflow_id=workflow_id,
        name=name,
        description=description,
        tasks=tasks,
    )
    return {
        "status": "success",
        "workflow_id": workflow.workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "task_count": len(workflow.tasks),
        "message": "Workflow updated successfully",
    }


def delete_workflow(workflow_id: str) -> Dict[str, Any]:
    """Delete a workflow using shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    manager.delete_workflow(workflow_id)
    return {
        "status": "success",
        "workflow_id": workflow_id,
        "message": "Workflow deleted successfully",
    }


def start_workflow(workflow_id: str) -> Dict[str, Any]:
    """Start a workflow using shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    manager.start_workflow(workflow_id)
    return {
        "status": "success",
        "workflow_id": workflow_id,
        "message": "Workflow started successfully",
    }


def pause_workflow(workflow_id: str) -> Dict[str, Any]:
    """Pause a workflow using shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    manager.pause_workflow(workflow_id)
    return {
        "status": "success",
        "workflow_id": workflow_id,
        "message": "Workflow paused successfully",
    }


def stop_workflow(workflow_id: str) -> Dict[str, Any]:
    """Stop a workflow using shared workflow manager when available."""
    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "error": "Workflow manager not available",
        }

    manager.stop_workflow(workflow_id)
    return {
        "status": "success",
        "workflow_id": workflow_id,
        "message": "Workflow stopped successfully",
    }


def register_native_workflow_tools(manager: Any) -> None:
    """Register native workflow tools in the unified hierarchical manager."""
    manager.register_tool(
        category="workflow",
        name="get_workflow_templates",
        func=get_workflow_templates,
        description="Return pre-built workflow templates using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="list_workflows",
        func=list_workflows,
        description="List workflows using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="get_workflow",
        func=get_workflow,
        description="Get workflow details using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="create_workflow",
        func=create_workflow,
        description="Create workflow using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "tasks": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["name", "description", "tasks"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="update_workflow",
        func=update_workflow,
        description="Update workflow using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "name": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "tasks": {
                    "type": ["array", "null"],
                    "items": {"type": "object"},
                },
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="delete_workflow",
        func=delete_workflow,
        description="Delete workflow using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="start_workflow",
        func=start_workflow,
        description="Start workflow using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="pause_workflow",
        func=pause_workflow,
        description="Pause workflow using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )

    manager.register_tool(
        category="workflow",
        name="stop_workflow",
        func=stop_workflow,
        description="Stop workflow using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )
