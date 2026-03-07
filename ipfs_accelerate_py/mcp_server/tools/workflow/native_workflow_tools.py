"""Native workflow tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _error_result(message: str) -> Dict[str, Any]:
    return {"status": "error", "success": False, "error": message}


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
        "success": True,
        "templates": templates,
        "total": len(templates),
    }


def list_workflows(status: str | None = None) -> Dict[str, Any]:
    """List workflows using the shared workflow manager when available."""
    if status is not None and not isinstance(status, str):
        return _error_result("status must be a string or null")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    normalized_status = (status or "").strip() or None
    workflows = manager.list_workflows(status=normalized_status)
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
        "success": True,
        "workflows": workflow_list,
        "total": len(workflow_list),
    }


def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get detailed workflow information using shared workflow manager when available."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    workflow = manager.get_workflow(workflow_id.strip())
    if not workflow:
        return {
            "status": "error",
            "success": False,
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
        "success": True,
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
    if not isinstance(name, str) or not name.strip():
        return _error_result("name must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        return _error_result("description must be a non-empty string")
    if not isinstance(tasks, list):
        return _error_result("tasks must be a list")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    workflow = manager.create_workflow(name.strip(), description.strip(), tasks)
    return {
        "status": "success",
        "success": True,
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
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        return _error_result("name must be null or a non-empty string")
    if description is not None and (not isinstance(description, str) or not description.strip()):
        return _error_result("description must be null or a non-empty string")
    if tasks is not None and not isinstance(tasks, list):
        return _error_result("tasks must be null or a list")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    workflow = manager.update_workflow(
        workflow_id=workflow_id.strip(),
        name=name.strip() if isinstance(name, str) else name,
        description=description.strip() if isinstance(description, str) else description,
        tasks=tasks,
    )
    return {
        "status": "success",
        "success": True,
        "workflow_id": workflow.workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "task_count": len(workflow.tasks),
        "message": "Workflow updated successfully",
    }


def delete_workflow(workflow_id: str) -> Dict[str, Any]:
    """Delete a workflow using shared workflow manager when available."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    manager.delete_workflow(workflow_id.strip())
    return {
        "status": "success",
        "success": True,
        "workflow_id": workflow_id,
        "message": "Workflow deleted successfully",
    }


def start_workflow(workflow_id: str) -> Dict[str, Any]:
    """Start a workflow using shared workflow manager when available."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    manager.start_workflow(workflow_id.strip())
    return {
        "status": "success",
        "success": True,
        "workflow_id": workflow_id,
        "message": "Workflow started successfully",
    }


def pause_workflow(workflow_id: str) -> Dict[str, Any]:
    """Pause a workflow using shared workflow manager when available."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    manager.pause_workflow(workflow_id.strip())
    return {
        "status": "success",
        "success": True,
        "workflow_id": workflow_id,
        "message": "Workflow paused successfully",
    }


def stop_workflow(workflow_id: str) -> Dict[str, Any]:
    """Stop a workflow using shared workflow manager when available."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_workflow_manager()
    if not manager:
        return {
            "status": "error",
            "success": False,
            "error": "Workflow manager not available",
        }

    manager.stop_workflow(workflow_id.strip())
    return {
        "status": "success",
        "success": True,
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
                "workflow_id": {"type": "string", "minLength": 1},
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
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
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
                "workflow_id": {"type": "string", "minLength": 1},
                "name": {"type": ["string", "null"], "minLength": 1},
                "description": {"type": ["string", "null"], "minLength": 1},
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
                "workflow_id": {"type": "string", "minLength": 1},
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
                "workflow_id": {"type": "string", "minLength": 1},
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
                "workflow_id": {"type": "string", "minLength": 1},
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
                "workflow_id": {"type": "string", "minLength": 1},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "workflow"],
    )
