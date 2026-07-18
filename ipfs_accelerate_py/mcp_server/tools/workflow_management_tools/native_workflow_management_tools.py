"""Native workflow management tool implementations for unified mcp_server.

Migrated from ipfs_accelerate_py/mcp/tools/workflows.py.
Operations: create_workflow, list_workflows, get_workflow, start_workflow,
pause_workflow, stop_workflow, update_workflow, delete_workflow,
get_workflow_templates, create_workflow_from_template.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_workflow_management_api() -> Dict[str, Any]:
    """Resolve source workflow management APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.workflow_manager import (  # type: ignore
            get_workflow_manager as _get_wm,
        )

        return {"get_workflow_manager": _get_wm}
    except Exception:
        logger.warning("Workflow manager unavailable, using fallback stubs")
        return {}


_API = _load_workflow_management_api()


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
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _get_manager():
    """Return the workflow manager or None if unavailable."""
    factory = _API.get("get_workflow_manager")
    if callable(factory):
        try:
            return factory()
        except Exception as exc:
            logger.warning("get_workflow_manager() failed: %s", exc)
    return None


async def workflow_management_inventory() -> Dict[str, Any]:
    """Return inventory metadata for workflow management tools."""
    return _normalize_payload(
        {
            "category": "workflow_management_tools",
            "tools": [
                "create_workflow",
                "list_workflows",
                "get_workflow",
                "start_workflow",
                "pause_workflow",
                "stop_workflow",
                "update_workflow",
                "delete_workflow",
                "get_workflow_templates",
                "create_workflow_from_template",
            ],
            "description": "Local workflow CRUD and lifecycle management",
            "source": "mcp/tools/workflows.py",
        }
    )


async def create_workflow(
    name: str,
    description: str,
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a new AI model pipeline workflow."""
    if not isinstance(name, str) or not name.strip():
        return _error_result("name must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        workflow = manager.create_workflow(
            name.strip(), description or "", tasks or []
        )
        return _normalize_payload(
            {
                "workflow_id": getattr(workflow, "workflow_id", None),
                "name": getattr(workflow, "name", name.strip()),
                "description": getattr(workflow, "description", description),
                "task_count": len(tasks or []),
                "created_at": getattr(workflow, "created_at", None),
            }
        )
    except Exception as exc:
        return _error_result(str(exc), name=name)


async def list_workflows(status: Optional[str] = None) -> Dict[str, Any]:
    """List all workflows, optionally filtered by status."""
    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        workflows = manager.list_workflows(status=status)
        if not isinstance(workflows, list):
            workflows = []
        return _normalize_payload({"workflows": workflows, "count": len(workflows)})
    except Exception as exc:
        return _error_result(str(exc))


async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get details of a specific workflow."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        workflow = manager.get_workflow(workflow_id.strip())
        if workflow is None:
            return _error_result(
                f"Workflow '{workflow_id}' not found", workflow_id=workflow_id
            )
        if isinstance(workflow, dict):
            return _normalize_payload(workflow)
        return _normalize_payload(
            {
                "workflow_id": getattr(workflow, "workflow_id", workflow_id),
                "name": getattr(workflow, "name", ""),
                "description": getattr(workflow, "description", ""),
                "status": getattr(workflow, "status", ""),
                "tasks": getattr(workflow, "tasks", []),
            }
        )
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def start_workflow(workflow_id: str) -> Dict[str, Any]:
    """Start a workflow by ID."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        result = manager.start_workflow(workflow_id.strip())
        return _normalize_payload(
            result if isinstance(result, dict) else {"workflow_id": workflow_id, "started": True}
        )
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def pause_workflow(workflow_id: str) -> Dict[str, Any]:
    """Pause a running workflow."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        result = manager.pause_workflow(workflow_id.strip())
        return _normalize_payload(
            result if isinstance(result, dict) else {"workflow_id": workflow_id, "paused": True}
        )
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def stop_workflow(workflow_id: str) -> Dict[str, Any]:
    """Stop a running or paused workflow."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        result = manager.stop_workflow(workflow_id.strip())
        return _normalize_payload(
            result if isinstance(result, dict) else {"workflow_id": workflow_id, "stopped": True}
        )
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def update_workflow(
    workflow_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Update workflow properties."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        updates: Dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if tasks is not None:
            updates["tasks"] = tasks
        result = manager.update_workflow(workflow_id.strip(), **updates)
        return _normalize_payload(
            result if isinstance(result, dict) else {"workflow_id": workflow_id, "updated": True}
        )
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def delete_workflow(workflow_id: str) -> Dict[str, Any]:
    """Delete a workflow by ID."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result("workflow_id must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        result = manager.delete_workflow(workflow_id.strip())
        return _normalize_payload(
            result if isinstance(result, dict) else {"workflow_id": workflow_id, "deleted": True}
        )
    except Exception as exc:
        return _error_result(str(exc), workflow_id=workflow_id)


async def get_workflow_templates() -> Dict[str, Any]:
    """Get available workflow templates."""
    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        templates = manager.get_workflow_templates()
        if not isinstance(templates, list):
            templates = []
        return _normalize_payload({"templates": templates, "count": len(templates)})
    except Exception as exc:
        return _error_result(str(exc))


async def create_workflow_from_template(
    template_id: str,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a workflow from a named template."""
    if not isinstance(template_id, str) or not template_id.strip():
        return _error_result("template_id must be a non-empty string")
    if not isinstance(name, str) or not name.strip():
        return _error_result("name must be a non-empty string")

    manager = _get_manager()
    if manager is None:
        return _error_result("Workflow manager unavailable")
    try:
        result = manager.create_workflow_from_template(
            template_id=template_id.strip(),
            name=name.strip(),
            config=config or {},
        )
        if isinstance(result, dict):
            return _normalize_payload(result)
        return _normalize_payload(
            {
                "workflow_id": getattr(result, "workflow_id", None),
                "name": name.strip(),
                "template_id": template_id.strip(),
            }
        )
    except Exception as exc:
        return _error_result(str(exc), template_id=template_id, name=name)


def register_native_workflow_management_tools(manager: Any) -> None:
    """Register native workflow management tools in the unified hierarchical manager."""
    manager.register_tool(
        category="workflow_management_tools",
        name="workflow_management_inventory",
        func=workflow_management_inventory,
        description="Return inventory metadata for workflow management tools.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="create_workflow",
        func=create_workflow,
        description="Create a new AI model pipeline workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Workflow name."},
                "description": {"type": "string", "description": "Workflow description."},
                "tasks": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of task definitions.",
                },
            },
            "required": ["name", "description"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="list_workflows",
        func=list_workflows,
        description="List all workflows, optionally filtered by status.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: pending, running, paused, completed, failed, stopped.",
                }
            },
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="get_workflow",
        func=get_workflow,
        description="Get details of a specific workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"}
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="start_workflow",
        func=start_workflow,
        description="Start a workflow by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"}
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="pause_workflow",
        func=pause_workflow,
        description="Pause a running workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"}
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="stop_workflow",
        func=stop_workflow,
        description="Stop a running or paused workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"}
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="update_workflow",
        func=update_workflow,
        description="Update workflow properties (name, description, tasks).",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "tasks": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="delete_workflow",
        func=delete_workflow,
        description="Delete a workflow by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"}
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="get_workflow_templates",
        func=get_workflow_templates,
        description="Get available workflow templates.",
        input_schema={"type": "object", "properties": {}},
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
    manager.register_tool(
        category="workflow_management_tools",
        name="create_workflow_from_template",
        func=create_workflow_from_template,
        description="Create a workflow from a named template.",
        input_schema={
            "type": "object",
            "properties": {
                "template_id": {"type": "string"},
                "name": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["template_id", "name"],
        },
        runtime="fastapi",
        tags=["native", "workflow", "management"],
    )
