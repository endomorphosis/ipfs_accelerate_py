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
        "success": False,
        "available": bool(_API),
        "engine": engine,
        "method": method,
        "error": message,
    }


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        if "status" not in envelope:
            if envelope.get("error") or envelope.get("success") is False:
                envelope["status"] = "error"
            else:
                envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


async def _invoke_engine_method(engine_name: str, method_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Invoke an engine method with graceful fallback when unavailable."""
    cls = _API.get(engine_name)
    if cls is None:
        return {
            "status": "error",
            "success": False,
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
                "success": False,
                "available": True,
                "engine": engine_name,
                "method": method_name,
                "error": "method_unavailable",
            }

        result = method(**kwargs)
        if hasattr(result, "__await__"):
            result = await result

        envelope = _normalize_payload(result)
        envelope.setdefault("engine", engine_name)
        envelope.setdefault("method", method_name)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
        else:
            envelope.setdefault("success", False)
        return envelope
    except Exception as exc:
        return {
            "status": "error",
            "success": False,
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
            "success": True,
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
        "success": True,
        "available": True,
        "engines": engines,
    }


async def mcplusplus_list_engines() -> Dict[str, List[str]]:
    """List MCP++ engine classes exposed through source compatibility shims."""
    if not _API:
        return {"status": "success", "success": True, "engines": [], "fallback": True}

    return {
        "status": "success",
        "success": True,
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
    retry_policy: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
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
    if retry_policy is not None and not isinstance(retry_policy, dict):
        return _error_result(
            "retry_policy must be an object when provided",
            engine="TaskQueueEngine",
            method="submit",
        )
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result(
            "metadata must be an object when provided",
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
        retry_policy=dict(retry_policy or {}),
        metadata=dict(metadata or {}),
    )


async def mcplusplus_taskqueue_priority(
    task_id: str,
    new_priority: float,
    requeue: bool = True,
) -> Dict[str, Any]:
    """Set task priority through MCP++ TaskQueueEngine adapter."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result(
            "task_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="set_priority",
        )
    if not isinstance(new_priority, (int, float)):
        return _error_result(
            "new_priority must be a number",
            engine="TaskQueueEngine",
            method="set_priority",
        )
    if float(new_priority) <= 0:
        return _error_result(
            "new_priority must be > 0",
            engine="TaskQueueEngine",
            method="set_priority",
        )
    if not isinstance(requeue, bool):
        return _error_result(
            "requeue must be a boolean",
            engine="TaskQueueEngine",
            method="set_priority",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "set_priority",
        task_id=task_id.strip(),
        new_priority=float(new_priority),
        requeue=requeue,
    )


async def mcplusplus_taskqueue_cancel(
    task_id: str,
    reason: str | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Cancel a task through MCP++ TaskQueueEngine adapter."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result(
            "task_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="cancel",
        )
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        return _error_result(
            "reason must be a non-empty string when provided",
            engine="TaskQueueEngine",
            method="cancel",
        )
    if not isinstance(force, bool):
        return _error_result(
            "force must be a boolean",
            engine="TaskQueueEngine",
            method="cancel",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "cancel",
        task_id=task_id.strip(),
        reason=reason.strip() if isinstance(reason, str) else None,
        force=force,
    )


async def mcplusplus_taskqueue_list(
    status_filter: str | None = None,
    worker_filter: str | None = None,
    tag_filter: List[str] | None = None,
    priority_min: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """List tasks through MCP++ TaskQueueEngine adapter."""
    if status_filter is not None and not isinstance(status_filter, str):
        return _error_result(
            "status_filter must be a string",
            engine="TaskQueueEngine",
            method="list_tasks",
        )
    if worker_filter is not None and not isinstance(worker_filter, str):
        return _error_result(
            "worker_filter must be a string",
            engine="TaskQueueEngine",
            method="list_tasks",
        )
    if tag_filter is not None and (
        not isinstance(tag_filter, list)
        or any(not isinstance(item, str) or not item.strip() for item in tag_filter)
    ):
        return _error_result(
            "tag_filter must be an array of non-empty strings",
            engine="TaskQueueEngine",
            method="list_tasks",
        )
    if priority_min is not None and not isinstance(priority_min, (int, float)):
        return _error_result(
            "priority_min must be a number when provided",
            engine="TaskQueueEngine",
            method="list_tasks",
        )
    if not isinstance(limit, int) or limit < 1:
        return _error_result(
            "limit must be an integer >= 1",
            engine="TaskQueueEngine",
            method="list_tasks",
        )
    if not isinstance(offset, int) or offset < 0:
        return _error_result(
            "offset must be an integer >= 0",
            engine="TaskQueueEngine",
            method="list_tasks",
        )

    result = await _invoke_engine_method(
        "TaskQueueEngine",
        "list_tasks",
        status_filter=status_filter.strip() if isinstance(status_filter, str) and status_filter.strip() else None,
        worker_filter=worker_filter.strip() if isinstance(worker_filter, str) and worker_filter.strip() else None,
        tag_filter=[str(item).strip() for item in (tag_filter or [])] or None,
        priority_min=float(priority_min) if isinstance(priority_min, (int, float)) else None,
        limit=limit,
        offset=offset,
    )
    if result.get("status") == "success":
        result.setdefault("tasks", [])
        result.setdefault("limit", limit)
        result.setdefault("offset", offset)
    return result


async def mcplusplus_taskqueue_set_priority(
    task_id: str,
    new_priority: float,
    requeue: bool = True,
) -> Dict[str, Any]:
    """Compatibility alias for task priority updates through TaskQueueEngine."""
    return await mcplusplus_taskqueue_priority(
        task_id=task_id,
        new_priority=new_priority,
        requeue=requeue,
    )


async def mcplusplus_taskqueue_stats(
    include_worker_stats: bool = False,
    include_historical: bool = False,
) -> Dict[str, Any]:
    """Get taskqueue statistics through MCP++ TaskQueueEngine adapter."""
    if not isinstance(include_worker_stats, bool):
        return _error_result(
            "include_worker_stats must be a boolean",
            engine="TaskQueueEngine",
            method="get_stats",
        )
    if not isinstance(include_historical, bool):
        return _error_result(
            "include_historical must be a boolean",
            engine="TaskQueueEngine",
            method="get_stats",
        )

    result = await _invoke_engine_method(
        "TaskQueueEngine",
        "get_stats",
        include_worker_stats=include_worker_stats,
        include_historical=include_historical,
    )
    if result.get("status") == "success":
        result.setdefault("stats", {})
        result.setdefault("include_worker_stats", include_worker_stats)
        result.setdefault("include_historical", include_historical)
    return result


async def mcplusplus_taskqueue_retry(
    task_id: str,
    retry_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Retry a failed task through MCP++ TaskQueueEngine adapter."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result(
            "task_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="retry",
        )
    if retry_config is not None and not isinstance(retry_config, dict):
        return _error_result(
            "retry_config must be an object when provided",
            engine="TaskQueueEngine",
            method="retry",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "retry",
        task_id=task_id.strip(),
        retry_config=dict(retry_config or {}),
    )


async def mcplusplus_taskqueue_pause(
    reason: str | None = None,
) -> Dict[str, Any]:
    """Pause task queue processing through MCP++ TaskQueueEngine adapter."""
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        return _error_result(
            "reason must be a non-empty string when provided",
            engine="TaskQueueEngine",
            method="pause",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "pause",
        reason=(reason.strip() if isinstance(reason, str) else None),
    )


async def mcplusplus_taskqueue_resume(
    reorder_by_priority: bool = True,
) -> Dict[str, Any]:
    """Resume task queue processing through MCP++ TaskQueueEngine adapter."""
    if not isinstance(reorder_by_priority, bool):
        return _error_result(
            "reorder_by_priority must be a boolean",
            engine="TaskQueueEngine",
            method="resume",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "resume",
        reorder_by_priority=reorder_by_priority,
    )


async def mcplusplus_taskqueue_clear(
    status_filter: str | None = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Clear queue entries through MCP++ TaskQueueEngine adapter."""
    if status_filter is not None and not isinstance(status_filter, str):
        return _error_result(
            "status_filter must be a string when provided",
            engine="TaskQueueEngine",
            method="clear",
        )
    if not isinstance(confirm, bool):
        return _error_result(
            "confirm must be a boolean",
            engine="TaskQueueEngine",
            method="clear",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "clear",
        status_filter=(status_filter.strip() if isinstance(status_filter, str) and status_filter.strip() else None),
        confirm=confirm,
    )


async def mcplusplus_worker_register(
    worker_id: str,
    capabilities: List[str],
    max_concurrent_tasks: int = 5,
    resource_limits: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Register worker through MCP++ TaskQueueEngine adapter."""
    if not isinstance(worker_id, str) or not worker_id.strip():
        return _error_result(
            "worker_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="register_worker",
        )
    if not isinstance(capabilities, list) or not capabilities or any(
        not isinstance(item, str) or not item.strip() for item in capabilities
    ):
        return _error_result(
            "capabilities must be a non-empty array of non-empty strings",
            engine="TaskQueueEngine",
            method="register_worker",
        )
    if not isinstance(max_concurrent_tasks, int) or max_concurrent_tasks < 1:
        return _error_result(
            "max_concurrent_tasks must be an integer >= 1",
            engine="TaskQueueEngine",
            method="register_worker",
        )
    if resource_limits is not None and not isinstance(resource_limits, dict):
        return _error_result(
            "resource_limits must be an object when provided",
            engine="TaskQueueEngine",
            method="register_worker",
        )
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result(
            "metadata must be an object when provided",
            engine="TaskQueueEngine",
            method="register_worker",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "register_worker",
        worker_id=worker_id.strip(),
        capabilities=[str(item).strip() for item in capabilities],
        max_concurrent_tasks=max_concurrent_tasks,
        resource_limits=dict(resource_limits or {}),
        metadata=dict(metadata or {}),
    )


async def mcplusplus_worker_unregister(
    worker_id: str,
    graceful: bool = True,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Unregister worker through MCP++ TaskQueueEngine adapter."""
    if not isinstance(worker_id, str) or not worker_id.strip():
        return _error_result(
            "worker_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="unregister_worker",
        )
    if not isinstance(graceful, bool):
        return _error_result(
            "graceful must be a boolean",
            engine="TaskQueueEngine",
            method="unregister_worker",
        )
    if not isinstance(timeout, int) or timeout < 1:
        return _error_result(
            "timeout must be an integer >= 1",
            engine="TaskQueueEngine",
            method="unregister_worker",
        )

    return await _invoke_engine_method(
        "TaskQueueEngine",
        "unregister_worker",
        worker_id=worker_id.strip(),
        graceful=graceful,
        timeout=timeout,
    )


async def mcplusplus_worker_status(
    worker_id: str,
    include_tasks: bool = False,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """Get worker status through MCP++ TaskQueueEngine adapter."""
    if not isinstance(worker_id, str) or not worker_id.strip():
        return _error_result(
            "worker_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="get_worker_status",
        )
    if not isinstance(include_tasks, bool):
        return _error_result(
            "include_tasks must be a boolean",
            engine="TaskQueueEngine",
            method="get_worker_status",
        )
    if not isinstance(include_metrics, bool):
        return _error_result(
            "include_metrics must be a boolean",
            engine="TaskQueueEngine",
            method="get_worker_status",
        )

    result = await _invoke_engine_method(
        "TaskQueueEngine",
        "get_worker_status",
        worker_id=worker_id.strip(),
        include_tasks=include_tasks,
        include_metrics=include_metrics,
    )
    if result.get("status") == "success":
        result.setdefault("worker_id", worker_id.strip())
        result.setdefault("status", "unknown")
        result.setdefault("running_tasks", 0)
        result.setdefault("completed_tasks", 0)
        result.setdefault("failed_tasks", 0)
        result.setdefault("tasks", [] if include_tasks else None)
        result.setdefault("metrics", {} if include_metrics else None)
        result.setdefault("include_tasks", include_tasks)
        result.setdefault("include_metrics", include_metrics)
    return result


async def mcplusplus_taskqueue_result(
    task_id: str,
    include_output: bool = True,
    include_logs: bool = False,
) -> Dict[str, Any]:
    """Fetch task result through MCP++ TaskQueueEngine adapter."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result(
            "task_id must be a non-empty string",
            engine="TaskQueueEngine",
            method="get_result",
        )
    if not isinstance(include_output, bool):
        return _error_result(
            "include_output must be a boolean",
            engine="TaskQueueEngine",
            method="get_result",
        )
    if not isinstance(include_logs, bool):
        return _error_result(
            "include_logs must be a boolean",
            engine="TaskQueueEngine",
            method="get_result",
        )

    result = await _invoke_engine_method(
        "TaskQueueEngine",
        "get_result",
        task_id=task_id.strip(),
        include_output=include_output,
        include_logs=include_logs,
    )
    if result.get("status") == "success":
        result.setdefault("task_id", task_id.strip())
        result.setdefault("result", None)
        result.setdefault("output", None if include_output else None)
        result.setdefault("logs", [] if include_logs else None)
        result.setdefault("execution_time", 0)
        result.setdefault("include_output", include_output)
        result.setdefault("include_logs", include_logs)
    return result


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

    result = await _invoke_engine_method(
        "WorkflowEngine",
        "get_status",
        workflow_id=workflow_id.strip(),
        include_steps=include_steps,
        include_metrics=include_metrics,
    )
    if result.get("status") == "success":
        result.setdefault("workflow_id", workflow_id.strip())
        result.setdefault("progress", 0)
        result.setdefault("current_step", None)
        result.setdefault("peer_id", None)
        result.setdefault("start_time", None)
        result.setdefault("end_time", None)
        result.setdefault("steps", [] if include_steps else None)
        result.setdefault("metrics", {} if include_metrics else None)
        result.setdefault("include_steps", include_steps)
        result.setdefault("include_metrics", include_metrics)
    return result


async def mcplusplus_workflow_submit(
    workflow_id: str,
    name: str,
    steps: List[Dict[str, Any]],
    priority: float = 1.0,
    tags: List[str] | None = None,
    dependencies: List[str] | None = None,
    metadata: Dict[str, Any] | None = None,
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
    if dependencies is not None and (
        not isinstance(dependencies, list)
        or any(not isinstance(item, str) or not item.strip() for item in dependencies)
    ):
        return _error_result(
            "dependencies must be an array of non-empty strings",
            engine="WorkflowEngine",
            method="submit",
        )
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result(
            "metadata must be an object when provided",
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
        dependencies=[str(item).strip() for item in (dependencies or [])],
        metadata=dict(metadata or {}),
    )


async def mcplusplus_workflow_cancel(
    workflow_id: str,
    reason: str | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Cancel workflow through MCP++ WorkflowEngine adapter."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result(
            "workflow_id must be a non-empty string",
            engine="WorkflowEngine",
            method="cancel",
        )
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        return _error_result(
            "reason must be a non-empty string when provided",
            engine="WorkflowEngine",
            method="cancel",
        )
    if not isinstance(force, bool):
        return _error_result(
            "force must be a boolean",
            engine="WorkflowEngine",
            method="cancel",
        )

    return await _invoke_engine_method(
        "WorkflowEngine",
        "cancel",
        workflow_id=workflow_id.strip(),
        reason=reason.strip() if isinstance(reason, str) else None,
        force=force,
    )


async def mcplusplus_workflow_list(
    status_filter: str | None = None,
    peer_filter: str | None = None,
    tag_filter: List[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List workflows through MCP++ WorkflowEngine adapter."""
    if status_filter is not None and not isinstance(status_filter, str):
        return _error_result(
            "status_filter must be a string",
            engine="WorkflowEngine",
            method="list_workflows",
        )
    if peer_filter is not None and not isinstance(peer_filter, str):
        return _error_result(
            "peer_filter must be a string",
            engine="WorkflowEngine",
            method="list_workflows",
        )
    if tag_filter is not None and (
        not isinstance(tag_filter, list)
        or any(not isinstance(item, str) or not item.strip() for item in tag_filter)
    ):
        return _error_result(
            "tag_filter must be an array of non-empty strings",
            engine="WorkflowEngine",
            method="list_workflows",
        )
    if not isinstance(limit, int) or limit < 1:
        return _error_result(
            "limit must be an integer >= 1",
            engine="WorkflowEngine",
            method="list_workflows",
        )
    if not isinstance(offset, int) or offset < 0:
        return _error_result(
            "offset must be an integer >= 0",
            engine="WorkflowEngine",
            method="list_workflows",
        )

    result = await _invoke_engine_method(
        "WorkflowEngine",
        "list_workflows",
        status_filter=status_filter.strip() if isinstance(status_filter, str) and status_filter.strip() else None,
        peer_filter=peer_filter.strip() if isinstance(peer_filter, str) and peer_filter.strip() else None,
        tag_filter=[str(item).strip() for item in (tag_filter or [])] or None,
        limit=limit,
        offset=offset,
    )
    if result.get("status") == "success":
        result.setdefault("workflows", [])
        result.setdefault("limit", limit)
        result.setdefault("offset", offset)
    return result


async def mcplusplus_workflow_dependencies(
    workflow_id: str,
    fmt: str = "json",
) -> Dict[str, Any]:
    """Get workflow dependency graph via MCP++ WorkflowEngine adapter."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result(
            "workflow_id must be a non-empty string",
            engine="WorkflowEngine",
            method="get_dependencies",
        )
    if not isinstance(fmt, str) or not fmt.strip():
        return _error_result(
            "fmt must be a non-empty string",
            engine="WorkflowEngine",
            method="get_dependencies",
        )

    normalized_fmt = fmt.strip().lower()
    if normalized_fmt not in {"json", "dot", "mermaid"}:
        return _error_result(
            "fmt must be one of: json, dot, mermaid",
            engine="WorkflowEngine",
            method="get_dependencies",
        )

    result = await _invoke_engine_method(
        "WorkflowEngine",
        "get_dependencies",
        workflow_id=workflow_id.strip(),
        fmt=normalized_fmt,
    )
    if result.get("status") == "success":
        result.setdefault("workflow_id", workflow_id.strip())
        result.setdefault("dag", None)
        result.setdefault("nodes", [])
        result.setdefault("edges", [])
        result.setdefault("critical_path", [])
        result.setdefault("fmt", normalized_fmt)
    return result


async def mcplusplus_workflow_result(
    workflow_id: str,
    include_outputs: bool = True,
    include_logs: bool = False,
) -> Dict[str, Any]:
    """Fetch workflow result through MCP++ WorkflowEngine adapter."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        return _error_result(
            "workflow_id must be a non-empty string",
            engine="WorkflowEngine",
            method="get_result",
        )
    if not isinstance(include_outputs, bool):
        return _error_result(
            "include_outputs must be a boolean",
            engine="WorkflowEngine",
            method="get_result",
        )
    if not isinstance(include_logs, bool):
        return _error_result(
            "include_logs must be a boolean",
            engine="WorkflowEngine",
            method="get_result",
        )

    result = await _invoke_engine_method(
        "WorkflowEngine",
        "get_result",
        workflow_id=workflow_id.strip(),
        include_outputs=include_outputs,
        include_logs=include_logs,
    )
    if result.get("status") == "success":
        result.setdefault("workflow_id", workflow_id.strip())
        result.setdefault("result", None)
        result.setdefault("execution_time", None)
        result.setdefault("outputs", [] if include_outputs else None)
        result.setdefault("logs", [] if include_logs else None)
        result.setdefault("include_outputs", include_outputs)
        result.setdefault("include_logs", include_logs)
    return result


async def mcplusplus_peer_list(
    status_filter: str = "",
    capability_filter: List[str] | None = None,
    sort_by: str = "last_seen",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List peers via MCP++ PeerEngine adapter."""
    if status_filter is not None and not isinstance(status_filter, str):
        return _error_result(
            "status_filter must be a string",
            engine="PeerEngine",
            method="list_peers",
        )
    if capability_filter is not None and (
        not isinstance(capability_filter, list)
        or any(not isinstance(item, str) or not item.strip() for item in capability_filter)
    ):
        return _error_result(
            "capability_filter must be an array of non-empty strings",
            engine="PeerEngine",
            method="list_peers",
        )
    if not isinstance(sort_by, str) or not sort_by.strip():
        return _error_result(
            "sort_by must be a non-empty string",
            engine="PeerEngine",
            method="list_peers",
        )
    if not isinstance(limit, int) or limit < 1:
        return _error_result(
            "limit must be an integer >= 1",
            engine="PeerEngine",
            method="list_peers",
        )
    if not isinstance(offset, int) or offset < 0:
        return _error_result(
            "offset must be an integer >= 0",
            engine="PeerEngine",
            method="list_peers",
        )

    normalized_status_filter = (status_filter or "").strip()

    result = await _invoke_engine_method(
        "PeerEngine",
        "list_peers",
        status_filter=(normalized_status_filter or None),
        capability_filter=[str(item).strip() for item in (capability_filter or [])] or None,
        sort_by=sort_by.strip(),
        limit=limit,
        offset=offset,
    )
    if result.get("status") == "success":
        result.setdefault("peers", [])
        result.setdefault("limit", limit)
        result.setdefault("offset", offset)
        result.setdefault("sort_by", sort_by.strip())
    return result


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

    result = await _invoke_engine_method(
        "PeerEngine",
        "discover",
        capability_filter=[str(item).strip() for item in (capability_filter or [])] or None,
        max_peers=max_peers,
        timeout=timeout,
        include_metrics=include_metrics,
    )
    if result.get("status") == "success":
        result.setdefault("peers", [])
        result.setdefault("discovered_count", 0)
        result.setdefault("search_time", 0)
        result.setdefault("capability_filter", [str(item).strip() for item in (capability_filter or [])] or None)
        result.setdefault("max_peers", max_peers)
        result.setdefault("timeout", timeout)
        result.setdefault("include_metrics", include_metrics)
    return result


async def mcplusplus_peer_connect(
    peer_id: str,
    multiaddr: str,
    timeout: int = 30,
    retry_count: int = 3,
    persist: bool = True,
) -> Dict[str, Any]:
    """Connect peer through MCP++ PeerEngine adapter."""
    if not isinstance(peer_id, str) or not peer_id.strip():
        return _error_result(
            "peer_id must be a non-empty string",
            engine="PeerEngine",
            method="connect",
        )
    if not isinstance(multiaddr, str) or not multiaddr.strip():
        return _error_result(
            "multiaddr must be a non-empty string",
            engine="PeerEngine",
            method="connect",
        )
    if not isinstance(timeout, int) or timeout < 1:
        return _error_result(
            "timeout must be an integer >= 1",
            engine="PeerEngine",
            method="connect",
        )
    if not isinstance(retry_count, int) or retry_count < 0:
        return _error_result(
            "retry_count must be an integer >= 0",
            engine="PeerEngine",
            method="connect",
        )
    if not isinstance(persist, bool):
        return _error_result(
            "persist must be a boolean",
            engine="PeerEngine",
            method="connect",
        )

    return await _invoke_engine_method(
        "PeerEngine",
        "connect",
        peer_id=peer_id.strip(),
        multiaddr=multiaddr.strip(),
        timeout=timeout,
        retry_count=retry_count,
        persist=persist,
    )


async def mcplusplus_peer_disconnect(
    peer_id: str,
    reason: str | None = None,
    graceful: bool = True,
) -> Dict[str, Any]:
    """Disconnect peer through MCP++ PeerEngine adapter."""
    if not isinstance(peer_id, str) or not peer_id.strip():
        return _error_result(
            "peer_id must be a non-empty string",
            engine="PeerEngine",
            method="disconnect",
        )
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        return _error_result(
            "reason must be a non-empty string when provided",
            engine="PeerEngine",
            method="disconnect",
        )
    if not isinstance(graceful, bool):
        return _error_result(
            "graceful must be a boolean",
            engine="PeerEngine",
            method="disconnect",
        )

    return await _invoke_engine_method(
        "PeerEngine",
        "disconnect",
        peer_id=peer_id.strip(),
        reason=reason.strip() if isinstance(reason, str) else None,
        graceful=graceful,
    )


async def mcplusplus_peer_metrics(
    peer_id: str,
    include_history: bool = False,
    history_hours: int = 24,
) -> Dict[str, Any]:
    """Get peer metrics through MCP++ PeerEngine adapter."""
    if not isinstance(peer_id, str) or not peer_id.strip():
        return _error_result(
            "peer_id must be a non-empty string",
            engine="PeerEngine",
            method="get_metrics",
        )
    if not isinstance(include_history, bool):
        return _error_result(
            "include_history must be a boolean",
            engine="PeerEngine",
            method="get_metrics",
        )
    if not isinstance(history_hours, int) or history_hours < 1:
        return _error_result(
            "history_hours must be an integer >= 1",
            engine="PeerEngine",
            method="get_metrics",
        )

    result = await _invoke_engine_method(
        "PeerEngine",
        "get_metrics",
        peer_id=peer_id.strip(),
        include_history=include_history,
        history_hours=history_hours,
    )
    if result.get("status") == "success":
        result.setdefault("peer_id", peer_id.strip())
        result.setdefault("current_metrics", {})
        result.setdefault("connection_info", {})
        result.setdefault("timestamp", None)
        if include_history:
            result.setdefault("history", [])
            result.setdefault("history_hours", history_hours)
        result.setdefault("include_history", include_history)
    return result


async def mcplusplus_peer_bootstrap_network(
    bootstrap_nodes: List[str] | None = None,
    timeout: int = 60,
    min_connections: int = 3,
    max_connections: int = 10,
) -> Dict[str, Any]:
    """Bootstrap network through MCP++ PeerEngine adapter."""
    if bootstrap_nodes is not None and (
        not isinstance(bootstrap_nodes, list)
        or any(not isinstance(item, str) or not item.strip() for item in bootstrap_nodes)
    ):
        return _error_result(
            "bootstrap_nodes must be an array of non-empty strings",
            engine="PeerEngine",
            method="bootstrap",
        )
    if not isinstance(timeout, int) or timeout < 1:
        return _error_result(
            "timeout must be an integer >= 1",
            engine="PeerEngine",
            method="bootstrap",
        )
    if not isinstance(min_connections, int) or min_connections < 0:
        return _error_result(
            "min_connections must be an integer >= 0",
            engine="PeerEngine",
            method="bootstrap",
        )
    if not isinstance(max_connections, int) or max_connections < 1:
        return _error_result(
            "max_connections must be an integer >= 1",
            engine="PeerEngine",
            method="bootstrap",
        )
    if min_connections > max_connections:
        return _error_result(
            "min_connections must be <= max_connections",
            engine="PeerEngine",
            method="bootstrap",
        )

    return await _invoke_engine_method(
        "PeerEngine",
        "bootstrap",
        bootstrap_nodes=[str(item).strip() for item in (bootstrap_nodes or [])] or None,
        timeout=timeout,
        min_connections=min_connections,
        max_connections=max_connections,
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
                "retry_policy": {"type": ["object", "null"]},
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["task_id", "task_type", "payload"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_priority",
        func=mcplusplus_taskqueue_priority,
        description="Set task priority through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "new_priority": {"type": "number", "exclusiveMinimum": 0},
                "requeue": {"type": "boolean", "default": True},
            },
            "required": ["task_id", "new_priority"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_cancel",
        func=mcplusplus_taskqueue_cancel,
        description="Cancel task through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["task_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_list",
        func=mcplusplus_taskqueue_list,
        description="List tasks through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "status_filter": {"type": "string"},
                "worker_filter": {"type": "string"},
                "tag_filter": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "priority_min": {"type": "number"},
                "limit": {"type": "integer", "minimum": 1, "default": 100},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_set_priority",
        func=mcplusplus_taskqueue_set_priority,
        description="Update task priority through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "new_priority": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
                "requeue": {"type": "boolean", "default": True},
            },
            "required": ["task_id", "new_priority"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_stats",
        func=mcplusplus_taskqueue_stats,
        description="Get task queue statistics through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "include_worker_stats": {"type": "boolean", "default": False},
                "include_historical": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_retry",
        func=mcplusplus_taskqueue_retry,
        description="Retry a failed task through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "retry_config": {"type": ["object", "null"]},
            },
            "required": ["task_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_pause",
        func=mcplusplus_taskqueue_pause,
        description="Pause task queue processing through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "reason": {"type": ["string", "null"], "minLength": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_resume",
        func=mcplusplus_taskqueue_resume,
        description="Resume task queue processing through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "reorder_by_priority": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_clear",
        func=mcplusplus_taskqueue_clear,
        description="Clear task queue entries through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "status_filter": {"type": ["string", "null"], "minLength": 1},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_worker_register",
        func=mcplusplus_worker_register,
        description="Register worker through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "worker_id": {"type": "string", "minLength": 1},
                "capabilities": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "max_concurrent_tasks": {"type": "integer", "minimum": 1, "default": 5},
                "resource_limits": {"type": ["object", "null"]},
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["worker_id", "capabilities"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_worker_unregister",
        func=mcplusplus_worker_unregister,
        description="Unregister worker through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "worker_id": {"type": "string", "minLength": 1},
                "graceful": {"type": "boolean", "default": True},
                "timeout": {"type": "integer", "minimum": 1, "default": 300},
            },
            "required": ["worker_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_worker_status",
        func=mcplusplus_worker_status,
        description="Get worker status through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "worker_id": {"type": "string", "minLength": 1},
                "include_tasks": {"type": "boolean", "default": False},
                "include_metrics": {"type": "boolean", "default": False},
            },
            "required": ["worker_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_taskqueue_result",
        func=mcplusplus_taskqueue_result,
        description="Get task result through TaskQueueEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "include_output": {"type": "boolean", "default": True},
                "include_logs": {"type": "boolean", "default": False},
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
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "metadata": {"type": ["object", "null"]},
            },
            "required": ["workflow_id", "name", "steps"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_workflow_cancel",
        func=mcplusplus_workflow_cancel,
        description="Cancel workflow through WorkflowEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_workflow_list",
        func=mcplusplus_workflow_list,
        description="List workflows through WorkflowEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "status_filter": {"type": "string"},
                "peer_filter": {"type": "string"},
                "tag_filter": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "limit": {"type": "integer", "minimum": 1, "default": 50},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_workflow_dependencies",
        func=mcplusplus_workflow_dependencies,
        description="Get workflow dependency graph through WorkflowEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
                "fmt": {"type": "string", "enum": ["json", "dot", "mermaid"], "default": "json"},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_workflow_result",
        func=mcplusplus_workflow_result,
        description="Get workflow result through WorkflowEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
                "include_outputs": {"type": "boolean", "default": True},
                "include_logs": {"type": "boolean", "default": False},
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
                "capability_filter": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "sort_by": {"type": "string", "minLength": 1, "default": "last_seen"},
                "limit": {"type": "integer", "minimum": 1, "default": 50},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
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

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_peer_connect",
        func=mcplusplus_peer_connect,
        description="Connect peer through PeerEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string", "minLength": 1},
                "multiaddr": {"type": "string", "minLength": 1},
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
                "retry_count": {"type": "integer", "minimum": 0, "default": 3},
                "persist": {"type": "boolean", "default": True},
            },
            "required": ["peer_id", "multiaddr"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_peer_disconnect",
        func=mcplusplus_peer_disconnect,
        description="Disconnect peer through PeerEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
                "graceful": {"type": "boolean", "default": True},
            },
            "required": ["peer_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_peer_metrics",
        func=mcplusplus_peer_metrics,
        description="Get peer metrics through PeerEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string", "minLength": 1},
                "include_history": {"type": "boolean", "default": False},
                "history_hours": {"type": "integer", "minimum": 1, "default": 24},
            },
            "required": ["peer_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

    manager.register_tool(
        category="mcplusplus",
        name="mcplusplus_peer_bootstrap_network",
        func=mcplusplus_peer_bootstrap_network,
        description="Bootstrap peer network through PeerEngine adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "bootstrap_nodes": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "timeout": {"type": "integer", "minimum": 1, "default": 60},
                "min_connections": {"type": "integer", "minimum": 0, "default": 3},
                "max_connections": {"type": "integer", "minimum": 1, "default": 10},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "mcplusplus"],
    )

