"""Native p2p tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


async def _request_status(
    remote_multiaddr: str = "",
    peer_id: str = "",
    timeout_s: float = 10.0,
    detail: bool = False,
) -> Any:
    """Request TaskQueue status using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import request_status

    remote = _remote_queue(peer_id=peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        request_status,
        remote=remote,
        timeout_s=float(timeout_s),
        detail=bool(detail),
    )


async def p2p_taskqueue_status(
    remote_multiaddr: str = "",
    peer_id: str = "",
    timeout_s: float = 10.0,
    detail: bool = False,
) -> Dict[str, Any]:
    """Get TaskQueue service status from native unified p2p tool path."""
    try:
        resp = await _request_status(
            remote_multiaddr=remote_multiaddr,
            peer_id=peer_id,
            timeout_s=timeout_s,
            detail=detail,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _list_tasks(
    status: str = "",
    limit: int = 50,
    task_types: Optional[List[str]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """List TaskQueue tasks using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import list_tasks

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    parsed_status = str(status).strip().lower() if status else None
    parsed_task_types = None
    if task_types is not None:
        parsed_task_types = [str(x) for x in (task_types or []) if str(x).strip()]

    return await _run_in_trio(
        list_tasks,
        remote=remote,
        status=parsed_status,
        limit=int(limit),
        task_types=parsed_task_types,
    )


async def p2p_taskqueue_list_tasks(
    status: str = "",
    limit: int = 50,
    task_types: Optional[List[str]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """List TaskQueue tasks from native unified p2p tool path."""
    try:
        tasks = await _list_tasks(
            status=status,
            limit=limit,
            task_types=task_types,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return {"ok": True, "tasks": tasks}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _get_task(
    task_id: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Get a single TaskQueue task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import get_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        get_task,
        remote=remote,
        task_id=str(task_id),
    )


async def p2p_taskqueue_get_task(
    task_id: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Get a single TaskQueue task from native unified p2p tool path."""
    try:
        task = await _get_task(
            task_id=task_id,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return {"ok": True, "task": task}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _wait_task(
    task_id: str,
    timeout_s: float = 60.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Wait for a TaskQueue task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import wait_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        wait_task,
        remote=remote,
        task_id=str(task_id),
        timeout_s=float(timeout_s),
    )


async def p2p_taskqueue_wait_task(
    task_id: str,
    timeout_s: float = 60.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Wait for a TaskQueue task from native unified p2p tool path."""
    try:
        task = await _wait_task(
            task_id=task_id,
            timeout_s=timeout_s,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return {"ok": True, "task": task}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _complete_task(
    task_id: str,
    status: str = "completed",
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Complete a TaskQueue task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import complete_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        complete_task,
        remote=remote,
        task_id=str(task_id),
        status=str(status),
        result=(result if isinstance(result, dict) else None),
        error=str(error) if error else None,
    )


async def p2p_taskqueue_complete_task(
    task_id: str,
    status: str = "completed",
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Complete a TaskQueue task from native unified p2p tool path."""
    try:
        resp = await _complete_task(
            task_id=task_id,
            status=status,
            result=result,
            error=error,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _heartbeat(
    peer_id: str,
    clock: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Send heartbeat using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import heartbeat

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        heartbeat,
        remote=remote,
        peer_id=str(peer_id),
        clock=(clock if isinstance(clock, dict) else None),
    )


async def p2p_taskqueue_heartbeat(
    peer_id: str,
    clock: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Send heartbeat from native unified p2p tool path."""
    try:
        resp = await _heartbeat(
            peer_id=peer_id,
            clock=clock,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def register_native_p2p_tools(manager: Any) -> None:
    """Register native p2p tools in the unified hierarchical manager."""
    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_status",
        func=p2p_taskqueue_status,
        description="Get p2p TaskQueue service status using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "remote_multiaddr": {"type": "string", "default": ""},
                "peer_id": {"type": "string", "default": ""},
                "timeout_s": {"type": "number", "default": 10.0},
                "detail": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_list_tasks",
        func=p2p_taskqueue_list_tasks,
        description="List p2p TaskQueue tasks using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "default": ""},
                "limit": {"type": "integer", "default": 50},
                "task_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": [],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_get_task",
        func=p2p_taskqueue_get_task,
        description="Get p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["task_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_wait_task",
        func=p2p_taskqueue_wait_task,
        description="Wait for p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "timeout_s": {"type": "number", "default": 60.0},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["task_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_complete_task",
        func=p2p_taskqueue_complete_task,
        description="Complete p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "status": {"type": "string", "default": "completed"},
                "result": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "error": {"type": "string", "default": ""},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["task_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_heartbeat",
        func=p2p_taskqueue_heartbeat,
        description="Send p2p TaskQueue heartbeat using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string"},
                "clock": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["peer_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )
