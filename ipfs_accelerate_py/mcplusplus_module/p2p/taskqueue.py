"""Task queue compatibility adapter for MCP++.

This module keeps the historical ``mcplusplus_module.p2p.taskqueue`` imports
working while delegating behavior to the canonical MCP++ runtime wrapper at
``ipfs_accelerate_py.mcp_server.mcplusplus.task_queue``.
"""

from __future__ import annotations

from typing import Any, Optional

from ipfs_accelerate_py.mcp_server.mcplusplus.task_queue import (
    HAVE_TASK_QUEUE,
    TaskQueueWrapper,
    create_task_queue,
)

try:
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
except ImportError:  # pragma: no cover - optional dependency
    RemoteQueue = None  # type: ignore[assignment]


class P2PTaskQueue(TaskQueueWrapper):
    """Backward-compatible alias for the canonical task queue wrapper."""


def build_task_queue(
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> P2PTaskQueue:
    """Return a ``P2PTaskQueue`` instance backed by canonical wrappers."""
    wrapper = create_task_queue(
        queue_path=queue_path,
        peer_id=peer_id,
        multiaddr=multiaddr,
    )
    # Re-wrap to preserve the historical concrete type name for callers.
    queue = P2PTaskQueue(
        queue_path=wrapper.queue_path,
        peer_id=wrapper.peer_id,
        multiaddr=wrapper.multiaddr,
    )
    return queue


async def submit_task(
    *,
    task_type: str,
    payload: dict[str, Any],
    priority: int = 0,
    model_name: str = "default",
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
    **kwargs: Any,
) -> Optional[str]:
    """Submit a task using the canonical task queue implementation."""
    queue = build_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
    return await queue.submit(
        task_type=task_type,
        payload=payload,
        priority=priority,
        model_name=model_name,
        **kwargs,
    )


async def get_task_status(
    *,
    task_id: str,
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> Optional[dict[str, Any]]:
    """Get task status via canonical task queue implementation."""
    queue = build_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
    return await queue.get_status(task_id=task_id)


async def cancel_task(
    *,
    task_id: str,
    reason: str | None = None,
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> bool:
    """Cancel task via canonical task queue implementation."""
    queue = build_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
    return await queue.cancel(task_id=task_id, reason=reason)


async def list_tasks(
    *,
    status: str | None = None,
    limit: int = 100,
    task_types: Optional[list[str]] = None,
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> list[dict[str, Any]]:
    """List tasks via canonical task queue implementation."""
    queue = build_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
    return await queue.list(status=status, limit=limit, task_types=task_types)


__all__ = [
    "HAVE_TASK_QUEUE",
    "P2PTaskQueue",
    "RemoteQueue",
    "build_task_queue",
    "submit_task",
    "get_task_status",
    "cancel_task",
    "list_tasks",
]
