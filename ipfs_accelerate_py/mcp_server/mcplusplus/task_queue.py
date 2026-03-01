"""Task queue primitive for MCP++ runtime integration.

This module provides a small, testable wrapper around the existing
`ipfs_accelerate_py.p2p_tasks.client` queue client APIs so the unified MCP
runtime can depend on a stable task queue interface.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from ipfs_accelerate_py.p2p_tasks.client import (
        RemoteQueue,
        cancel_task as _client_cancel_task,
        get_task as _client_get_task,
        list_tasks as _client_list_tasks,
        submit_task_with_info as _client_submit_task_with_info,
    )

    HAVE_TASK_QUEUE = True
except ImportError:
    HAVE_TASK_QUEUE = False
    RemoteQueue = None  # type: ignore[assignment]
    _client_submit_task_with_info = None  # type: ignore[assignment]
    _client_get_task = None  # type: ignore[assignment]
    _client_cancel_task = None  # type: ignore[assignment]
    _client_list_tasks = None  # type: ignore[assignment]


def _build_remote(*, peer_id: str = "", multiaddr: str = "") -> Any:
    """Create queue client connection descriptor."""
    if RemoteQueue is None:
        raise RuntimeError("Task queue remote client is unavailable")
    return RemoteQueue(peer_id=str(peer_id or "").strip(), multiaddr=str(multiaddr or "").strip())


class TaskQueueWrapper:
    """Wrapper around P2P task queue operations."""

    def __init__(self, queue_path: Optional[str] = None, peer_id: str = "", multiaddr: str = ""):
        """Initialize wrapper.

        Args:
            queue_path: Reserved compatibility field; not required for remote queue operations.
            peer_id: Optional remote peer ID hint.
            multiaddr: Optional remote multiaddr hint.
        """
        self.queue_path = queue_path
        self.peer_id = str(peer_id or "").strip()
        self.multiaddr = str(multiaddr or "").strip()
        self.available = HAVE_TASK_QUEUE

        if not self.available:
            logger.warning("Task queue client not available")

    def _remote(self) -> Any:
        return _build_remote(peer_id=self.peer_id, multiaddr=self.multiaddr)

    async def submit(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 0,
        model_name: str = "default",
        **kwargs: Any,
    ) -> Optional[str]:
        """Submit a task and return task ID when successful."""
        if not self.available or _client_submit_task_with_info is None:
            return None

        merged_payload = dict(payload if isinstance(payload, dict) else {})
        merged_payload.setdefault("priority", int(priority))
        if kwargs:
            merged_payload.update(kwargs)

        try:
            response = await _client_submit_task_with_info(
                remote=self._remote(),
                task_type=str(task_type),
                model_name=str(model_name),
                payload=merged_payload,
            )
            if isinstance(response, dict):
                task_id = response.get("task_id")
                return str(task_id) if task_id else None
            return None
        except Exception as exc:
            logger.error("Failed to submit task: %s", exc)
            return None

    async def get_status(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get current task status/details by task ID."""
        if not self.available or _client_get_task is None:
            return None

        try:
            response = await _client_get_task(remote=self._remote(), task_id=str(task_id))
            return response if isinstance(response, dict) else None
        except Exception as exc:
            logger.error("Failed to get task status: %s", exc)
            return None

    async def cancel(self, task_id: str, reason: str | None = None) -> bool:
        """Cancel a queued/running task by task ID."""
        if not self.available or _client_cancel_task is None:
            return False

        try:
            response = await _client_cancel_task(
                remote=self._remote(),
                task_id=str(task_id),
                reason=(str(reason) if reason else None),
            )
            return bool(isinstance(response, dict) and response.get("ok"))
        except Exception as exc:
            logger.error("Failed to cancel task: %s", exc)
            return False

    async def list(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        task_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """List tasks in the queue using optional filters."""
        if not self.available or _client_list_tasks is None:
            return []

        try:
            response = await _client_list_tasks(
                remote=self._remote(),
                status=(str(status).strip() if status else None),
                limit=int(limit),
                task_types=[str(x) for x in (task_types or []) if str(x).strip()],
            )
            if not isinstance(response, dict):
                return []
            tasks = response.get("tasks")
            return list(tasks) if isinstance(tasks, list) else []
        except Exception as exc:
            logger.error("Failed to list tasks: %s", exc)
            return []


def create_task_queue(
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> TaskQueueWrapper:
    """Create a task queue wrapper instance."""
    return TaskQueueWrapper(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)


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
    """Module-level submit helper for compatibility with legacy call sites."""
    queue = create_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
    return await queue.submit(task_type=task_type, payload=payload, priority=priority, model_name=model_name, **kwargs)


async def get_task_status(
    *,
    task_id: str,
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> Optional[dict[str, Any]]:
    """Module-level status helper for compatibility with legacy call sites."""
    queue = create_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
    return await queue.get_status(task_id=task_id)


async def cancel_task(
    *,
    task_id: str,
    reason: str | None = None,
    queue_path: Optional[str] = None,
    peer_id: str = "",
    multiaddr: str = "",
) -> bool:
    """Module-level cancel helper for compatibility with legacy call sites."""
    queue = create_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
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
    """Module-level list helper for compatibility with legacy call sites."""
    queue = create_task_queue(queue_path=queue_path, peer_id=peer_id, multiaddr=multiaddr)
    return await queue.list(status=status, limit=limit, task_types=task_types)


__all__ = [
    "HAVE_TASK_QUEUE",
    "TaskQueueWrapper",
    "create_task_queue",
    "submit_task",
    "get_task_status",
    "cancel_task",
    "list_tasks",
]
