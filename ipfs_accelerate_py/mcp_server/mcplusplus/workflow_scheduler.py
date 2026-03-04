"""Workflow scheduler primitive for MCP++ runtime integration."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

P2PWorkflowScheduler: Any = None
_get_scheduler: Optional[Callable[[], Any]] = None
_reset_scheduler: Optional[Callable[[], None]] = None

try:
    from ipfs_accelerate_py.mcplusplus_module.p2p.workflow import (
        HAVE_P2P_SCHEDULER,
        P2PWorkflowScheduler as _P2PWorkflowSchedulerImpl,
        get_scheduler as _get_scheduler_impl,
        reset_scheduler as _reset_scheduler_impl,
    )

    HAVE_WORKFLOW_SCHEDULER = bool(HAVE_P2P_SCHEDULER)
    P2PWorkflowScheduler = _P2PWorkflowSchedulerImpl
    _get_scheduler = _get_scheduler_impl
    _reset_scheduler = _reset_scheduler_impl
except ImportError:
    HAVE_WORKFLOW_SCHEDULER = False
    P2PWorkflowScheduler = None
    _get_scheduler = None
    _reset_scheduler = None


def create_workflow_scheduler(
    peer_id: Optional[str] = None,
    context: Optional[Any] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Create or retrieve scheduler instance.

    Args:
        peer_id: Reserved compatibility argument.
        context: Optional context object with `workflow_scheduler` attribute.
        **kwargs: Reserved compatibility arguments.
    """
    del peer_id, kwargs

    if not HAVE_WORKFLOW_SCHEDULER:
        logger.warning("Workflow scheduler unavailable")
        return None

    try:
        scheduler = _get_scheduler() if _get_scheduler is not None else None
        if context is not None and scheduler is not None:
            setattr(context, "workflow_scheduler", scheduler)
        return scheduler
    except Exception as exc:
        logger.error("Failed to create workflow scheduler: %s", exc)
        return None


def get_scheduler(context: Optional[Any] = None) -> Optional[Any]:
    """Get scheduler from context when provided, else use global scheduler."""
    if context is not None:
        return getattr(context, "workflow_scheduler", None)

    if not HAVE_WORKFLOW_SCHEDULER:
        return None

    try:
        return _get_scheduler() if _get_scheduler is not None else None
    except Exception as exc:
        logger.error("Failed to get workflow scheduler: %s", exc)
        return None


def reset_scheduler() -> None:
    """Reset global scheduler instance."""
    if not HAVE_WORKFLOW_SCHEDULER:
        return
    try:
        if _reset_scheduler is not None:
            _reset_scheduler()
    except Exception as exc:
        logger.error("Failed to reset workflow scheduler: %s", exc)


async def submit_workflow(
    workflow_name: str,
    tasks: list[dict[str, Any]],
    **kwargs: Any,
) -> Optional[str]:
    """Submit workflow using available scheduler API when present.

    This probes common scheduler method names for compatibility.
    """
    scheduler = get_scheduler()
    if scheduler is None:
        return None

    try:
        if hasattr(scheduler, "submit_workflow"):
            result = scheduler.submit_workflow(workflow_name=workflow_name, tasks=tasks, **kwargs)
        elif hasattr(scheduler, "create_workflow"):
            result = scheduler.create_workflow(workflow_name=workflow_name, tasks=tasks, **kwargs)
        elif hasattr(scheduler, "submit"):
            result = scheduler.submit(workflow_name=workflow_name, tasks=tasks, **kwargs)
        else:
            logger.warning("Scheduler has no known submit method")
            return None

        if hasattr(result, "__await__"):
            result = await result

        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            workflow_id = result.get("workflow_id")
            return str(workflow_id) if workflow_id else None
        return None
    except Exception as exc:
        logger.error("Failed to submit workflow: %s", exc)
        return None


__all__ = [
    "HAVE_WORKFLOW_SCHEDULER",
    "P2PWorkflowScheduler",
    "create_workflow_scheduler",
    "get_scheduler",
    "reset_scheduler",
    "submit_workflow",
]
