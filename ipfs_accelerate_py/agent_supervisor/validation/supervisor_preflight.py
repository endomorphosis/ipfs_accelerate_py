"""Fail-closed validation for supervisor taskboard preflight state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

ACTIVE_PHASES: Final[frozenset[str]] = frozenset(
    {"implementing", "validating", "committing", "merging"}
)


class SupervisorPreflightError(ValueError):
    """Raised when an incomplete board has neither runnable nor live work."""


def _count(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SupervisorPreflightError(f"{key} must be a non-negative integer")
    return value


def _task_ids(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    if not isinstance(value, (list, tuple)):
        raise SupervisorPreflightError(f"{key} must be a sequence")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise SupervisorPreflightError(f"{key} contains an empty or duplicate task id")
    return result


def _active_claimed_task_ids(
    lane_states: Sequence[Mapping[str, Any]],
    *,
    ready_task_ids: frozenset[str],
) -> tuple[str, ...]:
    active: set[str] = set()
    for state in lane_states:
        task_id = str(state.get("active_task_id") or "").strip()
        phase = str(state.get("active_phase") or "").strip().lower()
        in_progress = state.get("implementation_in_progress") is True
        if task_id and task_id in ready_task_ids and (in_progress or phase in ACTIVE_PHASES):
            active.add(task_id)
    return tuple(sorted(active))


def summarize_supervisor_preflight(
    payload: Mapping[str, Any],
    *,
    expected_task_count: int,
    live_lane_states: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a bounded readiness summary or reject a false-green preflight.

    A structurally ready task can be absent from ``eligible_ready_task_ids``
    while another healthy lane owns its implementation lease.  Such a task is
    progress only when a caller supplies a matching live lane state; a mere
    ready count never masks exhausted retries, stale claims, or a dead worker.
    """

    if isinstance(expected_task_count, bool) or expected_task_count <= 0:
        raise SupervisorPreflightError("expected_task_count must be positive")

    task_count = _count(payload, "task_count")
    completed_count = _count(payload, "completed_count")
    ready_count = _count(payload, "ready_count")
    eligible_ready_count = _count(payload, "eligible_ready_count")
    blocked_count = _count(payload, "blocked_count")
    ready_task_ids = _task_ids(payload, "ready_task_ids")
    eligible_ready_task_ids = _task_ids(payload, "eligible_ready_task_ids")
    blocked_task_ids = _task_ids(payload, "blocked_task_ids")

    if task_count != expected_task_count:
        raise SupervisorPreflightError(
            f"unexpected task count: {task_count} != {expected_task_count}"
        )
    if completed_count > task_count:
        raise SupervisorPreflightError("completed_count exceeds task_count")
    if ready_count != len(ready_task_ids):
        raise SupervisorPreflightError("ready_count does not match ready_task_ids")
    if eligible_ready_count != len(eligible_ready_task_ids):
        raise SupervisorPreflightError(
            "eligible_ready_count does not match eligible_ready_task_ids"
        )
    if blocked_count != len(blocked_task_ids):
        raise SupervisorPreflightError("blocked_count does not match blocked_task_ids")
    if blocked_task_ids:
        raise SupervisorPreflightError(f"preflight found blocked tasks: {blocked_task_ids!r}")

    active_task_ids = _active_claimed_task_ids(
        live_lane_states,
        ready_task_ids=frozenset(ready_task_ids),
    )
    drained = completed_count == task_count
    if drained:
        progress_state = "drained"
    elif eligible_ready_task_ids:
        progress_state = "runnable"
    elif active_task_ids:
        progress_state = "active_claimed"
    else:
        raise SupervisorPreflightError(
            "incomplete board has neither eligible work nor a matching live claim"
        )

    return {
        "task_count": task_count,
        "completed_count": completed_count,
        "ready_count": ready_count,
        "eligible_ready_count": eligible_ready_count,
        "active_claimed_count": len(active_task_ids),
        "active_claimed_task_ids": list(active_task_ids),
        "progress_state": progress_state,
        "drained": drained,
    }
