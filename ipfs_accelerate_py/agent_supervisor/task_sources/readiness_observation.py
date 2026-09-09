"""Bounded task readiness reads through an already admitted state owner."""
from dataclasses import dataclass
from typing import Any, Callable

from .database_task_source import MAX_QUERY_LIMIT


@dataclass(frozen=True)
class ReadinessObservation:
    ready: tuple[Any, ...]
    active: tuple[Any, ...]
    blocked_recoverable: tuple[Any, ...]
    revision: int


def observe_readiness(
    source: Any,
    *,
    in_scope: Callable[[Any], bool],
    blocked_recoverable: Callable[[Any, Any], bool],
    include_blocked: bool,
) -> ReadinessObservation:
    """Reuse one intent client across task enumeration and dependency reads.

    The caller supplies an admitted source and retains all owner/fence checks.
    This observation does not authorize claims, recovery, or completion. Failed
    or truncated required pages propagate; optional blocked-frontier failures
    provide no recoverable work. The session closes on every exit.
    """
    with source.intent.read_session():
        ready_page = source.ready_tasks(limit=MAX_QUERY_LIMIT)
        active_page = source.list_tasks(
            status=("claimed", "in_progress", "running"), limit=MAX_QUERY_LIMIT,
        )
        if ready_page.next_cursor or active_page.next_cursor:
            raise RuntimeError("authoritative readiness projection is truncated")
        ready = tuple(task for task in ready_page.tasks if in_scope(task))
        active = tuple(task for task in active_page.tasks if in_scope(task))
        blocked = ()
        blocked_revision = 0
        if include_blocked:
            try:
                page = source.list_tasks(status=("blocked",), limit=MAX_QUERY_LIMIT)
                if not page.next_cursor:
                    blocked = tuple(
                        task for task in page.tasks
                        if in_scope(task) and blocked_recoverable(task, source)
                    )
                    blocked_revision = int(page.revision)
            except Exception:
                blocked = ()
                blocked_revision = 0
        return ReadinessObservation(
            ready, active, blocked,
            max(int(ready_page.revision), int(active_page.revision), blocked_revision),
        )
