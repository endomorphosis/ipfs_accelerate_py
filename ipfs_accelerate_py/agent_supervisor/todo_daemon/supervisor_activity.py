"""Typed, read-only activity facts for a supervisor whose process scope is current.

Callers must independently verify supervisor/daemon births and parentage. These
facts distinguish a bounded maintenance pass from readiness to dispatch work;
they never authorize a restart, task mutation, or release of a native fence.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any

MAINTENANCE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/supervisor-maintenance@1"
ACTIVITY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/supervisor-activity@1"


def _timestamp(value: Any) -> datetime:
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError("timestamp_required")
    result = datetime.fromisoformat(value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone_required")
    return result.astimezone(timezone.utc)


def _duration(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("finite_positive_timeout_required")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("finite_positive_timeout_required")
    return result


@dataclass(frozen=True)
class SupervisorMaintenanceWindow:
    """The original start and deadline, captured once before a maintenance pass."""

    started_at: str
    timeout_seconds: float
    deadline_at: str

    def __post_init__(self) -> None:
        start = _timestamp(self.started_at)
        timeout = _duration(self.timeout_seconds)
        deadline = _timestamp(self.deadline_at)
        expected = start + timedelta(seconds=timeout)
        if deadline != expected or deadline <= start:
            raise ValueError("maintenance_deadline_mismatch")
        object.__setattr__(self, "started_at", start.isoformat())
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "deadline_at", deadline.isoformat())

    @classmethod
    def begin(cls, *, started_at: str, timeout_seconds: float) -> SupervisorMaintenanceWindow:
        start = _timestamp(started_at)
        timeout = _duration(timeout_seconds)
        return cls(start.isoformat(), timeout, (start + timedelta(seconds=timeout)).isoformat())

    def event(self, *, phase: str, observed_at: str) -> dict[str, Any]:
        if phase not in {"running", "completed", "failed"}:
            raise ValueError("maintenance_phase_unknown")
        observed = _timestamp(observed_at)
        return {
            "schema": MAINTENANCE_SCHEMA,
            **asdict(self),
            "phase": phase,
            "finished_at": None if phase == "running" else observed.isoformat(),
        }


@dataclass(frozen=True)
class SupervisorActivity:
    status: str
    phase_ready: bool
    permits_dispatch: bool
    heartbeat_age_seconds: float | None
    reason: str
    maintenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ACTIVITY_SCHEMA, **asdict(self)}


def observe_supervisor_activity(
    status: Mapping[str, Any], *, now: datetime, heartbeat_max_age_seconds: float = 60.0
) -> SupervisorActivity:
    """Validate heartbeat and a closed maintenance window, without process inference.

    Ordinary running is dispatch-ready. Active or successfully completed
    maintenance can remain supervised while dispatch stays unavailable. Missing
    legacy maintenance evidence, failure, expiry, and malformed times all refuse.
    """
    state = str(status.get("status") or "")
    age = None
    maintenance = None

    def unavailable(reason: str) -> SupervisorActivity:
        return SupervisorActivity(state, False, False, age, reason, maintenance)

    try:
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            return unavailable("observation_clock_invalid")
        current = now.astimezone(timezone.utc)
        updated = _timestamp(status.get("updated_at"))
        age = (current - updated).total_seconds()
        maximum = _duration(heartbeat_max_age_seconds)
        if not 0 <= age <= maximum:
            return unavailable("heartbeat_not_current")
        if state == "running":
            return SupervisorActivity(state, True, True, age, "running")
        phases = {
            "agentic_maintenance_started": "running",
            "agentic_maintenance_completed": "completed",
            "agentic_maintenance_failed": "failed",
        }
        phase = phases.get(state)
        if phase is None:
            return unavailable("supervisor_phase_not_ready")
        value = status.get("supervisor_maintenance")
        if not isinstance(value, Mapping) or set(value) != {
            "schema", "started_at", "timeout_seconds", "deadline_at", "phase", "finished_at"
        } or value.get("schema") != MAINTENANCE_SCHEMA or value.get("phase") != phase:
            return unavailable("typed_maintenance_window_required")
        window = SupervisorMaintenanceWindow(value["started_at"], value["timeout_seconds"], value["deadline_at"])
        maintenance = dict(value)
        if status.get("active_agentic_maintenance_has_daemon") is not True:
            return unavailable("maintenance_without_managed_daemon")
        if status.get("last_agentic_maintenance_status") != phase or status.get("last_agentic_maintenance_error"):
            return unavailable("maintenance_outcome_not_ready")
        if _duration(status.get("active_agentic_maintenance_timeout_seconds")) != window.timeout_seconds:
            return unavailable("maintenance_timeout_disagrees")
        start, deadline = _timestamp(window.started_at), _timestamp(window.deadline_at)
        if not start <= updated <= current:
            return unavailable("maintenance_start_not_current")
        if phase == "running":
            if value["finished_at"] is not None or _timestamp(status.get("active_agentic_maintenance_started_at")) != start:
                return unavailable("active_maintenance_window_disagrees")
            if current > deadline:
                return unavailable("maintenance_deadline_expired")
        elif phase == "completed":
            finished = _timestamp(value["finished_at"])
            if (
                status.get("active_agentic_maintenance_started_at") != ""
                or finished != updated
                or not start <= finished <= deadline
            ):
                return unavailable("maintenance_completion_not_current")
        else:
            return unavailable("maintenance_failed")
        return SupervisorActivity(state, True, False, age, "bounded_maintenance_" + phase, maintenance)
    except (ValueError, TypeError, OverflowError):
        return unavailable("maintenance_or_heartbeat_invalid")
