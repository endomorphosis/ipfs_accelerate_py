"""Gate coordinator wakes through recovery leases. TypeSafe is not required.

Callers:

- ``AutonomyRuntime.handle_wake`` — task then lane; keep until ack if
  progressing without auto-ack
- Portal ``ImplementationDaemon.run_once`` — after preflight, including
  active-task extras; ack + TTL backoff on ``lease_held``
- ``DatabaseImplementationDaemon`` resume — ``session_for_task(task_cid)``

Window/safety ticks do not take a lease unless an active-task extra is
supplied. Errors fail open (no leak, do not skip work). A live holder
still blocks. TypeSafe is optional triage on the closed recovery catalog,
never a requirement and never process authority.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .recovery_leases import (
    DEFAULT_RECOVERY_LEASE_TTL_SECONDS,
    RECOVERY_LEASE_KINDS,
    RecoveryLeaseTable,
    recovery_leases_from_env,
)


@dataclass
class WakeLeaseSession:
    blocked: bool
    reason: str = ""
    owner_id: str = ""
    held: tuple[tuple[str, str], ...] = ()
    table: RecoveryLeaseTable | None = None
    reason_codes: tuple[str, ...] = ()
    events: tuple[Any, ...] = field(default_factory=tuple)
    retry_after_seconds: float | None = None
    expires_at: float | None = None
    ttl_seconds: float = DEFAULT_RECOVERY_LEASE_TTL_SECONDS
    _stop: threading.Event | None = field(default=None, repr=False, compare=False)
    _thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    _released: bool = field(default=False, repr=False, compare=False)
    _mutex: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def start_heartbeat(self, *, interval_seconds: float | None = None) -> None:
        """Renew in the background until ``release``. Same owner only."""

        if self._released or self.blocked or not self.held or self._thread is not None:
            return
        interval = float(
            max(0.05, min(self.ttl_seconds / 4.0, self.ttl_seconds * 0.5))
            if interval_seconds is None
            else interval_seconds
        )
        if interval <= 0:
            return
        stop = threading.Event()
        self._stop = stop
        self.renew()

        def _loop() -> None:
            while not stop.wait(interval):
                try:
                    if not self.renew():
                        return
                except Exception:
                    if self._released:
                        return

        thread = threading.Thread(
            target=_loop,
            name="wake-lease-heartbeat",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop_heartbeat(self) -> None:
        stop = self._stop
        thread = self._thread
        self._stop = None
        self._thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def renew(self, *, ttl_seconds: float | None = None) -> bool:
        """Extend TTL for every held key. Same owner only; never steals."""

        with self._mutex:
            if self._released or self.table is None or self.blocked or not self.held:
                return False
            held = self.held
            table = self.table
            owner = self.owner_id
            ttl = float(self.ttl_seconds if ttl_seconds is None else ttl_seconds)
        if ttl <= 0:
            return False
        for kind, resource_id in held:
            with self._mutex:
                if self._released:
                    return False
            ok, _view = table.try_acquire(
                kind, resource_id, owner, ttl_seconds=ttl
            )
            if not ok:
                return False
        with self._mutex:
            return not self._released

    def detach(self) -> None:
        """Stop heartbeat without dropping table keys (session transfer)."""

        with self._mutex:
            self._released = True
            self.held = ()
        self.stop_heartbeat()

    def release(self) -> None:
        with self._mutex:
            self._released = True
            held = self.held
            table = self.table
            owner = self.owner_id
            self.held = ()
        self.stop_heartbeat()
        if table is None or not held:
            return
        for kind, resource_id in reversed(held):
            try:
                table.release(kind, resource_id, owner)
            except Exception:
                pass


def begin_wake_leases(
    events: Sequence[Any],
    *,
    owner_id: str,
    table: RecoveryLeaseTable | None = None,
    extra_state: Mapping[str, Any] | None = None,
    extra_resources: Sequence[tuple[str, str]] = (),
    ttl_seconds: float = DEFAULT_RECOVERY_LEASE_TTL_SECONDS,
) -> WakeLeaseSession:
    """Acquire task then lane for each non-window wake. Steal only after TTL.

    ``extra_resources`` covers the active task when the wake is only a window
    tick, so two lanes still cannot run the same work.
    """

    owner = str(owner_id or "").strip()
    if not owner:
        return WakeLeaseSession(blocked=False, reason="no_owner")
    leases = table or recovery_leases_from_env()
    extra = extra_state or {}
    ttl = float(ttl_seconds)
    if ttl <= 0:
        ttl = DEFAULT_RECOVERY_LEASE_TTL_SECONDS
    held: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _blocked(view: Mapping[str, Any]) -> WakeLeaseSession:
        try:
            expires_at = float(view.get("expires_at") or 0.0)
        except (TypeError, ValueError):
            expires_at = 0.0
        retry = max(0.0, expires_at - time.time()) if expires_at else None
        session = WakeLeaseSession(
            blocked=True,
            reason=str(view.get("reason") or "recovery_lease_held"),
            owner_id=owner,
            held=tuple(held),
            table=leases,
            reason_codes=("lease_held",),
            events=tuple(events),
            retry_after_seconds=retry,
            expires_at=expires_at or None,
        )
        session.release()
        session.blocked = True
        session.reason_codes = ("lease_held",)
        return session

    from .runtime import AutonomyWakeEvent

    for raw in events:
        try:
            bound = (
                raw
                if isinstance(raw, AutonomyWakeEvent)
                else AutonomyWakeEvent.from_runtime_wake(raw)
            )
        except Exception:
            continue
        if bound.safety_timer:
            continue
        resource_id = bound.subject_id or bound.cursor_id
        if not resource_id:
            continue
        task_key = ("task", resource_id)
        if task_key not in seen:
            ok, view = leases.try_acquire(
                "task", resource_id, owner, ttl_seconds=ttl
            )
            if not ok:
                return _blocked(view)
            held.append(task_key)
            seen.add(task_key)
        lane_id = str(bound.lane_id or extra.get("lane_id") or "").strip()
        if lane_id:
            lane_key = ("lane", lane_id)
            if lane_key not in seen:
                ok, view = leases.try_acquire(
                    "lane", lane_id, owner, ttl_seconds=ttl
                )
                if not ok:
                    return _blocked(view)
                held.append(lane_key)
                seen.add(lane_key)
    for kind, resource_id in extra_resources:
        ident = str(resource_id or "").strip()
        if not ident:
            continue
        chosen = str(kind or "task").strip() or "task"
        if chosen not in RECOVERY_LEASE_KINDS:
            continue
        key = (chosen, ident)
        if key in seen:
            continue
        ok, view = leases.try_acquire(key[0], ident, owner, ttl_seconds=ttl)
        if not ok:
            return _blocked(view)
        held.append(key)
        seen.add(key)
    return WakeLeaseSession(
        blocked=False,
        reason="recovery_lease_acquired" if held else "no_lease_required",
        owner_id=owner,
        held=tuple(held),
        table=leases,
        events=tuple(events),
        ttl_seconds=ttl,
    )


def session_for_task(
    task_id: str,
    *,
    owner_id: str,
    table: RecoveryLeaseTable | None = None,
    ttl_seconds: float = DEFAULT_RECOVERY_LEASE_TTL_SECONDS,
    extra_resources: Sequence[tuple[str, str]] = (),
) -> WakeLeaseSession:
    """Lease one task (and optional extras) with fail-open arming."""

    ident = str(task_id or "").strip()
    resources: list[tuple[str, str]] = []
    if ident:
        resources.append(("task", ident))
    resources.extend(tuple(extra_resources))
    return safe_begin_wake_leases(
        (),
        owner_id=owner_id,
        table=table,
        extra_resources=tuple(resources),
        ttl_seconds=ttl_seconds,
    )


def safe_begin_wake_leases(
    events: Sequence[Any],
    *,
    owner_id: str,
    table: RecoveryLeaseTable | None = None,
    extra_state: Mapping[str, Any] | None = None,
    extra_resources: Sequence[tuple[str, str]] = (),
    ttl_seconds: float = DEFAULT_RECOVERY_LEASE_TTL_SECONDS,
) -> WakeLeaseSession:
    """Acquire and arm heartbeat. Errors fail open (no leak, do not skip work)."""

    session: WakeLeaseSession | None = None
    try:
        session = begin_wake_leases(
            events,
            owner_id=owner_id,
            table=table,
            extra_state=extra_state,
            extra_resources=extra_resources,
            ttl_seconds=ttl_seconds,
        )
        return arm_heartbeat(session)
    except Exception:
        if session is not None:
            try:
                session.release()
            except Exception:
                pass
        return WakeLeaseSession(
            blocked=False,
            reason="lease_gate_fail_open",
            reason_codes=("lease_gate_fail_open",),
        )


def apply_lease_held_backoff(
    result: dict[str, Any],
    session: WakeLeaseSession | None,
) -> dict[str, Any]:
    """Stamp ``lease_held`` and ``next_wake_after_seconds`` from the session."""

    retry_after = getattr(session, "retry_after_seconds", None) if session else None
    result["blocked"] = True
    result["reason"] = "lease_held"
    result["wake_lease"] = {
        "reason": str(getattr(session, "reason", "") or "") if session else "",
        "reason_codes": list(getattr(session, "reason_codes", ()) or ()),
        "expires_at": getattr(session, "expires_at", None) if session else None,
        "retry_after_seconds": retry_after,
    }
    if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
        wait = max(0.0, float(retry_after))
        existing = result.get("next_wake_after_seconds")
        if existing is None:
            result["next_wake_after_seconds"] = wait
        else:
            try:
                result["next_wake_after_seconds"] = min(float(existing), wait)
            except (TypeError, ValueError):
                result["next_wake_after_seconds"] = wait
    return result


def arm_heartbeat(session: WakeLeaseSession) -> WakeLeaseSession:
    """Start heartbeat; on failure release keys so they cannot leak until TTL."""

    if session.blocked or not session.held:
        return session
    try:
        session.start_heartbeat()
    except Exception:
        session.release()
        raise
    return session


__all__ = [
    "WakeLeaseSession",
    "apply_lease_held_backoff",
    "arm_heartbeat",
    "begin_wake_leases",
    "safe_begin_wake_leases",
    "session_for_task",
]
