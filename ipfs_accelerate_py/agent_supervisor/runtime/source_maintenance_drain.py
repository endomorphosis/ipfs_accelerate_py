"""Retain one native parent during a separately admitted source-maintenance drain.

This process-local API issues no signals, credentials, task mutations or leases.
The original parent supplies its retained child and native admission/closure
gates. A bounded incomplete observation never authorizes parent exit or cleanup.
There is deliberately no JSON or signal-to-admission adapter here.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import select
import threading
import time
from typing import Callable


@dataclass(frozen=True)
class MaintenanceChildBinding:
    parent_pid: int
    parent_start_time_ticks: int
    child_pid: int
    child_start_time_ticks: int
    boot_id: str
    source_head: str
    source_tree: str


def _birth(pid: int) -> tuple[int, int, int, int]:
    with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise RuntimeError("maintenance process observation oversized")
    fields = raw[raw.rfind(")") + 2:].split()
    return int(fields[1]), int(fields[19]), int(fields[2]), int(fields[3])


class SourceMaintenanceDrain:
    """A request held by the original native parent until positive closure.

    ``admission_gate`` rechecks the original source, native launch locks,
    original actor identities and current canonical custody on every live
    observation. ``closed_gate`` must positively verify the complete original
    cohort, stopped native stores/listeners and retained state before cleanup.
    Neither gate may infer those facts from this object's child-exit bit.
    Gates run in the native parent and must use their own bounded operations.
    """

    def __init__(
        self,
        *,
        admission_gate: Callable[[MaintenanceChildBinding], None],
        closed_gate: Callable[[MaintenanceChildBinding], None],
        publish: Callable[[dict[str, object]], None],
    ) -> None:
        if not all(callable(fn) for fn in (admission_gate, closed_gate, publish)):
            raise TypeError("native maintenance gates are required")
        self._admission_gate = admission_gate
        self._closed_gate = closed_gate
        self._publish = publish
        self._request = threading.Event()
        self._transition_lock = threading.Lock()
        self._ordinary_shutdown = False
        self.binding: MaintenanceChildBinding | None = None
        self._descriptor: int | None = None
        self.active = False
        self.complete = False

    @property
    def requested(self) -> bool:
        return self._request.is_set()

    def request(self) -> None:
        """Called only by a separately admitted native parent control path."""
        with self._transition_lock:
            if self.binding is None or self._descriptor is None or self._ordinary_shutdown or self.complete:
                raise RuntimeError("original parent cannot admit maintenance now")
            self._request.set()

    def reserve_ordinary_shutdown(self) -> bool:
        """Serialize an ordinary signal/cleanup against maintenance admission."""
        with self._transition_lock:
            if self.requested:
                return False
            self._ordinary_shutdown = True
            return True

    def bind(self, child, *, start_time_ticks: int, source_head: str, source_tree: str) -> None:
        if self.binding is not None or self._descriptor is not None:
            raise RuntimeError("maintenance child already bound")
        if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (source_head, source_tree)):
            raise ValueError("maintenance source identity unavailable")
        parent = os.getpid()
        _, parent_start, _, _ = _birth(parent)
        before = _birth(child.pid)
        if before != (parent, start_time_ticks, child.pid, child.pid):
            raise RuntimeError("maintenance child is not the retained dedicated child")
        descriptor = os.pidfd_open(child.pid)
        try:
            if _birth(child.pid) != before:
                raise RuntimeError("maintenance child birth changed")
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
                boot = stream.read().strip()
            self.binding = MaintenanceChildBinding(
                parent, parent_start, child.pid, start_time_ticks, boot,
                source_head, source_tree,
            )
            self._descriptor = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def observe(self) -> dict[str, object]:
        """Return incomplete while retaining custody; never raise on gate refusal."""
        if not self.requested or self.binding is None or self._descriptor is None:
            raise RuntimeError("maintenance observation lacks native request/binding")
        self.active = True
        binding = self.binding
        reason = "native_child_drain_pending"
        complete = False
        try:
            if os.getpid() != binding.parent_pid or _birth(os.getpid())[1] != binding.parent_start_time_ticks:
                raise RuntimeError("original maintenance parent changed")
            poller = select.poll()
            poller.register(self._descriptor, select.POLLIN)
            events = poller.poll(0)
            if any(not (mask & select.POLLIN)
                   or mask & ~(select.POLLIN | select.POLLHUP)
                   for _, mask in events):
                raise RuntimeError("native maintenance pidfd observation unavailable")
            exited = bool(events)
            if exited:
                self._closed_gate(binding)
                complete = True
                reason = "original_native_cohort_closed"
            else:
                if _birth(binding.child_pid) != (
                    binding.parent_pid, binding.child_start_time_ticks,
                    binding.child_pid, binding.child_pid,
                ):
                    raise RuntimeError("original maintenance child changed")
                self._admission_gate(binding)
        except BaseException as exc:
            reason = "native_maintenance_unverified:" + type(exc).__name__
        observation: dict[str, object] = {
            "schema": "ipfs_accelerate_py/source-maintenance-drain@1",
            "complete": complete,
            "reason": reason,
            "parent_pid": binding.parent_pid,
            "child_pid": binding.child_pid,
            "source_head": binding.source_head,
            "source_tree": binding.source_tree,
            "callback_settlement_authority": False,
            "task_retry_authority": False,
            "source_transition_authority": False,
            "observed_at": time.time(),
        }
        try:
            self._publish(dict(observation))
        except BaseException:
            complete = False
            observation = {**observation, "complete": False,
                           "reason": "native_maintenance_observation_unpublished"}
        self.complete = complete
        return observation

    def retain_until_closed(self, *, interval_seconds: float = 0.2) -> None:
        """Keep the native parent alive; incomplete observations never unwind it."""
        if not 0.01 <= interval_seconds <= 1.0:
            raise ValueError("native maintenance observation interval out of bounds")
        while not self.observe()["complete"]:
            try:
                time.sleep(interval_seconds)
            except (KeyboardInterrupt, SystemExit):
                # An incomplete native drain is not permission to abandon its
                # parent. Original stop handlers normally absorb these signals.
                continue

    def close(self) -> None:
        if self.requested and not self.complete:
            raise RuntimeError("incomplete maintenance cannot release parent custody")
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
