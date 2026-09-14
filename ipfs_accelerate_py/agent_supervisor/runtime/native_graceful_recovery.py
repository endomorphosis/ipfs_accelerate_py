"""Exact-process graceful recovery, with the coordinator suspended temporarily.

This is a mechanism for an already admitted native operator.  Its gates must
bind the configured source, process roster, launch fences and protected state.
It supplies no task, callback, source-transition or completion authority.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import select
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Iterator, Sequence


class GracefulRecoveryUnverified(RuntimeError):
    """A static refusal; partially closed actors and all durable state remain."""


@dataclass(frozen=True)
class ProcessBinding:
    pid: int
    birth: int
    parent: int
    process_group: int
    session: int
    boot_id: str
    uid: int
    argv_sha256: str
    cwd: str


@dataclass(frozen=True)
class LaneBinding:
    supervisor: ProcessBinding
    daemon: ProcessBinding


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise GracefulRecoveryUnverified(reason)


def _read_proc(path: Path, limit: int = 131072) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    _require(len(value) <= limit, "process_observation_bound")
    return value


def _stat(pid: int, tid: int | None = None) -> list[str]:
    path = Path("/proc") / str(pid)
    if tid is not None:
        path = path / "task" / str(tid)
    raw = _read_proc(path / "stat").decode("ascii")
    fields = raw[raw.rfind(")") + 2 :].split()
    _require(len(fields) >= 20, "process_stat_unavailable")
    return fields


def observe_process(pid: int) -> ProcessBinding:
    _require(type(pid) is int and pid > 1, "process_pid_invalid")
    path = Path("/proc") / str(pid)
    before = _stat(pid)
    _require(before[0] not in {"Z", "X"}, "process_already_exited")
    result = ProcessBinding(
        pid=pid,
        birth=int(before[19]),
        parent=int(before[1]),
        process_group=int(before[2]),
        session=int(before[3]),
        boot_id=_read_proc(Path("/proc/sys/kernel/random/boot_id")).decode().strip(),
        uid=path.stat().st_uid,
        argv_sha256=hashlib.sha256(_read_proc(path / "cmdline")).hexdigest(),
        cwd=os.readlink(path / "cwd"),
    )
    after = _stat(pid)
    _require(
        before[19] == after[19] and before[1:4] == after[1:4], "process_birth_changed"
    )
    return result


def require_exact_process(binding: ProcessBinding) -> None:
    _require(observe_process(binding.pid) == binding, "process_binding_changed")


def all_threads_stopped(binding: ProcessBinding) -> bool:
    require_exact_process(binding)
    task = Path("/proc") / str(binding.pid) / "task"
    tids = sorted(int(path.name) for path in task.iterdir())
    _require(0 < len(tids) <= 512, "controller_thread_bound")
    stopped = all(_stat(binding.pid, tid)[0] == "T" for tid in tids)
    _require(
        tids == sorted(int(path.name) for path in task.iterdir()),
        "controller_threads_changed",
    )
    require_exact_process(binding)
    return stopped


def _exited(descriptor: int, milliseconds: int = 0) -> bool:
    poll = select.poll()
    poll.register(descriptor, select.POLLIN)
    events = poll.poll(milliseconds)
    _require(
        all(mask & select.POLLIN for _, mask in events), "pidfd_observation_unavailable"
    )
    return bool(events)


def _wait_exit(descriptor: int, deadline: float) -> None:
    while not _exited(descriptor):
        remaining = deadline - time.monotonic()
        _require(remaining > 0, "graceful_exit_timeout_no_escalation")
        _exited(descriptor, min(100, max(1, int(remaining * 1000))))


@contextlib.contextmanager
def _exact_pidfd(binding: ProcessBinding) -> Iterator[int]:
    descriptor = os.pidfd_open(binding.pid, 0)
    try:
        require_exact_process(binding)
        yield descriptor
    finally:
        os.close(descriptor)


def gracefully_close_native_lanes(
    *,
    controller: ProcessBinding,
    lanes: Sequence[LaneBinding],
    lane_fence: Callable[[int], ContextManager[object]],
    effect_gate: Callable[[], None],
    lane_children_gate: Callable[[int], None],
    closed_children_gate: Callable[[], None],
    record_phase: Callable[[str], None],
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    """Close exact native actors without running the old forced tail on survivors.

    Caller-supplied native fences block daemon birth while each daemon exits.
    The supervisor receives TERM before that fence is released; its installed
    native handler raises into cleanup.  Releasing the fence before CONT also
    permits native cleanup implementations that acquire that same fence.  The
    suspended coordinator cannot replace either actor during this interval.

    Every exception after STOP attempts CONT in finally.  A timeout never sends
    KILL, discards process markers, settles a callback or rolls back old state.
    """
    _require(
        type(timeout_seconds) in {int, float} and 0 < timeout_seconds <= 300,
        "recovery_timeout_invalid",
    )
    _require(0 < len(lanes) <= 32, "lane_roster_bound")
    bindings = [
        controller,
        *(item for lane in lanes for item in (lane.supervisor, lane.daemon)),
    ]
    _require(
        len({item.pid for item in bindings}) == len(bindings),
        "process_roster_duplicate",
    )
    _require(
        all(
            lane.supervisor.parent == controller.pid
            and lane.daemon.parent == lane.supervisor.pid
            for lane in lanes
        ),
        "lane_parent_binding_changed",
    )
    closed_lanes: list[int] = []
    suspended_supervisors: list[int] = []
    stop_attempted = False
    controller_term_sent = False
    with contextlib.ExitStack() as descriptors:
        master_fd = descriptors.enter_context(_exact_pidfd(controller))
        lane_fds = [
            (
                descriptors.enter_context(_exact_pidfd(lane.supervisor)),
                descriptors.enter_context(_exact_pidfd(lane.daemon)),
            )
            for lane in lanes
        ]
        _require(
            _stat(controller.pid)[0] not in {"T", "t"}, "controller_already_stopped"
        )
        try:
            record_phase("controller_suspend_prepared")
            effect_gate()
            require_exact_process(controller)
            stop_attempted = True
            signal.pidfd_send_signal(master_fd, signal.SIGSTOP)
            deadline = time.monotonic() + timeout_seconds
            while not all_threads_stopped(controller):
                _require(time.monotonic() < deadline, "controller_suspend_unverified")
                time.sleep(0.01)
            record_phase("controller_all_threads_stopped")
            for index, (lane, (supervisor_fd, daemon_fd)) in enumerate(
                zip(lanes, lane_fds)
            ):
                with lane_fence(index):
                    record_phase("supervisor_suspend_prepared")
                    effect_gate()
                    require_exact_process(lane.supervisor)
                    _require(
                        _stat(lane.supervisor.pid)[0] not in {"T", "t"},
                        "supervisor_already_stopped",
                    )
                    suspended_supervisors.append(supervisor_fd)
                    signal.pidfd_send_signal(supervisor_fd, signal.SIGSTOP)
                    deadline = time.monotonic() + timeout_seconds
                    while not all_threads_stopped(lane.supervisor):
                        _require(
                            time.monotonic() < deadline, "supervisor_suspend_unverified"
                        )
                        time.sleep(0.01)
                    record_phase("supervisor_all_threads_stopped")
                    record_phase("daemon_graceful_exit_prepared")
                    effect_gate()
                    _require(
                        all_threads_stopped(controller), "controller_suspension_lost"
                    )
                    require_exact_process(lane.supervisor)
                    require_exact_process(lane.daemon)
                    signal.pidfd_send_signal(daemon_fd, signal.SIGTERM)
                    _wait_exit(daemon_fd, time.monotonic() + timeout_seconds)
                    record_phase("daemon_exit_observed")
                    # Old wrappers have a forced tree-cleanup tail. Closing the
                    # daemon alone does not prove its other children are gone.
                    lane_children_gate(index)
                    effect_gate()
                    _require(
                        all_threads_stopped(controller), "controller_suspension_lost"
                    )
                    _require(
                        all_threads_stopped(lane.supervisor),
                        "supervisor_suspension_lost",
                    )
                    signal.pidfd_send_signal(supervisor_fd, signal.SIGTERM)
                # Release before cleanup, including implementations using this fence.
                signal.pidfd_send_signal(supervisor_fd, signal.SIGCONT)
                suspended_supervisors.remove(supervisor_fd)
                _wait_exit(supervisor_fd, time.monotonic() + timeout_seconds)
                _require(all_threads_stopped(controller), "controller_suspension_lost")
                closed_lanes.append(index)
                record_phase("lane_exit_observed")
            record_phase("controller_graceful_exit_prepared")
            closed_children_gate()
            effect_gate()
            _require(
                all(_exited(fd) for pair in lane_fds for fd in pair),
                "old_actor_exit_unverified",
            )
            _require(all_threads_stopped(controller), "controller_suspension_lost")
            signal.pidfd_send_signal(master_fd, signal.SIGTERM)
            controller_term_sent = True
        finally:
            try:
                for descriptor in reversed(suspended_supervisors):
                    try:
                        signal.pidfd_send_signal(descriptor, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
            finally:
                if stop_attempted:
                    # Never leave a live coordinator suspended after any refusal,
                    # timeout or journal error. This descriptor still names our birth.
                    try:
                        signal.pidfd_send_signal(master_fd, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
        if controller_term_sent:
            _wait_exit(master_fd, time.monotonic() + timeout_seconds)
        record_phase("controller_exit_observed")
    return {
        "closed_lanes": closed_lanes,
        "controller_exited": True,
        "callback_settlement_authority": False,
        "completion_authority": False,
    }
