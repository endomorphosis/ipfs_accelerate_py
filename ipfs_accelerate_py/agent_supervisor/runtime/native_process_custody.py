"""Linux process identities and original pidfd custody for admitted recovery.

This is a mechanism for an already admitted native operator.  Its gates must
bind the configured source, process roster, launch fences and protected state.
It supplies no task, callback, source-transition or completion authority.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import select
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


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
