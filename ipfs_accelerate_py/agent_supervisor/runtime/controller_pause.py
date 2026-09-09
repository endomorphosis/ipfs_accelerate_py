"""Bounded exact-controller pause for native source qualification windows.

This does not quiesce children or establish task recovery authority. Callers
must separately hold the native mutation and daemon launch fences, keep their
CAS/preflight/rollback inside the deadline, and preserve provider workspaces.
"""

from __future__ import annotations

import json
import math
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from ..merge.worktree_lifecycle import ProcessBirthIdentity, read_process_birth

# A distinct systemd user scope owns the same pidfd before STOP. EOF,
# explicit cleanup, or the wall-clock bound all resume that exact process. It
# does not inherit owner/launch locks, credentials, or a process-group target.
_RESUMER = r"""
import json, os, select, signal, sys, time
pidfd, watch, ready = map(int, sys.argv[1:4])
bound = float(sys.argv[4])
expected = json.loads(sys.argv[5])
with open('/proc/self/cgroup') as stream:
    cgroup = stream.read().strip()
def reply(value):
    os.write(ready, (json.dumps(value) + '\n').encode())
def current():
    with open('/proc/%s/stat' % expected['pid']) as stream:
        raw = stream.read()
    fields = raw[raw.rfind(')')+2:].split()
    with open('/proc/sys/kernel/random/boot_id') as stream:
        boot_id = stream.read().strip()
    birth = {'pid': expected['pid'], 'parent_pid': int(fields[1]),
             'start_time_ticks': int(fields[19]), 'boot_id': boot_id}
    if birth != expected: raise RuntimeError('guardian birth changed')
    return fields[0]
reply({'pid': os.getpid(), 'cgroup': cgroup})
stopped = False
try:
    if not select.select([watch], [], [], bound)[0] or os.read(watch, 1) != b'A':
        raise RuntimeError('pause request expired or caller closed')
    if current() in {'T','t','Z','X'}:
        raise RuntimeError('controller not independently runnable')
    # This guardian exclusively owns STOP and its bounded CONT. A delayed
    # caller can never send a late STOP after the resumer has already exited.
    stopped = True
    signal.pidfd_send_signal(pidfd, signal.SIGSTOP)
    deadline = time.monotonic() + bound
    until = min(deadline, time.monotonic() + 1)
    while current() not in {'T','t'}:
        if time.monotonic() >= until: raise RuntimeError('controller did not stop')
        time.sleep(.005)
    reply({'paused': True, 'deadline': deadline})
    select.select([watch], [], [], max(0, deadline-time.monotonic()))
finally:
    if stopped:
        try: signal.pidfd_send_signal(pidfd, signal.SIGCONT)
        except ProcessLookupError: pass
    os.close(ready)
    os.close(watch)
    os.close(pidfd)
"""


class ControllerPauseError(RuntimeError):
    """The exact controller could not enter or retain its bounded pause."""


def _state(pid: int) -> str:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
        raw = stream.read()
    return raw[raw.rfind(")") + 2 :].split()[0]


@dataclass(frozen=True)
class ControllerPause:
    identity: ProcessBirthIdentity
    deadline: float
    resumer_pid: int
    resumer_unit: str
    resumer_cgroup: str

    def require_paused(self) -> None:
        if time.monotonic() >= self.deadline:
            raise ControllerPauseError("controller pause deadline elapsed")
        if read_process_birth(self.identity.pid) != self.identity:
            raise ControllerPauseError("controller process birth changed")
        if _state(self.identity.pid) not in {"T", "t"}:
            raise ControllerPauseError("controller is no longer paused")


@contextmanager
def pause_exact_controller(
    identity: ProcessBirthIdentity,
    *,
    max_seconds: float = 30.0,
) -> Iterator[ControllerPause]:
    """Pause one proven birth, with timed SIGCONT in an independent user scope.

    A working systemd user manager is required; no same-cgroup fallback exists.
    Already-stopped processes are refused so cleanup cannot resume somebody
    else's pause. No descendant, process group, task record, or lease is touched.
    The caller must abort and roll back before the bound; the failsafe resumes
    the controller even if the caller dies or becomes stuck.
    """
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 60:
        raise ValueError("controller pause bound must be in (0, 60] seconds")
    if (
        identity.pid <= 1
        or identity.pid == os.getpid()
        or identity.start_time_ticks <= 0
        or not identity.boot_id
        or identity.parent_pid <= 0
    ):
        raise ControllerPauseError("complete external controller birth is required")
    if read_process_birth(identity.pid) != identity:
        raise ControllerPauseError("controller process birth differs")
    if _state(identity.pid) in {"T", "t", "Z", "X"}:
        raise ControllerPauseError("controller is not independently runnable")
    pidfd = os.pidfd_open(identity.pid, 0)
    watch_read, watch_write = os.pipe()
    ready_read, ready_write = os.pipe()
    guardian = None
    stopped = False
    try:
        if read_process_birth(identity.pid) != identity:
            raise ControllerPauseError("controller process birth changed at pidfd bind")
        unit = "ipfs-controller-resumer-" + uuid.uuid4().hex + ".scope"
        guardian = subprocess.Popen(
            [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "--unit",
                unit,
                sys.executable,
                "-I",
                "-c",
                _RESUMER,
                str(pidfd),
                str(watch_read),
                str(ready_write),
                str(max_seconds),
                json.dumps(identity.to_dict()),
            ],
            pass_fds=(pidfd, watch_read, ready_write),
            env={
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"}
            },
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(watch_read)
        watch_read = -1
        os.close(ready_write)
        ready_write = -1
        if not select.select([ready_read], [], [], 5.0)[0]:
            raise ControllerPauseError("independent controller resumer is not ready")
        try:
            ready = json.loads(os.read(ready_read, 4096))
            guardian_pid = ready["pid"]
            guardian_cgroup = ready["cgroup"]
            with open("/proc/self/cgroup", encoding="utf-8") as stream:
                caller_cgroup = stream.read().strip()
            with open(f"/proc/{guardian_pid}/cgroup", encoding="utf-8") as stream:
                observed_cgroup = stream.read().strip()
            if (
                type(guardian_pid) is not int
                or guardian_pid <= 1
                or not isinstance(guardian_cgroup, str)
                or not guardian_cgroup.endswith("/" + unit)
                or guardian_cgroup != observed_cgroup
                or guardian_cgroup == caller_cgroup
                or guardian_cgroup.startswith(caller_cgroup + "/")
            ):
                raise ValueError("resumer is not in its independent user scope")
        except (KeyError, ValueError, TypeError, OSError) as exc:
            raise ControllerPauseError(
                "independent resumer cgroup admission failed"
            ) from exc
        if read_process_birth(identity.pid) != identity or _state(identity.pid) in {
            "T",
            "t",
        }:
            raise ControllerPauseError("controller changed before pause request")
        try:
            os.write(watch_write, b"A")
            if not select.select([ready_read], [], [], 2.0)[0]:
                raise ValueError("guardian did not acknowledge pause")
            acknowledgement = json.loads(os.read(ready_read, 4096))
            deadline = acknowledgement.get("deadline")
            if (
                acknowledgement.get("paused") is not True
                or type(deadline) not in {float, int}
                or not math.isfinite(deadline)
            ):
                raise ValueError("invalid paused acknowledgement")
        except (OSError, ValueError, TypeError) as exc:
            raise ControllerPauseError(
                "independent guardian pause request expired or failed"
            ) from exc
        stopped = True
        pause = ControllerPause(identity, deadline, guardian_pid, unit, guardian_cgroup)
        pause.require_paused()
        yield pause
    finally:
        if stopped:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            except ProcessLookupError:
                pass
        # Closing the only writer is the guardian's immediate cleanup request.
        for fd in (watch_write, watch_read, ready_read, ready_write):
            if fd >= 0:
                os.close(fd)
        if guardian is not None:
            try:
                guardian.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                # It independently owns the pidfd and its hard resume bound.
                pass
        os.close(pidfd)
