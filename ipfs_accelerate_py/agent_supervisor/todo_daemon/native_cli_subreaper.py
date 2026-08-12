"""Private Linux subreaper wrapper for native provider CLI execution.

This module is an executable implementation detail of
``legacy_landed_provider_cli``.  It deliberately has no provider-specific
logic and emits no diagnostics: the supervising process owns all output and
failure classification.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_CONFINEMENT_FAILURE_EXIT_CODE = 125
_TERMINATION_POLL_SECONDS = 0.01

_termination_signal = 0


def _enable_child_subreaper() -> bool:
    """Enable and verify the Linux child-subreaper contract."""

    if sys.platform != "linux":
        return False
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    try:
        children_path.read_text(encoding="ascii")
    except OSError:
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.restype = ctypes.c_int
        if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            return False
        enabled = ctypes.c_int(0)
        if prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(enabled), 0, 0, 0) != 0:
            return False
        return enabled.value == 1
    except (AttributeError, OSError, TypeError):
        return False


def _record_termination(signum: int, _frame: FrameType | None) -> None:
    """Request cleanup from the ordinary interpreter control flow."""

    global _termination_signal
    if not _termination_signal:
        _termination_signal = int(signum)


def _direct_child_pids() -> set[int]:
    """Return all children of every thread in this wrapper process."""

    result: set[int] = set()
    task_root = Path("/proc/self/task")
    task_paths = tuple(task_root.iterdir())
    for task_path in task_paths:
        try:
            values = (task_path / "children").read_text(encoding="ascii").split()
        except FileNotFoundError:
            # A short-lived interpreter helper thread may disappear between
            # listing and reading.  Re-reading on the next pass is sufficient.
            continue
        for value in values:
            result.add(int(value))
    return result


def _kill_and_reap_orphans() -> None:
    """Do not return until the kernel reports that no child remains.

    Once this process is a subreaper, an orphan at any depth is reparented
    here.  Killing only the current direct children and repeating therefore
    closes the fork/setsid/double-fork race without relying on polling having
    observed a process before its original parent exited.
    """

    while True:
        try:
            children = _direct_child_pids()
        except (OSError, ValueError):
            # The prerequisite check succeeded before provider execution.  A
            # transient procfs read failure must delay result acceptance, not
            # allow descendants to escape by making the wrapper exit.
            time.sleep(_TERMINATION_POLL_SECONDS)
            continue
        for process_id in children:
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # A genuine child has the same credentials.  Treat a denial
                # as transient and retain the subreaper fence.
                pass

        try:
            waited_pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if waited_pid == 0:
            time.sleep(_TERMINATION_POLL_SECONDS)


def _normalized_exit_code(return_code: int) -> int:
    """Translate Popen's signal result into conventional shell status."""

    if return_code < 0:
        return min(255, 128 + abs(return_code))
    return min(255, int(return_code))


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one argv under the verified subreaper fence."""

    global _termination_signal
    _termination_signal = 0
    values = list(sys.argv[1:] if arguments is None else arguments)
    if len(values) < 2 or values[0] != "--" or not values[1]:
        return _CONFINEMENT_FAILURE_EXIT_CODE
    if not _enable_child_subreaper():
        return _CONFINEMENT_FAILURE_EXIT_CODE

    # Verify that procfs child enumeration works before executing untrusted
    # provider code.  If confinement is unavailable, the CLI is never run.
    try:
        if _direct_child_pids():
            return _CONFINEMENT_FAILURE_EXIT_CODE
    except (OSError, ValueError):
        return _CONFINEMENT_FAILURE_EXIT_CODE

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _record_termination)

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            values[1:],
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
            start_new_session=False,
        )
        while process.poll() is None:
            if _termination_signal:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                break
            try:
                process.wait(timeout=_TERMINATION_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                continue
        return_code = process.wait()
        _kill_and_reap_orphans()
        if _termination_signal:
            return min(255, 128 + _termination_signal)
        return _normalized_exit_code(return_code)
    except BaseException:
        if process is not None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait()
            except BaseException:
                pass
        _kill_and_reap_orphans()
        return _CONFINEMENT_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
