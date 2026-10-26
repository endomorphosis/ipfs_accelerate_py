"""Private Linux subreaper wrapper for native CLI and validation execution.

This module is an executable implementation detail of
``legacy_landed_provider_cli``.  It deliberately has no provider-specific
logic and emits no diagnostics: the supervising process owns all output and
failure classification.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
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


def _reap_exited_adopted_children(command_pid: int) -> None:
    """Reap completed adoptees without consuming the Popen leader's status.

    A nested owned process group can wait for its orphaned zombie to vanish
    while the main command is still running. Waiting until the command exits
    would make that group wait on its own outer subreaper. Kernel parentage
    confines this nonblocking wait to this dedicated invocation's children;
    live adoptees retain their existing terminal cleanup boundary.
    """
    for process_id in _direct_child_pids():
        if process_id == command_pid:
            continue
        try:
            os.waitpid(process_id, os.WNOHANG)
        except ChildProcessError:
            pass


def run_with_child_custody(callback: Callable[[], int]) -> int:
    """Run one route in a fresh, dedicated worker and reap its descendants.

    This belongs inside the isolated worker's Unix identity. It must never
    wrap a callback in a shared daemon: after the empty-child prerequisite,
    every child of this process must belong to this invocation. The caller's
    outer watchdog remains responsible for an unresponsive worker. Returning
    here supplies no task, settlement, retry, or provider-usage authority.
    """
    if not callable(callback):
        raise TypeError("worker callback must be callable")
    if not _enable_child_subreaper():
        raise RuntimeError("worker child custody unavailable")
    task_root = Path("/proc/self/task")
    before = tuple(sorted(entry.name for entry in task_root.iterdir()))
    first = _direct_child_pids()
    middle = tuple(sorted(entry.name for entry in task_root.iterdir()))
    second = _direct_child_pids()
    after = tuple(sorted(entry.name for entry in task_root.iterdir()))
    if not before or before != middle or middle != after or first or second:
        raise RuntimeError("worker child custody requires an empty stable process")
    try:
        return callback()
    finally:
        # Same-UID descendants, including setsid/double-fork children and
        # zombies, remain owned until the kernel reports no children. A
        # cleanup failure must not turn into an observed worker exit.
        _kill_and_reap_orphans()


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
    inherited_fds: list[int] = []
    result_fd: int | None = None
    timeout_seconds: float | None = None
    output_limit: int | None = None
    try:
        while values and values[0] != "--":
            option, value = values[:2]
            values = values[2:]
            if option == "--pass-fd":
                descriptor = int(value)
                if descriptor < 3 or descriptor in inherited_fds:
                    raise ValueError("invalid inherited descriptor")
                os.fstat(descriptor)
                inherited_fds.append(descriptor)
            elif option == "--result-fd" and result_fd is None:
                result_fd = int(value)
                if result_fd < 3:
                    raise ValueError("invalid result descriptor")
                os.fstat(result_fd)
            elif option == "--timeout-seconds" and timeout_seconds is None:
                timeout_seconds = float(value)
                if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                    raise ValueError("invalid timeout")
            elif option == "--output-limit-bytes" and output_limit is None:
                output_limit = int(value)
                if output_limit < 1:
                    raise ValueError("invalid output limit")
            else:
                raise ValueError("unknown option")
        if (len(values) < 2 or values[0] != "--" or not values[1]
                or result_fd in inherited_fds):
            raise ValueError("invalid command")
    except (OSError, ValueError, TypeError):
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
    timed_out = False
    output_limit_exceeded = False
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    try:
        process = subprocess.Popen(
            values[1:],
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
            pass_fds=tuple(inherited_fds),
            start_new_session=False,
        )
        while process.poll() is None:
            _reap_exited_adopted_children(process.pid)
            timed_out = deadline is not None and time.monotonic() >= deadline
            output_limit_exceeded = output_limit is not None and os.fstat(1).st_size > output_limit
            if _termination_signal or timed_out or output_limit_exceeded:
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
        # Only this private wrapper holds the result descriptor. The nested
        # command receives exactly inherited_fds; it cannot write this report
        # through an accidentally inherited result channel.
        if result_fd is not None:
            raw = json.dumps({"returncode": return_code, "timed_out": timed_out,
                "output_limit_exceeded": output_limit_exceeded,
                "termination_signal": _termination_signal, "children_reaped": True},
                sort_keys=True).encode()
            view = memoryview(raw)
            while view:
                count = os.write(result_fd, view)
                if count <= 0:
                    raise OSError("result write failed")
                view = view[count:]
        if timed_out:
            return 124
        if output_limit_exceeded:
            return 75
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
