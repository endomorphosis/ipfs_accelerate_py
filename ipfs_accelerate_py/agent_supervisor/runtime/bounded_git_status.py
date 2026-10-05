"""Capture a supervisor-owned Git status without buffering unbounded output."""
from __future__ import annotations

import locale
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from typing import Mapping, Sequence


def capture_git_status(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str],
                       input_bytes: bytes | None, maximum_stdout_bytes: int,
                       maximum_stderr_bytes: int, maximum_records: int,
                       timeout_seconds: float) -> subprocess.CompletedProcess:
    """Preserve caller argv/environment; bound both pipes before retaining bytes.

    Uses the selector capture pattern from production_context_slice, while the
    plan-bound caller retains its own fixed Git/environment/deadline contract.
    This helper provides no command, task, or process-adoption authority.
    """
    if any(type(value) is not int or value < 1 for value in (
            maximum_stdout_bytes, maximum_stderr_bytes, maximum_records)):
        raise ValueError("positive Git status capture bounds required")
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError("positive finite Git status deadline required")
    selector = selectors.DefaultSelector()
    process = None
    owned_group = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": maximum_stdout_bytes, "stderr": maximum_stderr_bytes}
    records = 0
    deadline = time.monotonic() + timeout_seconds
    remaining_input = memoryview(input_bytes or b"")
    try:
        process = subprocess.Popen(list(command), cwd=cwd, env=dict(environment),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        # Git status may spawn Git children for submodules. This new session
        # is owned by this invocation; never borrow the supervisor's group.
        if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
            raise RuntimeError("Git status child did not establish its own process group")
        owned_group = process.pid
        for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            if stream is None:
                raise RuntimeError("Git status capture pipe unavailable")
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        if process.stdin is not None:
            if remaining_input:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                process.stdin.close()
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(list(command), timeout_seconds)
            for key, _ in selector.select(min(remaining, 0.1)):
                stream, label = key.fileobj, key.data
                if label == "stdin":
                    try:
                        written = os.write(stream.fileno(), remaining_input[:65_536])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        remaining_input = remaining_input[len(remaining_input):]
                    else:
                        remaining_input = remaining_input[written:]
                    if not remaining_input:
                        selector.unregister(stream)
                        stream.close()
                    continue
                target = buffers[label]
                try:
                    chunk = os.read(stream.fileno(), min(65_536, limits[label] - len(target) + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    continue
                if len(target) + len(chunk) > limits[label]:
                    raise ValueError(f"plan-bound Git status {label} byte bound exceeded")
                if label == "stdout":
                    records += chunk.count(b"\0")
                    if records > maximum_records:
                        raise ValueError("plan-bound Git status record bound exceeded")
                target.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(list(command), timeout_seconds)
        returncode = process.wait(timeout=remaining)
        # Once reaped, the leader PID can be reused. Decode failures below
        # must never signal a process group using that expired identity.
        owned_group = None
        output = {name: bytes(value) for name, value in buffers.items()}
        if input_bytes is None:
            output = {name: value.decode(locale.getencoding()).replace("\r\n", "\n").replace("\r", "\n")
                      for name, value in output.items()}
        return subprocess.CompletedProcess(list(command), returncode, output["stdout"], output["stderr"])
    except BaseException:
        if process is not None:
            if owned_group is not None:
                # Do not reap the group leader before fencing its group: a
                # subordinate may still hold output pipes after leader exit.
                try:
                    os.killpg(owned_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            # Reap our exact Popen child using a bounded cleanup interval.
            process.wait(timeout=1.0)
        raise
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
