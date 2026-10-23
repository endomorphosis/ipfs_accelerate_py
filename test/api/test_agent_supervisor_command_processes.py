from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import command_processes


pytestmark = pytest.mark.skipif(
    not Path("/proc/self/task").exists()
    or not hasattr(os, "pidfd_open")
    or not hasattr(signal, "pidfd_send_signal"),
    reason="descendant identity fencing requires Linux pidfds",
)


def _until(predicate, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("process did not reach expected state")
        time.sleep(0.01)


def _stopped(pid: int) -> bool:
    identity = command_processes._identity(pid)
    return identity is None or identity[0] in {"Z", "X"}


def test_cleanup_reaches_detached_orphan_and_preserves_unrelated_process(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); "
        "time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    tree = command_processes.CommandProcessTree(parent.pid)
    child_pid = 0
    try:
        _until(lambda: marker.exists() and bool(marker.read_text()))
        child_pid = int(marker.read_text())
        assert os.getpgid(child_pid) != os.getpgid(parent.pid)
        tree.snapshot()
        parent.terminate()
        parent.wait(timeout=3)
        # Parentage has disappeared, but the observer retains the owned handle.
        tree.snapshot()
        tree.send_signal(signal.SIGTERM)
        assert not _stopped(child_pid)
        tree.send_signal(signal.SIGKILL)
        _until(lambda: _stopped(child_pid))
        assert outsider.poll() is None
    finally:
        tree.send_signal(signal.SIGKILL)
        tree.close()
        for process in (parent, outsider):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        if child_pid and not _stopped(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_changed_identity_while_opening_handle_is_never_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = command_processes.CommandProcessTree(os.getpid())
    versions = iter([("S", "original"), ("S", "reused")])
    closed: list[int] = []
    monkeypatch.setattr(command_processes, "_identity", lambda _pid: next(versions))
    monkeypatch.setattr(os, "pidfd_open", lambda _pid, _flags: 777)
    monkeypatch.setattr(os, "close", closed.append)
    assert tree._remember(12345) is False
    assert closed == [777]
    assert not tree._owned


def test_missing_pidfd_support_keeps_process_group_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "pidfd_open")
    tree = command_processes.CommandProcessTree(os.getpid())
    tree.snapshot()
    tree.send_signal(signal.SIGTERM)
    tree.close()
    assert not tree._owned
