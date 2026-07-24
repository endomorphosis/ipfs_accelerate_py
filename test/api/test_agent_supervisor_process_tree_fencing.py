from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import (
    pid_alive,
    terminate_pid_tree,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="process-group fencing regression requires Linux process sessions",
)


def _wait_until_dead(pid: int, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not pid_alive(pid)


def test_terminate_pid_tree_fences_descendant_in_separate_session(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "separate-session-child.pid"
    parent_script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(60)'], "
        "start_new_session=True"
        "); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    child_pid = 0
    try:
        deadline = time.monotonic() + 3.0
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert os.getsid(parent.pid) == parent.pid
        assert os.getsid(child_pid) == child_pid

        assert terminate_pid_tree(parent.pid, grace_seconds=0.2)
        _wait_until_dead(child_pid)
        _wait_until_dead(parent.pid)
        parent.wait(timeout=1.0)
    finally:
        if child_pid and pid_alive(child_pid):
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parent.poll() is None:
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            parent.wait(timeout=1.0)
