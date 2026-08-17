from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.leased_lane import (
    _capture_spawned_direct_child_start_time,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import core as core_module
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


def test_strict_fence_rejects_reused_root_before_any_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = 4242
    monkeypatch.setattr(
        core_module,
        "_process_identity_snapshot",
        lambda: {pid: ("S", 1, pid, pid, "999")},
    )
    monkeypatch.setattr(
        core_module.os,
        "kill",
        lambda *_args, **_kwargs: pytest.fail("reused PID was signalled"),
    )
    monkeypatch.setattr(
        core_module.os,
        "killpg",
        lambda *_args, **_kwargs: pytest.fail("reused process group was signalled"),
    )

    assert not terminate_pid_tree(
        pid,
        grace_seconds=0.0,
        freeze_first=True,
        require_gone=True,
        owned_process_group_id=pid,
        expected_root_start_time_ticks=123,
    )


def test_strict_fence_rejects_claimed_process_group_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = 4243
    monkeypatch.setattr(
        core_module,
        "_process_identity_snapshot",
        lambda: {pid: ("S", 1, 777, 777, "123")},
    )
    monkeypatch.setattr(
        core_module.os,
        "kill",
        lambda *_args, **_kwargs: pytest.fail(
            "process with a mismatched ownership group was signalled"
        ),
    )
    monkeypatch.setattr(
        core_module.os,
        "killpg",
        lambda *_args, **_kwargs: pytest.fail(
            "mismatched process group was signalled"
        ),
    )

    assert not terminate_pid_tree(
        pid,
        grace_seconds=0.0,
        freeze_first=True,
        require_gone=True,
        owned_process_group_id=pid,
        expected_root_start_time_ticks=123,
    )


def test_naturally_exited_direct_child_retains_empty_group_authority() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        birth = read_process_birth(child.pid)
        assert birth is not None
        assert birth.parent_pid == os.getpid()
        child.wait(timeout=3.0)

        assert terminate_pid_tree(
            child.pid,
            grace_seconds=0.0,
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=child.pid,
            expected_root_start_time_ticks=birth.start_time_ticks,
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1.0)


def test_fast_zombie_child_birth_is_captured_before_reap() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3.0
        state = ""
        while time.monotonic() < deadline:
            raw = Path(f"/proc/{child.pid}/stat").read_text(encoding="utf-8")
            close = raw.rfind(")")
            state = raw[close + 2 :].split()[0]
            if state == "Z":
                break
            time.sleep(0.005)
        assert state == "Z"
        start_time = _capture_spawned_direct_child_start_time(
            child.pid,
            expected_parent_pid=os.getpid(),
        )
        assert start_time is not None and start_time > 0
        child.wait(timeout=3.0)
        assert terminate_pid_tree(
            child.pid,
            grace_seconds=0.0,
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=child.pid,
            expected_root_start_time_ticks=start_time,
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1.0)


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
