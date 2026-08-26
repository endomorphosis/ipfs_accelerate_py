from __future__ import annotations

import errno
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
from ipfs_accelerate_py.agent_supervisor.todo_daemon.process_tree_fencing import (
    ProcessTreeFenceError,
    fence_process_tree,
    reject_unsafe_cleanup,
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
        lambda: core_module.ProcessIdentitySnapshot.observed(
            {pid: ("S", 1, pid, pid, "999")}
        ),
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
        lambda: core_module.ProcessIdentitySnapshot.observed(
            {pid: ("S", 1, 777, 777, "123")}
        ),
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


def _write_proc_stat(
    proc_root: Path,
    *,
    pid: int,
    start_time_ticks: int,
) -> Path:
    process_root = proc_root / str(pid)
    process_root.mkdir()
    stat_path = process_root / "stat"
    fields = [
        "S",
        "1",
        str(pid),
        str(pid),
        *("0" for _ in range(15)),
        str(start_time_ticks),
    ]
    assert len(fields) == 20
    stat_path.write_text(
        f"{pid} (worker) {' '.join(fields)}\n",
        encoding="utf-8",
    )
    return stat_path


@pytest.mark.parametrize(
    ("error_type", "error_number"),
    (
        (FileNotFoundError, errno.ENOENT),
        (ProcessLookupError, errno.ESRCH),
    ),
)
def test_process_identity_snapshot_skips_only_disappeared_proc_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
    error_number: int,
) -> None:
    retained_pid = 4244
    disappeared_pid = 4245
    _write_proc_stat(
        tmp_path,
        pid=retained_pid,
        start_time_ticks=123,
    )
    disappeared_stat = _write_proc_stat(
        tmp_path,
        pid=disappeared_pid,
        start_time_ticks=456,
    )
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == disappeared_stat:
            raise error_type(
                error_number,
                os.strerror(error_number),
                str(path),
            )
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    snapshot = core_module._process_identity_snapshot(proc_root=tmp_path)

    assert snapshot.available
    assert snapshot.error == ""
    assert snapshot.processes == {
        retained_pid: ("S", 1, retained_pid, retained_pid, "123")
    }


def test_process_identity_snapshot_keeps_other_proc_errors_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inaccessible_pid = 4246
    inaccessible_stat = _write_proc_stat(
        tmp_path,
        pid=inaccessible_pid,
        start_time_ticks=789,
    )
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == inaccessible_stat:
            raise PermissionError(
                errno.EACCES,
                os.strerror(errno.EACCES),
                str(path),
            )
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    snapshot = core_module._process_identity_snapshot(proc_root=tmp_path)

    assert not snapshot.available
    assert snapshot.processes == {}
    assert f"{inaccessible_pid}: PermissionError:" in snapshot.error


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


def test_fence_process_tree_stops_separate_session_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "published" / "receipt.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"keep":true}', encoding="utf-8")
    child_pid_path = tmp_path / "fenced-child.pid"
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
        receipt = fence_process_tree(
            parent.pid,
            grace_seconds=0.2,
            freeze_first=True,
            cleanup_paths=(tmp_path / "scratch.json",),
        )
        _wait_until_dead(child_pid)
        _wait_until_dead(parent.pid)
        parent.wait(timeout=1.0)
        assert receipt.terminated
        assert evidence.read_text(encoding="utf-8") == '{"keep":true}'
        with pytest.raises(ProcessTreeFenceError, match="unsafe cleanup"):
            reject_unsafe_cleanup([evidence], flags=("delete_published",))
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
