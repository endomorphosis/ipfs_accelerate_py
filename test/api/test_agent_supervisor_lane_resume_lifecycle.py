"""Lifecycle regressions for commands owned by the deterministic resume lane."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

pytestmark = pytest.mark.proof_reuse_disabled(reason="Must exercise real process cleanup")


def _command_tree(
    tmp_path: Path,
    *,
    leader_cooperates: bool = True,
    detached: bool = False,
    exit_after_release: bool = False,
    exit_code: int = 0,
) -> list[str]:
    """A cooperative leader with a descendant that refuses SIGTERM."""

    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(tmp_path / 'child.pid')!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    leader = tmp_path / "leader.py"
    handler = "lambda *_: sys.exit(0)" if leader_cooperates else "signal.SIG_IGN"
    wait = (
        f"while not Path({str(tmp_path / 'release')!r}).exists(): time.sleep(0.01)\n"
        if exit_after_release
        else "while True: time.sleep(1)\n"
    )
    leader.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"signal.signal(signal.SIGTERM, {handler})\n"
        f"Path({str(tmp_path / 'leader.pid')!r}).write_text(str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}], start_new_session={detached!r})\n"
        + wait
        + f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(leader)]


def _running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].strip().split()[0] != "Z"


def _cleanup_tree(tmp_path: Path) -> None:
    """Cleanup only the isolated session created by this test on failure."""

    path = tmp_path / "leader.pid"
    if path.is_file():
        try:
            os.killpg(int(path.read_text(encoding="utf-8")), signal.SIGKILL)
        except ProcessLookupError:
            pass
    child_path = tmp_path / "child.pid"
    if child_path.is_file():
        try:
            os.kill(int(child_path.read_text(encoding="utf-8")), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _assert_tree_stopped(tmp_path: Path) -> None:
    leader = int((tmp_path / "leader.pid").read_text(encoding="utf-8"))
    child = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while _running(child) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{leader}").exists(), "command leader was not reaped"
    assert not _running(child), "SIGTERM-resistant descendant survived cleanup"


@pytest.fixture(autouse=True)
def _short_command_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lane_resume, "_COMMAND_POLL_SECONDS", 0.05)
    monkeypatch.setattr(lane_resume, "_COMMAND_STOP_GRACE_SECONDS", 0.2)


@pytest.mark.parametrize("leader_cooperates", [True, False])
def test_command_timeout_kills_descendants_for_either_leader_behavior(
    tmp_path: Path, leader_cooperates: bool,
) -> None:
    argv = _command_tree(tmp_path, leader_cooperates=leader_cooperates)
    beats: list[float] = []
    started = time.monotonic()
    try:
        code = lane_resume._run_argv(
            argv,
            repo_root=tmp_path,
            timeout=1.0,
            log_path=tmp_path / "lane.log",
            heartbeat=lambda: beats.append(time.monotonic()),
        )
        elapsed = time.monotonic() - started
        assert code == 124, "a timed-out command must never become successful"
        assert 0.9 <= elapsed < 3
        assert len(beats) >= 2
        _assert_tree_stopped(tmp_path)
    finally:
        _cleanup_tree(tmp_path)


def test_heartbeat_error_stops_the_owned_command_tree(tmp_path: Path) -> None:
    argv = _command_tree(tmp_path)
    failure = RuntimeError("status storage is unavailable")

    def heartbeat() -> None:
        if (tmp_path / "child.pid").is_file():
            raise failure

    try:
        with pytest.raises(RuntimeError) as caught:
            lane_resume._run_argv(
                argv,
                repo_root=tmp_path,
                timeout=5,
                log_path=tmp_path / "lane.log",
                heartbeat=heartbeat,
            )
        assert caught.value is failure
        _assert_tree_stopped(tmp_path)
    finally:
        _cleanup_tree(tmp_path)


def test_successful_command_keeps_its_exit_code_and_output(tmp_path: Path) -> None:
    log = tmp_path / "lane.log"
    code = lane_resume._run_argv(
        [sys.executable, "-c", "print('finished')"],
        repo_root=tmp_path,
        timeout=5,
        log_path=log,
        heartbeat=lambda: None,
    )
    assert code == 0
    assert "finished\n" in log.read_text(encoding="utf-8")


def test_stop_during_successful_wait_is_checked_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False
    original_wait = lane_resume.subprocess.Popen.wait

    def wait(process, *args, **kwargs):
        nonlocal cancelled
        result = original_wait(process, *args, **kwargs)
        if result == 0:
            cancelled = True
        return result

    def heartbeat() -> None:
        if cancelled:
            raise lane_resume._LaneResumeInterrupted

    monkeypatch.setattr(lane_resume.subprocess.Popen, "wait", wait)
    with pytest.raises(lane_resume._LaneResumeInterrupted):
        lane_resume._run_argv(
            [sys.executable, "-c", "raise SystemExit(0)"],
            repo_root=tmp_path,
            timeout=5,
            log_path=tmp_path / "lane.log",
            heartbeat=heartbeat,
        )
    assert cancelled


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="detached descendant cleanup requires Linux pidfds",
)
@pytest.mark.parametrize("exit_code,cleanup", [(0, False), (0, True), (1, False)])
def test_command_cleans_children_according_to_exit_and_success_policy(
    tmp_path: Path, exit_code: int, cleanup: bool,
) -> None:
    argv = _command_tree(
        tmp_path, detached=True, exit_after_release=True, exit_code=exit_code,
    )

    def heartbeat() -> None:
        if (tmp_path / "child.pid").is_file():
            (tmp_path / "release").touch()

    try:
        options = {"cleanup_descendants_on_success": True} if cleanup else {}
        assert lane_resume._run_argv(
            argv,
            repo_root=tmp_path,
            timeout=5,
            log_path=tmp_path / "lane.log",
            heartbeat=heartbeat,
            **options,
        ) == exit_code
        if cleanup or exit_code != 0:
            _assert_tree_stopped(tmp_path)
        else:
            child = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
            assert _running(child), "successful service launch lost its background child"
    finally:
        _cleanup_tree(tmp_path)


@pytest.mark.parametrize("stop_reason", ["signal", "deadline"])
def test_serve_interrupts_active_command_before_publishing_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_reason: str,
) -> None:
    argv = _command_tree(tmp_path)
    handlers: dict[int, Any] = {}
    monkeypatch.setattr(lane_resume.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    board = tmp_path / "board.md"
    original = "## VOG-001 Task\n- Status: pending\n"
    board.write_text(original, encoding="utf-8")

    def run_cycle(**kwargs: Any) -> dict[str, Any]:
        def heartbeat() -> None:
            if stop_reason == "signal" and (tmp_path / "child.pid").is_file():
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            kwargs["heartbeat"](active_task_id="VOG-001", phase="deterministic_command")

        lane_resume._run_argv(
            argv,
            repo_root=tmp_path,
            timeout=30,
            log_path=tmp_path / "lane.log",
            heartbeat=heartbeat,
        )
        pytest.fail("interrupted command returned to task completion")

    monkeypatch.setattr(lane_resume, "run_resume_cycle", run_cycle)
    started = time.monotonic()
    try:
        assert lane_resume.serve(
            [
                "--repo-root", str(tmp_path),
                "--todo-path", str(board),
                "--task-header-prefix", "## VOG-",
                "--status-path", str(tmp_path / "status.json"),
                "--pid-path", str(tmp_path / "lane.pid"),
                "--log-path", str(tmp_path / "lane.log"),
                "--bundle-key", "test-lane",
                "--max-seconds", "1" if stop_reason == "deadline" else "30",
            ]
        ) == 0
        assert time.monotonic() - started < 3
        status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "stopped"
        assert status["phase"] == "lane_resume_exited"
        assert status["terminal_reason"] == (
            "resume_stopped" if stop_reason == "signal" else "resume_duration_elapsed"
        )
        assert status["work_complete"] is False
        assert status["completion_authority"] is False
        assert board.read_text(encoding="utf-8") == original
        _assert_tree_stopped(tmp_path)
    finally:
        _cleanup_tree(tmp_path)


def test_long_failed_command_starts_backoff_when_failure_is_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = tmp_path / "board.md"
    original = (
        "## TEST-001 Long browser command\n"
        "- Status: pending\n"
        "- Depends on:\n"
        "- Deterministic command: python3 -c 'raise SystemExit(1)'\n"
        "- Resource class: cpu-browser\n"
        "- Implementation timeout seconds: 7200\n"
    )
    board.write_text(original, encoding="utf-8")
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps({"TEST-001": 2}), encoding="utf-8")
    now = [1_000.0]
    clock = SimpleNamespace(time=lambda: now[0], monotonic_ns=time.monotonic_ns)
    monkeypatch.setattr(lane_resume, "time", clock)
    invocations: list[list[str]] = []

    def command(argv, **_kwargs):
        invocations.append(list(argv))
        now[0] += 3_600.0
        return 1

    monkeypatch.setattr(lane_resume, "_run_argv", command)
    arguments = {
        "repo_root": tmp_path,
        "todo_path": board,
        "task_header_prefix": "## TEST-",
        "attempt_path": attempts_path,
        "log_path": tmp_path / "lane.log",
        "heartbeat": lambda **_kwargs: None,
    }
    failed = lane_resume.run_resume_cycle(**arguments)
    assert failed["phase"] == "deterministic_command_failed"
    persisted = json.loads(attempts_path.read_text(encoding="utf-8"))
    assert persisted["TEST-001"] == 3
    assert persisted["_retry_after_ms"]["TEST-001"] >= (
        int(now[0] * 1000) + lane_resume._EXHAUSTED_RETRY_MS
    )
    waiting = lane_resume.run_resume_cycle(**arguments)
    assert waiting["phase"] == "attempt_retry_wait"
    assert len(invocations) == 1, "an hour-long failure immediately ran again"
    assert waiting["work_complete"] is waiting["completion_authority"] is False
    assert board.read_text(encoding="utf-8") == original


def test_resume_cycle_preserves_loaded_command_evidence(tmp_path, monkeypatch):
    board = tmp_path / "board.md"
    board.write_text("## TEST-001 Retry\n- Status: pending\n- Depends on:\n- Deterministic command: python3 -c 'raise SystemExit(1)'\n")
    attempts = tmp_path / "attempts.json"
    evidence = {"older-task": {"fingerprint": "keep"}}
    attempts.write_text(json.dumps({"_command_evidence": evidence}))
    monkeypatch.setattr(lane_resume, "_run_argv", lambda *args, **kwargs: 1)
    result = lane_resume.run_resume_cycle(
        repo_root=tmp_path, todo_path=board, task_header_prefix="## TEST-",
        attempt_path=attempts, log_path=tmp_path / "lane.log",
        heartbeat=lambda **kwargs: None,
    )
    assert result["phase"] == "deterministic_command_failed"
    assert json.loads(attempts.read_text())["_command_evidence"]["older-task"] == evidence["older-task"]
    assert result["work_complete"] is result["completion_authority"] is False
