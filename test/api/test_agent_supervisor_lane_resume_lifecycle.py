"""Lifecycle regressions for commands owned by the deterministic resume lane."""

from __future__ import annotations

import json
import fcntl
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


@pytest.mark.parametrize(("resource", "declared", "expected"), [
    ("cpu-browser", "25200", 25200.0),
    ("cpu-browser", "28800", 28800.0),
    ("cpu-browser", "46800", 46800.0),
    ("cpu-browser", "50400", 50400.0),
    ("cpu-browser", "50401", 50400.0),
    ("cpu-browser", "", 1800.0),
    ("cpu-browser", "invalid", 1800.0),
    ("cpu-small", "25200", 1800.0),
    ("cpu-small", "50400", 1800.0),
    ("cpu-small", "", 1800.0),
    ("cpu-other", "25200", 1800.0),
    ("cpu-other", "50400", 1800.0),
    ("", "25200", 1800.0),
])
def test_declared_browser_workload_bound_reaches_producer_without_extending_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    resource: str, declared: str, expected: float,
) -> None:
    board = tmp_path / "board.md"
    original = (
        "## TEST-001 Source-bound browser acquisition\n"
        "- Status: pending\n- Depends on:\n"
        "- Deterministic command: python3 -c 'raise SystemExit(1)'\n"
        f"- Resource class: {resource}\n"
        f"- Implementation timeout seconds: {declared}\n"
    )
    board.write_text(original, encoding="utf-8")
    observed = []
    def invoke(_argv, **kwargs):
        observed.append(kwargs["timeout"])
        return 1
    monkeypatch.setattr(lane_resume, "_run_argv", invoke)
    result = lane_resume.run_resume_cycle(
        repo_root=tmp_path, todo_path=board, task_header_prefix="## TEST-",
        attempt_path=tmp_path / "attempts.json", log_path=tmp_path / "lane.log",
        heartbeat=lambda **_kwargs: None,
    )
    assert observed == [expected]
    assert result["phase"] == "deterministic_command_failed"
    assert result["work_complete"] is result["completion_authority"] is False
    assert board.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(("lane_budget", "manifest_budget", "expected"), [
    (50400, 28800, "50400"),
    (None, 50400, "50400"),
    (None, None, "28800"),
])
def test_fenced_resume_uses_explicit_workload_lifetime_without_extending_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    lane_budget: int | None, manifest_budget: int | None, expected: str,
) -> None:
    lane = {
        "bundle_key": "test", "state_dir": str(tmp_path), "state_prefix": "test",
        "pid_path": str(tmp_path / "lane.pid"), "todo_path": "board.md",
        "task_header_prefix": "## TEST-",
    }
    manifest = {}
    if lane_budget is not None:
        lane["max_resume_seconds"] = lane_budget
    if manifest_budget is not None:
        manifest["max_resume_seconds"] = manifest_budget
    spawned = []

    def spawn(argv, **_kwargs):
        spawned.append(argv)
        return SimpleNamespace(pid=999999)

    monkeypatch.setattr(lane_resume.subprocess, "Popen", spawn)
    monkeypatch.setattr(lane_resume, "_await_resume_ready", lambda *_args, **_kwargs: "")
    result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path, manifest=manifest)
    assert result["restarted"] is True
    assert len(spawned) == 1
    assert spawned[0][spawned[0].index("--max-seconds") + 1] == expected
    assert result["work_complete"] is result["completion_authority"] is False
    assert result["readiness_verified"] is True


def _startup_lane(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "board.md").write_text(
        "## TEST-001 Real pending gate\n- Status: pending\n- Depends on:\n"
        "- Deterministic command: python3 -c 'raise SystemExit(75)'\n",
        encoding="utf-8",
    )
    return {"bundle_key": "test", "state_dir": str(tmp_path), "state_prefix": "test",
            "pid_path": str(tmp_path / "lane.pid"), "todo_path": "board.md",
            "task_header_prefix": "## TEST-", "max_resume_seconds": 2}


def _reap_startup_lane(pid: int) -> None:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        try:
            reaped, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if reaped == pid:
            return
        time.sleep(0.02)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    os.waitpid(pid, 0)
    pytest.fail("isolated startup lane did not terminate within its short test lifetime")


def test_real_resume_hands_lifetime_fence_to_child_while_parent_still_holds_it(tmp_path: Path) -> None:
    lane = _startup_lane(tmp_path)
    lock_path = tmp_path / "test.resume.lock"
    pid = None
    try:
        with lock_path.open("a") as parent_fence:
            fcntl.flock(parent_fence, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path, resume_lock_fd=parent_fence.fileno())
            assert result["restarted"] is result["readiness_verified"] is result["lock_handoff"] is True
            pid = result["new_pid"]
            assert _running(pid)
            assert int((tmp_path / "lane.pid").read_text()) == pid
            status = json.loads((tmp_path / "test_status.json").read_text())
            assert status["pid"] == pid and status["target_id"] == "test"
            assert status["state"] != "stopped" and status["heartbeat_at_ms"] > 0
            # Reproduce the formerly fatal parent delay after spawning but
            # before its with-statement closes the exact same lifetime fence.
            time.sleep(0.2)
            assert _running(pid)
            duplicate = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path)
            assert duplicate["restarted"] is False and duplicate["reason"] == "live_lane_pid"
        # The child's inherited open-file description must keep the exclusive
        # fence after the parent closes its own copy, with no unlock interval.
        with lock_path.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _running(pid)
    finally:
        if pid is not None:
            _reap_startup_lane(pid)
    with lock_path.open("a") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_real_resume_reports_lock_contention_as_failure_without_touching_owner_state(tmp_path: Path) -> None:
    lane = _startup_lane(tmp_path)
    status_path = tmp_path / "test_status.json"
    status_path.write_text('{"owner":"preserve exact bytes"}\n')
    original = status_path.read_bytes()
    lock_path = tmp_path / "test.resume.lock"
    with lock_path.open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Omitting the explicit handoff faithfully reproduces the old race.
        result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path)
        assert result["restarted"] is result["readiness_verified"] is False
        assert result["reason"] == "resume_exited_before_ready"
        assert not (tmp_path / "lane.pid").exists()
        assert status_path.read_bytes() == original
        with lock_path.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_real_resume_without_parent_fence_still_requires_child_readiness(tmp_path: Path) -> None:
    lane = _startup_lane(tmp_path)
    result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path)
    assert result["restarted"] is result["readiness_verified"] is True
    assert result["lock_handoff"] is False
    pid = result["new_pid"]
    try:
        assert _running(pid)
        assert json.loads((tmp_path / "test_status.json").read_text())["pid"] == pid
    finally:
        _reap_startup_lane(pid)


def test_handoff_descriptor_cannot_steal_or_unlock_another_open_descriptions_fence(tmp_path: Path) -> None:
    lane = _startup_lane(tmp_path)
    lock_path = tmp_path / "test.resume.lock"
    with lock_path.open("a") as owner, lock_path.open("a") as unowned:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path, resume_lock_fd=unowned.fileno())
        assert result["restarted"] is False and result["reason"] == "resume_lock_invalid"
        assert not (tmp_path / "lane.pid").exists()
        with lock_path.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("kind", ["closed", "mismatched", "readonly", "directory", "replaced"])
def test_resume_handoff_rejects_bad_descriptors_in_parent_and_child(tmp_path: Path, kind: str) -> None:
    lane = _startup_lane(tmp_path)
    lock_path = tmp_path / "test.resume.lock"
    lock_path.touch()
    descriptor = -1
    if kind == "mismatched":
        descriptor = os.open(tmp_path / "different.lock", os.O_CREAT | os.O_WRONLY, 0o600)
    elif kind == "readonly":
        descriptor = os.open(lock_path, os.O_RDONLY)
    elif kind == "directory":
        descriptor = os.open(tmp_path, os.O_RDONLY)
    elif kind == "replaced":
        descriptor = os.open(lock_path, os.O_WRONLY)
        lock_path.unlink()
        lock_path.touch()
    try:
        result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path, resume_lock_fd=descriptor)
        assert result["restarted"] is result["readiness_verified"] is False
        assert result["reason"] == "resume_lock_invalid"
        assert lane_resume.serve([
            "--repo-root", str(tmp_path), "--todo-path", "board.md", "--task-header-prefix", "## TEST-",
            "--status-path", str(tmp_path / "test_status.json"), "--pid-path", str(tmp_path / "lane.pid"),
            "--log-path", str(tmp_path / "lane.log"), "--bundle-key", "test", "--resume-lock-fd", str(descriptor),
        ]) == 0
        assert not (tmp_path / "lane.pid").exists()
        assert not (tmp_path / "test_status.json").exists()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@pytest.mark.parametrize("behavior", ["exit", "timeout", "invalid_ack"])
def test_real_spawn_without_valid_readiness_cannot_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: str,
) -> None:
    lane = _startup_lane(tmp_path)
    pid_path = tmp_path / "lane.pid"
    status_path = tmp_path / "test_status.json"
    pid_path.write_text("0\n")
    status_path.write_text('{"original":"unchanged"}\n')
    real_spawn = lane_resume.subprocess.Popen
    launched = []

    def spawn(argv, **kwargs):
        ready_fd = int(argv[argv.index("--ready-fd") + 1])
        code = {"exit": "raise SystemExit(0)", "timeout": "import time; time.sleep(60)",
                "invalid_ack": f"import os; os.write({ready_fd}, b'{{\"pid\":1}}\\n')"}[behavior]
        process = real_spawn([sys.executable, "-c", code], **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(lane_resume.subprocess, "Popen", spawn)
    monkeypatch.setattr(lane_resume, "_RESUME_READY_SECONDS", 0.2)
    result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path)
    assert result["restarted"] is result["readiness_verified"] is False
    assert result["reason"] == {"exit": "resume_exited_before_ready", "timeout": "resume_readiness_timeout",
                                "invalid_ack": "resume_readiness_invalid"}[behavior]
    assert pid_path.read_text() == "0\n"
    assert status_path.read_text() == '{"original":"unchanged"}\n'
    assert launched[0].poll() is not None
    assert not _running(launched[0].pid)


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
