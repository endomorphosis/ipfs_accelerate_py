"""Real owned-process and durable-write regressions for storage exhaustion."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume
from ipfs_accelerate_py.agent_supervisor.rescue import storage_pressure as storage
from ipfs_accelerate_py.agent_supervisor.rescue import supervisor_watchdog as watchdog

pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise real storage retry and owned subprocesses")


@pytest.fixture(autouse=True)
def private_runtime(tmp_path, monkeypatch):
    runtime = tmp_path / "private-runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(storage, "_runtime_directory", lambda: runtime)
    monkeypatch.setattr(lane_resume, "_COMMAND_POLL_SECONDS", 0.02)
    monkeypatch.setattr(lane_resume, "_COMMAND_STOP_GRACE_SECONDS", 0.1)
    return runtime


def _guard(root, **options):
    return storage.StorageGuard(root / "status.json", [root], check_continue=lambda: None,
                                minimum_free_bytes=1, retry_seconds=0.01, **options)


def _heartbeat(root):
    lane_resume.write_lane_status(root / "status.json", target_id="TEST", state="healthy",
                                 phase="deterministic_command", active_task_id="TEST-001")


@pytest.mark.parametrize("code,expected", [(errno.ENOSPC, True), (errno.EDQUOT, True),
                                         (errno.EACCES, False), (errno.EIO, False)])
def test_only_space_and_quota_failures_are_transient(code, expected):
    assert storage.is_storage_pressure(OSError(code, "injected")) is expected


def test_capacity_follows_output_symlink_and_performs_real_durable_probe(tmp_path):
    target = tmp_path / "actual-output"; target.mkdir()
    alias = tmp_path / "public-output"; alias.symlink_to(target, target_is_directory=True)
    result = storage.check_storage_capacity([alias], minimum_free_bytes=1)
    assert result["ok"] and result["durable_probe"]
    assert result["observations"][0]["path"] == str(target)
    assert list(target.iterdir()) == [], "capacity probes must clean their own temporary files"


def test_low_capacity_refuses_admission_without_attempting_a_write(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.os, "statvfs", lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096))
    monkeypatch.setattr(storage, "_durable_probe", lambda _: pytest.fail("low capacity must not allocate a probe"))
    result = _guard(tmp_path).admission()
    assert result["ok"] is False and result["storage_pressure"] is True
    assert result["reason"] == "insufficient_free_bytes"
    assert not (tmp_path / "attempts.json").exists()


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_available_blocks_do_not_hide_actual_durable_probe_failure(tmp_path, monkeypatch, code):
    def fail(_): raise OSError(code, "quota or disk full")
    monkeypatch.setattr(storage, "_durable_probe", fail)
    assert storage.check_storage_capacity([tmp_path], minimum_free_bytes=1)["ok"] is False


def test_capacity_does_not_swallow_permission_failure(tmp_path, monkeypatch):
    def fail(_): raise PermissionError(errno.EACCES, "not a space failure")
    monkeypatch.setattr(storage, "_durable_probe", fail)
    with pytest.raises(PermissionError): storage.check_storage_capacity([tmp_path], minimum_free_bytes=1)


def test_emergency_status_is_explicitly_degraded_and_bound_to_current_expected_role(tmp_path):
    path = tmp_path / "status.json"
    record = storage.publish_emergency_status(path, {"state": "healthy", "phase": "capture", "active_task_id": "TEST-001"},
                                              error=OSError(errno.ENOSPC, "full"))
    assert record and record["payload"]["state"] == "degraded"
    assert record["payload"]["storage_original_phase"] == "capture"
    assert record["payload"]["work_complete"] is False
    read = lambda **options: storage.read_emergency_status(path, expected_pid=os.getpid(), **options)
    assert read(identity_check=lambda pid: pid == os.getpid()) == record
    assert read(identity_check=lambda _: False) is None
    assert storage.read_emergency_status(path, expected_pid=os.getppid(), identity_check=lambda _: True) is None


@pytest.mark.parametrize("field,value", [("start_ticks", "0"), ("uid", -1), ("canonical_path", "/other/status.json"),
                                       ("command_sha256", "0" * 64), ("observed_at_ms", 1)])
def test_emergency_status_rejects_stale_or_mismatched_identity(tmp_path, field, value):
    path = tmp_path / "status.json"
    storage.publish_emergency_status(path, {"state": "healthy"})
    emergency = storage._emergency_path(path)
    record = json.loads(emergency.read_text()); record[field] = value
    emergency.write_text(json.dumps(record))
    assert storage.read_emergency_status(path, expected_pid=os.getpid(), identity_check=lambda _: True) is None


def test_emergency_channel_failure_does_not_escape_or_claim_healthy(tmp_path, monkeypatch):
    def full(): raise OSError(errno.ENOSPC, "secondary full")
    monkeypatch.setattr(storage, "_runtime_directory", full)
    assert storage.publish_emergency_status(tmp_path / "status.json", {"state": "healthy"}) is None


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_real_owned_command_keeps_progressing_across_failed_heartbeats_and_recovers(tmp_path, monkeypatch, code):
    original = lane_resume._atomic_write_once; failures = 0
    def write(path, text):
        nonlocal failures
        if path.name == "status.json" and failures < 3:
            failures += 1; raise OSError(code, "heartbeat cannot persist")
        original(path, text)
    monkeypatch.setattr(lane_resume, "_atomic_write_once", write)
    script = tmp_path / "produce.py"
    script.write_text("from pathlib import Path\nimport time\n"
                      "Path('invocations').write_text('1')\n"
                      "for i in range(15):\n Path('progress').write_text(str(i))\n time.sleep(.02)\n")
    with _guard(tmp_path).activate():
        code = lane_resume._run_argv([sys.executable, str(script)], repo_root=tmp_path, timeout=3,
                                    log_path=tmp_path / "log", heartbeat=lambda: _heartbeat(tmp_path))
    assert code == 0 and failures == 3
    assert (tmp_path / "progress").read_text() == "14"
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "healthy"
    assert not storage._emergency_path(tmp_path / "status.json").exists()


def test_a_real_task_producer_is_not_reexecuted_when_completion_commit_temporarily_fails(tmp_path, monkeypatch):
    (tmp_path / "produce.py").write_text("from pathlib import Path\n"
        "p=Path('invocations');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
        "Path('report.json').write_text('{\"decision\":\"GO\"}')\n")
    (tmp_path / "validate.py").write_text("import json\nfrom pathlib import Path\nassert json.loads(Path('report.json').read_text())['decision']=='GO'\n")
    board = tmp_path / "board.md"
    board.write_text("## TEST-001 Real producer\n- Status: pending\n- Depends on:\n"
        "- Deterministic command: python3 produce.py\n- Validation: python3 validate.py\n- Outputs: report.json\n")
    original = lane_resume._atomic_write_once; failed = []
    def write(path, text):
        if path == board and "- Status: completed" in text and len(failed) < 2:
            assert "- Status: pending" in board.read_text()
            assert (tmp_path / "invocations").read_text() == "1"
            failed.append(True); raise OSError(errno.ENOSPC, "cannot commit yet")
        original(path, text)
    monkeypatch.setattr(lane_resume, "_atomic_write_once", write)
    with _guard(tmp_path).activate():
        result = lane_resume.run_resume_cycle(repo_root=tmp_path, todo_path=board, task_header_prefix="## TEST-",
            attempt_path=tmp_path / "attempts.json", log_path=tmp_path / "log", heartbeat=lambda **_: _heartbeat(tmp_path))
    assert len(failed) == 2 and result["advanced_task_id"] == "TEST-001"
    assert (tmp_path / "invocations").read_text() == "1"
    assert "- Status: completed" in board.read_text()
    assert result["work_complete"] is False and result["completion_authority"] is False


def test_pending_durable_write_honors_real_stop_without_committing(tmp_path, monkeypatch):
    stop = False
    def check():
        if stop: raise lane_resume._LaneResumeInterrupted
    guard = storage.StorageGuard(tmp_path / "status.json", [tmp_path], check_continue=check,
                                 minimum_free_bytes=1, retry_seconds=0.01)
    def write(path, text):
        nonlocal stop
        stop = True; raise OSError(errno.ENOSPC, "full")
    with pytest.raises(lane_resume._LaneResumeInterrupted):
        guard.persist(tmp_path / "receipt.json", "real pending receipt", write)
    assert not (tmp_path / "receipt.json").exists()


def test_nonstorage_heartbeat_failure_still_cancels_and_reaps_owned_command(tmp_path, monkeypatch):
    script = tmp_path / "wait.py"
    script.write_text("import os,time\nfrom pathlib import Path\nPath('child.pid').write_text(str(os.getpid()))\ntime.sleep(30)\n")
    def failed():
        if (tmp_path / "child.pid").exists(): raise PermissionError(errno.EACCES, "permission revoked")
    with _guard(tmp_path).activate(), pytest.raises(PermissionError):
        lane_resume._run_argv([sys.executable, str(script)], repo_root=tmp_path, timeout=3,
                             log_path=tmp_path / "log", heartbeat=failed)
    assert not Path(f"/proc/{int((tmp_path / 'child.pid').read_text())}").exists()


def test_append_log_space_failure_waits_before_starting_the_real_producer(tmp_path, monkeypatch):
    original = Path.open
    failures = []
    def open_file(path, *args, **kwargs):
        if path == tmp_path / "command.log" and args and args[0] == "a" and len(failures) < 2:
            assert not (tmp_path / "producer-ran").exists()
            failures.append(True); raise OSError(errno.ENOSPC, "append log cannot allocate")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    with _guard(tmp_path).activate():
        code = lane_resume._run_argv([sys.executable, "-c", "from pathlib import Path;Path('producer-ran').write_text('once')"],
            repo_root=tmp_path, timeout=3, log_path=tmp_path / "command.log", heartbeat=lambda: _heartbeat(tmp_path))
    assert code == 0 and len(failures) == 2
    assert (tmp_path / "producer-ran").read_text() == "once"


def test_serve_storage_wait_does_not_admit_work_or_spend_attempts(tmp_path, monkeypatch):
    board = tmp_path / "board.md"
    board.write_text("## TEST-001 Pending\n- Status: pending\n")
    monkeypatch.setattr(storage, "check_storage_capacity", lambda *_, **__: {
        "ok": False, "storage_pressure": True, "reason": "ENOSPC"})
    monkeypatch.setattr(lane_resume, "run_resume_cycle", lambda **_: pytest.fail("no new task may be admitted"))
    assert lane_resume.serve(["--repo-root", str(tmp_path), "--todo-path", str(board), "--task-header-prefix", "## TEST-",
        "--status-path", str(tmp_path / "status.json"), "--pid-path", str(tmp_path / "lane.pid"),
        "--log-path", str(tmp_path / "command.log"), "--bundle-key", "TEST", "--max-seconds", "0.08"]) == 0
    assert not (tmp_path / "lane_resume_attempts.json").exists()
    assert "- Status: pending" in board.read_text()
    assert json.loads((tmp_path / "status.json").read_text())["work_complete"] is False


def test_owned_emergency_marker_can_clear_only_after_actual_recovery(tmp_path):
    path = tmp_path / "status.json"
    storage.publish_emergency_status(path, {"state": "healthy"})
    emergency = storage._emergency_path(path)
    original = json.loads(emergency.read_text())
    changed = {**original, "start_ticks": "0"}; emergency.write_text(json.dumps(changed))
    storage.clear_emergency_status(path)
    assert emergency.exists(), "do not delete a marker from another process lifetime"
    emergency.write_text(json.dumps(original)); storage.clear_emergency_status(path)
    assert not emergency.exists()


def test_watchdog_capacity_deferral_preserves_actual_dead_pid_and_consumes_no_restart(tmp_path, monkeypatch):
    state = tmp_path / "state"; state.mkdir()
    (state / "test_bundle_supervisor.pid").write_text("99999999")
    (state / "test_status.json").write_text(json.dumps({"schema": lane_resume._SCHEMA, "pid": 99999999,
        "state": "healthy", "phase": "deterministic_command", "heartbeat_at_ms": int(time.time() * 1000)}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"lanes": [{"bundle_key": "TEST", "state_dir": str(state), "state_prefix": "test"}]}))
    monkeypatch.setattr(watchdog, "check_storage_capacity", lambda *_, **__: {"ok": False, "reason": "ENOSPC"})
    controller = watchdog.SupervisorWatchdog(manifest_path=manifest, repo_root=tmp_path)
    result = controller._check_cycle()
    assert result["restarts"] == 0
    row = result["reports"][0]
    assert row["action"] == "storage_wait" and row["storage_pressure"] is True
    assert row["pid_check"]["alive"] is False, "a dead capture must remain explicitly dead"
    assert result["status"]["state"] != "healthy"


def test_independent_watchdog_role_check_rejects_unrelated_live_process(tmp_path):
    assert watchdog._lane_storage_identity(os.getpid(), tmp_path, tmp_path / "status.json") is False


@pytest.mark.parametrize("kind", ["genuine", "script-data", "duplicate-root", "relative-wrong-cwd", "wrong-executable"])
def test_watchdog_role_fence_checks_real_entrypoint_unique_options_and_process_cwd(tmp_path, monkeypatch, kind):
    module = "ipfs_accelerate_py.agent_supervisor.rescue.lane_resume"
    status = tmp_path / "status.json"
    argv = [sys.executable, "-B", "-m", module, "--repo-root", str(tmp_path), "--status-path", str(status)]
    if kind == "script-data": argv = [sys.executable, "other.py", "--evidence", *argv[2:]]
    elif kind == "duplicate-root": argv += ["--repo-root", str(tmp_path)]
    elif kind == "relative-wrong-cwd": argv[argv.index("--repo-root") + 1] = "."
    elif kind == "wrong-executable": argv[0] = "/bin/other-interpreter"
    original = Path.read_bytes
    target = Path(f"/proc/{os.getpid()}/cmdline")
    monkeypatch.setattr(Path, "read_bytes", lambda path: "\0".join(argv).encode() if path == target else original(path))
    assert watchdog._lane_storage_identity(os.getpid(), tmp_path, status) is (kind == "genuine")


def test_fenced_resume_forwards_declared_storage_roots_and_floor(tmp_path, monkeypatch):
    lane = {"bundle_key": "test", "state_dir": str(tmp_path), "state_prefix": "test", "pid_path": str(tmp_path / "lane.pid"),
            "todo_path": "board.md", "task_header_prefix": "## TEST-", "minimum_free_bytes": 134217728,
            "storage_paths": ["outputs", str(tmp_path / "physical-output")]}
    commands = []
    def spawn(argv, **_):
        commands.append(argv); return SimpleNamespace(pid=99999999)
    monkeypatch.setattr(lane_resume.subprocess, "Popen", spawn)
    monkeypatch.setattr(lane_resume, "_await_resume_ready", lambda *_, **__: "")
    result = lane_resume.resume_stopped_lane(lane, repo_root=tmp_path)
    assert result["readiness_verified"] is True
    argv = commands[0]
    assert argv[argv.index("--minimum-free-bytes") + 1] == "134217728"
    assert [argv[index + 1] for index, item in enumerate(argv) if item == "--storage-path"] == [str(tmp_path / "outputs"), str(tmp_path / "physical-output")]


def _live_watchdog(tmp_path):
    state = tmp_path / "state"; state.mkdir()
    path = state / "test_status.json"
    (state / "test_bundle_supervisor.pid").write_text(str(os.getpid()))
    path.write_text(json.dumps({"schema": lane_resume._SCHEMA, "pid": os.getpid(), "state": "healthy",
        "phase": "deterministic_command", "active_task_id": "TEST-001", "heartbeat_at_ms": 1}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"report_path": str(tmp_path / "watchdog-report.json"),
        "lanes": [{"bundle_key": "TEST", "state_dir": str(state), "state_prefix": "test"}]}))
    return watchdog.SupervisorWatchdog(manifest_path=manifest, repo_root=tmp_path), path


def test_authenticated_emergency_heartbeat_prevents_stale_restart_even_if_probe_is_writable(tmp_path, monkeypatch):
    controller, path = _live_watchdog(tmp_path)
    storage.publish_emergency_status(path, {"schema": lane_resume._SCHEMA, "pid": os.getpid(), "state": "healthy",
        "phase": "deterministic_command", "active_task_id": "TEST-001"}, error=OSError(errno.EDQUOT, "file quota"))
    # Process-role parsing is exercised separately; use the real test process's
    # uid/startticks here to isolate reader-to-controller integration.
    monkeypatch.setattr(watchdog, "_lane_storage_identity", lambda *_: True)
    monkeypatch.setattr(watchdog, "check_storage_capacity", lambda *_, **__: {"ok": True})
    result = controller._check_cycle()
    row = result["reports"][0]
    assert row["action"] == "storage_wait" and result["restarts"] == 0
    assert row["pid_check"]["alive"] is True
    assert row["heartbeat_check"]["emergency_identity_verified"] is True
    assert row["heartbeat_check"]["active_task_id"] == "TEST-001"
    assert result["status"]["state"] == "degraded"


def test_watchdog_report_enospc_publishes_degraded_without_crashing_or_restarting(tmp_path, monkeypatch):
    controller, path = _live_watchdog(tmp_path)
    payload = json.loads(path.read_text()); payload["heartbeat_at_ms"] = int(time.time() * 1000)
    path.write_text(json.dumps(payload))
    original = Path.open
    def open_file(file, *args, **kwargs):
        if file.name.startswith(".watchdog-report.json.") and args and args[0] == "w":
            raise OSError(errno.ENOSPC, "report volume full")
        return original(file, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    result = controller._check_cycle()
    assert result["restarts"] == 0 and result["storage_pressure"] is True
    assert result["status"]["state"] == "degraded"
    emergency = storage._emergency_path(tmp_path / "watchdog-report.json")
    assert json.loads(emergency.read_text())["payload"]["state"] == "degraded"
