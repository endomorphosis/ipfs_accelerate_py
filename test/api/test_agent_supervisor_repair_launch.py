"""Unavailable launch infrastructure must not consume task-repair backoff."""
import json
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import read_json, write_json


@pytest.fixture
def launch(tmp_path, monkeypatch):
    executable = tmp_path / "coder"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    policy = {"cwd": str(tmp_path), "argv": [str(executable)], "retry_seconds": 1800}
    board = {"id": "spar", "cwd": str(tmp_path), "probe": {"argv": ["probe"]}}
    config = {"repair_worker": policy, "boards": [board], "state_dir": str(tmp_path)}
    path = tmp_path / "repairs/spar/job.json"
    write_json(path, {"board_id": "spar", "status": "queued", "attempts": 13,
                      "latest_incident": {"observation": {"health": "blocked"}}})
    which = repair.shutil.which
    monkeypatch.setattr(repair.shutil, "which", lambda name:
                        "/bin/systemd-run" if name == "systemd-run" else which(name))
    monkeypatch.setattr(repair, "command", lambda spec, **kw:
                        {"returncode": 3, "stdout": "inactive"}
                        if spec["argv"][0] == "systemctl" else
                        {"returncode": 0, "stdout": json.dumps({"health": "blocked"})})
    return config, board, path, executable


@pytest.mark.parametrize("failure", ["missing", "not_executable", "missing_checkout"])
def test_preflight_retries_without_charging_work_then_recovers(launch, monkeypatch, failure):
    config, board, path, executable = launch
    old_cwd = config["repair_worker"]["cwd"]
    if failure == "missing":
        executable.unlink()
    elif failure == "not_executable":
        executable.chmod(0o600)
    else:
        config["repair_worker"]["cwd"] += "/missing"
    calls = []
    class Process:
        def __init__(self, *args, **kwargs):
            calls.append(args)
        def wait(self, timeout):
            return 1
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    now = time.time()
    assert repair.run_job(config, board, path)["status"] == "launch_deferred"
    job = read_json(path)
    assert job["attempts"] == 13 and job["launch_failures"] == 1
    assert now + 30 <= job["next_attempt_at"] <= time.time() + 60
    assert "report_path" not in job
    assert not calls
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    config["repair_worker"]["cwd"] = old_cwd
    # A process that actually started and returned 1 keeps its normal task
    # budget, even if it emits no report. There is no log-based refund.
    assert repair.run_job(config, board, path)["status"] == "queued"
    job = read_json(path)
    assert calls and job["attempts"] == 14
    assert job["next_attempt_at"] > time.time() + 21000


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_launcher_spawn_failure_refunds_only_unstarted_attempt(launch, monkeypatch, error):
    config, board, path, _ = launch
    def fail(*args, **kwargs):
        raise error("launch infrastructure disappeared")
    monkeypatch.setattr(repair.subprocess, "Popen", fail)
    result = repair.run_job(config, board, path)
    job = read_json(path)
    assert result["status"] == "launch_deferred"
    assert job["attempts"] == 13
    assert job["last_launch_failure"]["attempt"]["attempts"] == 14
    assert job["next_attempt_at"] <= time.time() + 60


def test_spawn_failure_cannot_refund_superseding_attempt(launch, monkeypatch):
    config, board, path, _ = launch
    def supersede(*args, **kwargs):
        job = read_json(path)
        job.update(attempts=15, report_path="newer-report", next_attempt_at=9999999999)
        write_json(path, job)
        raise FileNotFoundError("launcher disappeared")
    monkeypatch.setattr(repair.subprocess, "Popen", supersede)
    assert repair.run_job(config, board, path)["status"] == "completion_superseded"
    job = read_json(path)
    assert job["attempts"] == 15 and job["next_attempt_at"] == 9999999999
    assert "launch_failures" not in job


def test_running_cgroup_is_adopted_even_when_executable_disappears(launch, monkeypatch):
    config, board, path, executable = launch
    executable.unlink()
    monkeypatch.setattr(repair, "command", lambda *args, **kwargs:
                        {"returncode": 0, "stdout": "active"})
    assert repair.run_job(config, board, path)["status"] == "existing_repair_job_running"
    assert "launch_failures" not in read_json(path)
