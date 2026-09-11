"""Unavailable launch infrastructure must not consume task-repair backoff."""
import json
import time
import os
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr(repair.os, "fstatvfs", lambda fd: SimpleNamespace(
        f_frsize=4096, f_bavail=25 * 1024 ** 2, f_favail=1_000_000))
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


@pytest.mark.parametrize("field", ["next_attempt_at", "finished_at"])
def test_spawn_failure_preserves_updated_deadline_on_same_attempt(launch, monkeypatch, field):
    config, board, path, _ = launch
    def supersede(*args, **kwargs):
        job = read_json(path)
        job[field] = 9999999999
        write_json(path, job)
        raise FileNotFoundError("launcher disappeared")
    monkeypatch.setattr(repair.subprocess, "Popen", supersede)
    assert repair.run_job(config, board, path)["status"] == "completion_superseded"
    job = read_json(path)
    assert job[field] == 9999999999 and job["attempts"] == 14
    assert "launch_failures" not in job


def test_running_cgroup_is_adopted_even_when_executable_disappears(launch, monkeypatch):
    config, board, path, executable = launch
    executable.unlink()
    config["repair_worker"]["cwd"] += "/missing"
    def status(*args, **kwargs):
        assert kwargs["cwd"] == "/"
        return {"returncode": 0, "stdout": "active"}
    monkeypatch.setattr(repair, "command", status)
    assert repair.run_job(config, board, path)["status"] == "existing_repair_job_running"
    assert "launch_failures" not in read_json(path)


@pytest.mark.parametrize("mutation", [
    {"status": "verified_healthy"}, {"status": "running", "attempts": 14},
    {"next_attempt_at": 9999999999}, {"finished_at": 9999999999},
])
def test_preflight_cannot_requeue_superseding_job(launch, monkeypatch, mutation):
    config, board, path, _ = launch
    def preflight(_):
        job = read_json(path)
        job.update(mutation)
        write_json(path, job)
        return "repair_executable_unavailable"
    monkeypatch.setattr(repair, "_launch_preflight", preflight)
    assert repair.run_job(config, board, path)["status"] == "completion_superseded"
    job = read_json(path)
    assert all(job[k] == value for k, value in mutation.items())
    assert "launch_failures" not in job


def test_unknown_cgroup_liveness_preserves_queue(launch, monkeypatch):
    config, board, path, executable = launch
    executable.unlink()
    before = read_json(path)
    monkeypatch.setattr(repair, "command", lambda *args, **kwargs:
                        {"returncode": 1, "stdout": "", "stderr": "bus unavailable"})
    assert repair.run_job(config, board, path)["status"] == "repair_job_liveness_unknown"
    assert read_json(path) == before


def test_collected_transient_unit_is_inactive_with_exit_four(launch, monkeypatch):
    config, board, path, executable = launch
    executable.unlink()
    monkeypatch.setattr(repair, "command", lambda *args, **kwargs:
                        {"returncode": 4, "stdout": "inactive"})
    assert repair.run_job(config, board, path)["status"] == "launch_deferred"


def test_systemctl_spawn_failure_preserves_queue(launch, monkeypatch):
    config, board, path, _ = launch
    before = read_json(path)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("systemctl unavailable")
    monkeypatch.setattr(repair, "command", unavailable)
    assert repair.run_job(config, board, path)["status"] == "repair_job_liveness_unknown"
    assert read_json(path) == before


@pytest.mark.parametrize("status", ["verified_healthy", "verified_published", "missing"])
def test_selected_job_completed_or_removed_before_run_is_not_resurrected(launch, status):
    config, board, path, executable = launch
    executable.unlink()
    if status == "missing":
        path.unlink()
    else:
        job = read_json(path)
        job["status"] = status
        write_json(path, job)
    before = read_json(path)
    assert repair.run_job(config, board, path)["status"] == "completion_superseded"
    assert read_json(path) == before
    if status == "missing":
        assert not path.exists()


def test_package_disappearing_during_native_probe_does_not_charge_work(launch, monkeypatch):
    config, board, path, executable = launch
    def command(spec, **kwargs):
        if spec["argv"][0] == "systemctl":
            return {"returncode": 4, "stdout": "inactive"}
        executable.unlink()
        return {"returncode": 0, "stdout": json.dumps({"health": "blocked"})}
    monkeypatch.setattr(repair, "command", command)
    assert repair.run_job(config, board, path)["status"] == "launch_deferred"
    job = read_json(path)
    assert job["attempts"] == 13 and "report_path" not in job


@pytest.mark.parametrize("resource,reason", [
    ("bytes", "repair_storage_bytes_low"),
    ("inodes", "repair_storage_inodes_low"),
    ("unknown", "repair_storage_unavailable"),
])
def test_storage_pressure_defers_without_probe_or_attempt(launch, monkeypatch, resource, reason):
    config, board, path, _ = launch
    def sample(fd):
        if resource == "unknown":
            raise PermissionError("filesystem query denied")
        return SimpleNamespace(f_frsize=4096,
            f_bavail=0 if resource == "bytes" else 25 * 1024 ** 2,
            f_favail=0 if resource == "inodes" else 1_000_000)
    monkeypatch.setattr(repair.os, "fstatvfs", sample)
    calls = []
    def command(spec, **kwargs):
        calls.append(spec["argv"][0])
        return {"returncode": 4, "stdout": "inactive"}
    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned"))
    assert repair.run_job(config, board, path)["reason"] == reason
    job = read_json(path)
    assert job["attempts"] == 13 and job["status"] == "queued"
    detail = job["last_launch_failure"]["storage"]
    assert detail["role"] == "cwd" and detail["path"] == config["repair_worker"]["cwd"]
    assert detail["minimum_disk_available_bytes"] == 8 * 1024 ** 3
    assert detail["minimum_disk_available_inodes"] == 50_000
    assert calls == ["systemctl"] and "report_path" not in job


@pytest.mark.parametrize("field", ["minimum_disk_available_bytes", "minimum_disk_available_inodes"])
@pytest.mark.parametrize("value", [-1, True, 8.0, "8589934592", None])
def test_invalid_storage_override_defers_without_attempt(launch, field, value):
    config, board, path, _ = launch
    config["repair_worker"][field] = value
    result = repair.run_job(config, board, path)
    assert result["reason"] == "repair_storage_policy_invalid"
    job = read_json(path)
    assert job["attempts"] == 13
    assert job["last_launch_failure"]["storage"]["field"] == field


def test_storage_overrides_have_exact_floor_and_explicit_zero(launch, monkeypatch):
    config, _, path, _ = launch
    policy = config["repair_worker"]
    policy.update(minimum_disk_available_bytes=8192, minimum_disk_available_inodes=2)
    monkeypatch.setattr(repair.os, "fstatvfs", lambda fd:
                        SimpleNamespace(f_frsize=4096, f_bavail=2, f_favail=2))
    assert repair._storage_preflight(policy, path, config["state_dir"]) is None
    policy["minimum_disk_available_bytes"] = 8193
    assert repair._storage_preflight(policy, path, config["state_dir"])["reason"] == "repair_storage_bytes_low"
    policy.update(minimum_disk_available_bytes=0, minimum_disk_available_inodes=0)
    monkeypatch.setattr(repair.os, "fstatvfs", lambda fd:
                        SimpleNamespace(f_frsize=4096, f_bavail=0, f_favail=0))
    assert repair._storage_preflight(policy, path, config["state_dir"]) is None
    # An explicit zero floor does not make unavailable measurements known.
    monkeypatch.setattr(repair.os, "fstatvfs", lambda fd:
                        SimpleNamespace(f_frsize=4096, f_bavail=0, f_favail=-1))
    assert repair._storage_preflight(policy, path, config["state_dir"])["reason"] == "repair_storage_unavailable"


@pytest.mark.parametrize("role", ["state", "job", "temporary"])
def test_pressure_on_distinct_allocation_filesystem_is_reported(launch, monkeypatch, tmp_path, role):
    config, _, path, _ = launch
    directories = {name: tmp_path / name for name in ("cwd", "state", "job", "temporary")}
    for directory in directories.values():
        directory.mkdir()
    config["repair_worker"].update(cwd=str(directories["cwd"]),
                                  temporary_directory=str(directories["temporary"]))
    config["state_dir"] = str(directories["state"])
    path = directories["job"] / "job.json"
    # The paths are actual opened directories. Model independent filesystems
    # with per-descriptor device/free-space readings, without privileged mounts.
    def descriptor_role(fd):
        return Path(os.readlink(f"/proc/self/fd/{fd}")).name
    devices = {name: 100 + i for i, name in enumerate(directories)}
    monkeypatch.setattr(repair.os, "fstat", lambda fd:
                        SimpleNamespace(st_dev=devices[descriptor_role(fd)]))
    monkeypatch.setattr(repair.os, "fstatvfs", lambda fd: SimpleNamespace(
        f_frsize=4096, f_bavail=0 if descriptor_role(fd) == role else 25 * 1024 ** 2,
        f_favail=1_000_000))
    result = repair._storage_preflight(config["repair_worker"], path, config["state_dir"])
    assert result["role"] == role and result["device"] == devices[role]
    assert result["observed_directory"] == str(directories[role])


def test_future_state_uses_existing_ancestor_without_creating_it(launch, tmp_path):
    config, _, path, _ = launch
    state = tmp_path / "future/deep/state"
    assert repair._storage_preflight(config["repair_worker"], path, str(state)) is None
    assert not (tmp_path / "future").exists()


@pytest.mark.parametrize("temporary", ["missing", "relative", "wrong_type"])
def test_unavailable_temporary_root_does_not_fallback(launch, tmp_path, temporary):
    config, board, path, _ = launch
    config["repair_worker"]["temporary_directory"] = {
        "missing": str(tmp_path / "absent"), "relative": "relative", "wrong_type": 12}[temporary]
    result = repair.run_job(config, board, path)
    expected = "repair_storage_unavailable" if temporary == "missing" else "repair_storage_policy_invalid"
    assert result["reason"] == expected and read_json(path)["attempts"] == 13


def test_disk_pressure_arriving_during_probe_does_not_charge_attempt(launch, monkeypatch):
    config, board, path, _ = launch
    def command(spec, **kwargs):
        if spec["argv"][0] == "systemctl":
            return {"returncode": 4, "stdout": "inactive"}
        monkeypatch.setattr(repair.os, "fstatvfs", lambda fd:
                            SimpleNamespace(f_frsize=4096, f_bavail=0, f_favail=1_000_000))
        return {"returncode": 0, "stdout": json.dumps({"health": "blocked"})}
    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned"))
    assert repair.run_job(config, board, path)["reason"] == "repair_storage_bytes_low"
    assert read_json(path)["attempts"] == 13 and "report_path" not in read_json(path)


@pytest.mark.parametrize("superseded", [False, True])
def test_final_storage_gate_refunds_only_its_unstarted_claim(launch, monkeypatch, superseded):
    config, board, path, _ = launch
    original = repair.repair_prompt
    def prompt(*args, **kwargs):
        text = original(*args, **kwargs)
        assert read_json(path)["attempts"] == 14
        if superseded:
            job = read_json(path)
            job.update(attempts=15, report_path="superseding-report")
            write_json(path, job)
        monkeypatch.setattr(repair.os, "fstatvfs", lambda fd:
                            SimpleNamespace(f_frsize=4096, f_bavail=0, f_favail=1_000_000))
        return text
    monkeypatch.setattr(repair, "repair_prompt", prompt)
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned"))
    result = repair.run_job(config, board, path)
    job = read_json(path)
    if superseded:
        assert result["status"] == "completion_superseded" and job["attempts"] == 15
        assert "launch_failures" not in job
    else:
        assert result["reason"] == "repair_storage_bytes_low" and job["attempts"] == 13
        assert job["last_launch_failure"]["attempt"]["attempts"] == 14
        assert job["status"] == "queued"


def test_running_repair_is_preserved_under_storage_pressure(launch, monkeypatch):
    config, board, path, _ = launch
    before = read_json(path)
    monkeypatch.setattr(repair, "command", lambda *args, **kwargs:
                        {"returncode": 0, "stdout": "active"})
    monkeypatch.setattr(repair.os, "fstatvfs", lambda fd: pytest.fail("storage must not revoke running work"))
    assert repair.run_job(config, board, path)["status"] == "existing_repair_job_running"
    assert read_json(path) == before


def test_launch_binds_measured_temp_path_to_systemd_child(launch, monkeypatch, tmp_path):
    config, board, path, _ = launch
    temporary = tmp_path / "worker temporary"
    temporary.mkdir()
    config["repair_worker"]["temporary_directory"] = str(temporary)
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            for key in ("TMPDIR", "TEMP", "TMP"):
                assert f"--setenv={key}={temporary}" in argv
        def wait(self, timeout):
            return 1
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    assert repair.run_job(config, board, path)["status"] == "queued"
    assert len(calls) == 1 and read_json(path)["attempts"] == 14


def test_native_storage_observation_closes_descriptors_and_creates_no_files(tmp_path):
    policy = {"cwd": str(tmp_path), "temporary_directory": str(tmp_path),
              "minimum_disk_available_bytes": 0, "minimum_disk_available_inodes": 0}
    path = tmp_path / "future/job.json"
    before = set(os.listdir("/proc/self/fd"))
    for _ in range(8):
        assert repair._storage_preflight(policy, path, str(tmp_path / "state")) is None
    assert set(os.listdir("/proc/self/fd")) == before
    assert list(tmp_path.iterdir()) == []


def test_deferred_storage_recovers_with_current_headroom(launch, monkeypatch):
    config, board, path, _ = launch
    healthy = repair.os.fstatvfs
    monkeypatch.setattr(repair.os, "fstatvfs", lambda fd:
                        SimpleNamespace(f_frsize=4096, f_bavail=0, f_favail=1_000_000))
    assert repair.run_job(config, board, path)["status"] == "launch_deferred"
    assert read_json(path)["attempts"] == 13
    monkeypatch.setattr(repair.os, "fstatvfs", healthy)
    calls = []
    class Process:
        def __init__(self, *args, **kwargs):
            calls.append(args)
        def wait(self, timeout):
            return 1
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    assert repair.run_job(config, board, path)["status"] == "queued"
    assert len(calls) == 1 and read_json(path)["attempts"] == 14


def test_temporary_binding_cannot_drift_during_probe(launch, monkeypatch, tmp_path):
    config, board, path, _ = launch
    temporary = tmp_path / "initial-temp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    def command(spec, **kwargs):
        if spec["argv"][0] == "systemctl":
            return {"returncode": 4, "stdout": "inactive"}
        monkeypatch.setenv("TMPDIR", str(tmp_path / "unavailable-temp"))
        return {"returncode": 0, "stdout": json.dumps({"health": "blocked"})}
    monkeypatch.setattr(repair, "command", command)
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            assert f"--setenv=TMPDIR={temporary}" in argv
        def wait(self, timeout):
            return 1
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    assert repair.run_job(config, board, path)["status"] == "queued"
    assert len(calls) == 1
    assert "temporary_directory" not in config["repair_worker"]
