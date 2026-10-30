"""Frozen candidate continuations cannot become another coding attempt."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate as candidate
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import read_json, write_json


def _case(tmp_path, *, status="validation_pending"):
    repository, state = tmp_path / "repository", tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    config_path = tmp_path / "fleet.json"
    config_path.write_text("{}\n")
    board = {"id": "spar", "cwd": str(repository), "hold_files": [],
             "max_repair_attempts": 3, "probe": {"argv": ["actual-probe"]}}
    cfg = {"state_dir": str(state), "boards": [board], "_config_path": str(config_path),
        "repair_worker": {"cwd": str(repository), "timeout_seconds": 2400,
            "temporary_directory": str(tmp_path),
            "job_unit": "ipfs-test-candidate-repair-job.service",
            "candidate_policy": {"schema": candidate.POLICY_SCHEMA,
                "repository_root": str(repository), "state_root": str(state),
                "namespace": "ptr_lane_9009", "base_commit": "a" * 40,
                "source_paths": ["source.py"], "allowed_paths": ["source.py"],
                "max_source_bytes": 1024, "max_candidate_bytes": 2048,
                "max_files": 8, "git_timeout_seconds": 10},
            "candidate_runtime": {"fixture": "the actual owner preflight is mocked only where named"}}}
    token = "b" * 32
    path = state / "repairs/spar/job.json"
    metadata = state / "repair-candidates/spar" / token
    metadata.mkdir(parents=True)
    job = {"board_id": "spar", "status": status, "attempts": 3,
        "max_repair_attempts": 3, "repair_unit_token": token,
        "repair_job_unit": cfg["repair_worker"]["job_unit"],
        "next_attempt_at": 999999, "queued_at": 1, "last_started_at": 900,
        "finished_at": 950, "returncode": 2,
        "escalation_reason": "repair_budget_exhausted", "prior_attempts": 1,
        "latest_incident": {"board_id": "spar", "signature": "the original incident"},
        "candidate_runtime": {"attempt_token": token, "attempt_number": 3,
            "candidate_metadata": str(metadata), "receipt": {"retained": True}},
        "candidate_next_resume_at": 1000}
    # Retained-file fixtures test only the cheap dispatch fence. They claim no
    # checkout/validation/adoption authority, and never execute an owner.
    scope = {"board_id": "spar", "attempt_id": token, "attempt_number": 3,
             "metadata_root": str(metadata), "base_commit": "a" * 40}
    ready = {"schema": candidate.FREEZE_SCHEMA, "stage": "candidate_ready", "binding": scope,
             "manifest": {}, "changed_paths": ["source.py"], "validation_passed": False,
             "runtime_adopted": False, "completion_authority": False,
             "coding_attempts_incremented": False}
    ready["freeze_sha256"] = candidate.digest(ready)
    terminal = {"returncode": 2, "binding_sha256": candidate.digest(scope),
                "known_cgroup_empty_except_original_owner": True}
    for name, value in [("candidate-ready.json", ready), ("provider-terminal.json", terminal)]:
        write_json(metadata / name, value)
        os.chmod(metadata / name, 0o400)
    write_json(path, job)
    return cfg, board, path, job


def _no_process(*args, **kwargs):
    pytest.fail("an observation/refusal entered a process or provider path")


@pytest.mark.parametrize("kind, expected", [("ordinary", False), ("fleet", True),
    ("infrastructure", True), ("explicit", True), ("malformed_explicit", True)])
def test_isolation_requirement_cannot_silently_fall_back_to_direct_coding(tmp_path, kind, expected):
    cfg, board, _, _ = _case(tmp_path)
    policy = dict(cfg["repair_worker"])
    policy.pop("candidate_policy")
    if kind == "fleet": board["service_fleet_binding"] = {}
    elif kind == "infrastructure":
        board["recovery_policy"] = {"profile": "infrastructure-only/v1",
            "repairable_stalls": ["probe_failed"], "max_actions_per_window": 3,
            "window_seconds": 86400, "circuit_breaker_seconds": 3600}
    elif kind == "explicit": policy["candidate_policy"] = {}
    elif kind == "malformed_explicit": policy["candidate_policy"] = None
    assert repair.isolated_candidate_required(board, policy) is expected


def test_required_candidate_policy_absent_is_preflight_refusal(tmp_path):
    cfg, board, path, _ = _case(tmp_path)
    board["service_fleet_binding"] = {}
    cfg["repair_worker"].pop("candidate_policy")
    before = path.read_bytes()
    assert repair.candidate_dispatch_preflight(cfg, board) == "repair_candidate_policy_required"
    assert path.read_bytes() == before


@pytest.mark.parametrize("config_path", [None, "relative.json", "/nonexistent-test-candidate-config.json"])
def test_candidate_requires_real_absolute_original_config_before_charge(tmp_path, monkeypatch, config_path):
    cfg, board, path, _ = _case(tmp_path)
    cfg["_config_path"] = config_path
    before = path.read_bytes()
    monkeypatch.setattr(runtime, "candidate_runtime_preflight", _no_process)
    assert repair.candidate_dispatch_preflight(cfg, board) == "repair_candidate_config_path_required"
    assert path.read_bytes() == before


def test_candidate_runtime_owner_preflight_result_is_preserved(tmp_path, monkeypatch):
    cfg, board, path, _ = _case(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(runtime, "candidate_runtime_preflight", lambda config: "owner_fixture_blocked")
    assert repair.candidate_dispatch_preflight(cfg, board) == "owner_fixture_blocked"
    assert path.read_bytes() == before


def test_required_preflight_refusal_does_not_charge_or_enter_provider(tmp_path, monkeypatch):
    cfg, board, path, job = _case(tmp_path, status="queued")
    job.update(attempts=2, next_attempt_at=0)
    job.pop("candidate_runtime")
    write_json(path, job)
    monkeypatch.setattr(repair, "candidate_dispatch_preflight", lambda *a: "owner_fixture_blocked")
    monkeypatch.setattr(repair, "_launch_preflight", lambda *a: None)
    monkeypatch.setattr(repair, "_storage_preflight", lambda *a: None)
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    monkeypatch.setattr(repair, "llm_router_repair_argv", _no_process)
    monkeypatch.setattr(repair, "command", lambda spec, **kw: {"returncode": 3, "stdout": "inactive\n"})
    result = repair.run_job(cfg, board, path)
    current = read_json(path)
    assert result["status"] != "running" and current["attempts"] == 2
    assert current.get("repair_unit_token") == job["repair_unit_token"]
    assert "owner_fixture_blocked" in json.dumps(result)


@pytest.mark.parametrize("status", ["candidate_ready", "validation_pending", "adoption_pending",
    "candidate_pending", "adopted_pending_probe"])
def test_pending_owner_stages_survive_spent_budget_and_coding_backoff(tmp_path, monkeypatch, status):
    cfg, board, path, _ = _case(tmp_path, status=status)
    before = path.read_bytes()
    monkeypatch.setattr(repair, "_check_repair_budget", _no_process)
    assert repair.next_pending_candidate(cfg, 1001) == (board, path)
    assert repair.next_job(cfg, 1001) is None
    assert path.read_bytes() == before
    state = repair.queue_status(cfg, 1001)
    assert state["status"] == "candidate_pending"
    assert state["candidates"][0]["status"] == status
    assert state["candidates"][0]["attempts"] == 3


@pytest.mark.parametrize("damage", ["different_token", "different_attempt", "bool_attempt",
    "job_board", "empty_metadata", "missing_binding", "invalid_token", "resume_backoff", "nonfinite_due"])
def test_pending_selection_rejects_replaced_or_unrecoverable_original_attempt(tmp_path, damage):
    cfg, _, path, job = _case(tmp_path)
    if damage == "different_token": job["candidate_runtime"]["attempt_token"] = "c" * 32
    elif damage == "different_attempt": job["candidate_runtime"]["attempt_number"] = 2
    elif damage == "bool_attempt": job["candidate_runtime"]["attempt_number"] = True
    elif damage == "job_board": job["board_id"] = "another-board"
    elif damage == "empty_metadata": job["candidate_runtime"]["candidate_metadata"] = ""
    elif damage == "missing_binding": job.pop("candidate_runtime")
    elif damage == "invalid_token": job["repair_unit_token"] = "not-a-token"
    elif damage == "resume_backoff": job["candidate_next_resume_at"] = 1301
    else: job["candidate_next_resume_at"] = float("inf")
    write_json(path, job)
    before = path.read_bytes()
    assert repair.next_pending_candidate(cfg, 1001) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("denial", ["hold", "observation", "retention"])
def test_pending_selection_keeps_holds_and_retained_attempt_denials(tmp_path, monkeypatch, denial):
    cfg, board, path, _ = _case(tmp_path)
    if denial == "hold":
        held = tmp_path / "HOLD"
        held.write_text("operator hold\n")
        board["hold_files"] = [str(held)]
    elif denial == "observation": board["observation_only"] = True
    else: monkeypatch.setattr(repair, "_recovery_denial", lambda *a: "production_attempt_settlement_required")
    before = path.read_bytes()
    monkeypatch.setattr(repair, "command", _no_process)
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    assert repair.next_pending_candidate(cfg, 1001) is None
    assert repair.resume_pending_candidate(cfg, board, path)["status"] == "candidate_recovery_wait"
    assert path.read_bytes() == before


def test_pending_owners_are_not_retired_by_cached_complete_wait_or_new_incident(tmp_path, monkeypatch):
    cfg, board, path, job = _case(tmp_path)
    write_json(Path(cfg["state_dir"]) / "spar/state.json", {"health": "complete",
        "stall_class": "external_dependency_wait", "observation": {"complete": True}})
    monkeypatch.setattr(repair, "_check_repair_budget", _no_process)
    incident = tmp_path / "new-incident.json"
    write_json(incident, {"board_id": "spar", "signature": "another incident"})
    assert repair.next_pending_candidate(cfg, 1001) == (board, path)
    result = repair.enqueue(cfg, incident)
    assert result["status"] == job["status"]
    assert read_json(path)["attempts"] == 3 and read_json(path)["repair_unit_token"] == job["repair_unit_token"]


@pytest.mark.parametrize("snapshot", ["ActiveState=active\nMainPID=321\n",
    "ActiveState=inactive\nMainPID=321\n", "ActiveState=inactive\n",
    "ActiveState=unknown\nMainPID=0\n"])
def test_resume_requires_positive_closed_unit_custody(tmp_path, monkeypatch, snapshot):
    cfg, board, path, _ = _case(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(repair.time, "time", lambda: 1001)
    monkeypatch.setattr(repair, "command", lambda *a, **kw: {"returncode": 0, "stdout": snapshot})
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    monkeypatch.setattr(repair, "llm_router_repair_argv", _no_process)
    assert repair.resume_pending_candidate(cfg, board, path)["status"] == "candidate_owner_liveness_unknown"
    assert path.read_bytes() == before


def test_resume_only_dispatches_original_owner_stages_inside_bounded_same_unit(tmp_path, monkeypatch):
    cfg, board, path, job = _case(tmp_path)
    calls, waits = [], []
    monkeypatch.setattr(repair.time, "time", lambda: 1001)
    monkeypatch.setattr(repair, "candidate_dispatch_preflight", lambda *a: None)
    monkeypatch.setattr(repair, "_storage_preflight", lambda *a: None)
    monkeypatch.setattr(repair, "_check_repair_budget", _no_process)
    monkeypatch.setattr(repair, "llm_router_repair_argv", _no_process)
    monkeypatch.setattr(repair, "command", lambda spec, **kw: calls.append(spec["argv"]) or {
        "returncode": 0, "stdout": "ActiveState=inactive\nMainPID=0\n"})

    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            assert kwargs["stdin"] == subprocess.DEVNULL
        def wait(self, timeout):
            waits.append(timeout)
            return 0

    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    # No host watchdog environment is consulted by this deterministic fake.
    monkeypatch.setattr(repair, "_SERVICE_HEARTBEAT", repair.SystemdHeartbeat.from_environment({}))
    result = repair.resume_pending_candidate(cfg, board, path)
    argv = next(argv for argv in calls if argv[0] == "systemd-run")
    assert "--property=CPUQuota=100%" in argv and "--property=TasksMax=64" in argv
    assert "--property=RuntimeMaxSec=2400" in argv and "--property=KillMode=control-group" in argv
    assert "--unit=" + cfg["repair_worker"]["job_unit"] in argv
    module_at = argv.index("-m")
    assert argv[module_at + 1:module_at + 3] == [
        "ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_runtime", "resume"]
    assert argv[argv.index("--attempt-token") + 1] == job["repair_unit_token"]
    assert "--prompt" not in argv and waits == [2490]
    current = read_json(path)
    assert result["status"] == "validation_pending" and result["coding_attempts_incremented"] is False
    for key in ["attempts", "max_repair_attempts", "next_attempt_at", "prior_attempts", "escalation_reason",
                "repair_unit_token", "last_started_at"]:
        assert current[key] == job[key]
    assert current["candidate_next_resume_at"] == 1301
    assert repair.next_pending_candidate(cfg, 1002) is None


def test_missing_bootstrap_closeout_is_retained_unknown_and_never_recoded(tmp_path, monkeypatch):
    cfg, board, path, job = _case(tmp_path, status="running")
    job.pop("candidate_runtime")
    write_json(path, job)
    monkeypatch.setattr(repair.time, "time", lambda: 1001)
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    result = repair._candidate_finished(cfg, board, path, job["repair_unit_token"], 2)
    current = read_json(path)
    assert result["status"] == "candidate_pending" and current["attempts"] == 3
    assert current["candidate_runtime"]["candidate_metadata"] == ""
    assert repair.next_pending_candidate(cfg, 2000) is None
    assert repair.next_job(cfg, 2000) is None
    assert repair.queue_status(cfg, 2000)["status"] == "candidate_pending"


@pytest.mark.parametrize("damage", ["missing_ready", "invalid_ready_json", "freeze_digest", "freeze_schema",
    "scope_board", "scope_token", "scope_attempt", "missing_terminal", "terminal_binding",
    "terminal_unknown", "terminal_code_none", "terminal_code_bool", "metadata_relative", "metadata_mapping",
    "metadata_other_root", "attempt_zero", "attempt_negative"])
def test_pending_selection_refuses_malformed_retained_proofs_without_processes(tmp_path, monkeypatch, damage):
    cfg, _, path, job = _case(tmp_path)
    metadata = Path(job["candidate_runtime"]["candidate_metadata"])
    ready_path, terminal_path = metadata / "candidate-ready.json", metadata / "provider-terminal.json"
    ready, terminal = read_json(ready_path), read_json(terminal_path)
    if damage == "missing_ready": ready_path.unlink()
    elif damage == "invalid_ready_json":
        os.chmod(ready_path, 0o600)
        ready_path.write_text("not JSON\n")
    elif damage == "freeze_digest": ready["freeze_sha256"] = "0" * 64
    elif damage == "freeze_schema": ready["schema"] = "untrusted-freeze"
    elif damage == "scope_board": ready["binding"]["board_id"] = "another-board"
    elif damage == "scope_token": ready["binding"]["attempt_id"] = "c" * 32
    elif damage == "scope_attempt": ready["binding"]["attempt_number"] = 2
    elif damage == "missing_terminal": terminal_path.unlink()
    elif damage == "terminal_binding": terminal["binding_sha256"] = "0" * 64
    elif damage == "terminal_unknown": terminal["known_cgroup_empty_except_original_owner"] = False
    elif damage == "terminal_code_none": terminal["returncode"] = None
    elif damage == "terminal_code_bool": terminal["returncode"] = True
    elif damage == "metadata_relative": job["candidate_runtime"]["candidate_metadata"] = "repair-candidates/spar"
    elif damage == "metadata_mapping": job["candidate_runtime"]["candidate_metadata"] = {"root": str(metadata)}
    elif damage == "metadata_other_root": job["candidate_runtime"]["candidate_metadata"] = str(metadata) + "-another"
    else:
        job["attempts"] = job["candidate_runtime"]["attempt_number"] = 0 if damage == "attempt_zero" else -1
    if damage not in {"missing_ready", "invalid_ready_json"}:
        write_json(ready_path, ready)
        os.chmod(ready_path, 0o400)
    if damage != "missing_terminal":
        write_json(terminal_path, terminal)
        os.chmod(terminal_path, 0o400)
    write_json(path, job)
    before = path.read_bytes()
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    monkeypatch.setattr(repair, "command", _no_process)
    assert repair.next_pending_candidate(cfg, 1001) is None
    assert path.read_bytes() == before


def _bootstrap_fixture(cfg, path, job, *, stage="adopted_pending_probe"):
    invocation = Path(cfg["state_dir"]) / "repair-runtime/spar" / job["repair_unit_token"] / ("c" * 32)
    invocation.mkdir(parents=True)
    receipt = {"schema": runtime.SCHEMA, "stage": stage, "board_id": "spar",
        "attempt_token": job["repair_unit_token"], "attempts": job["attempts"],
        "original_checkout_lease_closed": True, "completion_authority": False,
        "coding_attempts_incremented": False}
    bootstrap = invocation / "bootstrap-result.json"
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False) + "\n").encode()
    bootstrap.write_bytes(raw)
    os.chmod(bootstrap, 0o400)
    job["candidate_runtime"]["bootstrap_result"] = {"path": str(bootstrap), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest()}
    write_json(path, job)
    return bootstrap, receipt


@pytest.mark.parametrize("damage", ["missing", "descriptor_hash", "wrong_token", "unclosed_lease",
    "wrong_attempt", "wrong_stage", "authority_claim", "writable", "wrong_invocation"])
def test_adopted_candidate_without_original_bootstrap_closeout_never_probes(tmp_path, monkeypatch, damage):
    cfg, board, path, job = _case(tmp_path, status="adopted_pending_probe")
    if damage != "missing":
        bootstrap, receipt = _bootstrap_fixture(cfg, path, job)
        if damage == "descriptor_hash": job["candidate_runtime"]["bootstrap_result"]["sha256"] = "0" * 64
        elif damage == "wrong_token": receipt["attempt_token"] = "d" * 32
        elif damage == "unclosed_lease": receipt["original_checkout_lease_closed"] = False
        elif damage == "wrong_attempt": receipt["attempts"] = 2
        elif damage == "wrong_stage": receipt["stage"] = "validation_pending"
        elif damage == "authority_claim": receipt["completion_authority"] = True
        elif damage == "writable": os.chmod(bootstrap, 0o600)
        else: job["candidate_runtime"]["bootstrap_result"]["path"] = str(bootstrap.parent.parent / bootstrap.name)
        if damage in {"wrong_token", "unclosed_lease", "wrong_attempt", "wrong_stage", "authority_claim"}:
            raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False) + "\n").encode()
            os.chmod(bootstrap, 0o600)
            bootstrap.write_bytes(raw)
            os.chmod(bootstrap, 0o400)
            job["candidate_runtime"]["bootstrap_result"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        write_json(path, job)
    before = path.read_bytes()
    calls = []
    monkeypatch.setattr(repair, "load_config", lambda config_path: cfg)
    monkeypatch.setattr(repair, "verify_job_recovery", _no_process)
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)

    def command(spec, **kwargs):
        calls.append(spec["argv"])
        assert spec["argv"][:3] == ["systemctl", "--user", "show"]
        return {"returncode": 0, "stdout": "ActiveState=inactive\nMainPID=0\n"}

    monkeypatch.setattr(repair, "command", command)
    result = repair._verify_adopted_candidate(cfg, board, path, job["repair_unit_token"])
    assert result["status"] == "adopted_pending_probe"
    assert result["reason"] == "original_adoption_closeout_not_established"
    assert len(calls) == 1 and path.read_bytes() == before


def test_actual_scoped_bootstrap_descriptor_is_recognized_without_probe_authority(tmp_path):
    cfg, board, path, job = _case(tmp_path, status="adopted_pending_probe")
    _bootstrap_fixture(cfg, path, job)
    assert repair._candidate_bootstrap_closed(cfg, board, read_json(path)) is True
    assert read_json(path)["status"] == "adopted_pending_probe"  # Read-only; no completion is granted.


def test_healthy_diagnostic_readback_precedes_missing_candidate_owner_refusal(tmp_path, monkeypatch):
    cfg, board, path, job = _case(tmp_path, status="queued")
    cfg["repair_worker"].pop("candidate_policy")
    cfg["repair_worker"].pop("candidate_runtime")
    units = {f"lane-{i}": f"legal-ir-lane-{i}.service" for i in range(4)}
    board["service_fleet_binding"] = {"board_namespace": "legal-ir-autoencoder-v1",
        "lanes": units, "max_age_seconds": 180}
    lanes = [{"lane_id": lane, "service": unit, "healthy": True,
        "checked_at_epoch": 1000, "native_status_at_epoch": 1000,
        "native_daemon_pid_alive": True, "active": False,
        "ready_count": 0, "selectable_ready_count": 0, "global_ready_count": 0,
        "blocked_count": 0, "waiting_count": 0, "task_count": 1, "completed_count": 1}
        for lane, unit in units.items()]
    sample = {"board_id": "spar", "health": "healthy", "busy": False,
        "complete": False, "completion_candidate": False, "reason_codes": [],
        "details": {"completion_authority": False, "authenticated_task_observation": False,
            "service_fleet": {"schema": "service-fleet-diagnostic/v1", "checked_at_epoch": 1000,
                "completion_authority": False, "board_namespace": "legal-ir-autoencoder-v1",
                "work_availability": "no_runnable_work", "lanes": lanes}}}
    job.update(attempts=2, next_attempt_at=0)
    job.pop("candidate_runtime")
    write_json(path, job)
    monkeypatch.setattr(repair.time, "time", lambda: 1001)
    monkeypatch.setattr(repair, "_launch_preflight", lambda *a: None)
    monkeypatch.setattr(repair, "_storage_preflight", lambda *a: None)
    monkeypatch.setattr(repair, "_mirror_job_recovery", lambda payload: payload)
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    monkeypatch.setattr(repair, "llm_router_repair_argv", _no_process)
    # The real candidate preflight refuses this configuration. It must not run
    # until a coding slot is actually needed; this fresh observer proves health.
    assert repair.candidate_dispatch_preflight(cfg, board) == "repair_candidate_policy_required"
    calls = []

    def command(spec, **kwargs):
        calls.append(spec["argv"])
        if spec["argv"][0] == "systemctl":
            return {"returncode": 3, "stdout": "inactive\n"}
        assert spec["argv"] == ["actual-probe"]
        return {"returncode": 0, "stdout": json.dumps(sample)}

    monkeypatch.setattr(repair, "command", command)
    result = repair.run_job(cfg, board, path)
    current = read_json(path)
    assert result["status"] == "verified_healthy" and current["attempts"] == 2
    assert current["verification"]["scope"] == "service_fleet_diagnostic_runtime"
    assert all(current["verification"][key] is False for key in
        ["native_admission", "task_completion_authority", "goal_completion_authority",
         "completion_authority", "complete"])
    assert calls[-1] == ["actual-probe"]


@pytest.mark.parametrize("removed", ["candidate_policy", "candidate_runtime"])
def test_resume_without_current_original_owners_does_not_launch(tmp_path, monkeypatch, removed):
    cfg, board, path, job = _case(tmp_path)
    cfg["repair_worker"].pop(removed)
    before = path.read_bytes()
    monkeypatch.setattr(repair.time, "time", lambda: 1001)
    monkeypatch.setattr(repair, "command", lambda *a, **kw: {
        "returncode": 0, "stdout": "ActiveState=inactive\nMainPID=0\n"})
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    result = repair.resume_pending_candidate(cfg, board, path)
    assert result["status"] == "candidate_recovery_wait"
    assert result["reason"] == "original_candidate_owners_removed"
    assert path.read_bytes() == before


def _intercept_candidate_dependency_import(monkeypatch, blocked, calls):
    import builtins
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name in {"repair_candidate", "repair_candidate_runtime"}:
            calls.append(name)
            if blocked is None:
                pytest.fail("missing candidate policy imported optional candidate dependencies")
            if name == blocked:
                raise ModuleNotFoundError("selected candidate dependency is unavailable", name=name)
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_missing_policy_defers_without_importing_optional_candidate_dependencies(tmp_path, monkeypatch):
    cfg, board, path, _ = _case(tmp_path)
    board["service_fleet_binding"] = {}
    cfg["repair_worker"].pop("candidate_policy")
    before, calls = path.read_bytes(), []
    _intercept_candidate_dependency_import(monkeypatch, None, calls)
    assert repair.candidate_dispatch_preflight(cfg, board) == "repair_candidate_policy_required"
    assert calls == [] and path.read_bytes() == before


@pytest.mark.parametrize("blocked", ["repair_candidate", "repair_candidate_runtime"])
def test_missing_candidate_dependency_defers_without_coding_or_job_mutation(tmp_path, monkeypatch, blocked):
    cfg, board, path, _ = _case(tmp_path)
    before, calls = path.read_bytes(), []
    _intercept_candidate_dependency_import(monkeypatch, blocked, calls)
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    assert repair.candidate_dispatch_preflight(cfg, board) == "repair_candidate_runtime_unavailable"
    assert blocked in calls and path.read_bytes() == before


@pytest.mark.parametrize("helper,blocked", [
    ("_candidate_resume_ready", "repair_candidate_runtime"),
    ("_candidate_resume_ready", "repair_candidate"),
    ("_candidate_bootstrap_closed", "repair_candidate_runtime"),
])
def test_missing_candidate_dependency_never_resumes_or_accepts_bootstrap(tmp_path, monkeypatch, helper, blocked):
    cfg, board, path, job = _case(tmp_path)
    before, calls = path.read_bytes(), []
    _intercept_candidate_dependency_import(monkeypatch, blocked, calls)
    monkeypatch.setattr(repair.subprocess, "Popen", _no_process)
    assert getattr(repair, helper)(cfg, board, job) is False
    assert blocked in calls and path.read_bytes() == before
