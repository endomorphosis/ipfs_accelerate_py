"""Unresolved callback custody cannot become a coding replay or budget reset."""
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue import live_board_probe as probe
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import write_json


POLICY = {"schema": "agent-supervisor/owner-retention@1", "mode": "retain",
          "profile_id": "test/native-profile", "reason": "Retain the accepted owner"}
WAIT = {"health": "stalled", "details": {
    "selection_idle_reason": "expired_attempt_settlement_unavailable",
    "task_counts": {"retrying": 2, "todo": 23}}}


def fixture(tmp_path, monkeypatch, mode, status="queued"):
    def forbidden(*args, **kwargs):
        pytest.fail("retained repair reached a host effect or dispatch")
    monkeypatch.setattr(repair.subprocess, "run", forbidden)
    monkeypatch.setattr(repair.subprocess, "Popen", forbidden)
    monkeypatch.setattr(repair.os, "kill", forbidden)
    monkeypatch.setattr(repair.os, "killpg", forbidden)
    board = {"id": "retained", "cwd": str(tmp_path), "hold_files": []}
    if mode == "policy":
        board["recovery_policy"] = POLICY
    elif mode == "invalid":
        board["recovery_policy"] = {"mode": "restart"}
    config = {"state_dir": str(tmp_path), "boards": [board],
              "repair_worker": {"cwd": str(tmp_path)}}
    write_json(tmp_path / "retained/state.json", {
        "stall_class": "stalled_no_progress", "observation": WAIT if mode == "wait" else {}})
    incident = tmp_path / "incident.json"
    write_json(incident, {"board_id": "retained", "observation": WAIT if mode == "incident" else {}})
    path = tmp_path / "repairs/retained/job.json"
    write_json(path, {"board_id": "retained", "status": status, "attempts": 3,
                      "next_attempt_at": 999999, "last_started_at": 1,
                      "repair_workspace": "deleted/worktree", "pending_action": "old",
                      "latest_incident": json.loads(incident.read_text())})
    return config, board, path, incident


@pytest.mark.parametrize("mode", ["policy", "invalid", "wait", "incident"])
@pytest.mark.parametrize("status", ["queued", "running"])
def test_all_queue_routes_preserve_retained_history(tmp_path, monkeypatch, mode, status):
    config, board, path, incident = fixture(tmp_path, monkeypatch, mode, status)
    before = path.read_bytes()
    assert repair.enqueue(config, incident)["status"] == "recovery_wait"
    assert repair.reconcile_queued_jobs(config, 100) == []
    assert repair.next_job(config, 100) is None
    # A copied board cannot strip the configured retention policy.
    assert repair.run_job(config, {"id": board["id"]}, path)["status"] == "recovery_wait"
    assert repair.should_reclaim_repair_unit(config) is False
    assert repair.queue_status(config, 100)["held"][0]["reason"]
    assert path.read_bytes() == before


def test_incident_custody_denies_enqueue_and_direct_dispatch(tmp_path, monkeypatch):
    config, board, path, incident = fixture(tmp_path, monkeypatch, "incident")
    before = path.read_bytes()
    assert repair.enqueue(config, incident)["reason"] == "retained_attempt_settlement_wait"
    assert repair.run_job(config, board, path)["status"] == "recovery_wait"
    assert path.read_bytes() == before


def test_retained_running_job_prevents_shared_unit_reclaim(tmp_path, monkeypatch):
    config, _, path, _ = fixture(tmp_path, monkeypatch, "wait", "running")
    config["boards"].append({"id": "ordinary", "cwd": str(tmp_path), "hold_files": []})
    write_json(tmp_path / "repairs/ordinary/job.json", {
        "status": "running", "repair_workspace": "deleted", "last_started_at": 2})
    assert repair.should_reclaim_repair_unit(config) is False
    assert json.loads(path.read_text())["attempts"] == 3


def test_retention_does_not_starve_an_independent_board(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, monkeypatch, "policy")
    config["boards"].append({"id": "ordinary", "cwd": str(tmp_path), "hold_files": []})
    write_json(tmp_path / "repairs/ordinary/job.json", {
        "status": "queued", "next_attempt_at": 0, "queued_at": 1})
    selected = repair.next_job(config, 100)
    assert selected is not None and selected[0]["id"] == "ordinary"


def test_later_lane_settlement_is_not_hidden_by_first_idle_lane():
    result = probe._readiness({"selection_idle_reason": "no_ready_tasks"}, [
        {"selection_idle_reason": "no_ready_tasks"},
        {"selection_idle_reason": "expired_attempt_settlement_unavailable"}])
    assert result["selection_idle_reason"] == "expired_attempt_settlement_unavailable"
    assert result["ready_count"] is None


def test_readiness_without_retained_attempt_keeps_native_order():
    assert probe._readiness({"selection_idle_reason": "native_wait", "ready_count": 1}, [
        {"selection_idle_reason": "no_ready_tasks", "ready_count": 0}
    ]) == {"selection_idle_reason": "native_wait", "ready_count": 1,
           "eligible_ready_count": None}


def healthy():
    return {"board_id": "retained", "health": "healthy", "complete": False,
            "details": {"authenticated_task_observation": True,
                        "task_counts": {"completed": 2}, "completion_receipt_count": 2}}


@pytest.mark.parametrize("mode", ["fresh", "fresh_complete", "persisted_during_probe"])
def test_reconcile_new_retained_observation_preserves_all_queue_bytes(
        tmp_path, monkeypatch, mode):
    config, board, path, _ = fixture(tmp_path, monkeypatch, "ordinary")
    board["probe"] = {"argv": ["private-probe"]}
    state = tmp_path / "retained/state.json"
    write_json(state, {"board_id": "retained", "health": "healthy", "observed_at": 1000,
                       "observation": healthy()})
    before = path.read_bytes()
    calls = []

    def probe_only(spec, **kwargs):
        calls.append(spec)
        assert spec == board["probe"]
        sample = {**WAIT, "board_id": "retained"}
        if mode == "fresh_complete":
            sample.update(health="complete", complete=True, completion_candidate=True)
        if mode == "persisted_during_probe":
            write_json(state, {"stall_class": "retained_attempt_settlement_wait", "observation": {}})
            sample = healthy()
        return {"returncode": 0, "stdout": json.dumps(sample), "stderr": "", "timed_out": False}

    monkeypatch.setattr(repair, "verify_job_recovery",
                        lambda *a, **k: pytest.fail("fresh denial cannot reach verification"))
    assert repair.reconcile_queued_jobs(config, 1000, runner=probe_only) == []
    assert calls == [board["probe"]]
    assert path.read_bytes() == before
    assert not (path.parent / "queue.lock").exists()


@pytest.mark.parametrize("mode", ["fresh", "fresh_complete", "persisted_before_claim"])
def test_run_job_new_retained_observation_never_charges_or_dispatches(
        tmp_path, monkeypatch, mode):
    config, board, path, _ = fixture(tmp_path, monkeypatch, "ordinary")
    board["probe"] = {"argv": ["private-probe"]}
    state = tmp_path / "retained/state.json"
    before = path.read_bytes()
    calls = []
    samples = []

    def command(spec, **kwargs):
        calls.append(spec["argv"])
        if spec["argv"][0] == "systemctl":
            assert spec["argv"][1:3] == ["--user", "is-active"]
            return {"returncode": 3, "stdout": "inactive", "stderr": ""}
        assert spec == board["probe"]
        sample = {**WAIT, "board_id": "retained"}
        if mode == "fresh_complete":
            sample.update(health="complete", complete=True, completion_candidate=True)
        if mode == "persisted_before_claim":
            sample = healthy()
        samples.append(sample)
        return {"returncode": 0, "stdout": json.dumps(sample), "stderr": "", "timed_out": False}

    storage_calls = []

    def storage(*args, **kwargs):
        storage_calls.append(True)
        if mode == "persisted_before_claim" and len(storage_calls) == 2:
            write_json(state, {"stall_class": "retained_attempt_settlement_wait", "observation": {}})
        return None

    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(repair, "_launch_preflight", lambda *a, **k: None)
    monkeypatch.setattr(repair, "_storage_preflight", storage)
    monkeypatch.setattr(repair, "_repair_temporary_directory", lambda _: tmp_path)
    monkeypatch.setattr(repair, "verify_job_recovery",
                        lambda *a, **k: pytest.fail("fresh denial cannot verify, publish or dispatch"))
    result = repair.run_job(config, board, path)
    assert result["status"] == "recovery_wait"
    assert result["reason"] == "retained_attempt_settlement_wait"
    assert len(samples) == 1
    assert path.read_bytes() == before
    assert not list(path.parent.glob("prompt-*.txt"))
    assert not list(path.parent.glob("worker-*.log"))


@pytest.mark.parametrize("mode", ["current", "historical_incident", "policy", "invalid"])
def test_direct_verification_returns_unverified_wait_before_completion_effects(
        tmp_path, monkeypatch, mode):
    config, board, _, _ = fixture(tmp_path, monkeypatch, "ordinary")
    board["publication"] = {"manifest": "must-not-publish"}
    sample = {"health": "complete", "complete": True, "completion_candidate": True}
    incident = {}
    if mode == "current":
        sample.update(WAIT, health="complete")
    elif mode == "historical_incident":
        incident["observation"] = WAIT
    else:
        board["recovery_policy"] = POLICY if mode == "policy" else None
    monkeypatch.setattr(repair, "_mirror_job_recovery",
                        lambda *a, **k: pytest.fail("a denial cannot activate derived metadata"))
    result = repair.verify_job_recovery(board, incident, sample, tmp_path / "publication")
    assert result["verified"] is False and result["wait"] is True
    assert result["completion_authority"] is False
    assert not (tmp_path / "publication").exists()


def test_persisted_wait_and_cached_unknown_do_not_clear_from_marker_absence(tmp_path, monkeypatch):
    config, board, path, _ = fixture(tmp_path, monkeypatch, "ordinary")
    before = path.read_bytes()
    state = tmp_path / "retained/state.json"
    write_json(state, {"stall_class": "retained_attempt_settlement_wait", "observation": {}})
    assert repair._recovery_denial(config, board) == "retained_attempt_settlement_wait"
    assert repair.next_job(config, 100) is None
    assert path.read_bytes() == before
    # A historical UNKNOWN incident requires native evidence supersession;
    # a fresh healthy heartbeat or completion claim is not such evidence.
    write_json(state, {"observation": healthy()})
    job = json.loads(path.read_text())
    job["latest_incident"]["observation"] = WAIT
    write_json(path, job)
    before = path.read_bytes()
    assert repair._recovery_denial(config, board, healthy()) == "retained_attempt_settlement_wait"
    assert repair.reconcile_queued_jobs(config, 1000) == []
    assert path.read_bytes() == before


def test_ordinary_verification_without_retained_history_is_preserved(tmp_path, monkeypatch):
    _, board, _, _ = fixture(tmp_path, monkeypatch, "ordinary")
    monkeypatch.setattr(repair, "_mirror_job_recovery", lambda value: value)
    result = repair.verify_job_recovery(board, {"observation": healthy()}, healthy(),
                                       tmp_path / "publication")
    assert result == {"verified": True, "reason": "runtime_health_verified"}
