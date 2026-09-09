"""Durable repair scheduling must not duplicate jobs or starve another board."""
import time
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.fleet_repair import enqueue, next_job, repair_prompt
from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import read_json, write_json


def config(tmp_path):
    return {"state_dir": str(tmp_path), "boards": [
        {"id": "spar", "cwd": str(tmp_path), "hold_files": []},
        {"id": "sawm", "cwd": str(tmp_path), "hold_files": []}],
        "repair_worker": {"cwd": str(tmp_path)}}


def test_duplicate_incident_preserves_running_job_and_cooldown(tmp_path):
    cfg = config(tmp_path)
    incident = tmp_path / "incident.json"
    write_json(incident, {"board_id": "spar", "signature": "one"})
    enqueue(cfg, incident)
    path = tmp_path / "repairs/spar/job.json"
    job = read_json(path)
    job.update(status="running", attempts=2, next_attempt_at=9000)
    write_json(path, job)
    write_json(incident, {"board_id": "spar", "signature": "changed"})
    result = enqueue(cfg, incident)
    assert result["status"] == "running"
    assert read_json(path)["next_attempt_at"] == 9000
    assert read_json(path)["attempts"] == 2
    assert read_json(path)["latest_incident"]["signature"] == "changed"


def test_repair_fairness_and_backoff(tmp_path):
    cfg = config(tmp_path)
    for identifier in ("spar", "sawm"):
        write_json(tmp_path / f"repairs/{identifier}/job.json", {
            "board_id": identifier, "status": "queued", "queued_at": 1,
            "last_started_at": 10 if identifier == "spar" else 0})
    assert next_job(cfg, 100)[0]["id"] == "sawm"
    write_json(tmp_path / "repairs/sawm/job.json", {"status": "queued", "next_attempt_at": 200})
    assert next_job(cfg, 100)[0]["id"] == "spar"


def test_hold_prevents_coding_repair(tmp_path):
    cfg = config(tmp_path)
    hold = tmp_path / "OPERATOR_STOP"
    hold.touch()
    cfg["boards"][0]["hold_files"] = [str(hold)]
    write_json(tmp_path / "repairs/spar/job.json", {"status": "queued"})
    assert next_job(cfg, time.time()) is None


def test_hold_arriving_during_job_prevents_publication(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_repair import verify_job_recovery
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion
    hold = tmp_path / "HOLD"
    hold.touch()
    calls = []
    monkeypatch.setattr(fleet_completion, "publish_completed_board", lambda *args: calls.append(args))
    result = verify_job_recovery({"hold_files": [str(hold)], "publication": {"board_id": "spar"}},
                                 {}, {"completion_candidate": True}, tmp_path)
    assert result == {"verified": False, "reason": "operator_hold"}
    assert not calls


def test_unknown_board_cannot_create_job(tmp_path):
    import pytest
    incident = tmp_path / "incident.json"
    write_json(incident, {"board_id": "../../unrelated"})
    with pytest.raises(ValueError, match="not authorized"):
        enqueue(config(tmp_path), incident)


def test_repair_prompt_preserves_authority_and_llama_stop(tmp_path):
    cfg = config(tmp_path)
    prompt = repair_prompt(cfg["boards"][0], {"board_id": "spar"}, cfg, tmp_path / "report.json")
    assert "Never start llama-server" in prompt
    assert "regression test" in prompt
    assert "UNTRUSTED DIAGNOSTIC DATA" in prompt
    assert "active claims zero" in prompt
    assert "never force push" in prompt
    assert "Production supervisors default to DuckDB + Quack" in prompt
    assert "AST/hash/state in a separate DuckDB + Quack instance" in prompt


@pytest.mark.parametrize("health,busy,token,verified", [
    ("healthy", False, "old", False),
    ("healthy", False, None, False),
    ("healthy", True, "old", True),
    ("healthy", False, "new", True),
    ("blocked", True, "new", False),
])
def test_stall_recovery_requires_task_progress(tmp_path, health, busy, token, verified):
    incident = {"observation": {"health": "stalled", "progress_token": "old",
                               "reason_codes": ["no_task_progress"]}}
    result = repair.verify_job_recovery({}, incident,
        {"health": health, "busy": busy, "progress_token": token}, tmp_path)
    assert result["verified"] is verified


def test_raw_completion_does_not_verify_publication(tmp_path):
    result = repair.verify_job_recovery({}, {}, {"health": "complete"}, tmp_path)
    assert result == {"verified": False, "reason": "publication_configuration_required"}


@pytest.mark.parametrize("status,verified", [("held", False), ("published", True)])
def test_completion_runs_current_publication_gate(tmp_path, monkeypatch, status, verified):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion
    calls = []
    def publish(manifest, directory):
        calls.append((manifest, directory))
        return {"status": status}
    monkeypatch.setattr(fleet_completion, "publish_completed_board", publish)
    manifest = {"board_id": "spar"}
    result = repair.verify_job_recovery({"publication": manifest}, {},
        {"health": "healthy", "completion_candidate": True}, tmp_path)
    assert result["verified"] is verified
    assert calls == [(manifest, tmp_path)]


def test_run_job_keeps_backoff_for_idle_healthy_probe(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["repair_worker"].update(argv=["codex", "exec"], retry_seconds=100)
    board = dict(cfg["boards"][0], probe={"argv": ["probe"]})
    directory = tmp_path / "repairs/spar"
    prior_report = directory / "report-previous.json"
    write_json(prior_report, {"status": "blocked", "next_action": "continue regression fix"})
    path = directory / "job.json"
    write_json(path, {"status": "queued", "attempts": 2,
        "report_path": str(prior_report), "latest_incident": {
            "observation": {"health": "stalled", "progress_token": "old",
                            "reason_codes": ["no_task_progress"]}}})
    def command(spec, **kwargs):
        if spec["argv"][0] == "systemctl":
            return {"returncode": 3, "stdout": "inactive"}
        return {"returncode": 0, "stdout": json.dumps({
            "board_id": "spar", "health": "healthy", "busy": False,
            "progress_token": "old", "reason_codes": []})}
    class Process:
        def __init__(self, *args, **kwargs):
            self.prompt = kwargs["stdin"].read().decode()
            assert "continue regression fix" in self.prompt
            active = read_json(path)
            assert active["status"] == "running"
            assert active["log_path"] == kwargs["stdout"].name
            assert Path(active["log_path"]).exists()
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    result = repair.run_job(cfg, board, path)
    durable = read_json(path)
    assert result["status"] == "queued"
    assert durable["attempts"] == 3
    assert durable["next_attempt_at"] > time.time()
    assert durable["verification"]["reason"] == "task_progress_not_verified"
    assert durable["report_path"] != str(prior_report)


def test_prior_report_continuation_is_bounded_and_confined(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    prior = directory / "prior.json"
    prior.write_text("x" * 20000)
    context = repair._prior_report_context(str(prior), directory)
    assert len(context["content"]) == 16000
    assert context["truncated"] is True
    assert context["path"] == str(prior)
    foreign = tmp_path / "foreign.json"
    foreign.write_text("secret")
    assert "content" not in repair._prior_report_context(str(foreign), directory)
    link = directory / "linked.json"
    link.symlink_to(foreign)
    assert "content" not in repair._prior_report_context(str(link), directory)


def test_queue_reports_future_and_held_work_instead_of_idle(tmp_path):
    cfg = config(tmp_path)
    hold = tmp_path / "HOLD"
    hold.touch()
    cfg["boards"][1]["hold_files"] = [str(hold)]
    for identifier, due in (("spar", 900), ("sawm", 50)):
        write_json(tmp_path / f"repairs/{identifier}/job.json", {
            "status": "queued", "next_attempt_at": due})
    assert next_job(cfg, 100) is None
    result = repair.queue_status(cfg, 100)
    assert result["status"] == "waiting"
    assert result["next_attempt_at"] == 900
    assert result["waiting"] == [{"board_id": "spar", "next_attempt_at": 900}]
    assert result["held"] == [{"board_id": "sawm", "next_attempt_at": 50}]


def test_dispatcher_finishes_job_before_adopting_staged_release(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    for board in cfg["boards"]:
        board["probe"] = {"argv": ["probe"]}
    cfg["runtime_release"] = str(Path(repair.__file__).resolve().parents[3])
    assert not repair.runtime_update_pending(cfg)
    assert not repair.runtime_update_pending({})
    config_path = tmp_path / "config.json"
    write_json(config_path, cfg)
    write_json(tmp_path / "repairs/spar/job.json", {"status": "queued"})
    calls = []
    def run_job(current, board, path):
        calls.append(board["id"])
        assert read_json(tmp_path / "repair-worker.json")["status"] == "running"
        changed = dict(cfg, runtime_release=str(tmp_path / "new-release"))
        write_json(config_path, changed)
        # Updating the on-disk config cannot interrupt the active job.
        assert not repair.runtime_update_pending(current)
        return {"status": "verified_healthy"}
    class Event:
        def is_set(self):
            return False
        def wait(self, seconds):
            pass
        def set(self):
            pass
    monkeypatch.setattr(repair, "run_job", run_job)
    monkeypatch.setattr(repair.threading, "Event", Event)
    monkeypatch.setattr(repair.signal, "signal", lambda *args: None)
    assert repair.main(["run", "--config", str(config_path)]) == 0
    assert calls == ["spar"]
    assert read_json(tmp_path / "repair-worker.json")["status"] == "runtime_update_ready"
