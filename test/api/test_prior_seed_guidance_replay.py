"""Bounded same-task retry advice survives restart without candidate writes."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from test.api.test_agent_supervisor_prior_attempt_seed import _daemon, _task, _git


def record(daemon, task, candidate):
    daemon._record_prior_attempt_seed_failure(task=task, attempt=2,
        seed_plan={"prior_commit": "authored-prior-commit"},
        seed_apply={"reason": "prior_seed_accepted_proposal_missing"},
        worktree_path=candidate, branch_name="authored-retry")
    return daemon._iter_events()[-1]


def test_exact_guidance_restart_replay_and_consumption_do_not_scan_events(tmp_path, monkeypatch):
    daemon, task = _daemon(tmp_path), _task("answer.py")
    event = record(daemon, task, tmp_path / "candidate")
    restarted = _daemon(tmp_path)
    monkeypatch.setattr(restarted, "_iter_events", lambda: pytest.fail("guidance scanned event history"))
    guidance = restarted._prior_attempt_seed_recovery_guidance(task, 2)
    assert guidance == event["guidance"] and "MUST NOT cherry-pick" in guidance
    assert restarted._prior_attempt_seed_recovery_guidance(task, 3) == ""
    assert restarted._prior_attempt_seed_recovery_guidance(replace(task, title="foreign revision"), 2) == ""
    restarted._consume_prior_attempt_seed_recovery_guidance(task, 2, guidance)
    assert _daemon(tmp_path)._prior_attempt_seed_recovery_guidance(task, 2) == ""
    assert json.loads(Path(event["guidance_artifact"]).read_text())["status"] == "consumed"


@pytest.mark.parametrize("mutation", ["attempt", "boolean_attempt", "task", "key", "schema",
    "guidance", "authority", "extra", "oversize", "symlink", "duplicate_key"])
def test_foreign_stale_or_malformed_guidance_is_not_replayed(tmp_path, mutation):
    daemon, task = _daemon(tmp_path), _task("answer.py")
    event = record(daemon, task, tmp_path / "candidate")
    path = Path(event["guidance_artifact"])
    value = json.loads(path.read_text())
    if mutation == "attempt": value["attempt"] = 1
    elif mutation == "boolean_attempt": value["attempt"] = True
    elif mutation == "task": value["canonical_task_cid"] = "foreign"
    elif mutation == "key": value["canonical_task_key"] = "foreign"
    elif mutation == "schema": value["schema"] = "foreign@1"
    elif mutation == "guidance": value["guidance"] = "changed advisory without its hash"
    elif mutation == "authority": value["execution_authority"] = True
    elif mutation == "extra": value["foreign"] = True
    elif mutation == "oversize":
        value["guidance"] = "x" * 70_000
        value["guidance_sha256"] = hashlib.sha256(value["guidance"].encode()).hexdigest()
    elif mutation == "symlink":
        target = path.with_suffix(".other")
        path.rename(target)
        path.symlink_to(target)
    elif mutation == "duplicate_key":
        path.write_text('{"status":"consumed",' + json.dumps(value)[1:])
    if mutation not in {"symlink", "duplicate_key"}: path.write_text(json.dumps(value))
    assert _daemon(tmp_path)._prior_attempt_seed_recovery_guidance(task, 2) == ""


def test_symlinked_log_directory_does_not_contaminate_candidate(tmp_path):
    daemon, task = _daemon(tmp_path), _task("answer.py")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    _git(candidate, "init", "-q")
    external_link = tmp_path / "logs"
    external_link.symlink_to(candidate, target_is_directory=True)
    daemon.implementation_log_dir = external_link / "nested"
    event = record(daemon, task, candidate)
    assert not (candidate / "nested").exists()
    assert _git(candidate, "status", "--short") == ""
    assert event["guidance_file_skipped_reason"] == "implementation_log_dir_within_candidate"
    assert _daemon(tmp_path)._prior_attempt_seed_recovery_guidance(task, 2) == event["guidance"]


def test_candidate_event_state_is_refused_before_any_guidance_write(tmp_path):
    daemon, task = _daemon(tmp_path), _task("answer.py")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    daemon.events_path = candidate / "events.jsonl"
    with pytest.raises(ValueError, match="outside candidate"):
        record(daemon, task, candidate)
    assert list(candidate.iterdir()) == []


def test_guidance_budget_refusal_preserves_pending_advice(tmp_path, monkeypatch):
    daemon, task = _daemon(tmp_path), _task("answer.py")
    event = record(daemon, task, tmp_path / "candidate")
    monkeypatch.setattr(daemon, "_require_implementation_prompt_byte_budget",
        lambda *args: (_ for _ in ()).throw(ValueError("authored budget refusal")))
    with pytest.raises(ValueError, match="authored budget refusal"):
        daemon._build_implementation_prompt(task, attempt=2)
    assert daemon._prior_attempt_seed_recovery_guidance(task, 2) == event["guidance"]
