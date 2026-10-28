"""Repeated failures are not semantic progress, even across runner restarts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume as lane


@pytest.fixture
def task(tmp_path, monkeypatch):
    root = tmp_path
    (root / "helper.py").write_text("ready = False\n")
    (root / "produce.py").write_text("# producer\n")
    (root / "validate.py").write_text("# validator\n")
    (root / "board.md").write_text(
        "## TEST-001 Required implementation\n- Status: ready\n- Depends on:\n"
        "- Deterministic command: python3 produce.py\n- Validation: python3 validate.py\n"
        "- Execution inputs: helper.py\n- Predicted files: output.json\n- Acceptance: Real validated output\n")
    now, calls = [1000.0], []
    monkeypatch.setattr(lane.time, "time", lambda: now[0])
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[-1] == "validate.py":
            return 0
        if "True" in (root / "helper.py").read_text():
            (root / "output.json").write_text('{"decision":"GO"}')
            return 0
        return 1
    monkeypatch.setattr(lane, "_run_argv", run)
    def cycle():
        return lane.run_resume_cycle(repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-",
                                    attempt_path=root / "attempts.json", log_path=root / "lane.log", heartbeat=lambda **_kw: None)
    return root, now, calls, cycle


def test_unchanged_failures_have_a_finite_budget_despite_clock_and_retry_churn(task):
    root, now, calls, cycle = task
    for _ in range(6):
        assert cycle()["phase"] == "deterministic_command_failed"
        now[0] += lane._EXHAUSTED_RETRY_MS / 1000 + 1
    assert cycle()["phase"] == "semantic_failure_wait"
    now[0] += 86400
    assert cycle()["phase"] == "semantic_failure_wait"
    assert len(calls) == 6
    record = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
    assert record["semantic_failures"] == 6 and record["escalated"] is True
    assert "Status: ready" in (root / "board.md").read_text()


def test_only_changed_declared_content_rearms_an_escalated_contract(task):
    root, now, calls, cycle = task
    for _ in range(6):
        cycle();now[0] += lane._EXHAUSTED_RETRY_MS / 1000 + 1
    before = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
    os.utime(root / "helper.py", (1, 1))
    assert cycle()["phase"] == "semantic_failure_wait" and len(calls) == 6
    (root / "helper.py").write_text("ready = True\n")
    assert cycle()["phase"] == "task_command_passed"
    after = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
    assert after["state"] == "validated" and after["semantic_failures"] == 0
    assert after["contract"] != before["contract"]


def test_interrupted_invocation_preserves_failures_before_proof_is_replaced(task, monkeypatch):
    root, _now, _calls, cycle = task
    cycle()
    before = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
    def interrupt(*_args, **_kwargs): raise lane._LaneResumeInterrupted
    monkeypatch.setattr(lane, "_run_argv", interrupt)
    with pytest.raises(lane._LaneResumeInterrupted): cycle()
    after = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
    assert after["state"] == "running" and after["semantic_failures"] == 1
    assert before["first_started_at_ms"] == after["first_started_at_ms"]


def test_validation_failures_and_producer_owned_source_edits_do_not_reset_history(task, monkeypatch):
    root, _now, _calls, cycle = task
    def run(argv, **kwargs):
        if argv[-1] == "produce.py":
            with (root / "produce.py").open("a") as stream: stream.write("# owned edit\n")
            (root / "output.json").write_text('{"decision":"NO_GO"}')
            return 0
        return 1
    monkeypatch.setattr(lane, "_run_argv", run)
    assert cycle()["phase"] == "validation_failed"
    assert cycle()["phase"] == "validation_failed"
    record = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
    assert record["semantic_failures"] == 2 and record["last_stage"] == "validation"


def test_retry_wait_is_degraded_and_publishes_its_actual_deadline(task, monkeypatch):
    root, _now, _calls, cycle = task
    monkeypatch.setattr(lane, "_ATTEMPT_LIMIT", 1)
    cycle()
    result = cycle()
    assert result["phase"] == "attempt_retry_wait" and result["state"] == "degraded"
    assert result["next_retry_at_ms"] > 0
    lane.write_lane_status(root / "status.json", target_id="test", state=result["state"],
                           phase=result["phase"], next_retry_at_ms=result["next_retry_at_ms"])
    assert json.loads((root / "status.json").read_text())["next_retry_at_ms"] == result["next_retry_at_ms"]


def test_live_retry_heartbeat_does_not_claim_health_after_a_semantic_failure(task, monkeypatch):
    root, _now, _calls, cycle = task
    cycle()
    (root / "lane_resume_attempts.json").write_bytes((root / "attempts.json").read_bytes())
    captured = []
    def run(argv, **kwargs):
        kwargs["heartbeat"]()
        captured.append(json.loads((root / "status.json").read_text()))
        raise lane._LaneResumeInterrupted
    monkeypatch.setattr(lane, "_run_argv", run)
    monkeypatch.setattr(lane.signal, "signal", lambda *_args: None)
    assert lane.serve(["--repo-root", str(root), "--todo-path", "board.md", "--task-header-prefix", "## TEST-",
                       "--status-path", str(root / "status.json"), "--pid-path", str(root / "pid"),
                       "--log-path", str(root / "lane.log"), "--bundle-key", "test", "--max-seconds", "1",
                       "--minimum-free-bytes", "1"]) == 0
    assert captured[0]["state"] == "degraded"
    assert captured[0]["execution_progress"]["semantic_failures"] == 1


def test_contract_repairs_get_fresh_budgets_but_remain_unresolved_until_validation(task):
    root, now, calls, cycle = task
    for _ in range(6):
        assert cycle()["phase"] == "deterministic_command_failed"
        now[0] += lane._EXHAUSTED_RETRY_MS / 1000 + 1
    assert cycle()["phase"] == "semantic_failure_wait"
    for revision in range(10):
        (root / "helper.py").write_text(f"ready = False\n# revision {revision}\n")
        assert cycle()["phase"] == "deterministic_command_failed"
        record = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
        assert record["semantic_failures"] == 1
        assert record["attempts_started"] == 1
        assert record["unresolved_prior_failure"] is True
        assert record["unresolved_prior_failure_reason"] == "prior_semantic_failure_not_validated"
    ledger = json.loads((root / "attempts.json").read_text())
    assert len(ledger["_validation_fixed_points"]["TEST-001"]["execution_progress_history"]) == 8
    (root / "helper.py").write_text("ready = True\n")
    assert cycle()["phase"] == "task_command_passed"
    record = lane.execution_progress_snapshot(root / "attempts.json", "TEST-001")
    assert record["state"] == "validated" and record["semantic_failures"] == 0
    assert record["unresolved_prior_failure"] is False
    assert "unresolved_prior_failure_reason" not in record
    assert len(calls) == 18  # 16 failed producers, one producer and its validator.


def test_legacy_history_migrates_across_contracts_and_validation_stops_old_failures(tmp_path):
    fixed = {"TEST-001": {"execution_progress": {"contract": "b", "state": "running", "semantic_failures": 0},
                          "execution_progress_history": [{"contract": "a", "state": "failed", "semantic_failures": 6}]}}
    lane._begin_execution_progress(fixed, "TEST-001", "c", now_ms=1000, previous_evidence={}, previous_attempts=0)
    progress = fixed["TEST-001"]["execution_progress"]
    assert progress["semantic_failures"] == 0 and progress["unresolved_prior_failure"] is True
    lane._finish_execution_progress(fixed, "TEST-001", code=0, stage="command", contract="c", now_ms=2000)
    assert lane._unresolved_execution_failure(fixed["TEST-001"]) is True  # producer-only success is insufficient.
    lane._finish_execution_progress(fixed, "TEST-001", code=0, stage="validation", contract="c", now_ms=3000, validated=True)
    lane._begin_execution_progress(fixed, "TEST-001", "d", now_ms=4000, previous_evidence={}, previous_attempts=0)
    assert fixed["TEST-001"]["execution_progress"]["unresolved_prior_failure"] is False
    lane._begin_execution_progress(fixed, "NEW-001", "a", now_ms=4000, previous_evidence={}, previous_attempts=0)
    assert fixed["NEW-001"]["execution_progress"]["unresolved_prior_failure"] is False


def test_changed_contract_live_heartbeat_remains_degraded_with_zero_new_failures(task, monkeypatch):
    root, _now, _calls, cycle = task
    cycle()
    (root / "helper.py").write_text("ready = False\n# new source contract\n")
    (root / "lane_resume_attempts.json").write_bytes((root / "attempts.json").read_bytes())
    captured = []
    def run(_argv, **kwargs):
        kwargs["heartbeat"]()
        captured.append(json.loads((root / "status.json").read_text()))
        raise lane._LaneResumeInterrupted
    monkeypatch.setattr(lane, "_run_argv", run)
    monkeypatch.setattr(lane.signal, "signal", lambda *_args: None)
    assert lane.serve(["--repo-root", str(root), "--todo-path", "board.md", "--task-header-prefix", "## TEST-",
                       "--status-path", str(root / "status.json"), "--pid-path", str(root / "pid"),
                       "--log-path", str(root / "lane.log"), "--bundle-key", "test", "--max-seconds", "1",
                       "--minimum-free-bytes", "1"]) == 0
    status = captured[0]
    assert status["state"] == "degraded" and status["execution_progress"]["semantic_failures"] == 0
    assert status["unresolved_prior_failure"] is status["execution_progress"]["unresolved_prior_failure"] is True
    assert status["unresolved_prior_failure_reason"] == "prior_semantic_failure_not_validated"


def test_fresh_active_task_heartbeat_exposes_other_pending_failure_history_without_borrowing_it(task, monkeypatch):
    root, _now, _calls, cycle = task
    cycle()  # A genuine failed TEST-001 remains pending.
    (root / "board.md").write_text((root / "board.md").read_text()
        + "\n## TEST-002 Fresh unrelated implementation\n- Status: ready\n- Depends on:\n"
        "- Deterministic command: python3 fresh.py\n- Validation: python3 validate.py\n- Priority: P0\n")
    (root / "fresh.py").write_text("# fresh producer\n")
    # TEST-001's old slot must sort behind the explicitly preferred fresh task.
    (root / "board.md").write_text((root / "board.md").read_text().replace("- Status: ready\n- Depends on:", "- Status: ready\n- Priority: P1\n- Depends on:", 1))
    (root / "lane_resume_attempts.json").write_bytes((root / "attempts.json").read_bytes())
    captured = []
    def run(_argv, **kwargs):
        kwargs["heartbeat"]()
        captured.append(json.loads((root / "status.json").read_text()))
        raise lane._LaneResumeInterrupted
    monkeypatch.setattr(lane, "_run_argv", run)
    monkeypatch.setattr(lane.signal, "signal", lambda *_args: None)
    assert lane.serve(["--repo-root", str(root), "--todo-path", "board.md", "--task-header-prefix", "## TEST-",
                       "--status-path", str(root / "status.json"), "--pid-path", str(root / "pid"),
                       "--log-path", str(root / "lane.log"), "--bundle-key", "test", "--max-seconds", "1",
                       "--minimum-free-bytes", "1"]) == 0
    status = captured[0]
    assert status["active_task_id"] == "TEST-002" and status["state"] == "degraded"
    assert status["execution_progress"]["semantic_failures"] == 0
    assert status["execution_progress"]["unresolved_prior_failure"] is False
    assert status["unresolved_prior_failure"] is False
    assert status["lane_unresolved_prior_failure"] is True
    assert status["lane_execution_health"]["pending_failure_task_ids"] == ["TEST-001"]
    assert "lane:pending_semantic_failures" in status["backpressure_reasons"]


def test_lane_health_clears_only_validated_pending_failures_and_excludes_completed_history(task):
    root, _now, _calls, cycle = task
    cycle()
    read = lambda: lane.lane_execution_health(repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-", attempt_path=root / "attempts.json")
    assert read()["pending_failure_count"] == 1
    (root / "helper.py").write_text("ready = True\n")
    assert read()["health"] == "degraded"  # A source change is not semantic validation.
    assert cycle()["advanced_task_id"] == "TEST-001"
    assert read()["pending_failure_count"] == 0 and read()["health"] == "operational"
    payload = json.loads((root / "attempts.json").read_text())
    payload["_validation_fixed_points"]["TEST-001"]["execution_progress"].update(state="failed", semantic_failures=6, unresolved_prior_failure=True)
    (root / "attempts.json").write_text(json.dumps(payload))
    assert read()["pending_failure_count"] == 0  # Completed historical labels stay separate.


def test_lane_health_reports_unreadable_failure_history_without_claiming_healthy(task):
    root, _now, _calls, _cycle = task
    (root / "attempts.json").write_text("not-json")
    value = lane.lane_execution_health(repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-", attempt_path=root / "attempts.json")
    assert value["health"] == "degraded" and value["reason"] == "lane_failure_history_unavailable"
