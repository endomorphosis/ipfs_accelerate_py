"""Finite foreground repair admissions; real producers keep validation authority."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume, semantic_recovery

pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise persisted repair admission and real original validation")


def task(root: Path, *, body: str = "Path('helper.py').write_text('def ready(): return True\\n')\n", limit: int = 1) -> None:
    (root / "helper.py").write_text("def ready(): return False\n")
    (root / "feedback.json").write_text('{"code":"unique_anchor_failed"}')
    (root / "produce.py").write_text(
        "from pathlib import Path\nimport sys\nsys.dont_write_bytecode=True\nfrom helper import ready\n"
        "p=Path('producer-count');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
        "assert ready()\nPath('report.json').write_text('{\"actual\":50}')\n")
    (root / "validate.py").write_text("from pathlib import Path\nimport json\nPath('validator-ran').touch()\nassert json.loads(Path('report.json').read_text())['actual']==50\n")
    (root / "recover.py").write_text(
        "import json,os\nfrom pathlib import Path\n"
        "packet=json.loads(Path(os.environ['AGENT_SUPERVISOR_RECOVERY_CONTEXT_PATH']).read_text())\n"
        "assert packet['task_id']=='TEST-001' and packet['completion_authority'] is False\n"
        "Path('context-seen.json').write_text(json.dumps(packet))\n"
        "p=Path('recovery-count');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n" + body)
    (root / "board.md").write_text(
        "## TEST-001 Actual workflow\n- Status: ready\n- Depends on:\n"
        "- Deterministic command: python3 produce.py\n- Validation: python3 validate.py\n"
        "- Execution inputs: helper.py\n- Predicted files: helper.py\n- Outputs: report.json\n"
        "- Acceptance: Actual 50 from the original validated producer\n"
        "- Semantic recovery command: python3 recover.py --task-id TEST-001\n"
        "- Semantic recovery inputs: feedback.json, helper.py\n"
        f"- Semantic recovery limit: {limit}\n- Semantic recovery timeout seconds: 2\n")
    fields = _fields(root)
    contract = lane_resume._command_contract(fields, ["python3", "produce.py"], ["python3", "validate.py"], repo_root=root)
    lane_resume._store_attempt_state(root / "attempts.json", {"TEST-001": 6}, {},
        {"TEST-001": {"execution_progress": {"contract": contract, "state": "failed", "semantic_failures": 6,
            "attempts_started": 6, "last_stage": "command", "last_exit_code": 1, "unresolved_prior_failure": True}}},
        {"TEST-001": {"contract": contract, "retry_contract": contract, "before": {}, "after": {}, "exit_code": 1}})


def _fields(root: Path) -> dict[str, str]:
    from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import parse_todo_task_fields
    return dict(parse_todo_task_fields((root / "board.md").read_text(), task_header_prefix="## TEST-"))["TEST-001"]


def cycle(root: Path, heartbeat=lambda **_kwargs: None) -> dict:
    return lane_resume.run_resume_cycle(repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-",
        attempt_path=root / "attempts.json", log_path=root / "lane.log", heartbeat=heartbeat)


def summary(root: Path) -> dict:
    return semantic_recovery.summary(root / "attempts.json", "TEST-001")


def test_repair_changes_real_source_without_completing_then_original_producer_and_validator_complete(tmp_path):
    task(tmp_path)
    before_board = (tmp_path / "board.md").read_bytes()
    result = cycle(tmp_path)
    assert result["phase"] == "semantic_recovery_applied"
    assert not result["completion_authority"] and not result["work_complete"]
    assert (tmp_path / "board.md").read_bytes() == before_board
    assert not (tmp_path / "producer-count").exists() and not (tmp_path / "validator-ran").exists()
    attempts = json.loads((tmp_path / "attempts.json").read_text())
    assert attempts["TEST-001"] == 6
    assert attempts["_validation_fixed_points"]["TEST-001"]["execution_progress"]["semantic_failures"] == 6
    packet = json.loads((tmp_path / "context-seen.json").read_text())
    assert packet["admission_index"] == 1 and packet["failure_summary"]["semantic_failures"] == 6
    assert packet["feedback_inputs"]["feedback.json"] != '{"code":"unique_anchor_failed"}'
    assert packet["source_before"] == {"helper.py": lane_resume.hashlib.sha256(b"def ready(): return False\n").hexdigest()}
    assert summary(tmp_path)["spent"] == 1
    result = cycle(tmp_path)
    assert result["advanced_task_id"] == "TEST-001"
    assert (tmp_path / "producer-count").read_text() == "1"
    assert (tmp_path / "validator-ran").is_file()
    assert "- Status: completed" in (tmp_path / "board.md").read_text()
    progress = lane_resume.execution_progress_snapshot(tmp_path / "attempts.json", "TEST-001")
    assert progress["state"] == "validated" and progress["unresolved_prior_failure"] is False
    assert progress["semantic_recovery"]["spent"] == 1


@pytest.mark.parametrize("body", ["pass\n", "Path('feedback.json').write_text('{\"code\":\"different_feedback\"}')\n",
                                 "Path('receipt.progress.json').write_text('{\"heartbeat\":123}')\n"])
def test_successful_noop_feedback_or_native_progress_never_resets_or_probes(tmp_path, body):
    task(tmp_path, body=body)
    assert cycle(tmp_path)["phase"] == "semantic_recovery_noop"
    for _ in range(3):
        os.utime(tmp_path / "helper.py", None)
        assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert summary(tmp_path)["spent"] == 1
    assert (tmp_path / "recovery-count").read_text() == "1"
    assert not (tmp_path / "producer-count").exists()
    assert json.loads((tmp_path / "attempts.json").read_text())["TEST-001"] == 6


def test_runtime_source_named_output_cannot_qualify_material_repair(tmp_path):
    task(tmp_path, body="Path('data').mkdir(exist_ok=True)\nPath('data/generated.py').write_text('ready=True')\n")
    board = tmp_path / "board.md"
    board.write_text(board.read_text().replace("- Execution inputs: helper.py", "- Execution inputs: helper.py, data/generated.py"))
    # Qualify the current contract without running the intentionally broken producer.
    fields = _fields(tmp_path)
    digest = lane_resume._command_contract(fields, ["python3", "produce.py"], ["python3", "validate.py"], repo_root=tmp_path)
    data = json.loads((tmp_path / "attempts.json").read_text())
    data["_validation_fixed_points"]["TEST-001"]["execution_progress"]["contract"] = digest
    data["_command_evidence"]["TEST-001"].update(contract=digest, retry_contract=digest)
    (tmp_path / "attempts.json").write_text(json.dumps(data))
    assert cycle(tmp_path)["phase"] == "semantic_recovery_noop"
    (tmp_path / "data/generated.py").write_text("ready=True # runtime churn\n")
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert not (tmp_path / "producer-count").exists()


def test_failed_partial_source_change_is_blocked_without_reset_then_external_repair_rearms(tmp_path):
    task(tmp_path, body="Path('helper.py').write_text('def ready(): return True\\n')\nraise SystemExit(1)\n")
    assert cycle(tmp_path)["phase"] == "semantic_recovery_failed"
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert not (tmp_path / "producer-count").exists()
    assert json.loads((tmp_path / "attempts.json").read_text())["TEST-001"] == 6
    (tmp_path / "helper.py").write_text("def ready(): return 'external fixed source'\n")
    assert cycle(tmp_path)["advanced_task_id"] == "TEST-001"
    assert summary(tmp_path)["spent"] == 1


def test_unchanged_diagnostics_do_not_admit_second_recovery_but_changed_diagnostics_use_finite_budget(tmp_path):
    task(tmp_path, body="raise SystemExit(1)\n", limit=2)
    assert cycle(tmp_path)["phase"] == "semantic_recovery_failed"
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert summary(tmp_path)["spent"] == 1
    (tmp_path / "feedback.json").write_text('{"code":"new_qualified_anchor_diagnostic"}')
    assert cycle(tmp_path)["phase"] == "semantic_recovery_failed"
    assert summary(tmp_path)["spent"] == 2
    (tmp_path / "feedback.json").write_text('{"code":"another_diagnostic"}')
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert (tmp_path / "recovery-count").read_text() == "2"
    assert not (tmp_path / "producer-count").exists()


def test_recovery_budget_survives_contract_changes_and_new_cycle_owners(tmp_path):
    task(tmp_path, body="raise SystemExit(1)\n")
    assert cycle(tmp_path)["phase"] == "semantic_recovery_failed"
    (tmp_path / "helper.py").write_text("def ready(): return False # new contract\n")
    fields = _fields(tmp_path)
    contract = lane_resume._command_contract(fields, ["python3", "produce.py"], ["python3", "validate.py"], repo_root=tmp_path)
    data = json.loads((tmp_path / "attempts.json").read_text())
    data["_validation_fixed_points"]["TEST-001"]["execution_progress"]["contract"] = contract
    data["_command_evidence"]["TEST-001"].update(contract=contract, retry_contract=contract)
    (tmp_path / "attempts.json").write_text(json.dumps(data))
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert summary(tmp_path)["spent"] == 1 and (tmp_path / "recovery-count").read_text() == "1"


def test_clock_pid_and_json_formatting_changes_do_not_qualify_another_paid_repair(tmp_path):
    task(tmp_path, body="raise SystemExit(1)\n", limit=2)
    assert cycle(tmp_path)["phase"] == "semantic_recovery_failed"
    (tmp_path / "feedback.json").write_text('{ "timestamp": 999, "pid": 1234, "code": "unique_anchor_failed" }')
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert summary(tmp_path)["spent"] == 1
    assert (tmp_path / "recovery-count").read_text() == "1"


def test_validator_or_controller_edits_cannot_authorize_a_fresh_probe(tmp_path):
    task(tmp_path, body="Path('helper.py').write_text('def ready(): return True\\n')\nPath('validate.py').write_text('pass\\n')\n")
    assert cycle(tmp_path)["phase"] == "semantic_recovery_blocked"
    assert summary(tmp_path)["reason"] == "recovery_changed_controller_validator_or_taskboard"
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert not (tmp_path / "producer-count").exists()


def test_admission_is_durable_before_interruption_and_cannot_be_replayed(tmp_path, monkeypatch):
    task(tmp_path)
    def interrupted(*_args, **_kwargs):
        assert summary(tmp_path)["spent"] == 1 and summary(tmp_path)["state"] == "running"
        raise lane_resume._LaneResumeInterrupted
    monkeypatch.setattr(lane_resume, "_run_argv", interrupted)
    with pytest.raises(lane_resume._LaneResumeInterrupted):
        cycle(tmp_path)
    assert summary(tmp_path)["state"] == "interrupted"
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"


def test_an_abandoned_running_journal_is_spent_and_reconciled_before_new_source_probe(tmp_path):
    task(tmp_path)
    journal = semantic_recovery.Journal(semantic_recovery.journal_path(tmp_path / "attempts.json"), lane_resume._atomic_write)
    contract = json.loads((tmp_path / "attempts.json").read_text())["_command_evidence"]["TEST-001"]["contract"]
    journal.admit("TEST-001", contract=contract, fingerprint="a" * 64, limit=1, context_path="context.json", now_ms=1)
    (tmp_path / "helper.py").write_text("def ready(): return True\n")
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert summary(tmp_path)["state"] == "interrupted"
    assert not (tmp_path / "producer-count").exists()


@pytest.mark.parametrize("replacement", [
    ("python3 recover.py --task-id TEST-001", "grok --model grok4.7"),
    ("python3 recover.py --task-id TEST-001", "python3 -c pass TEST-001"),
    ("python3 recover.py --task-id TEST-001", "python3 recover.py --task-id FOREIGN-001"),
    ("Semantic recovery limit: 1", "Semantic recovery limit: 99"),
    ("Semantic recovery timeout seconds: 2", "Semantic recovery timeout seconds: nan"),
    ("feedback.json, helper.py", "../outside.json"),
    ("feedback.json, helper.py", "missing-feedback.json"),
])
def test_unsafe_or_unqualified_recovery_is_refused_without_spending_or_invoking(tmp_path, replacement):
    task(tmp_path)
    board = tmp_path / "board.md";board.write_text(board.read_text().replace(*replacement))
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert summary(tmp_path)["spent"] == 0 and not (tmp_path / "recovery-count").exists()


def test_corrupt_journal_fails_closed_instead_of_resetting_budget(tmp_path):
    task(tmp_path)
    semantic_recovery.journal_path(tmp_path / "attempts.json").write_text('{"tasks":{}}')
    result = cycle(tmp_path)
    assert result["phase"] == "semantic_failure_wait"
    assert "TEST-001:recovery_journal_invalid" in result["backpressure_reasons"]
    assert not (tmp_path / "recovery-count").exists()


def test_operator_hold_prevents_admission_and_interrupts_only_the_owned_hook(tmp_path, monkeypatch):
    task(tmp_path)
    (tmp_path / "HOLD").touch()
    assert cycle(tmp_path)["phase"] == "operator_hold"
    assert summary(tmp_path)["spent"] == 0
    (tmp_path / "HOLD").unlink()
    def hold_inside(*_args, **options):
        (tmp_path / "HOLD").touch()
        options["heartbeat"]()
        raise AssertionError("hold heartbeat must unwind")
    monkeypatch.setattr(lane_resume, "_run_argv", hold_inside)
    assert cycle(tmp_path)["phase"] == "operator_hold"
    assert summary(tmp_path)["spent"] == 1 and summary(tmp_path)["state"] == "interrupted"


def test_canary_limit_one_uses_one_real_failed_producer_then_one_scoped_fixture_repair(tmp_path):
    fixture = "data/campaign/fixtures/automatic-repair-canary.json"
    path = tmp_path / fixture;path.parent.mkdir(parents=True);path.write_text('{"recovery_state":"pending"}')
    task(tmp_path, body=f"Path({fixture!r}).write_text('{{\"recovery_state\":\"repaired_by_scoped_model\"}}')\n")
    (tmp_path / "produce.py").write_text(
        "from pathlib import Path\nimport json\n"
        "p=Path('producer-count');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
        f"assert json.loads(Path({fixture!r}).read_text())['recovery_state']=='repaired_by_scoped_model'\n"
        "Path('report.json').write_text('{\"actual\":50}')\n")
    board = tmp_path / "board.md"
    board.write_text(board.read_text().replace("- Execution inputs: helper.py", f"- Execution inputs: {fixture}")
                     .replace("- Predicted files: helper.py", f"- Predicted files: {fixture}")
                     + f"- Semantic failure limit: 1\n- Semantic recovery fixture inputs: {fixture}\n")
    (tmp_path / "attempts.json").unlink()  # This canary starts genuinely unattempted.
    assert cycle(tmp_path)["phase"] == "deterministic_command_failed"
    assert (tmp_path / "producer-count").read_text() == "1"
    progress = lane_resume.execution_progress_snapshot(tmp_path / "attempts.json", "TEST-001")
    assert progress["semantic_failures"] == 1
    assert cycle(tmp_path)["phase"] == "semantic_recovery_applied"
    packet = json.loads((tmp_path / "context-seen.json").read_text())
    assert list(packet["source_before"]) == [fixture]
    assert (tmp_path / "producer-count").read_text() == "1"
    assert cycle(tmp_path)["advanced_task_id"] == "TEST-001"
    assert (tmp_path / "producer-count").read_text() == "2"
    assert summary(tmp_path)["spent"] == 1


@pytest.mark.parametrize("value", ["0", "7", "-1", "1.5", "nan", "True"])
def test_invalid_semantic_failure_limit_never_invokes_a_command(tmp_path, value):
    task(tmp_path)
    board = tmp_path / "board.md";board.write_text(board.read_text() + f"- Semantic failure limit: {value}\n")
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert not (tmp_path / "producer-count").exists() and not (tmp_path / "recovery-count").exists()


@pytest.mark.parametrize("declaration", ["data/campaign/fixtures/unknown.json", "data/campaign/fixtures/feedback.json", "helper.py"])
def test_fixture_role_cannot_broaden_undeclared_sources_or_admit_proof_artifacts(tmp_path, declaration):
    task(tmp_path)
    board = tmp_path / "board.md";board.write_text(board.read_text() + f"- Semantic recovery fixture inputs: {declaration}\n")
    assert cycle(tmp_path)["phase"] == "semantic_failure_wait"
    assert not (tmp_path / "recovery-count").exists() and not (tmp_path / "producer-count").exists()


def test_authored_workflow_json_is_bound_with_app_source_and_scenario_only_repair_rearms(tmp_path):
    scenario = "app/test/e2e/functional-workflows/image-viewer.json"
    path = tmp_path / scenario;path.parent.mkdir(parents=True);path.write_text('{"action":"broken"}\n')
    task(tmp_path, body=f"Path({scenario!r}).write_text('{{\"action\":\"fixed\"}}\\n')\n")
    (tmp_path / "produce.py").write_text(
        "from pathlib import Path\nimport json\n"
        "p=Path('producer-count');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
        f"assert json.loads(Path({scenario!r}).read_text())['action']=='fixed'\n"
        "Path('report.json').write_text('{\"actual\":50}')\n")
    board = tmp_path / "board.md";board.write_text(board.read_text().replace("- Predicted files: helper.py", f"- Predicted files: helper.py, {scenario}"))
    fields = _fields(tmp_path)
    before = lane_resume._command_contract(fields, ["python3", "produce.py"], ["python3", "validate.py"], repo_root=tmp_path)
    data = json.loads((tmp_path / "attempts.json").read_text())
    data["_validation_fixed_points"]["TEST-001"]["execution_progress"]["contract"] = before
    data["_command_evidence"]["TEST-001"].update(contract=before, retry_contract=before)
    (tmp_path / "attempts.json").write_text(json.dumps(data))
    assert cycle(tmp_path)["phase"] == "semantic_recovery_applied"
    packet = json.loads((tmp_path / "context-seen.json").read_text())
    assert set(packet["source_before"]) == {"helper.py", scenario}
    assert packet["source_before"][scenario] == lane_resume.hashlib.sha256(b'{"action":"broken"}\n').hexdigest()
    after = lane_resume._command_contract(fields, ["python3", "produce.py"], ["python3", "validate.py"], repo_root=tmp_path)
    assert before != after
    assert summary(tmp_path)["spent"] == 1
    assert cycle(tmp_path)["advanced_task_id"] == "TEST-001"


def test_missing_declared_authored_scenario_retains_none_hash_until_created(tmp_path):
    scenario = "project/tests/e2e/functional-workflows/new-app.json"
    fields = {"predicted_files": scenario}
    assert lane_resume._recovery_source_snapshot(fields, repo_root=tmp_path) == {scenario: None}
    path = tmp_path / scenario;path.parent.mkdir(parents=True);path.write_text('{"steps":[]}')
    assert lane_resume._recovery_source_snapshot(fields, repo_root=tmp_path)[scenario] is not None


@pytest.mark.parametrize("name", [
    "app/test/e2e/functional-workflows/receipt.json", "app/test/e2e/functional-workflows/app.progress.json",
    "data/app/test/e2e/functional-workflows/app.json", "test-results/test/e2e/functional-workflows/app.json",
    "app/test/e2e/functional-workflows/reports/app.json", "app/test/e2e/functional-workflows/model-invocations/app.json",
    "../outside/test/e2e/functional-workflows/app.json", "app/test/e2e/results/app.json",
])
def test_authored_scenario_classification_does_not_admit_generated_proof_or_outside_paths(name):
    assert semantic_recovery.is_authored_scenario_source(Path(name)) is False
