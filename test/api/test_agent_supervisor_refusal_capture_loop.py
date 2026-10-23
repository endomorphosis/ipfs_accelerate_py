"""Withheld producers must not repeatedly capture unchanged failing evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import gate_recovery, lane_resume
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import parse_todo_task_fields


def _report(trace: str = "first", reason: str = "owner_unavailable") -> dict[str, Any]:
    return {"schema": "test.structured-gate@1", "decision": "no_go", "passed": False,
            "generated_at": trace, "gap_count": 1, "gap_counts": {reason: 1},
            "gaps": [{"code": reason, "app_id": "editor", "selector": "#save", "tool_id": "owner:write",
                      "transport": "http", "correlation_id": trace, "receipt_id": "receipt-" + trace,
                      "event_dag_cid": "event-" + trace}],
            "input_fingerprint": "fingerprint-" + trace,
            "evidence_sources": [{"path": "runs/" + trace + ".json", "sha256": trace}]}


@pytest.fixture
def capture_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    board = tmp_path / "board.md"
    board.write_text(
        "## TEST-010 Shared source\n- Status: completed\n- Depends on:\n"
        "- Validation: python3 source.py\n- Predicted files: source.py\n\n"
        "## TEST-001 Capture\n- Status: pending\n- Depends on: TEST-010\n"
        "- Deterministic command: python3 producer.py --fresh http\n"
        "- Validation: python3 validator.py\n"
        "- Predicted files: producer.py, validator.py, report.json\n\n"
        "## TEST-002 Other transport\n- Status: pending\n- Depends on: TEST-001\n"
        "- Deterministic command: python3 repair.py libp2p\n- Validation: python3 validator.py\n",
    )
    for name in ("source.py", "producer.py", "validator.py", "repair.py", "transport_check.py"):
        (tmp_path / name).write_text("# fixture source\n")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()))
    state = SimpleNamespace(producer_calls=0, validator_calls=0, producer_exit=75, validator_exit=75, other_calls=0)
    now = [1000.0]
    monkeypatch.setattr(lane_resume.time, "time", lambda: now[0])
    monkeypatch.setattr(lane_resume, "_refill_module", lambda _root: None)

    def run(argv: list[str], **_kwargs: Any) -> int:
        if "producer.py" in argv:
            state.producer_calls += 1
            payload = _report(f"fresh-{state.producer_calls}")
            if state.producer_exit == 0:
                payload.update(decision="GO", passed=True, gap_count=0, gap_counts={}, gaps=[])
            report.write_text(json.dumps(payload))
            return state.producer_exit
        if "validator.py" in argv:
            state.validator_calls += 1
            return state.validator_exit
        state.other_calls += 1
        return 0

    monkeypatch.setattr(lane_resume, "_run_argv", run)
    kwargs = dict(repo_root=tmp_path, todo_path=board, task_header_prefix="## TEST-",
                  attempt_path=tmp_path / "attempts.json", log_path=tmp_path / "lane.log",
                  heartbeat=lambda **_kwargs: None)
    return SimpleNamespace(root=tmp_path, board=board, report=report, state=state, now=now, kwargs=kwargs,
                           cycle=lambda: lane_resume.run_resume_cycle(**kwargs))


def test_coded_failure_fingerprint_ignores_new_trace_links_but_retains_outcomes(capture_lane: SimpleNamespace) -> None:
    lane = capture_lane
    blocks = dict(parse_todo_task_fields(lane.board.read_text(), task_header_prefix="## TEST-"))
    kwargs = dict(repo_root=lane.root, todo_path=lane.board, log_path=lane.root / "lane.log")
    first, _ = gate_recovery.gate_evidence("TEST-001", blocks, **kwargs)
    raw = lane_resume._file_digest(lane.report)
    diagnostic = gate_recovery.evidence_file_fingerprint(lane.report)
    lane.report.write_text(json.dumps(_report("different-run-and-cids")))
    assert gate_recovery.gate_evidence("TEST-001", blocks, **kwargs)[0] == first
    assert gate_recovery.evidence_file_fingerprint(lane.report) == diagnostic
    assert lane_resume._file_digest(lane.report) != raw, "raw proof identity must remain strict"
    lane.report.write_text(json.dumps(_report("new", "policy_denied")))
    assert gate_recovery.gate_evidence("TEST-001", blocks, **kwargs)[0] != first


def test_unknown_json_does_not_discard_receipt_or_correlation_identity(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"schema": "raw-proof@1", "receipt_id": "first", "correlation_id": "first"}))
    before = gate_recovery.evidence_file_fingerprint(path)
    path.write_text(json.dumps({"schema": "raw-proof@1", "receipt_id": "second", "correlation_id": "second"}))
    assert gate_recovery.evidence_file_fingerprint(path) != before


def test_real_descriptor_json_rearms_source_contract_but_generated_reports_do_not() -> None:
    def contract(descriptor: str, run: str) -> str:
        return gate_recovery.gate_source_contract({"tasks": {}, "artifacts": {
            "swissknife/src/services/ipfs/tools-manifest.json": {"sha256": descriptor},
            "swissknife/test-results/family.json": {"sha256": run},
            "data/watchdog/backend-readiness.json": {"sha256": run},
        }}, "same-command")
    assert contract("real-first", "trace-first") == contract("real-first", "trace-next")
    assert contract("real-first", "trace-next") != contract("real-fixed", "trace-next")


def test_forced_capture_runs_once_then_cheap_probes_escalate_unchanged_refusal(capture_lane: SimpleNamespace) -> None:
    lane = capture_lane
    original = lane.board.read_text()
    assert lane.cycle()["phase"] == "gate_withheld"
    assert lane.cycle()["phase"] == "gate_withheld"
    third = lane.cycle()
    assert third["phase"] == "gate_recovery_needed"
    assert lane.state.producer_calls == 1, "--fresh producer was repeated instead of checking its existing evidence"
    packet = json.loads(Path(third["repair_packet"]).read_text())
    assert packet["source_refusals"] == 3
    assert packet["last_probe"] == "validation_only"
    assert packet["pending_dependents"][0]["task_id"] == "TEST-002"
    warnings = {warning["code"] for warning in packet["contract_warnings"]}
    assert warnings == {"forced_fresh_producer", "single_transport_producer_validation_scope_unspecified"}
    calls = lane.state.validator_calls
    assert lane.cycle()["phase"] == "gate_retry_wait"
    assert lane.state.validator_calls == calls
    lane.now[0] = third["next_retry_at_ms"] / 1000 + 1
    assert lane.cycle()["phase"] == "gate_recovery_needed"
    assert lane.state.producer_calls == 1
    assert lane.board.read_text() == original


def test_source_budget_survives_changing_failure_details_and_rearms_after_code_fix(capture_lane: SimpleNamespace) -> None:
    lane = capture_lane
    lane.cycle()
    lane.report.write_text(json.dumps(_report("second", "another_gap")))
    assert lane.cycle()["phase"] == "gate_withheld"
    lane.report.write_text(json.dumps(_report("third", "still_another_gap")))
    third = lane.cycle()
    assert third["phase"] == "gate_recovery_needed"
    record = gate_recovery.GateRecovery(lane.root / "attempts.json").records["TEST-001"]
    assert record["unchanged_probes"] == 1
    assert record["source_refusals"] == 3
    (lane.root / "producer.py").write_text("# repaired producer behavior\n")
    restored = lane.cycle()
    assert restored["phase"] == "gate_withheld"
    assert restored["gate_recovery_needed"] is False
    assert lane.state.producer_calls == 2
    assert gate_recovery.GateRecovery(lane.root / "attempts.json").records["TEST-001"]["source_refusals"] == 1


def test_green_probe_cannot_complete_a_still_refusing_producer(capture_lane: SimpleNamespace) -> None:
    lane = capture_lane
    lane.cycle()
    lane.state.validator_exit = 0
    result = lane.cycle()
    assert result["phase"] == "gate_withheld"
    assert lane.state.producer_calls == 2
    assert dict(parse_todo_task_fields(lane.board.read_text(), task_header_prefix="## TEST-"))["TEST-001"]["status"] == "pending"


def test_new_accepted_evidence_still_requires_original_producer_and_validation(capture_lane: SimpleNamespace) -> None:
    lane = capture_lane
    lane.cycle()
    lane.state.validator_exit = 0
    lane.state.producer_exit = 0
    result = lane.cycle()
    assert result["advanced_task_id"] == "TEST-001"
    assert lane.state.producer_calls == 2
    assert lane.state.validator_calls == 2, "a green cheap probe skipped the final strict validation"
    assert "TEST-001" not in gate_recovery.GateRecovery(lane.root / "attempts.json").records


def test_hold_preserves_withheld_history_without_probe_or_capture(capture_lane: SimpleNamespace) -> None:
    lane = capture_lane
    lane.cycle()
    original = (lane.root / "attempts.gates.json").read_bytes()
    (lane.root / "operator.hold").touch()
    assert lane.cycle()["phase"] == "operator_hold"
    assert lane.state.producer_calls == 1 and lane.state.validator_calls == 0
    assert (lane.root / "attempts.gates.json").read_bytes() == original


@pytest.mark.parametrize("bypass", ["", " --fresh", " --no-cache", " --force-refresh"])
def test_admission_rejects_recapture_only_repair_dependency(capture_lane: SimpleNamespace, bypass: str) -> None:
    lane = capture_lane
    with pytest.raises(ValueError, match="recapture-only repair"):
        gate_recovery.admit_gate_repair_task(
            lane.board.read_text(), repo_root=lane.root, task_id="TEST-001", repair_task_id="TEST-R001",
            command="python3 producer.py http" + bypass, validation="python3 validator.py",
            predicted_files=["producer.py"], acceptance="Run the same evidence again", depends_on=["TEST-010"],
        )


def test_changed_validation_contract_can_admit_transport_phase_repair(capture_lane: SimpleNamespace) -> None:
    lane = capture_lane
    result = gate_recovery.admit_gate_repair_task(
        lane.board.read_text(), repo_root=lane.root, task_id="TEST-001", repair_task_id="TEST-R001",
        command="python3 producer.py http --fresh", validation="python3 transport_check.py http",
        predicted_files=["producer.py"], acceptance="Verify complete capture on the selected transport only", depends_on=["TEST-010"],
    )
    assert result["admitted"] is True
    tasks = dict(parse_todo_task_fields(result["todo_text"], task_header_prefix="## TEST-"))
    assert tasks["TEST-001"]["status"] == "pending"
    assert tasks["TEST-R001"]["status"] == "pending"
