"""Current audit joins and bounded transport; authored failures confer no authority."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_context_evidence as evidence
from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import compiled
from test.api.test_terminal_decision_input_audit import dialogue_case


def observe(case):
    return audit.audit_terminal_context(state=case["state"], receipts=case["receipts"],
        workspace_root=case["workspace_root"], decision_report=case["report"])


def test_current_capsule_keeps_three_value_contract_and_native_dialogue_anchor(dialogue_case):
    raw = dialogue_case["capsule"].read_bytes()
    expected = json.loads((dialogue_case["state"] / "context-result.json").read_bytes())
    prepared = json.loads((dialogue_case["state"] / "prepared.json").read_bytes())
    observed, rendered, formalization = audit._capsule(raw, expected=expected, prepared=prepared)
    assert formalization is None and observed["all_context_checks_passed"] is True
    assert rendered == dialogue_case["original"]
    result = observe(dialogue_case)
    assert result["all_observed_coding_inputs_verified"] is True
    assert result["semantic_decision_inputs"]["all_native_invocations_matched"] is True
    assert all(row["historical_dialogue_replay"] is True for row in result["matches"])
    assert result["provider_calls"] == 0 and result["completion_authority"] is False
    assert rendered not in json.dumps(result)


@pytest.mark.parametrize("field", ["provider", "model"])
def test_missing_independently_recorded_route_never_infers_dialogue_identity(dialogue_case, field):
    dialogue_case["receipts"][1].pop(field)
    result = observe(dialogue_case)
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["semantic_decision_inputs"]["all_native_invocations_matched"] is False
    assert len(result["matches"]) == 1


def test_collector_bounds_closed_envelope_and_keeps_real_route_fields(compiled, monkeypatch):
    captured = []
    def child(command, **kwargs):
        captured.append((command, kwargs))
        return 0, json.dumps(audit._base()).encode()
    monkeypatch.setattr(audit, "_bounded_audit_child", child)
    rows = [{**compiled["receipts"][0], "provider": "codex_cli", "model": "pinned"}]
    report = {"semantic_decision_dialogue_expected": True, "private_unrelated_state": "PRIVATE"}
    result = audit.collect_terminal_context_audit(state=compiled["state"], receipts=rows,
        workspace_root=compiled["workspace_root"], decision_report=report, timeout_seconds=1)
    command, options = captured[0]
    assert command[1:3] == ["-B", "-P"]
    assert command[3] == str(Path(audit.__file__).resolve()) and "-m" not in command
    supplied = json.loads(options["input_bytes"])
    assert supplied["decision_report"] == {"semantic_decision_dialogue_expected": True}
    assert supplied["receipts"][0]["provider"] == "codex_cli"
    assert supplied["receipts"][0]["model"] == "pinned"
    assert "PRIVATE" not in json.dumps(supplied)
    assert 0 < options["timeout"] <= 1 and options["limit"] == audit.MAX_RESULT_BYTES
    assert result["schema"] == audit.SCHEMA


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"nested": [object()]}])
def test_request_rejects_nonfinite_or_non_json_values_before_child(compiled, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid request launched a child")
    monkeypatch.setattr(audit, "_bounded_audit_child", forbidden)
    result = audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"],
        decision_report={"semantic_decision_dialogues": [{"invalid": value}]})
    assert result["reason"] == "audit_request_exceeds_bound"


@pytest.mark.parametrize("row,reason", [
    ({"phase": "coding", "world_task_packet": {}}, "world_packet_export_not_supported"),
    ({"phase": "coding", "source_formalization_context": {}}, "formalization_export_not_supported"),
])
def test_closed_export_profile_refuses_current_unsupported_dependency_families(row, reason):
    with pytest.raises(ValueError, match=reason):
        evidence._request([row], {})


def test_typed_owner_sidecar_cannot_be_silently_dropped_from_closed_export():
    with pytest.raises(ValueError, match="typed_nomination_independent_expectation_required"):
        evidence._request([], {"semantic_decision_owner_bindings": [{"typed_nomination": {}}]})


def test_actual_bounded_audit_transport_refuses_output_flood_and_reaps_owned_child(tmp_path):
    pid = tmp_path / "child.pid"
    command = [sys.executable, "-c",
        "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "os.write(1,b'x'*65536); time.sleep(10)", str(pid)]
    with pytest.raises(ValueError, match="child_output_bound"):
        audit._bounded_audit_child(command, input_bytes=b"", timeout=2,
            limit=64, environment=dict(os.environ))
    child = int(pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_actual_bounded_audit_transport_times_out_blocked_input_without_waiting_for_stdout(tmp_path):
    pid = tmp_path / "blocked.pid"
    command = [sys.executable, "-c",
        "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(10)", str(pid)]
    with pytest.raises(subprocess.TimeoutExpired):
        audit._bounded_audit_child(command, input_bytes=b"x"*65536, timeout=.5,
            limit=64, environment=dict(os.environ))
    child = int(pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
