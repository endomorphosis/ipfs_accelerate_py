"""Learned clauses retain exact scope when supplied finite workflow premises.

The numerical fixture still predicts one clause. Guarded/retry graph structure
is never injected into the decoded candidate by a projection request.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_autoencoder import _plan
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_roundtrip import (
    PROBE, PROBE_FRAME, roundtrip_checkpoint,  # noqa: F401
)
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor
from ipfs_datasets_py.logic.intent_ir.formalize import extended_preplanning, roundtrip
from ipfs_datasets_py.logic.intent_ir.formalize.projection_contracts import canonical_bytes, source_ir_sha256
from ipfs_datasets_py.logic.intent_ir.formalize.projection_request import make_intent_projection_request
from ipfs_datasets_py.logic.intent_ir.schema import ControlEdgeKind, IntentControlEdge


def _document():
    return roundtrip.frame_to_intent_ir(PROBE_FRAME, instruction=PROBE)


def _write(path, value):
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _finite_request(checkpoint):
    """Explicit data premise; no guards/retries exist in the learned clause."""
    source = _document()
    workflow = {"semantics": "finite_guarded_state_flow", "source_ir_sha256": source_ir_sha256(source),
        "evidence_ref": "source", "variables": [{"variable_id": "read_flag", "kind": "boolean",
            "domain": [False, True], "initial_values": [False], "evidence_ref": "source"}],
        "predicate_bindings": [], "retry_bounds": [],
        "action_updates": [{"action_id": "action", "evidence_ref": "source",
            "outcomes": [{"values": {"read_flag": True}, "evidence_ref": "source"}]}]}
    return make_intent_projection_request(PROBE, source, checkpoint_sha256=checkpoint["sha256"],
        requested_families=["transition_system", "tdfol"], context={
            "state": {"max_steps": 4, "workflow": workflow},
            "modal": {"temporal_bindings": [{"statement_id": "goal", "operator": "next",
                                            "evidence_ref": "source"}]}})


def _assert_finite_advice(advice, request):
    assert advice["status"] == "semantic_candidate_advice"
    report = advice["report"]
    assert report["candidate_intent_ir"] == _document().to_dict()
    assert report["candidate_intent_ir"]["control_edges"] == []
    assert report["projection_request"] == request
    assert report["extension_status"] == "projected_with_explicit_frontiers"
    assert report["learned"]["frame"] == PROBE_FRAME
    assert report["learned"]["encoder"]["checkpoint_weights_sha256"]
    assert report["training_steps"] == report["provider_calls"] == report["download_calls"] == 0
    rows = report["extended_projections"]["projections"]
    state = next(row for row in rows if row["family_id"] == "transition_system" and row["profile_id"] is None)
    graph = state["representation"]["payload"]["metadata"]["guarded_graph"]
    assert state["status"] == "partial" and state["semantics"] == "finite_guarded_state_flow"
    assert graph["deadlock_free"] and graph["initial_valuation_count"] == 1
    assert graph["premises_verified"] is False
    assert {row["values"]["read_flag"] for row in graph["configurations"] if row["phase"] == "ready"} == {False}
    assert {row["values"]["read_flag"] for row in graph["configurations"] if row["terminal"]} == {True}
    assert all(row["retry_counts"] == {} for row in graph["configurations"])
    formula = next(row for row in rows if row["family_id"] == "tdfol")["representation"]["payload"]["formulas"][0]
    assert formula["source"].startswith("X(O(")
    assert formula["temporal_scope"]["evidence_verified"] is False
    assert all(advice[key] is False for key in advisor.AUTHORITY_FIELDS)
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    assert len(advisor._encoded(report)) <= advisor.MAX_REPORT_BYTES
    return graph


def test_learned_clause_with_explicit_data_premises_reaches_goal_planner(
        original, roundtrip_checkpoint, monkeypatch):
    repository, instruction, state = original
    instruction.write_text(PROBE)
    descriptor_path, _ = _write(instruction.parent / "checkpoint.json", roundtrip_checkpoint)
    request = _finite_request(roundtrip_checkpoint)
    request_path, request_digest = _write(instruction.parent / "finite-request.json", request)
    manifest = Path(roundtrip_checkpoint["path"])
    before = manifest.read_bytes()
    prepared = preparation.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=descriptor_path, intent_projection_request=request_path,
        intent_projection_request_sha256=request_digest)
    advice = json.loads((state / "intent-advice.json").read_bytes())
    _assert_finite_advice(advice, request)
    summary, selected = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary and selected == advice and len(summary.encode()) <= 8192
    compact = json.loads(summary)
    assert compact["projection_request_sha256"] == request["request_sha256"]
    assert compact["external_provers_executed"] is False
    assert any(row["semantics"] == "finite_guarded_state_flow" for row in compact["extended_families"])
    assert compact["abstract_state_diagnostics"] == {
        "validator": "finite_guarded_deadlock_scan", "status": "passed", "deadlock_count": 0,
        "initial_valuation_count": 1, "all_configurations_enumerated": True, "premises_verified": False}
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] and result["intent_preplanning"]["supplied_to_router"]
    constraints = json.loads(prompt)["constraints"]["constraint_summaries"]
    assert summary in constraints
    assert json.loads(prompt)["terminal_public_instruction"]["text"] == PROBE
    assert prepared["query"] == PROBE == (repository / preparation.INSTRUCTION).read_text()
    assert manifest.read_bytes() == before


def test_guarded_premise_request_relocates_with_checkpoint_and_exact_runtime_pin(tmp_path, roundtrip_checkpoint):
    request = _finite_request(roundtrip_checkpoint)
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint,
        intent_projection_request=request, **_inputs(tmp_path))
    binding = built["intent_projection_request"]
    assert built["task_modeling_premises_included"] and built["task_inputs_in_archive"]
    restored = tmp_path / "restored"
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        for name in archive.getnames():
            if name.startswith("models/intent-"):
                path = restored / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(name).read())
    descriptor = json.loads((restored / deployment.INTENT_CHECKPOINT_DESCRIPTOR).read_bytes())
    descriptor["path"] = str(restored / deployment.INTENT_ROUNDTRIP_PATH)
    selected = restored / deployment.INTENT_PROJECTION_REQUEST_PATH
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == binding["sha256"]
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=descriptor,
        projection_request_path=selected, projection_request_sha256=binding["sha256"])
    _assert_finite_advice(advice, request)
    selected.write_bytes(selected.read_bytes() + b" ")
    rejected = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=descriptor,
        projection_request_path=selected, projection_request_sha256=binding["sha256"])
    assert rejected["status"] == "semantic_candidate_advice"
    assert rejected["report"]["candidate_intent_ir"] == _document().to_dict()
    assert rejected["report"]["extension_status"] == "fail_open_projection_error"
    assert rejected["report"]["extended_projections"] is None
    assert advisor.validate_intent_advice(rejected, instruction=PROBE) == rejected


@pytest.mark.parametrize("defect", ["unknown_action", "undeclared_retry", "boolean_as_integer", "stale_workflow_source"])
def test_invalid_explicit_model_retains_real_base_inference(roundtrip_checkpoint, defect):
    request = _finite_request(roundtrip_checkpoint)
    workflow = request["context"]["state"]["workflow"]
    if defect == "unknown_action":
        workflow["action_updates"][0]["action_id"] = "not_in_learned_ir"
    elif defect == "undeclared_retry":
        workflow["retry_bounds"] = [{"edge_id": "retry", "max_traversals": 1, "evidence_ref": "source"}]
    elif defect == "boolean_as_integer":
        workflow["action_updates"][0]["outcomes"][0]["values"]["read_flag"] = 1
    else:
        workflow["source_ir_sha256"] = "b" * 64
    request["request_sha256"] = hashlib.sha256(canonical_bytes(
        {key: value for key, value in request.items() if key != "request_sha256"})).hexdigest()
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint,
        projection_request=request)
    assert advice["status"] == "semantic_candidate_advice"
    assert advice["report"]["candidate_intent_ir"] == _document().to_dict()
    assert advice["report"]["projection_request"] == request
    assert advice["report"]["extension_status"] == "fail_open_projection_error"
    assert advice["report"]["extended_projections"] is None
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice


def test_projection_over_budget_preserves_real_inference_and_goal_planning(
        original, roundtrip_checkpoint, monkeypatch):
    repository, instruction, state = original
    instruction.write_text(PROBE)
    descriptor_path, _ = _write(instruction.parent / "checkpoint.json", roundtrip_checkpoint)
    request = make_intent_projection_request(PROBE, _document(),
        checkpoint_sha256=roundtrip_checkpoint["sha256"], requested_families=["transition_system"])
    request_path, request_digest = _write(instruction.parent / "request.json", request)
    # Exercise a real projection at a smaller advisory boundary, with enough
    # room for the independently valid base plus the retained inert request.
    base = roundtrip.prepare_roundtrip_intent_instruction(PROBE, roundtrip_checkpoint)
    base_bound = len(advisor._encoded(base)) + len(advisor._encoded(request)) + 4096
    monkeypatch.setattr(extended_preplanning, "MAX_ADVISORY_REPORT_BYTES", base_bound)
    prepared = preparation.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=descriptor_path, intent_projection_request=request_path,
        intent_projection_request_sha256=request_digest)
    advice = json.loads((state / "intent-advice.json").read_bytes())
    report = advice["report"]
    assert advice["status"] == "semantic_candidate_advice"
    assert report["candidate_intent_ir"] == base["candidate_intent_ir"]
    assert report["learned"] == base["learned"]
    assert report["projection_request"] == request
    assert report["extended_projections"] is None
    assert report["extension_status"] == "fail_open_projection_error"
    assert report["gaps"][-1] == "extended_projection_error_category:ProjectionReportBudgetExceeded"
    assert len(advisor._encoded(report)) < base_bound < advisor.MAX_REPORT_BYTES
    # Replaying the base must remain valid even if the resource failure no
    # longer occurs. Neither failed projection output nor authority is retained.
    monkeypatch.setattr(extended_preplanning, "MAX_ADVISORY_REPORT_BYTES", advisor.MAX_REPORT_BYTES)
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    summary, selected = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary and selected == advice
    assert json.loads(summary)["extended_families"] == []
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] and result["intent_preplanning"]["supplied_to_router"]
    assert summary in json.loads(prompt)["constraints"]["constraint_summaries"]
    assert prepared["query"] == PROBE == (repository / preparation.INSTRUCTION).read_text()
    assert all(advice[key] is False for key in advisor.AUTHORITY_FIELDS)


def test_authored_retry_graph_cannot_replace_single_clause_decoded_by_weights(roundtrip_checkpoint):
    learned = _document()
    authored = replace(learned, control_edges=(IntentControlEdge("retry", "action", "action",
        ControlEdgeKind.RETRY, source_ref_ids=("source",)),))
    authored.validate()
    assert source_ir_sha256(authored) != source_ir_sha256(learned)
    request = make_intent_projection_request(PROBE, authored,
        checkpoint_sha256=roundtrip_checkpoint["sha256"], requested_families=["transition_system"])
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint,
        projection_request=request)
    report = advice["report"]
    assert advice["status"] == "semantic_candidate_advice"
    assert report["candidate_intent_ir"] == learned.to_dict()
    assert report["candidate_intent_ir"]["control_edges"] == []
    assert report["projection_request"] is None and report["extended_projections"] is None
    assert report["extension_status"] == "fail_open_projection_error"
    assert report["learned"]["frame"] == PROBE_FRAME
    assert report["training_steps"] == report["provider_calls"] == report["download_calls"] == 0
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    assert advice["continue_planning"] and advice["raw_instruction_preserved"]
