"""Shared Intent family projections remain optional, replayed planner advice."""
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
    PROBE, PROBE_FRAME, roundtrip_checkpoint)  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor
from ipfs_datasets_py.logic.intent_ir.formalize import roundtrip as native
from ipfs_datasets_py.logic.intent_ir.formalize import extended_preplanning, extended_projections
from ipfs_datasets_py.logic.intent_ir.formalize.projection_contracts import canonical_bytes
from ipfs_datasets_py.logic.backends.process import BoundedToolRunner


FAMILIES = {"dcec", "tdfol", "event_calculus", "frame_logic", "datalog", "horn_chc",
            "transition_system", "higher_order"}


def _resign_advice(advice):
    advice["advice_sha256"] = hashlib.sha256(advisor._encoded({k: v for k, v in advice.items()
        if k != "advice_sha256"})).hexdigest()


def test_actual_consumer_adds_families_without_retraining_or_replacing_native_views(roundtrip_checkpoint, monkeypatch):
    loaded = native.load_intent_roundtrip_checkpoint(roundtrip_checkpoint)
    paths = [Path(roundtrip_checkpoint["path"]), Path(loaded["backend_descriptor"]["path"])]
    before = [p.read_bytes() for p in paths]
    base = native.prepare_roundtrip_intent_instruction(PROBE, roundtrip_checkpoint)
    def no_process(*args, **kwargs):
        raise AssertionError("ordinary advice must not execute an external prover")
    monkeypatch.setattr(BoundedToolRunner, "run", no_process)
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint)
    assert advice["status"] == "semantic_candidate_advice"
    report = advice["report"]
    assert report["schema"] == extended_preplanning.SCHEMA
    assert report["base_report_sha256"] == base["report_sha256"]
    assert report["learned"] == base["learned"]
    assert report["candidate_intent_ir"] == base["candidate_intent_ir"]
    assert report["projections"] == base["projections"]
    extended = report["extended_projections"]
    assert {p["family_id"] for p in extended["projections"]} == FAMILIES
    assert len(extended["projections"]) == 9
    assert extended["native_targets"] == base["projections"]
    assert extended["external_backend_calls"] == extended["provider_calls"] == 0
    assert extended["training_executed"] is False
    event = next(r for r in extended["projections"] if r["family_id"] == "event_calculus")
    assert event["status"] == "unsupported"
    assert any(r["profile_id"] == "tla_plus" and r["status"] == "partial" for r in extended["projections"])
    assert len(advisor._encoded(report)) <= advisor.MAX_REPORT_BYTES
    assert before == [p.read_bytes() for p in paths]
    native.validate_roundtrip_intent_report(base, instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint)


@pytest.mark.parametrize("instruction,modality,operator", [
    (PROBE, "required", "O"),
    ("agent must not delete cache.", "prohibited", "F"),
    ("developer may update report.", "permitted", "P"),
])
def test_actual_weights_preserve_normative_operator_in_modal_families(roundtrip_checkpoint, instruction, modality, operator):
    advice = advisor.prepare_intent_advice(instruction=instruction, checkpoint_descriptor=roundtrip_checkpoint)
    assert advice["status"] == "semantic_candidate_advice"
    report = advice["report"]
    assert report["learned"]["frame"]["modality"] == modality
    for row in report["extended_projections"]["projections"]:
        if row["family_id"] not in {"dcec", "tdfol"}:
            continue
        formulas = row["representation"]["payload"]["formulas"]
        assert len(formulas) == 1 and formulas[0]["source"].startswith(operator + "(")
        assert formulas[0]["original_modality"] == modality
        assert formulas[0]["backend_proof_executed"] is False
        assert row["proof_authority"] is row["execution_authority"] is False


def test_compact_family_summary_enters_real_planner_request(original, roundtrip_checkpoint, monkeypatch):
    repository, instruction, state = original
    instruction.write_text(PROBE)
    selected = instruction.parent / "intent-roundtrip.json"
    selected.write_text(json.dumps(roundtrip_checkpoint))
    prepared = preparation.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=selected)
    sidecar = json.loads((state / "intent-advice.json").read_bytes())
    summary, advice = advisor.intent_planner_summary(sidecar, instruction=PROBE)
    assert summary and len(summary.encode()) <= 8192
    value = json.loads(summary)
    assert {p["family"] for p in value["extended_families"]} == FAMILIES
    assert value["frame"] == PROBE_FRAME
    assert value["external_provers_executed"] is False
    assert all("representation" not in row and "model_text" not in row for row in value["extended_families"])
    assert PROBE not in summary and "generated_text" not in summary
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] and result["intent_preplanning"]["supplied_to_router"]
    constraints = json.loads(prompt)["constraints"]["constraint_summaries"]
    assert summary in constraints
    assert prepared["query"] == PROBE == (repository / preparation.INSTRUCTION).read_text()
    assert all(value[key] is False for key in advisor.AUTHORITY_FIELDS)


def test_rehashed_projection_forgery_does_not_survive_full_sidecar_replay(roundtrip_checkpoint):
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint)
    report = advice["report"]
    extended = report["extended_projections"]
    row = next(p for p in extended["projections"] if p["family_id"] == "dcec")
    row["representation"]["source"] = "forged_formula"
    row["projection_sha256"] = hashlib.sha256(canonical_bytes({k: v for k, v in row.items()
        if k != "projection_sha256"})).hexdigest()
    extended["report_sha256"] = hashlib.sha256(canonical_bytes({k: v for k, v in extended.items()
        if k != "report_sha256"})).hexdigest()
    report["report_sha256"] = hashlib.sha256(canonical_bytes({k: v for k, v in report.items()
        if k != "report_sha256"})).hexdigest()
    _resign_advice(advice)
    with pytest.raises(ValueError, match="replay"):
        advisor.validate_intent_advice(advice, instruction=PROBE)
    summary, fallback = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary is None and fallback["status"] == "fail_open_advice_rejected"
    assert fallback["continue_planning"] is fallback["raw_instruction_preserved"] is True


@pytest.mark.parametrize("transient", [False, True])
def test_projection_exception_keeps_valid_base_candidate(roundtrip_checkpoint, monkeypatch, transient):
    real_project = extended_projections.project_intent_families
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(1)
        if not transient or len(calls) == 1:
            raise RuntimeError("optional projection fixture failure")
        return real_project(*args, **kwargs)
    monkeypatch.setattr(extended_projections, "project_intent_families", unavailable)
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint)
    assert advice["status"] == "semantic_candidate_advice"
    assert advice["report"]["extension_status"] == "fail_open_projection_error"
    assert advice["report"]["extended_projections"] is None
    assert advice["report"]["learned"]["frame"] == PROBE_FRAME
    assert advice["report"]["projections"]["projections"]
    summary, selected = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary and selected["status"] == "semantic_candidate_advice"
    assert json.loads(summary)["extended_families"] == []
    assert json.loads(summary)["frame"] == PROBE_FRAME


def test_previous_base_sidecar_remains_valid_with_same_frozen_checkpoint(roundtrip_checkpoint):
    base = native.prepare_roundtrip_intent_instruction(PROBE, roundtrip_checkpoint)
    old = advisor._base(PROBE, status=base["status"], report=base,
                        checkpoint_descriptor=roundtrip_checkpoint)
    assert advisor.validate_intent_advice(old, instruction=PROBE) == old
    summary, _ = advisor.intent_planner_summary(old, instruction=PROBE)
    assert summary and "extended_families" not in json.loads(summary)


def test_archive_includes_all_new_family_sources_and_unchanged_inert_checkpoint(tmp_path, roundtrip_checkpoint):
    inputs = _inputs(tmp_path)
    datasets_root = Path(native.__file__).resolve().parents[3]
    relative_names = ["logic/intent_ir/formalize/" + name + ".py" for name in (
        "projection_contracts", "extended_preplanning", "extended_projections", "modal_projections",
        "structural_projections", "state_projections", "lean_projection")]
    relative_names.append("logic/backends/tla/compiler.py")
    for relative in relative_names:
        target = inputs["datasets"] / "ipfs_datasets_py" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((datasets_root / relative).read_bytes())
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint, **inputs)
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        for relative in relative_names:
            stored = archive.extractfile("datasets/ipfs_datasets_py/" + relative).read()
            assert stored == (datasets_root / relative).read_bytes()
        checkpoint = archive.extractfile(deployment.INTENT_ROUNDTRIP_PATH).read()
        assert hashlib.sha256(checkpoint).hexdigest() == roundtrip_checkpoint["sha256"]
        names = archive.getnames()
        assert not any(name.endswith(".jar") or "corpus.json" in name for name in names)
    assert built["intent_checkpoint"]["runtime_training_steps"] == 0
    assert built["intent_checkpoint"]["source_training_data_included"] is False
