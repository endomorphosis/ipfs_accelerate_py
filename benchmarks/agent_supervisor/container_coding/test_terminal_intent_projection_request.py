"""Source-bound Intent modeling context survives native supervisor transport."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import (
    FullSupervisorAgent, intent_asset_arguments,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_autoencoder import _plan
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_roundtrip import (
    PROBE, PROBE_FRAME, roundtrip_checkpoint,  # noqa: F401
)
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor
from ipfs_datasets_py.logic.intent_ir.formalize import roundtrip as native
from ipfs_datasets_py.logic.intent_ir.formalize.extended_preplanning import CONTEXT_SCHEMA
from ipfs_datasets_py.logic.intent_ir.formalize.projection_contracts import canonical_bytes
from ipfs_datasets_py.logic.intent_ir.formalize.projection_request import make_intent_projection_request


def _request(checkpoint):
    document = native.frame_to_intent_ir(PROBE_FRAME, instruction=PROBE)
    return make_intent_projection_request(PROBE, document, checkpoint_sha256=checkpoint["sha256"],
        requested_families=["tdfol", "event_calculus", "frame_logic"], context={"modal": {
            "temporal_bindings": [{"statement_id": "goal", "operator": "always", "evidence_ref": "source"}],
            "event_occurrences": [{"action_id": "action", "time": 4, "evidence_ref": "source"}],
        }})


def _write(path, value):
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _resign(request):
    request["request_sha256"] = hashlib.sha256(canonical_bytes(
        {k: v for k, v in request.items() if k != "request_sha256"})).hexdigest()
    return request


def _assert_base_retained(advice):
    assert advice["status"] == "semantic_candidate_advice"
    assert advice["continue_planning"] is advice["raw_instruction_preserved"] is True
    assert advice["report"]["schema"] == CONTEXT_SCHEMA
    assert advice["report"]["learned"]["frame"] == PROBE_FRAME
    assert advice["report"]["extension_status"] == "fail_open_projection_error"
    assert advice["report"]["extended_projections"] is None
    assert advice["report"]["projection_request"] is None
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    summary, selected = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary and selected == advice
    assert json.loads(summary)["extended_families"] == []
    assert all(advice[key] is False for key in advisor.AUTHORITY_FIELDS)


def test_source_bound_temporal_context_reaches_existing_goal_planner(original, roundtrip_checkpoint, monkeypatch):
    repository, instruction, state = original
    instruction.write_text(PROBE)
    checkpoint_path, _ = _write(instruction.parent / "checkpoint.json", roundtrip_checkpoint)
    request = _request(roundtrip_checkpoint)
    request_path, request_file_sha = _write(instruction.parent / "projection-request.json", request)
    prepared = preparation.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=checkpoint_path, intent_projection_request=request_path,
        intent_projection_request_sha256=request_file_sha)
    advice = json.loads((state / "intent-advice.json").read_bytes())
    assert advice["report"]["schema"] == CONTEXT_SCHEMA
    assert advice["report"]["projection_request"] == request
    extensions = {r["family_id"]: r for r in advice["report"]["extended_projections"]["projections"]}
    assert set(extensions) == set(request["requested_families"])
    temporal = extensions["tdfol"]["representation"]["payload"]["formulas"][0]
    assert temporal["source"].startswith("□(O(")
    assert temporal["temporal_scope"]["evidence_verified"] is False
    event = extensions["event_calculus"]["representation"]["payload"]["formulas"][0]
    assert event["semantic_role"] == "caller_declared_occurrence_not_an_observation_by_this_projector"
    assert event["evidence_verified"] is False
    summary, _ = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary and len(summary.encode()) <= 8192
    assert {r["family"] for r in json.loads(summary)["extended_families"]} == set(extensions)
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] and result["intent_preplanning"]["supplied_to_router"]
    assert summary in json.loads(prompt)["constraints"]["constraint_summaries"]
    assert prepared["query"] == PROBE == (repository / preparation.INSTRUCTION).read_text()
    assert "Public instruction.md (the complete authorized task):\n" + PROBE in json.loads(prompt)["constraints"]["constraint_summaries"]
    assert all(advice[key] is False for key in advisor.AUTHORITY_FIELDS)


@pytest.mark.parametrize("field", ["instruction_sha256", "source_ir_sha256", "checkpoint_sha256"])
def test_validly_rehashed_wrong_source_ir_or_checkpoint_request_retains_base(roundtrip_checkpoint, field):
    request = _request(roundtrip_checkpoint)
    request[field] = "a" * 64
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint,
                                           projection_request=_resign(request))
    _assert_base_retained(advice)


@pytest.mark.parametrize("defect", ["drop_to_default", "substitute_request"])
def test_frontend_cannot_silently_drop_or_substitute_selected_context(roundtrip_checkpoint, monkeypatch, defect):
    from ipfs_datasets_py.logic.intent_ir.formalize import extended_preplanning
    original_prepare = extended_preplanning.prepare_extended_intent_instruction
    request = _request(roundtrip_checkpoint)
    replacement = deepcopy(request)
    replacement["context"]["modal"]["temporal_bindings"][0]["operator"] = "eventually"
    _resign(replacement)
    def altered(instruction, checkpoint_descriptor=None, **kwargs):
        return original_prepare(instruction, checkpoint_descriptor,
            projection_request=None if defect == "drop_to_default" else replacement)
    monkeypatch.setattr(extended_preplanning, "prepare_extended_intent_instruction", altered)
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint,
                                           projection_request=request)
    assert advice["status"] == "fail_open_optional_error"
    assert advice["report"] is None
    assert advice["continue_planning"] is advice["raw_instruction_preserved"] is True
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    summary, _ = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary is None


@pytest.mark.parametrize("defect", ["missing", "symlink", "hash", "missing_pin", "invalid_json", "oversized", "null"])
def test_optional_request_transport_failures_preserve_replayed_base(tmp_path, roundtrip_checkpoint, defect):
    request_path, digest = _write(tmp_path / "request.json", _request(roundtrip_checkpoint))
    if defect == "missing":
        request_path.unlink()
    elif defect == "symlink":
        target = tmp_path / "actual.json"
        request_path.rename(target)
        request_path.symlink_to(target)
    elif defect == "hash":
        digest = "b" * 64
    elif defect == "missing_pin":
        digest = None
    elif defect in {"invalid_json", "oversized", "null"}:
        request_path.write_bytes({"invalid_json": b"{", "oversized": b" " * 65537, "null": b"null"}[defect])
        digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint,
        projection_request_path=request_path, projection_request_sha256=digest)
    _assert_base_retained(advice)


def test_optional_file_failure_keeps_raw_planning_admission(original, roundtrip_checkpoint, monkeypatch):
    repository, instruction, state = original
    instruction.write_text(PROBE)
    checkpoint_path, _ = _write(instruction.parent / "checkpoint.json", roundtrip_checkpoint)
    prepared = preparation.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=checkpoint_path,
        intent_projection_request=instruction.parent / "missing-request.json",
        intent_projection_request_sha256="c" * 64)
    advice = json.loads((state / "intent-advice.json").read_bytes())
    _assert_base_retained(advice)
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] and prepared["query"] == PROBE
    assert "Public instruction.md (the complete authorized task):\n" + PROBE in json.loads(prompt)["constraints"]["constraint_summaries"]


def test_request_archive_relocation_keeps_exact_file_pin_and_runtime_source_binding(tmp_path, roundtrip_checkpoint):
    request = _request(roundtrip_checkpoint)
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint,
        intent_projection_request=request, **_inputs(tmp_path))
    binding = built["intent_projection_request"]
    assert built["task_inputs_in_archive"] is built["task_modeling_premises_included"] is True
    assert binding["request_sha256"] == request["request_sha256"]
    assert binding["checkpoint_sha256"] == roundtrip_checkpoint["sha256"]
    restored = tmp_path / "restored"
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        raw = archive.extractfile(deployment.INTENT_PROJECTION_REQUEST_PATH).read()
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
        assert json.loads(raw) == request and len(raw) == binding["bytes"]
        assert binding["sha256"] != request["request_sha256"]  # Two different documented digest scopes.
        assets = [name for name in archive.getnames() if name.startswith("models/intent-")]
        assert set(assets) == {deployment.INTENT_PROJECTION_REQUEST_PATH,
            deployment.INTENT_CHECKPOINT_DESCRIPTOR, deployment.INTENT_ROUNDTRIP_PATH,
            "models/intent-autoencoder/model/candidate.json"}
        for name in assets:
            path = restored / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(name).read())
    descriptor = json.loads((restored / deployment.INTENT_CHECKPOINT_DESCRIPTOR).read_bytes())
    descriptor["path"] = str(restored / deployment.INTENT_ROUNDTRIP_PATH)
    selected = restored / deployment.INTENT_PROJECTION_REQUEST_PATH
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=descriptor,
        projection_request_path=selected, projection_request_sha256=binding["sha256"])
    assert advice["report"]["projection_request"] == request
    assert advice["report"]["extension_status"] == "projected_with_explicit_frontiers"
    selected.write_bytes(selected.read_bytes() + b" ")
    _assert_base_retained(advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=descriptor,
        projection_request_path=selected, projection_request_sha256=binding["sha256"]))


def test_no_request_keeps_original_archive_and_harbor_contract(tmp_path, roundtrip_checkpoint):
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint,
        **_inputs(tmp_path))
    assert built["task_inputs_in_archive"] is False
    assert "task_modeling_premises_included" not in built
    assert "intent_projection_request" not in built
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        assert deployment.INTENT_PROJECTION_REQUEST_PATH not in archive.getnames()
    arguments, observation = intent_asset_arguments(built)
    assert arguments == ["--intent-checkpoint-descriptor", deployment.ROOT + "/" + deployment.INTENT_CHECKPOINT_DESCRIPTOR]
    assert "projection_request" not in observation


@pytest.mark.parametrize("defect", ["no_checkpoint", "different_checkpoint", "legacy_checkpoint", "extra_request_field"])
def test_bundle_rejects_incompatible_request_before_creating_archive(tmp_path, roundtrip_checkpoint, defect):
    request = _request(roundtrip_checkpoint)
    checkpoint = roundtrip_checkpoint
    if defect == "no_checkpoint":
        checkpoint = None
    elif defect == "different_checkpoint":
        request["checkpoint_sha256"] = "a" * 64
        _resign(request)
    elif defect == "legacy_checkpoint":
        # Exercise the request transport owner independently; a legacy model
        # cannot qualify contextual semantic interpretation.
        checkpoint = {**roundtrip_checkpoint, "schema": "intent-projection-feature-checkpoint/v1"}
    else:
        request["proof_authority"] = True
        _resign(request)
    with pytest.raises(ValueError):
        deployment._intent_projection_request_assets(request, checkpoint)
    assert not (tmp_path / "archive").exists()


@pytest.mark.parametrize("field,value", [
    ("path", "../../foreign.json"), ("schema", "other/v1"), ("sha256", "not-a-digest"),
    ("bytes", True), ("bytes", 65537), ("checkpoint_sha256", "a" * 64),
    ("proof_authority", True),
])
def test_harbor_rejects_retargeted_or_malformed_request_binding(tmp_path, roundtrip_checkpoint, field, value):
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint,
        intent_projection_request=_request(roundtrip_checkpoint), **_inputs(tmp_path))
    altered = deepcopy(built)
    altered["intent_projection_request"][field] = value
    with pytest.raises(ValueError, match="request binding"):
        intent_asset_arguments(altered)


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("disabled", [False, True])
def test_harbor_forwards_selected_request_in_both_arms_and_respects_ablation(tmp_path, roundtrip_checkpoint, arm, disabled):
    from harbor.models.agent.context import AgentContext
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint,
        intent_projection_request=_request(roundtrip_checkpoint), **_inputs(tmp_path))
    commands = []
    report = {"task_completed": False, "seconds": 0, "phases": {}, "provider_invocations": []}
    class Environment:
        async def upload_file(self, *args): pass
        async def exec(self, **kwargs):
            commands.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol",
        runtime_archive=str(tmp_path / "archive"), arm=arm, disable_intent_autoencoder=disabled)
    context = AgentContext()
    asyncio.run(agent.run(PROBE, Environment(), context))
    command = next(row for row in commands if "--arm" in row)
    assert command[command.index("--arm") + 1] == arm
    assert ("--intent-projection-request" in command) is not disabled
    assert ("--intent-projection-request-sha256" in command) is not disabled
    if not disabled:
        assert command[command.index("--intent-projection-request") + 1] == deployment.ROOT + "/" + deployment.INTENT_PROJECTION_REQUEST_PATH
        assert command[command.index("--intent-projection-request-sha256") + 1] == built["intent_projection_request"]["sha256"]
        assert context.metadata["intent_model_assets"]["projection_request"]["request_sha256"] == built["intent_projection_request"]["request_sha256"]


def test_bundle_cli_accepts_explicit_request_with_matching_checkpoint(tmp_path, roundtrip_checkpoint, monkeypatch, capsys):
    inputs = _inputs(tmp_path)
    checkpoint_path, _ = _write(tmp_path / "checkpoint.json", roundtrip_checkpoint)
    request_path, _ = _write(tmp_path / "request.json", _request(roundtrip_checkpoint))
    output = tmp_path / "archive"
    argv = ["terminal_deployment", "bundle", "--output", str(output)]
    for key, value in inputs.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    argv += ["--intent-checkpoint-descriptor", str(checkpoint_path), "--intent-projection-request", str(request_path)]
    monkeypatch.setattr(sys, "argv", argv)
    deployment.main()
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert json.loads(capsys.readouterr().out)["archive_sha256"] == manifest["archive_sha256"]
    assert manifest["intent_projection_request"]["request_sha256"] == _request(roundtrip_checkpoint)["request_sha256"]


@pytest.mark.parametrize("entrypoint", ["prepare", "driver"])
def test_cli_forwards_request_path_and_file_digest(entrypoint, tmp_path, monkeypatch, capsys):
    captured = {}
    request_path = tmp_path / "request.json"
    arguments = ["--instruction", str(tmp_path / "instruction.md"), "--state", str(tmp_path / "state"),
                 "--intent-projection-request", str(request_path), "--intent-projection-request-sha256", "c" * 64]
    def observe(**kwargs):
        captured.update(kwargs)
        return {"task_completed": True, "arm": "no-index", "seconds": 0, "provider_invocations": []}
    if entrypoint == "prepare":
        monkeypatch.setattr(preparation, "prepare", observe)
        monkeypatch.setattr(sys, "argv", ["prepare", "prepare", "--repository", str(tmp_path), *arguments])
        preparation.main()
    else:
        monkeypatch.setattr(driver, "run", observe)
        monkeypatch.setattr(sys, "argv", ["supervisor", "--arm", "no-index", *arguments])
        assert driver.main() == 0
    assert captured["intent_projection_request"] == request_path
    assert captured["intent_projection_request_sha256"] == "c" * 64
    assert json.loads(capsys.readouterr().out)
