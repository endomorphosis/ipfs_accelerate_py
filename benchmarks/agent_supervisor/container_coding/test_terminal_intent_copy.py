"""Independent consumer boundaries for the optional copied-token Intent codec.

The transport fixtures below test installed dispatch and schema boundaries;
they make no learned-accuracy claim. Real weight consumption is tested separately.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from types import ModuleType

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import intent_asset_arguments
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_autoencoder import _plan
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor


PROBE = "agent must inspect quasar-cache."
FRAME = {"actor": "agent", "action": "inspect", "object": "quasar-cache", "modality": "required"}
DESCRIPTOR = {"schema": "intent-copy-roundtrip-checkpoint/v1", "path": "/not-loaded-by-fixture/manifest.json",
              "sha256": "a" * 64}


def _report():
    return {"schema": "intent-instruction-copy-roundtrip/v1", "status": "semantic_candidate_advice",
        "instruction_sha256": hashlib.sha256(PROBE.encode()).hexdigest(), "instruction_bytes": len(PROBE.encode()),
        "checkpoint_sha256": DESCRIPTOR["sha256"], "learned": {"frame": dict(FRAME),
            "encoder": {"fixture_only": True}, "decoder": {"fixture_only": True}, "normalized_text": PROBE},
        "candidate_intent_ir": {"fixture_only": True}, "projections": {"projections": []},
        "extended_projections": {"projections": []}, "extension_status": "projected_with_explicit_frontiers",
        "projection_request": None, "gaps": ["fixture_not_numerical_evidence"], "report_sha256": "b" * 64,
        "continue_planning": True, "raw_instruction_preserved": True,
        "authority": "unverified_candidate_only", **{key: False for key in advisor.AUTHORITY_FIELDS}}


@pytest.fixture
def installed_copy_fixture(monkeypatch):
    report = _report()
    calls = []
    module = ModuleType("ipfs_datasets_py.logic.intent_ir.formalize.copy_roundtrip")
    def prepare(instruction, checkpoint_descriptor=None, projection_request=None):
        calls.append(("prepare", instruction, deepcopy(checkpoint_descriptor), deepcopy(projection_request)))
        return deepcopy(report)
    def validate(candidate, *, instruction, checkpoint_descriptor=None):
        calls.append(("validate", instruction, deepcopy(checkpoint_descriptor)))
        if candidate != report or instruction != PROBE or checkpoint_descriptor != DESCRIPTOR:
            raise ValueError("fixture replay differs")
        return candidate
    module.prepare_copy_intent_instruction = prepare
    module.validate_copy_intent_report = validate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return report, calls, module


def test_copy_dispatch_and_summary_are_explicit_and_non_authoritative(installed_copy_fixture):
    _, calls, _ = installed_copy_fixture
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    assert advice["status"] == "semantic_candidate_advice"
    assert advice["checkpoint_descriptor"] == DESCRIPTOR
    assert calls[0] == ("prepare", PROBE, DESCRIPTOR, None)
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    summary, selected = advisor.intent_planner_summary(advice, instruction=PROBE)
    value = json.loads(summary)
    assert selected == advice and value["frame"] == FRAME
    assert value["extended_families"] == [] and value["projection_request_sha256"] is None
    assert value["checkpoint_sha256"] == DESCRIPTOR["sha256"]
    assert value["external_provers_executed"] is value["reconstruction_is_proof"] is False
    assert all(value[key] is False for key in advisor.AUTHORITY_FIELDS)
    assert PROBE not in summary and "fixture_only" not in summary
    assert "encoder" not in value and "decoder" not in value


@pytest.mark.parametrize("schema", [advisor.ROUNDTRIP_REPORT_SCHEMA, advisor.EXTENDED_REPORT_SCHEMA,
    advisor.CONTEXT_REPORT_SCHEMA, "intent-instruction-copy-roundtrip/v2"])
def test_copy_checkpoint_cannot_substitute_another_report_schema(installed_copy_fixture, schema):
    report, _, _ = installed_copy_fixture
    report["schema"] = schema
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    assert advice["status"] == "fail_open_optional_error"
    assert advice["report"] is advice["checkpoint_descriptor"] is None
    assert advice["continue_planning"] is advice["raw_instruction_preserved"] is True


@pytest.mark.parametrize("schema", [advisor.ROUNDTRIP_CHECKPOINT_SCHEMA, "intent-projection-feature-checkpoint/v1"])
def test_copy_report_cannot_be_retargeted_to_another_checkpoint_schema(installed_copy_fixture, schema):
    report, _, _ = installed_copy_fixture
    descriptor = {**DESCRIPTOR, "schema": schema}
    advice = advisor._base(PROBE, status=report["status"], report=report, checkpoint_descriptor=descriptor)
    with pytest.raises(ValueError, match="schemas differ"):
        advisor.validate_intent_advice(advice, instruction=PROBE)


def test_copy_checkpoint_identity_is_bound_even_if_adapter_validation_accepts(installed_copy_fixture):
    report, _, module = installed_copy_fixture
    module.validate_copy_intent_report = lambda *args, **kwargs: args[0]
    report["checkpoint_sha256"] = "c" * 64
    advice = advisor._base(PROBE, status=report["status"], report=report, checkpoint_descriptor=DESCRIPTOR)
    with pytest.raises(ValueError, match="different checkpoint"):
        advisor.validate_intent_advice(advice, instruction=PROBE)


def test_copy_projection_request_cannot_be_silently_dropped(installed_copy_fixture):
    report, _, _ = installed_copy_fixture
    request = {"request_sha256": "c" * 64}
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR,
        projection_request=request)
    assert advice["status"] == "fail_open_optional_error"
    report["projection_request"] = request
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR,
        projection_request=request)
    assert advice["status"] == "semantic_candidate_advice"
    summary, _ = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert json.loads(summary)["projection_request_sha256"] == request["request_sha256"]


def test_copy_optional_projection_failure_keeps_the_base_candidate(installed_copy_fixture):
    report, _, _ = installed_copy_fixture
    report.update(extended_projections=None, extension_status="fail_open_projection_error")
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR,
        projection_request={"invalid_fixture": True})
    assert advice["status"] == "semantic_candidate_advice"
    summary, _ = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert json.loads(summary)["extended_families"] == []


@pytest.mark.parametrize("key", advisor.AUTHORITY_FIELDS)
def test_rehashed_copy_advice_cannot_gain_authority(installed_copy_fixture, key):
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    advice[key] = True
    advice["advice_sha256"] = hashlib.sha256(advisor._encoded({k: v for k, v in advice.items()
        if k != "advice_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="authority binding"):
        advisor.validate_intent_advice(advice, instruction=PROBE)
    summary, fallback = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary is None and fallback["status"] == "fail_open_advice_rejected"


def test_copy_failure_and_advice_budget_preserve_original_instruction(installed_copy_fixture):
    report, _, _ = installed_copy_fixture
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    summary, fallback = advisor.intent_planner_summary(advice, instruction=PROBE, maximum_bytes=1)
    assert summary is None and fallback["status"] == "fail_open_advice_over_budget"
    assert fallback["instruction_sha256"] == hashlib.sha256(PROBE.encode()).hexdigest()
    report.update(status="fail_open_encoder_generation", candidate_intent_ir=None, projections=None,
        extended_projections=None, extension_status="not_applicable")
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    assert advice["status"] == "fail_open_encoder_generation"
    assert advice["checkpoint_descriptor"] == DESCRIPTOR
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    assert advisor.intent_planner_summary(advice, instruction=PROBE)[0] is None


def _binding():
    descriptor = {**DESCRIPTOR, "path": deployment.ROOT + "/" + deployment.INTENT_ROUNDTRIP_PATH}
    return {"intent_checkpoint": {"path": deployment.INTENT_ROUNDTRIP_PATH,
        "descriptor_path": deployment.INTENT_CHECKPOINT_DESCRIPTOR, "descriptor": descriptor,
        "sha256": descriptor["sha256"], "mode": "frozen_copy_roundtrip_inference",
        "runtime_training_steps": 0, "runtime_download_calls": 0, "source_training_data_included": False}}


def test_copy_runtime_binding_selects_the_pinned_descriptor():
    arguments, observation = intent_asset_arguments(_binding())
    assert arguments == ["--intent-checkpoint-descriptor", deployment.ROOT + "/" + deployment.INTENT_CHECKPOINT_DESCRIPTOR]
    assert observation["mode"] == "frozen_copy_roundtrip_inference"
    assert observation["execution_authority"] is False


@pytest.mark.parametrize("field,value", [("schema", "intent-roundtrip-checkpoint/v1"),
    ("path", "/tmp/retargeted.json"), ("sha256", "b" * 64)])
def test_copy_runtime_binding_rejects_schema_path_and_digest_retargeting(field, value):
    manifest = _binding()
    manifest["intent_checkpoint"]["descriptor"][field] = value
    with pytest.raises(ValueError):
        intent_asset_arguments(manifest)


def test_copy_checkpoint_can_transport_a_bound_projection_request():
    request = {"schema": "intent-projection-request/v1", "path": deployment.INTENT_PROJECTION_REQUEST_PATH,
        "bytes": 100, "sha256": "b" * 64, "request_sha256": "c" * 64,
        "checkpoint_sha256": DESCRIPTOR["sha256"], "instruction_sha256": "d" * 64, "source_ir_sha256": "e" * 64}
    manifest = {**_binding(), "intent_projection_request": request}
    arguments, observation = intent_asset_arguments(manifest)
    assert "--intent-projection-request" in arguments
    assert observation["projection_request"]["checkpoint_sha256"] == DESCRIPTOR["sha256"]
    request["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="request binding"):
        intent_asset_arguments(manifest)


REAL_PROBE = "agent must inspect quantumcache."
REAL_FRAME = {**FRAME, "object": "quantumcache"}


@pytest.fixture(scope="module")
def copy_checkpoint(tmp_path_factory):
    """Train an authored fixture; its nonce probe never enters fitted vocabulary."""
    import torch
    from ipfs_datasets_py.logic.intent_ir.formalize import copy_roundtrip as native
    from ipfs_datasets_py.logic.intent_ir.formalize import roundtrip as codec
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_paired_copy as paired

    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("copy-intent-consumer")
    rows = []
    for modality in codec.MODALS:
        for action in ("inspect", "update"):
            for obj in ("cache", "report", "buffer", "package"):
                frame = {"actor": "agent", "action": action, "object": obj, "modality": modality}
                text, sequence = codec.canonical_frame_text(frame), codec.frame_to_sequence(frame)
                index = len(rows)
                rows.extend([{"id": f"encode-{index}", "source": text, "target": sequence, "direction": "encode"},
                             {"id": f"decode-{index}", "source": sequence, "target": text, "direction": "decode"}])
    validation_frame = {**REAL_FRAME, "object": "validationprobe"}
    validation = [{"id": "validation-encode", "source": codec.canonical_frame_text(validation_frame),
        "target": codec.frame_to_sequence(validation_frame), "direction": "encode"}]
    backend = paired.train_paired_copy(rows, validation, output_dir=root / "model", epochs=160,
        max_seconds=60, hidden_size=48, embedding_dim=24)
    descriptor = native.register_intent_copy_checkpoint(backend, output=root,
        corpus_sha256=hashlib.sha256(codec._raw(rows)).hexdigest())
    assert "quantumcache" not in paired.load_paired_copy(backend)["config"]["vocabulary"]
    return descriptor


def test_actual_copy_weights_reach_supervisor_and_reload_without_target_access(copy_checkpoint, tmp_path, monkeypatch):
    from ipfs_datasets_py.logic.intent_ir.formalize import copy_roundtrip as native
    from ipfs_datasets_py.logic.intent_ir.formalize import roundtrip as codec
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_paired_copy as paired

    loaded = native.load_intent_copy_checkpoint(copy_checkpoint)
    paths = [Path(copy_checkpoint["path"]), Path(loaded["backend_descriptor"]["path"])]
    before = [path.read_bytes() for path in paths]
    calls, original = [], paired.infer_paired_copy
    def traced(descriptor, source, direction, *args, **kwargs):
        result = original(descriptor, source, direction, *args, **kwargs)
        calls.append((deepcopy(descriptor), source, direction, result["checkpoint_weights_sha256"]))
        return result
    monkeypatch.setattr(paired, "infer_paired_copy", traced)
    advice = advisor.prepare_intent_advice(instruction=REAL_PROBE, checkpoint_descriptor=copy_checkpoint)
    assert advice["status"] == "semantic_candidate_advice", advice
    report = advice["report"]
    assert report["learned"]["frame"] == REAL_FRAME
    assert report["learned"]["normalized_text"] == REAL_PROBE
    assert report["schema"] == advisor.COPY_REPORT_SCHEMA
    assert report["training_steps"] == report["provider_calls"] == report["download_calls"] == 0
    for direction in ("encoder", "decoder"):
        inference = report["learned"][direction]
        assert inference["input_oov_tokens"] == ["quantumcache"]
        assert inference["input_coverage_complete"] is True
        assert inference["target_access"] is inference["training_executed"] is False
        copied = [row for row in inference["copy_trace"] if row["token"] == "quantumcache"]
        assert copied and all(row["extended_copy_token"] and row["copy_source_positions"] for row in copied)
    summary, _ = advisor.intent_planner_summary(advice, instruction=REAL_PROBE)
    assert json.loads(summary)["frame"] == REAL_FRAME
    sidecar = tmp_path / "advice.json"
    sidecar.write_bytes(advisor._encoded(advice))
    restored = advisor.load_intent_advice(path=sidecar,
        expected_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(), instruction=REAL_PROBE)
    assert restored == advice
    assert len(calls) >= 6
    for selected, source, direction, state_sha in calls:
        assert selected == loaded["backend_descriptor"]
        assert state_sha == loaded["backend"]["training"]["final_state_sha256"]
        assert source == (REAL_PROBE if direction == "encode" else codec.frame_to_sequence(REAL_FRAME))
    assert before == [path.read_bytes() for path in paths]


def test_actual_copy_without_copy_branch_cannot_reconstruct_unseen_literal(copy_checkpoint):
    from ipfs_datasets_py.logic.intent_ir.formalize import copy_roundtrip as native
    from ipfs_datasets_py.logic.intent_ir.formalize import roundtrip as codec
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_paired_copy as paired

    loaded = native.load_intent_copy_checkpoint(copy_checkpoint)
    before = Path(loaded["backend_descriptor"]["path"]).read_bytes()
    encoded = paired.infer_paired_copy(loaded["backend_descriptor"], REAL_PROBE, "encode", weight_ablation="disable_copy")
    assert encoded["input_coverage_complete"] is False and encoded["uncovered_input_tokens"] == ["quantumcache"]
    assert "quantumcache" not in encoded["tokens"]
    assert native.generation_usable(encoded) is False
    document = codec.frame_to_intent_ir(REAL_FRAME, instruction="independent provenance string")
    decoded = native.decode_copy_intent_text(copy_checkpoint, document, weight_ablation="disable_copy")
    assert "quantumcache" not in decoded["tokens"]
    assert Path(loaded["backend_descriptor"]["path"]).read_bytes() == before


def test_actual_copy_candidate_enters_goal_planner_without_replacing_instruction(copy_checkpoint, original, monkeypatch):
    repository, instruction, state = original
    instruction.write_text(REAL_PROBE)
    descriptor_file = instruction.parent / "copy-descriptor.json"
    descriptor_file.write_text(json.dumps(copy_checkpoint))
    prepared = preparation.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=descriptor_file)
    assert prepared["intent_preplanning"]["status"] == "semantic_candidate_advice"
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] and result["intent_preplanning"]["supplied_to_router"]
    assert prepared["query"] == REAL_PROBE == (repository / preparation.INSTRUCTION).read_text()
    summaries = json.loads(prompt)["constraints"]["constraint_summaries"]
    candidates = [json.loads(value) for value in summaries if value.startswith('{"advice_sha256"')]
    assert len(candidates) == 1 and candidates[0]["frame"] == REAL_FRAME
    assert all(candidates[0][key] is False for key in advisor.AUTHORITY_FIELDS)


def test_rehashed_actual_copy_trace_does_not_survive_numerical_replay(copy_checkpoint):
    advice = advisor.prepare_intent_advice(instruction=REAL_PROBE, checkpoint_descriptor=copy_checkpoint)
    report = advice["report"]
    trace = next(row for row in report["learned"]["encoder"]["copy_trace"] if row["token"] == "quantumcache")
    trace["copy_source_positions"] = [0]
    report["report_sha256"] = hashlib.sha256(advisor._encoded({k: v for k, v in report.items()
        if k != "report_sha256"})).hexdigest()
    advice["advice_sha256"] = hashlib.sha256(advisor._encoded({k: v for k, v in advice.items()
        if k != "advice_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="replay"):
        advisor.validate_intent_advice(advice, instruction=REAL_PROBE)
    summary, fallback = advisor.intent_planner_summary(advice, instruction=REAL_PROBE)
    assert summary is None and fallback["status"] == "fail_open_advice_rejected"


def test_actual_copy_model_relocates_with_inert_weights_and_installed_sources(copy_checkpoint, tmp_path):
    from ipfs_datasets_py.logic.intent_ir.formalize import copy_roundtrip as native
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_paired_copy as paired

    inputs = _inputs(tmp_path)
    roots = Path(native.__file__).resolve().parents[3]
    source_names = ["logic/intent_ir/formalize/copy_roundtrip.py",
                    "optimizers/logic_theorem_optimizer/autoencoder_paired_copy.py"]
    for relative in source_names:
        destination = inputs["datasets"] / "ipfs_datasets_py" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((roots / relative).read_bytes())
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=copy_checkpoint, **inputs)
    assert built["intent_checkpoint"]["mode"] == "frozen_copy_roundtrip_inference"
    restored_root = tmp_path / "relocated"
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        for relative in source_names:
            assert archive.extractfile("datasets/ipfs_datasets_py/" + relative).read() == (roots / relative).read_bytes()
        names = [name for name in archive.getnames() if name.startswith("models/intent-autoencoder/")]
        assert set(names) == {deployment.INTENT_ROUNDTRIP_PATH, "models/intent-autoencoder/model/candidate.json"}
        for name in names:
            target = restored_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(name).read())
        selected = json.load(archive.extractfile(deployment.INTENT_CHECKPOINT_DESCRIPTOR))
        assert not any("corpus.json" in name or "training-receipt" in name for name in archive.getnames())
    selected["path"] = str(restored_root / deployment.INTENT_ROUNDTRIP_PATH)
    advice = advisor.prepare_intent_advice(instruction=REAL_PROBE, checkpoint_descriptor=selected)
    assert advice["status"] == "semantic_candidate_advice" and advice["report"]["learned"]["frame"] == REAL_FRAME
    assert native.load_intent_copy_checkpoint(selected)["backend_descriptor"]["schema"] == paired.SCHEMA
    arguments, observation = intent_asset_arguments(built)
    assert arguments and observation["mode"] == "frozen_copy_roundtrip_inference"
