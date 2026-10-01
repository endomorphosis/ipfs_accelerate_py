"""Real paired weights remain advisory through native planning and packaging."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import intent_asset_arguments
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_autoencoder import _plan
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor
from ipfs_datasets_py.logic.intent_ir.formalize import roundtrip as native
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_paired_text as paired


PROBE = "agent must read report."
PROBE_FRAME = {"actor": "agent", "action": "read", "object": "report", "modality": "required"}


@pytest.fixture(scope="module")
def roundtrip_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("roundtrip-consumer")
    frames = [PROBE_FRAME,
        {"actor": "agent", "action": "delete", "object": "cache", "modality": "prohibited"},
        {"actor": "developer", "action": "update", "object": "report", "modality": "permitted"},
        {"actor": "developer", "action": "validate", "object": "cache", "modality": "recommended"},
        {"actor": "agent", "action": "validate", "object": "report", "modality": "intended"}]
    heldout = [{"actor": "agent", "action": "validate", "object": "cache", "modality": "required"}]
    def examples(rows, prefix):
        pairs = []
        for index, frame in enumerate(rows):
            source, target = native.canonical_frame_text(frame), native.frame_to_sequence(frame)
            pairs.extend([{"id": f"{prefix}-{index}-encode", "source": source, "target": target, "direction": "encode"},
                          {"id": f"{prefix}-{index}-decode", "source": target, "target": source, "direction": "decode"}])
        return pairs
    training, tuning = examples(frames, "train"), examples(heldout, "tune")
    backend = paired.train_paired_text(training, tuning, output_dir=root / "model",
        epochs=140, max_seconds=40, hidden_size=48, embedding_dim=24)
    descriptor = native.register_intent_roundtrip_checkpoint(backend, output=root,
        corpus_sha256=hashlib.sha256(json.dumps(training, sort_keys=True).encode()).hexdigest())
    report = native.prepare_roundtrip_intent_instruction(PROBE, descriptor)
    assert report["status"] == "semantic_candidate_advice", report
    assert report["learned"]["frame"] == PROBE_FRAME
    return descriptor


def test_real_roundtrip_inference_reaches_bounded_candidate_summary(roundtrip_checkpoint):
    loaded = native.load_intent_roundtrip_checkpoint(roundtrip_checkpoint)
    files = [Path(roundtrip_checkpoint["path"]), Path(loaded["backend_descriptor"]["path"])]
    before = [path.read_bytes() for path in files]
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint)
    assert advice["status"] == "semantic_candidate_advice"
    report = advice["report"]
    assert report["learned"]["frame"] == PROBE_FRAME
    assert report["candidate_intent_ir"] and report["projections"]["projections"]
    assert report["training_steps"] == report["download_calls"] == report["provider_calls"] == 0
    summary, selected = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert selected == advice and summary and len(summary.encode()) <= 8192
    value = json.loads(summary)
    assert value["frame"] == PROBE_FRAME and value["mode"] == "semantic_roundtrip_candidate"
    assert value["source_semantics_verified"] is value["reconstruction_is_proof"] is False
    assert PROBE not in summary and "normalized_text" not in summary and "generated_tokens" not in summary
    assert "encoder" not in value and "decoder" not in value
    assert before == [path.read_bytes() for path in files]


def test_roundtrip_candidate_enters_existing_planner_without_replacing_instruction(original, roundtrip_checkpoint, monkeypatch):
    repository, instruction, state = original
    instruction.write_text(PROBE)
    descriptor_file = instruction.parent / "roundtrip-descriptor.json"
    descriptor_file.write_text(json.dumps(roundtrip_checkpoint))
    prepared = preparation.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=descriptor_file)
    assert prepared["intent_preplanning"]["status"] == "semantic_candidate_advice"
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] and result["intent_preplanning"]["supplied_to_router"]
    assert prepared["query"] == PROBE == (repository / preparation.INSTRUCTION).read_text()
    summaries = json.loads(prompt)["constraints"]["constraint_summaries"]
    candidates = [json.loads(value) for value in summaries if value.startswith('{"advice_sha256"')]
    assert candidates and candidates[0]["frame"] == PROBE_FRAME
    assert all(candidates[0][name] is False for name in advisor.AUTHORITY_FIELDS)


@pytest.mark.parametrize("instruction", ["unseen_symbol_xyz must read report.", "x " * 49])
def test_roundtrip_fail_open_retains_replayable_descriptor(roundtrip_checkpoint, instruction):
    advice = advisor.prepare_intent_advice(instruction=instruction, checkpoint_descriptor=roundtrip_checkpoint)
    assert advice["status"].startswith("fail_open_")
    assert advice["checkpoint_descriptor"] == roundtrip_checkpoint
    assert advisor.validate_intent_advice(advice, instruction=instruction) == advice
    summary, selected = advisor.intent_planner_summary(advice, instruction=instruction)
    assert summary is None and selected["continue_planning"]
    assert selected["instruction_sha256"] == hashlib.sha256(instruction.encode()).hexdigest()


def test_roundtrip_sidecar_numeric_or_frame_forgery_is_rejected(roundtrip_checkpoint):
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=roundtrip_checkpoint)
    advice["report"]["learned"]["frame"]["modality"] = "prohibited"
    report = advice["report"]
    report["report_sha256"] = native._sha(native._raw({key: value for key, value in report.items() if key != "report_sha256"}))
    advice["advice_sha256"] = hashlib.sha256(advisor._encoded({key: value for key, value in advice.items() if key != "advice_sha256"})).hexdigest()
    with pytest.raises(ValueError, match="replay"):
        advisor.validate_intent_advice(advice, instruction=PROBE)
    summary, fallback = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary is None and fallback["status"] == "fail_open_advice_rejected"


def test_roundtrip_checkpoint_relocates_as_two_inert_files(tmp_path, roundtrip_checkpoint):
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint,
        **_inputs(tmp_path))
    binding = built["intent_checkpoint"]
    assert binding["path"] == deployment.INTENT_ROUNDTRIP_PATH
    assert binding["mode"] == "frozen_semantic_roundtrip_inference"
    restored = tmp_path / "restored"
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        names = [name for name in archive.getnames() if name.startswith("models/intent-autoencoder/")]
        assert sorted(names) == sorted([deployment.INTENT_ROUNDTRIP_PATH,
            "models/intent-autoencoder/model/candidate.json"])
        for name in names:
            target = restored / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(name).read())
        descriptor = json.load(archive.extractfile(deployment.INTENT_CHECKPOINT_DESCRIPTOR))
    descriptor["path"] = str(restored / deployment.INTENT_ROUNDTRIP_PATH)
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=descriptor)
    assert advice["status"] == "semantic_candidate_advice" and advice["report"]["learned"]["frame"] == PROBE_FRAME
    args, observation = intent_asset_arguments(built)
    assert args == ["--intent-checkpoint-descriptor", deployment.ROOT + "/" + deployment.INTENT_CHECKPOINT_DESCRIPTOR]
    assert observation["mode"] == "frozen_semantic_roundtrip_inference"


@pytest.mark.parametrize("field,value", [("schema", "intent-projection-feature-checkpoint/v1"),
    ("path", "/tmp/foreign.json")])
def test_roundtrip_transport_rejects_descriptor_retargeting(tmp_path, roundtrip_checkpoint, field, value):
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=roundtrip_checkpoint,
        **_inputs(tmp_path))
    changed = deepcopy(built)
    changed["intent_checkpoint"]["descriptor"][field] = value
    with pytest.raises(ValueError):
        intent_asset_arguments(changed)


@pytest.mark.parametrize("descriptor", [
    {"schema": advisor.ROUNDTRIP_CHECKPOINT_SCHEMA, "path": "/missing", "sha256": "0" * 64, "extra": "x" * 40_000},
    {"schema": advisor.ROUNDTRIP_CHECKPOINT_SCHEMA, "path": "x" * 5000, "sha256": "0" * 64},
])
def test_malformed_direct_descriptor_is_not_persisted_in_fallback(descriptor):
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=descriptor)
    assert advice["status"] == "fail_open_optional_error" and advice["checkpoint_descriptor"] is None
    assert advice["report"] is None and len(advisor._encoded(advice)) < 2048
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
