"""Independent planner and archive boundaries for optional Intent scope advice.

The archive uses explicit inert transport fixtures, not learned-quality evidence.
Neither these fixtures nor the rejected instructions require model inference.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import intent_asset_arguments
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_autoencoder import _original_prompt, _plan
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor
from ipfs_datasets_py.logic.intent_ir.formalize import copy_roundtrip, extended_preplanning, instruction_scope


UNSUPPORTED = "Read ledger and delete cache."


def _forbid_native_calls(monkeypatch):
    calls = []

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("scope rejection must precede native model work")

    for module, names in (
        (copy_roundtrip, ("prepare_copy_intent_instruction", "validate_copy_intent_report",
                          "load_intent_copy_checkpoint")),
        (extended_preplanning, ("prepare_extended_intent_instruction", "validate_extended_intent_report")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, unexpected)
    return calls


@pytest.mark.parametrize("schema", [advisor.COPY_CHECKPOINT_SCHEMA, advisor.ROUNDTRIP_CHECKPOINT_SCHEMA])
def test_scope_rejection_keeps_real_planning_and_independent_domain_roots(original, monkeypatch, schema):
    repository, instruction, state = original
    instruction.write_text(UNSUPPORTED)
    descriptor = {"schema": schema, "path": str(instruction.parent / "unavailable-checkpoint.json"),
                  "sha256": "b" * 64}
    descriptor_path = instruction.parent / "selected-checkpoint.json"
    descriptor_path.write_text(json.dumps(descriptor))
    calls = _forbid_native_calls(monkeypatch)
    declared = []
    original_domains = prep.local.local_planning_domain_declarations

    def independently_declared(**kwargs):
        domains = original_domains(**kwargs)
        declared.append(deepcopy(domains))
        return domains

    monkeypatch.setattr(prep.local, "local_planning_domain_declarations", independently_declared)
    prepared = prep.prepare(repository=repository, instruction=instruction, state=state,
        intent_checkpoint_descriptor=descriptor_path)
    assert prepared["intent_preplanning"]["status"] == "fail_open_instruction_scope"
    assert prepared["intent_preplanning"]["before_goal_decomposition"] is True
    assert prepared["provider_calls"] == 0
    assert declared == [prepared["manifest"]["payload"]["planning_inputs"]["domain_declarations"]]
    expected_roots = {name + "_ir_root": prep.local.content_identity(declared[0][name])
                      for name in ("intent", "legal", "security")}
    assert {key: prepared["request"][key] for key in expected_roots} == expected_roots
    before_request = deepcopy(prepared["request"])
    result, prompt = _plan(state, prepared, monkeypatch)
    assert result["qualified"] is True
    assert result["intent_preplanning"]["status"] == "fail_open_instruction_scope"
    assert result["intent_preplanning"]["supplied_to_router"] is False
    assert prompt == _original_prompt(prepared)
    assert prepared["request"] == before_request
    assert prepared["query"] == instruction.read_text() == (repository / prep.INSTRUCTION).read_text() == UNSUPPORTED
    for name in ("intent-advice.json", "planner-intent-advice.json"):
        advice = json.loads((state / name).read_text())
        assert advice["checkpoint_descriptor"] == descriptor and advice["report"] is None
        assert advice["scope_report"]["eligible_for_inference"] is False
        assert all(advice[key] is False for key in advisor.AUTHORITY_FIELDS)
        assert advisor.validate_intent_advice(advice, instruction=UNSUPPORTED) == advice
    assert calls == []


def _import_file(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_scope_source_and_selected_descriptor_survive_archive_relocation(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    source_assets = {
        "datasets/ipfs_datasets_py/logic/intent_ir/formalize/instruction_scope.py": Path(instruction_scope.__file__),
        "source/ipfs_accelerate_py/agent_supervisor/runtime/intent_autoencoder_advisor.py": Path(advisor.__file__),
    }
    for archive_name, source in source_assets.items():
        category, relative = archive_name.split("/", 1)
        destination = inputs[category] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    # Authored opaque assets exercise transport identity only. Packaging's native
    # manifest loader is explicit here; the relocated scope path never loads it.
    checkpoint_root = tmp_path / "transport-model"
    (checkpoint_root / "model").mkdir(parents=True)
    backend_path = checkpoint_root / "model/candidate.json"
    backend_path.write_text('{"transport_fixture_only":true}')
    manifest = {"backend": {"schema": "shared-paired-copy-autoencoder/v1",
        "file": "model/candidate.json", "sha256": hashlib.sha256(backend_path.read_bytes()).hexdigest()}}
    manifest_path = checkpoint_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    descriptor = {"schema": advisor.COPY_CHECKPOINT_SCHEMA, "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    monkeypatch.setattr(copy_roundtrip, "load_intent_copy_checkpoint", lambda selected: {"manifest": manifest})
    built = deployment.build_runtime_archive(output=tmp_path / "archive", intent_checkpoint=descriptor, **inputs)
    assert built["intent_checkpoint"]["mode"] == "frozen_copy_roundtrip_inference"
    arguments, observation = intent_asset_arguments(built)
    assert arguments == ["--intent-checkpoint-descriptor", deployment.ROOT + "/" + deployment.INTENT_CHECKPOINT_DESCRIPTOR]
    assert observation["mode"] == "frozen_copy_roundtrip_inference"
    assert built["intent_checkpoint"]["runtime_training_steps"] == 0
    assert built["intent_checkpoint"]["runtime_download_calls"] == 0
    assert built["intent_checkpoint"]["source_training_data_included"] is False

    restored = tmp_path / "relocated"
    assets = [*source_assets, deployment.INTENT_CHECKPOINT_DESCRIPTOR, deployment.INTENT_ROUNDTRIP_PATH,
              str(Path(deployment.INTENT_ROUNDTRIP_PATH).parent / "model/candidate.json")]
    with tarfile.open(tmp_path / "archive/runtime.tar.gz") as archive:
        assert not any("corpus.json" in name or "training-receipt" in name for name in archive.getnames())
        for name in assets:
            raw = archive.extractfile(name).read()
            if name in source_assets:
                assert raw == source_assets[name].read_bytes()
            destination = restored / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
    selected = json.loads((restored / deployment.INTENT_CHECKPOINT_DESCRIPTOR).read_text())
    selected["path"] = str(restored / deployment.INTENT_ROUNDTRIP_PATH)
    assert selected["sha256"] == hashlib.sha256(Path(selected["path"]).read_bytes()).hexdigest()
    restored_scope = _import_file(instruction_scope.__name__, restored / next(iter(source_assets)), monkeypatch)
    restored_advisor = _import_file("_relocated_intent_scope_advisor",
        restored / "source/ipfs_accelerate_py/agent_supervisor/runtime/intent_autoencoder_advisor.py", monkeypatch)
    assert Path(restored_scope.__file__).is_relative_to(restored)
    assert Path(restored_advisor.__file__).is_relative_to(restored)
    calls = _forbid_native_calls(monkeypatch)
    advice = restored_advisor.prepare_intent_advice(instruction=UNSUPPORTED, checkpoint_descriptor=selected)
    assert advice["status"] == "fail_open_instruction_scope" and advice["report"] is None
    assert advice["checkpoint_descriptor"] == selected
    assert advice["scope_report"] == restored_scope.assess_intent_instruction_scope(UNSUPPORTED)
    assert restored_advisor.validate_intent_advice(advice, instruction=UNSUPPORTED) == advice
    assert restored_advisor.intent_planner_summary(advice, instruction=UNSUPPORTED) == (None, advice)
    assert calls == []
