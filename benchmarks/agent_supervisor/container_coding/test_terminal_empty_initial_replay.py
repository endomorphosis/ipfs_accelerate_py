"""Signed empty-source contexts must reject rehashed transport tampering."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import terminal_program_population as population
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import (
    git, original, prepare,
)
from ipfs_accelerate_py.agent_supervisor.runtime import empty_code_retrieval as empty


def _write(path, value):
    raw = initial._bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _prepared_baseline(tmp_path, program):
    args = original(tmp_path, empty=not program)
    root, _, state, _ = args
    if program:
        (root / "source.py").write_text("# A real program input with no qualified code declarations.\n")
        git(root, "add", "source.py")
        git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "authored comment-only program baseline")
    prepared = prepare(args)
    return root, state, prepared


@pytest.fixture(params=[False, True], ids=["no-program-inputs", "comment-only-program"])
def prepared_only(tmp_path, request):
    return _prepared_baseline(tmp_path, request.param)


@pytest.fixture(params=[False, True], ids=["no-program-inputs", "comment-only-program"])
def captured(tmp_path, request):
    program = request.param
    root, state, prepared = _prepared_baseline(tmp_path, program)
    receipt = prep.initial_context(state=state)
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    assert receipt["indexed_symbols"] == 0 and receipt["index_id"] is None
    assert receipt["embedding_calls"] == 0 and receipt["learned_embeddings"] is False
    assert loaded["retrieval"]["disposition"] == ("zero_qualified_symbols" if program else "no_program_inputs")
    assert loaded["semantic"]["program_paths"] == (["source.py"] if program else [])
    return root, state, prepared, deepcopy(receipt), deepcopy(loaded["descriptor"])


def _persist_descriptor(root, state, receipt, descriptor):
    receipt["descriptor"]["sha256"] = _write(root / receipt["descriptor"]["artifact"], descriptor)
    _write(state / "initial-context-result.json", receipt)


def _reject_before_dispatch(root, state, prepared):
    calls = []
    def unexpected_provider(*args, **kwargs):
        calls.append(True)
        pytest.fail("tampered initial context reached the planner provider")
    with pytest.raises(ValueError):
        initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    with pytest.raises(ValueError):
        prep.plan(state=state, provider_callable=unexpected_provider)
    assert calls == []
    assert not (state / "planner-invoked.json").exists()
    assert not (state / "admission.json").exists()
    assert not (root / ".runtime/terminal-context").exists()
    assert not (root / "result.py").exists()


@pytest.mark.parametrize("change", ["schema", "index_id", "support_role", "population", "symbol_count", "authority"])
def test_rehashed_index_packet_cannot_change_qualified_empty_lane(captured, change):
    root, state, prepared, receipt, descriptor = captured
    path = root / descriptor["index"]["artifact"]
    indexed = json.loads(path.read_bytes())
    assert indexed["schema"] == population.EMPTY_INDEX_SCHEMA
    if change == "schema": indexed["schema"] = "terminal-qualified-vector-population@1"
    elif change == "index_id": indexed["index_id"] = "invented-vector-index"
    elif change == "support_role":
        indexed["support_hashes"][prep.SMOKE]["role"] = "instruction"
        indexed["observation"]["support_hashes"][prep.SMOKE]["role"] = "instruction"
    elif change == "population": indexed["source_population_cid"] = "invented-population"
    elif change == "symbol_count": indexed["symbols"] = 1
    else: indexed["proof_authority"] = True
    descriptor["index"]["sha256"] = _write(path, indexed)
    _persist_descriptor(root, state, receipt, descriptor)
    _reject_before_dispatch(root, state, prepared)


@pytest.mark.parametrize("change", ["support_role", "support_scope", "population", "hits", "authority", "index_id"])
def test_rehashed_retrieval_cannot_change_signed_empty_population(captured, change):
    root, state, prepared, receipt, descriptor = captured
    path = root / descriptor["metadata"]["Code retrieval artifact"]
    payload = json.loads(path.read_bytes())
    if change == "support_role": payload["support_hashes"][prep.SMOKE]["role"] = "instruction"
    elif change == "support_scope": payload["support_hashes"].pop(prep.INSTRUCTION)
    elif change == "population": payload["source_population_cid"] = "invented-population"
    elif change == "hits": payload["hits"] = [{"symbol": "invented-symbol"}]
    elif change == "authority": payload["completion_authority"] = True
    else: payload["index_id"] = "invented-vector-index"
    if change in {"support_role", "support_scope"}:
        # Keep every transport identity internally consistent. The independently
        # signed partition must reject the changed support role/population.
        payload["source_population_cid"] = empty._population(payload["program_paths"], payload["source_sha256"],
            payload["support_hashes"], payload["native_scan"], payload["disposition"])
        payload["query_id"] = empty._query_id(payload)
    payload["result_id"] = empty._result_id(payload)
    checksum = _write(path, payload)
    descriptor["metadata"]["Code retrieval sha256"] = checksum
    descriptor["retrieval"]["sha256"] = checksum
    descriptor["retrieval"]["metadata"]["Code retrieval sha256"] = checksum
    for key in ("source_population_cid", "query_id", "result_id"):
        descriptor["retrieval"][key] = payload[key]
        if key in receipt:
            receipt[key] = payload[key]
    _persist_descriptor(root, state, receipt, descriptor)
    _reject_before_dispatch(root, state, prepared)


@pytest.mark.parametrize("change", ["schema", "execution_authority", "completion_authority", "canonical_tasks_created",
    "provider_calls", "world_task_count", "semantic_root_cid", "world_snapshot_cid", "index_id",
    "indexed_symbols", "full_capsules", "learned_embeddings"])
def test_receipt_cannot_claim_authority_or_unobserved_roots_and_counts(captured, change):
    root, state, prepared, receipt, _ = captured
    if change == "schema": receipt[change] = "invented-initial-receipt@1"
    elif change in {"execution_authority", "completion_authority", "canonical_tasks_created", "learned_embeddings"}:
        receipt[change] = True
    elif change in {"provider_calls", "world_task_count", "indexed_symbols", "full_capsules"}:
        receipt[change] = 99
    else: receipt[change] = "invented-root"
    _write(state / "initial-context-result.json", receipt)
    _reject_before_dispatch(root, state, prepared)


@pytest.mark.parametrize("change", ["qualified", "authority", "model_consumed", "embedding_calls", "scan_count"])
def test_typed_packet_cannot_use_python_equality_to_replace_declared_values(captured, change):
    root, state, prepared, receipt, descriptor = captured
    path = root / descriptor["index"]["artifact"]
    indexed = json.loads(path.read_bytes())
    if change == "qualified": indexed["qualified"] = 1
    elif change == "authority": indexed["proof_authority"] = 0
    elif change == "model_consumed": indexed["selected_model"]["consumed"] = 0
    elif change == "embedding_calls": indexed["embedding_calls"] = False
    else: indexed["observation"]["native_scan"]["qualified_symbol_count"] = 0.0
    descriptor["index"]["sha256"] = _write(path, indexed)
    _persist_descriptor(root, state, receipt, descriptor)
    _reject_before_dispatch(root, state, prepared)


@pytest.mark.parametrize("change", ["provider_calls", "world_task_count", "indexed_symbols", "full_capsules",
    "learned_embeddings", "embedding_calls"])
def test_typed_receipt_cannot_replace_observation_counts_with_equal_json_values(captured, change):
    root, state, prepared, receipt, _ = captured
    if change in {"provider_calls", "indexed_symbols", "embedding_calls"}: receipt[change] = False
    elif change == "learned_embeddings": receipt[change] = 0
    else: receipt[change] = float(receipt[change])
    _write(state / "initial-context-result.json", receipt)
    _reject_before_dispatch(root, state, prepared)


@pytest.mark.parametrize("change", ["capsules", "worker_capsules", "doctor_findings"])
def test_rehashed_descriptor_cannot_invent_native_semantic_or_doctor_counts(captured, change):
    root, state, prepared, receipt, descriptor = captured
    descriptor["semantic"][change] = 99
    if change == "capsules":
        # Matching the loose result-side field must not hide an invented native
        # capsule count; replay must bind the complete producer index itself.
        receipt["full_capsules"] = 99
    _persist_descriptor(root, state, receipt, descriptor)
    _reject_before_dispatch(root, state, prepared)


@pytest.mark.parametrize("decoder", ["training", "legacy_security", "source384"])
def test_selected_decoders_abstain_before_consumption_for_empty_retrieval(prepared_only, monkeypatch, decoder):
    from ipfs_accelerate_py.agent_supervisor.runtime import security_autoencoder_advisor as legacy
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
    from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as training
    from ipfs_datasets_py.logic.software_contracts import codebase_source_384 as source384_parent
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as source384_units

    root, state, _ = prepared_only
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("an unavailable empty-scope decoder consumed weights or source")
    monkeypatch.setattr(legacy, "prepare_security_advice", forbidden)
    monkeypatch.setattr(source384, "prepare_source384_context", forbidden)
    monkeypatch.setattr(source384_parent, "register_shared_parent", forbidden)
    monkeypatch.setattr(source384_units, "infer_shared_parent_units", forbidden)
    monkeypatch.setattr(training, "train_codebase_autoencoder", forbidden)
    if decoder == "training":
        options = {"train_autoencoder": True}
    elif decoder == "legacy_security":
        options = {"security_checkpoint": {"explicit_unconsumed_selection": True}}
    else:
        selection = state / "requested-source384-config.json"
        selection.write_text("{}\n")
        options = {"source384_config": selection}
    with pytest.raises(ValueError, match="independent decoder abstention"):
        prep.initial_context(state=state, **options)
    assert calls == []
    assert not (state / "initial-context-result.json").exists()
    assert not (state / "planner-invoked.json").exists()
    assert not (state / "admission.json").exists()
    assert not (root / ".runtime/terminal-initial-context").exists()
    assert not (root / "result.py").exists()
