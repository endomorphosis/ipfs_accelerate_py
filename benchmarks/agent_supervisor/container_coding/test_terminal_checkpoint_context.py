"""Actual ready-root contexts authenticate opaque retained checkpoints on reopen.

The small checkpoint files are authored byte fixtures, not trained decoders.
Native admission, source/world/retrieval contexts and catalog reads are real.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_multitask_context as contexts
from benchmarks.agent_supervisor.container_coding.test_terminal_multitask_context import (
    admitted, retained_vectors, multitask_case, _assert_retained, _cold, _hash, _wire,
)
from ipfs_accelerate_py.agent_supervisor.runtime import task_ir_checkpoint as checkpoint_owner
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.test_ir_persistent_catalog import _raw, _record, _selection


def _catalog(tmp_path, *, matrix=False):
    pairs = ([(family, width) for family in ("codebase_ir", "security_ir", "legal_ir", "intent_ir")
              for width in (8, 384, 768)] if matrix else [("legal_ir", 384)])
    records = []
    files = []
    for family, width in pairs:
        raw = f"retained opaque byte fixture: {family}/{width}".encode()
        path = tmp_path / f"{family}-{width}.checkpoint"
        path.write_bytes(raw)
        record = _record(family, width, "latent" if width == 8 else "input_embedding",
                         checkpoint=raw.decode())
        record["huggingface_config"]["ir_checkpoint"]["original_checkpoint_pin"].update(
            path=str(path), bytes=len(raw))
        records.append(record)
        files.append(path)
    catalog = tmp_path / "retained-model-manager-catalog.json"
    catalog.write_bytes(_raw(records))
    return catalog, records, files


def _prepare(case, vectors, catalog, requests, **kwargs):
    return contexts.prepare_ready_task_contexts(state=case["state"], task_cids=case["roots"],
        output=case["output"], code_vector_snapshot=vectors["snapshot"],
        code_vector_result=vectors["result"], ir_catalog_path=catalog,
        ir_selections={cid: deepcopy(requests) for cid in case["roots"]}, **kwargs)


def _authenticated_cold(case):
    path = case["output"] / "result.json"
    return contexts.load_ready_task_contexts(state=case["state"],
        artifact=path.relative_to(case["repository"]).as_posix(), expected_sha256=_hash(path),
        require_checkpoint_authentication=True)


def _rewrite(case, result):
    (case["output"] / "result.json").write_bytes(_wire(result))


def test_two_actual_roots_authenticate_parallel_family_dimension_assets_without_runtime(admitted, retained_vectors, tmp_path):
    catalog, records, files = _catalog(tmp_path, matrix=True)
    before = {str(path): path.read_bytes() for path in [catalog, *files]}
    result = _prepare(admitted, retained_vectors, catalog, [_selection(row) for row in records],
                      authenticate_ir_checkpoints=True)
    assert result["schema"] == contexts.CHECKPOINT_SCHEMA
    assert result["ir_selection_mode"] == "authenticated_checkpoint_nomination"
    assert result["checkpoint_bytes_authenticated"] is True
    assert result["decoder_runtime_admitted"] is False
    assert result["execution_authority"] is result["completion_authority"] is result["proof_authority"] is False
    for cid in admitted["roots"]:
        observations = result["ir_checkpoint_observations"][cid]
        assert len(observations) == 12
        for observation, metadata in zip(observations, result["ir_metadata_nominations"][cid]):
            assert observation["checkpoint_bytes_authenticated"] is True
            assert observation["native_resolution"] == metadata
            assert metadata["authority"]["checkpoint_bytes_authenticated"] is False
            assert not any(observation["authority"].values())
            assert observation["selectors"]["schema_version"] is None
            pin = observation["original_checkpoint_pin"]
            info = Path(pin["path"]).stat()
            assert observation["file_witness"]["inode"] == info.st_ino
            assert observation["file_witness"]["device"] == info.st_dev
            assert observation["file_witness"]["bytes"] == info.st_size == pin["bytes"]
    assert _authenticated_cold(admitted) == result
    assert _cold(admitted) == result
    assert {str(path): path.read_bytes() for path in [catalog, *files]} == before
    _assert_retained(retained_vectors)


def test_default_receipt_preserves_metadata_only_scope_and_missing_assets(admitted, retained_vectors, tmp_path):
    catalog, records, files = _catalog(tmp_path)
    files[0].unlink()
    result = _prepare(admitted, retained_vectors, catalog, [_selection(records[0])])
    assert result["schema"] == contexts.SCHEMA
    assert result["checkpoint_bytes_authenticated"] is False
    assert "ir_checkpoint_observations" not in result
    assert _cold(admitted) == result
    with pytest.raises(ValueError, match="requires authenticated"):
        _authenticated_cold(admitted)
    assert not files[0].exists()


@pytest.mark.parametrize("mutation", ["bytes", "missing", "same_bytes_new_inode", "catalog_path"])
def test_cold_authenticated_receipt_reobserves_actual_checkpoint_owner(admitted, retained_vectors, tmp_path, mutation):
    catalog, records, files = _catalog(tmp_path)
    result = _prepare(admitted, retained_vectors, catalog, [_selection(records[0])],
                      authenticate_ir_checkpoints=True)
    checkpoint = files[0]
    raw = checkpoint.read_bytes()
    receipt_before = (admitted["output"] / "result.json").read_bytes()
    if mutation == "bytes":
        checkpoint.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif mutation == "missing":
        checkpoint.unlink()
    elif mutation == "same_bytes_new_inode":
        replacement = tmp_path / "replacement.checkpoint"
        replacement.write_bytes(raw)
        replacement.replace(checkpoint)
    else:
        alternate = tmp_path / "alternate.checkpoint"
        alternate.write_bytes(raw)
        records[0]["huggingface_config"]["ir_checkpoint"]["original_checkpoint_pin"]["path"] = str(alternate)
        catalog.write_bytes(_raw(records))
    with pytest.raises((ValueError, OSError)):
        _authenticated_cold(admitted)
    assert (admitted["output"] / "result.json").read_bytes() == receipt_before
    _assert_retained(retained_vectors)


@pytest.mark.parametrize("mutation", ["checkpoint_flag", "runtime_flag", "witness", "drop_observations", "downgrade", "task_binding",
    "checkpoint_flag_as_integer", "authority_flag_as_integer", "link_count_as_bool"])
def test_freshly_rehashed_checkpoint_claim_cannot_replace_live_observations(admitted, retained_vectors, tmp_path, mutation):
    catalog, records, _ = _catalog(tmp_path)
    result = _prepare(admitted, retained_vectors, catalog, [_selection(records[0])],
                      authenticate_ir_checkpoints=True)
    if mutation == "checkpoint_flag":
        result["checkpoint_bytes_authenticated"] = False
    elif mutation == "runtime_flag":
        result["decoder_runtime_admitted"] = True
    elif mutation == "witness":
        result["ir_checkpoint_observations"][admitted["roots"][0]][0]["file_witness"]["inode"] += 1
    elif mutation == "drop_observations":
        result.pop("ir_checkpoint_observations")
    elif mutation == "downgrade":
        result["schema"] = contexts.SCHEMA
        result["checkpoint_bytes_authenticated"] = False
        result["ir_selection_mode"] = "metadata_nomination"
        result.pop("ir_checkpoint_observations")
    elif mutation == "task_binding":
        result["ir_checkpoint_observations"][admitted["roots"][0]][0]["selectors"]["task_id"] = "foreign-head"
    elif mutation == "checkpoint_flag_as_integer":
        result["ir_checkpoint_observations"][admitted["roots"][0]][0]["checkpoint_bytes_authenticated"] = 1
    elif mutation == "authority_flag_as_integer":
        result["ir_checkpoint_observations"][admitted["roots"][0]][0]["authority"]["runtime_admitted"] = 0
    else:
        result["ir_checkpoint_observations"][admitted["roots"][0]][0]["file_witness"]["nlink"] = True
    _rewrite(admitted, result)
    with pytest.raises(ValueError):
        _authenticated_cold(admitted)


def test_missing_actual_checkpoint_refuses_before_context_output(admitted, retained_vectors, tmp_path):
    catalog, records, files = _catalog(tmp_path)
    files[0].unlink()
    with pytest.raises((ValueError, OSError)):
        _prepare(admitted, retained_vectors, catalog, [_selection(records[0])], authenticate_ir_checkpoints=True)
    assert not admitted["output"].exists()
    assert not files[0].exists()


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_typed_authentication_selection_refuses_before_state_or_artifacts(tmp_path, value):
    with pytest.raises(ValueError, match="explicit boolean"):
        contexts.prepare_ready_task_contexts(state=tmp_path / "uncreated-state", task_cids=["foreign"],
            output=tmp_path / "uncreated-output", authenticate_ir_checkpoints=value)
    with pytest.raises(ValueError, match="explicit boolean"):
        contexts.load_ready_task_contexts(state=tmp_path / "uncreated-state", artifact="missing",
            expected_sha256="0" * 64, require_checkpoint_authentication=value)
    assert not (tmp_path / "uncreated-state").exists()
    assert not (tmp_path / "uncreated-output").exists()


def test_authentication_requires_explicit_ir_selection_before_state_open(tmp_path):
    with pytest.raises(ValueError, match="exact per-task"):
        contexts.prepare_ready_task_contexts(state=tmp_path / "uncreated-state", task_cids=["foreign"],
            output=tmp_path / "uncreated-output", authenticate_ir_checkpoints=True)
    assert not (tmp_path / "uncreated-state").exists()


def test_later_task_authentication_closes_earlier_task_checkpoint_identity(admitted, retained_vectors, tmp_path, monkeypatch):
    """Inject drift after genuine later-task authentication, with real owners."""
    catalog, records, files = _catalog(tmp_path, matrix=True)
    requests = {cid: [_selection(records[index])] for index, cid in enumerate(admitted["roots"])}
    actual = checkpoint_owner.authenticate_task_ir_checkpoints
    calls = []
    def changed(**kwargs):
        observation = actual(**kwargs)
        calls.append(True)
        if len(calls) == 2:
            files[0].write_bytes(files[0].read_bytes() + b"changed after its own task read")
        return observation
    monkeypatch.setattr(checkpoint_owner, "authenticate_task_ir_checkpoints", changed)
    with pytest.raises(ValueError, match="checkpoint file identity"):
        contexts.prepare_ready_task_contexts(state=admitted["state"], task_cids=admitted["roots"],
            output=admitted["output"], code_vector_snapshot=retained_vectors["snapshot"],
            code_vector_result=retained_vectors["result"], ir_catalog_path=catalog,
            ir_selections=requests, authenticate_ir_checkpoints=True)
    assert calls == [True, True]
    assert not admitted["output"].exists()


@pytest.mark.parametrize("operation", ["prepare", "cold_reopen"])
@pytest.mark.parametrize("changed_owner", ["source", "native_event"])
def test_source_and_native_fences_close_after_genuine_checkpoint_authentication(admitted, retained_vectors, tmp_path,
        monkeypatch, operation, changed_owner):
    catalog, records, _ = _catalog(tmp_path)
    requests = [_selection(records[0])]
    if operation == "cold_reopen":
        _prepare(admitted, retained_vectors, catalog, requests, authenticate_ir_checkpoints=True)
    actual = contexts._authenticate_ir
    calls = []
    def changed(*args):
        observation = actual(*args)
        calls.append(True)
        if len(calls) == 2:
            if changed_owner == "source":
                with (admitted["repository"] / "left.py").open("a") as stream:
                    stream.write("\n# source changed during final checkpoint read\n")
            else:
                with IntentRepository(admitted["state"] / "intent.duckdb", install_schema=False) as intent:
                    intent.upsert_objective(objective_id="checkpoint-auth-race", objective_alias="CHECKPOINT-RACE",
                                            title="Actual native event after checkpoint read")
        return observation
    monkeypatch.setattr(contexts, "_authenticate_ir", changed)
    with pytest.raises(ValueError):
        if operation == "prepare":
            _prepare(admitted, retained_vectors, catalog, requests, authenticate_ir_checkpoints=True)
        else:
            _authenticated_cold(admitted)
    assert calls == [True, True]
    if operation == "prepare":
        assert not (admitted["output"] / "result.json").exists()
