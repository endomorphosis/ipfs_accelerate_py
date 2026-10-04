"""Frozen public learning capture: actual no-fit replay and isolated corruption.

The retained experiment is required, not skipped. Original sources, model
packages and metadata are never mutated by these controls.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_catalog_capture as capture
from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae


@pytest.fixture(scope="module")
def frozen_catalog():
    root = Path(os.environ.get("TERMINAL_CODEBASE_INTENT_EXPERIMENT",
        "/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/intent-qualification-20261001-02"))
    assert (root / "result.json").is_file(), "actual qualified public intent experiment required"
    intent = json.loads((root / "result.json").read_bytes())
    snapshot = intent["repository_planning_snapshot"]
    repository = Path(snapshot["native_input_snapshot"]["repository_root"])
    sources = {name: (repository / name).read_bytes() for name in snapshot["current_source_inventory"]}
    result = capture.capture_frozen_codebase_learning(intent_experiment=root,
        current_source_bytes=sources, fresh_process=True)
    lineage = capture._lineage(root)
    return {"root": root, "sources": sources, "result": result,
        "fixture": lineage[-1], "original": lineage[4], "qualification": lineage[5]}


def test_complete_actual_native_catalog_and_candidate_only_scope(frozen_catalog):
    result = frozen_catalog["result"]
    assert result["schema"] == capture.SCHEMA
    assert {name: len(rows) for name, rows in result["records"].items()} == {
        "features": 358, "vectors": 358, "feature_frontiers": 0, "training": 1}
    assert result["learning_validation"]["status"] == "verified"
    assert result["learning_validation"]["sample_count"] == 358
    assert result["metadata_validation"]["row_count"] == 8723
    assert len(result["provenance"]["metadata_exports"]) == 28
    assert result["provenance"]["complete_original_family_counts"]["kg"] == 4652
    assert result["provenance"]["complete_original_family_counts"]["ast"] == 2153
    assert result["provenance"]["complete_producer_sha256"] == (
        frozen_catalog["qualification"]["metadata_reconstruction"]["recovered_records_sha256"])
    assert all(result[field] is False for field in capture.AUTHORITY)
    assert result["source_executed"] is result["canonical_state_mutated"] is False
    assert result["training_steps"] == result["provider_calls"] == 0
    assert result["official_reward"] is None
    body = dict(result)
    identifier = body.pop("capture_sha256")
    assert identifier == capture._sha(capture._json_bytes(body))
    assert set(result["training_sources"]) == {"bottle.py", ".supervisor-public-smoke.py"}
    assert len(result["current_sources"]) == 4


def test_capture_is_deterministic_and_never_calls_training(frozen_catalog, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("frozen capture must never fit a model")
    monkeypatch.setattr(ae, "train_codebase_autoencoder", forbidden)
    observed = capture.capture_frozen_codebase_learning(intent_experiment=frozen_catalog["root"],
        current_source_bytes=frozen_catalog["sources"], fresh_process=False)
    assert observed == frozen_catalog["result"]
    assert "fresh_process_readback" not in observed["metadata_validation"]


@pytest.mark.parametrize("change", ["bottle", "smoke", "missing", "extra_python"])
def test_current_python_scope_changes_refuse_frozen_learning(frozen_catalog, change):
    sources = dict(frozen_catalog["sources"])
    if change == "bottle":
        sources["bottle.py"] += b"\n"
    elif change == "smoke":
        sources[".supervisor-public-smoke.py"] += b"\n"
    elif change == "missing":
        del sources[".supervisor-public-smoke.py"]
    else:
        sources["additional.py"] = b"x = 1\n"
    with pytest.raises(ValueError, match="current signed Python source scope"):
        capture.capture_frozen_codebase_learning(intent_experiment=frozen_catalog["root"],
            current_source_bytes=sources)


@pytest.mark.parametrize("sources", [{"../bottle.py": b"x"}, {"./bottle.py": b"x"},
    {".": b"x"}, {"/bottle.py": b"x"}, {"bottle.py": "not bytes"},
    {1: b"x", "bottle.py": b"x"}, {"bottle.py": b"x" * 2_000_001},
    {"a": b"x" * 2_000_000, "b": b"x" * 2_000_000, "c": b"x"}])
def test_current_sources_are_closed_and_bounded(sources):
    with pytest.raises(ValueError):
        capture._current_sources(sources)


def test_nonpython_context_is_retained_without_becoming_a_training_label(frozen_catalog):
    current = capture._current_sources({**frozen_catalog["sources"], "context.txt": b"new signed context"})
    assert current["context.txt"]["sha256"] == capture._sha(b"new signed context")
    training = {name: pin["sha256"] for name, pin in current.items() if name.endswith(".py")}
    assert training == frozen_catalog["result"]["training_sources"]


@pytest.mark.parametrize("corruption", ["missing", "bytes", "row_order"])
def test_pinned_exports_reject_private_copy_corruption(frozen_catalog, tmp_path, corruption):
    metadata = frozen_catalog["original"] / "metadata"
    manifest = json.loads((metadata / "manifest.json").read_bytes())
    family = "features"
    descriptor = copy.deepcopy(manifest["families"][family])
    private = tmp_path / "metadata"
    (private / "exports").mkdir(parents=True)
    raw = (metadata / descriptor["export"]["relative_path"]).read_bytes()
    path = private / descriptor["export"]["relative_path"]
    if corruption == "bytes":
        path.write_bytes(raw + b" ")
    elif corruption == "row_order":
        rows = [json.loads(line) for line in raw.splitlines()]
        rows[0]["row_ordinal"] = 1
        raw = b"".join(capture._json_bytes(row) + b"\n" for row in rows)
        path.write_bytes(raw)
        descriptor["export"] = {**descriptor["export"], "sha256": "sha256:" + capture._sha(raw), "bytes": len(raw)}
    with pytest.raises((ValueError, FileNotFoundError), match="changed|order|No such file"):
        capture._export_records(private, {**manifest, "families": {family: descriptor}})


@pytest.mark.parametrize("artifact", ["checkpoint", "index", "receipt"])
def test_native_model_artifact_drift_rejected_in_isolated_read_redirect(frozen_catalog, tmp_path, monkeypatch, artifact):
    output = Path(frozen_catalog["fixture"]["learner"]["output"])
    private = tmp_path / "package"
    private.mkdir()
    for name in ("receipt", "checkpoint", "features", "index"):
        (private / (name + ".json")).write_bytes((output / (name + ".json")).read_bytes())
    (private / (artifact + ".json")).write_bytes(b"{}")
    real_read = ae._read
    def redirected(path):
        path = Path(path)
        return real_read(private / path.name) if path.parent == output else real_read(path)
    monkeypatch.setattr(ae, "_read", redirected)
    with pytest.raises(ValueError, match="receipt changed|artifact digest changed"):
        capture._native_learning(fixture=frozen_catalog["fixture"], records=frozen_catalog["result"]["records"])


def test_rebased_index_receipt_still_requires_actual_native_inference(frozen_catalog, tmp_path, monkeypatch):
    fixture = copy.deepcopy(frozen_catalog["fixture"])
    output = Path(fixture["learner"]["output"])
    private = tmp_path / "package"
    private.mkdir()
    for name in ("receipt", "checkpoint", "features", "index"):
        (private / (name + ".json")).write_bytes((output / (name + ".json")).read_bytes())
    index = json.loads((private / "index.json").read_bytes())
    index["ranks"][0]["latent"][0] += 1
    raw = capture._json_bytes(index)
    (private / "index.json").write_bytes(raw)
    receipt = json.loads((private / "receipt.json").read_bytes())
    receipt["index_sha256"] = capture._sha(raw)
    raw = capture._json_bytes(receipt)
    (private / "receipt.json").write_bytes(raw)
    fixture["learner"]["receipt_sha256"] = capture._sha(raw)
    real_read = ae._read
    monkeypatch.setattr(ae, "_read", lambda path: real_read(private / Path(path).name)
        if Path(path).parent == output else real_read(path))
    with pytest.raises(ValueError, match="learned inference differs"):
        capture._native_learning(fixture=fixture, records=frozen_catalog["result"]["records"])


def test_no_caller_validation_marker_can_bypass_native_learning(frozen_catalog, monkeypatch):
    records = copy.deepcopy(frozen_catalog["result"]["records"])
    records["training"][0]["validated"] = True
    def reject(**kwargs):
        raise ValueError("actual native inference refused")
    monkeypatch.setattr(ae, "validate_codebase_autoencoder", reject)
    with pytest.raises(ValueError, match="actual native inference refused"):
        capture._native_learning(fixture=frozen_catalog["fixture"], records=records)


def test_incomplete_original_catalog_cannot_be_exported_as_complete(frozen_catalog):
    records = copy.deepcopy(frozen_catalog["result"]["records"])
    records["features"].pop()
    with pytest.raises(ValueError, match="complete original learning inventories"):
        capture._native_learning(fixture=frozen_catalog["fixture"], records=records)


def test_ancestor_pin_drift_rejected_from_private_intent_result(frozen_catalog, tmp_path, monkeypatch):
    intent = json.loads((frozen_catalog["root"] / "result.json").read_bytes())
    root = tmp_path / "intent"
    root.mkdir()
    intent["parent"]["decoder_manifest"]["sha256"] = "0" * 64
    private = root / "result.json"
    private.write_bytes(capture._json_bytes(intent))
    original_json = capture._json
    monkeypatch.setattr(capture, "_json", lambda path: original_json(private)
        if Path(path) == frozen_catalog["root"] / "result.json" else original_json(path))
    with pytest.raises(ValueError, match="ancestor artifact drift"):
        capture._lineage(frozen_catalog["root"])


def test_frozen_generation_requires_existing_producer_sidecar(frozen_catalog, monkeypatch):
    original_pin = capture._pin
    sidecar = frozen_catalog["original"] / "supervisor/metadata-records.json"
    def refuse(path):
        if Path(path) == sidecar:
            raise FileNotFoundError("absent producer sidecar")
        return original_pin(path)
    monkeypatch.setattr(capture, "_pin", refuse)
    with pytest.raises(FileNotFoundError, match="absent producer sidecar"):
        capture._lineage(frozen_catalog["root"])


def test_fresh_process_option_is_an_explicit_boolean(frozen_catalog):
    with pytest.raises(ValueError, match="Boolean"):
        capture.capture_frozen_codebase_learning(intent_experiment=frozen_catalog["root"],
            current_source_bytes=frozen_catalog["sources"], fresh_process=1)


def test_source_changed_during_native_inference_is_rejected(frozen_catalog, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata
    sources = dict(frozen_catalog["sources"])
    # The full native storage gate was exercised by the module fixture. Reuse
    # its exact report here to isolate the later concurrent input mutation.
    monkeypatch.setattr(metadata, "validate_codebase_ir_metadata", lambda **kwargs:
        copy.deepcopy(frozen_catalog["result"]["metadata_validation"]))
    native = capture._native_learning
    def observed(**kwargs):
        result = native(**kwargs)
        sources["bottle.py"] += b"\n"
        return result
    monkeypatch.setattr(capture, "_native_learning", observed)
    with pytest.raises(ValueError, match="current source changed during"):
        capture.capture_frozen_codebase_learning(intent_experiment=frozen_catalog["root"],
            current_source_bytes=sources)


def test_whole_producer_inventory_digest_rejects_a_dropped_row(frozen_catalog, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata
    monkeypatch.setattr(metadata, "validate_codebase_ir_metadata", lambda **kwargs:
        copy.deepcopy(frozen_catalog["result"]["metadata_validation"]))
    export = capture._export_records
    def incomplete(*args):
        records, pins = export(*args)
        records["features"].pop()
        return records, pins
    monkeypatch.setattr(capture, "_export_records", incomplete)
    with pytest.raises(ValueError, match="complete original producer reconstruction differs"):
        capture.capture_frozen_codebase_learning(intent_experiment=frozen_catalog["root"],
            current_source_bytes=frozen_catalog["sources"])


def test_native_metadata_failure_is_not_overridden_by_historical_markers(frozen_catalog, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata
    def refused(**kwargs):
        raise metadata.MetadataError("native original row root changed")
    monkeypatch.setattr(metadata, "validate_codebase_ir_metadata", refused)
    with pytest.raises(metadata.MetadataError, match="native original row root changed"):
        capture.capture_frozen_codebase_learning(intent_experiment=frozen_catalog["root"],
            current_source_bytes=frozen_catalog["sources"])
