"""Authored immutable views register only non-authoritative metadata links."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_catalog as bridge
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "owner-artifacts"
    root.mkdir(mode=0o700)
    artifact = root / "authored-metadata-view.json"
    raw = b'{"schema":"AUTHORED-inline-view@1","literal_fact":"preserved"}\n'
    artifact.write_bytes(raw)
    artifact.chmod(0o444)
    binding = dict(repository_id="repo:authored", tree_id="tree:authored",
        source_scope_cid="source:authored", task_id="authored-task", task_cid="task:authored",
        task_revision=1, context_cid="context:authored", native_prompt_sha256="1" * 64,
        native_evidence_sha256="2" * 64)
    receipt = dict(schema=bridge.OWNER_RECEIPT_SCHEMA, binding=copy.deepcopy(binding),
        artifact_sha256=hashlib.sha256(raw).hexdigest(), artifact_bytes=len(raw),
        native_evidence_ids=["evidence:authored:1", "evidence:authored:2"], provider_calls=0,
        **{field: False for field in bridge.DENIAL_FIELDS})
    return dict(index=SupervisorMetaIndex(tmp_path / "authored-meta.duckdb"), artifact=artifact,
                owner_receipt=receipt, expected_binding=binding,
                expected_native_evidence_ids=copy.deepcopy(receipt["native_evidence_ids"]))


def counts(index):
    with duckdb.connect(str(index.duckdb_path), read_only=True, config={"threads": 1}) as connection:
        return tuple(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                     for table in ("catalogs", "identity_links", "capsule_bindings"))


def test_actual_duckdb_registration_is_inert_owner_bound_and_idempotent(prepared):
    receipt_before = copy.deepcopy(prepared["owner_receipt"])
    binding_before = copy.deepcopy(prepared["expected_binding"])
    evidence_before = list(prepared["expected_native_evidence_ids"])
    raw_before = prepared["artifact"].read_bytes()
    result = bridge.register_semantic_metadata_catalog(**prepared)
    assert result["status"] == "registered_observational"
    assert result["link_count"] == 7
    assert counts(prepared["index"]) == (1, 7, 7)
    assert result == bridge.register_semantic_metadata_catalog(**prepared)
    assert counts(prepared["index"]) == (1, 7, 7)
    assert result["ducklake"] == dict(status="unconfigured", reason_code="ducklake_unconfigured",
                                     completion_authority=False, authoritative=False)
    assert all(result[field] is False for field in bridge.DENIAL_FIELDS)
    assert result["current_source_verified_by_bridge"] is False
    assert result["model_use_authorized"] is False
    assert prepared["owner_receipt"] == receipt_before
    assert prepared["expected_binding"] == binding_before
    assert prepared["expected_native_evidence_ids"] == evidence_before
    assert prepared["artifact"].read_bytes() == raw_before
    assert str(prepared["artifact"]) not in json.dumps(result)
    with duckdb.connect(str(prepared["index"].duckdb_path), read_only=True,
                        config={"threads": 1}) as connection:
        assert connection.execute("SELECT kind, exclusive_owner, attach_permitted, completion_authority "
            "FROM catalogs").fetchone() == ("capsule", "semantic_metadata_owner_prepared", False, False)
        assert connection.execute("SELECT count(*) FROM identity_links WHERE completion_authority").fetchone()[0] == 0
    linked = prepared["index"].compose_for_subject(subject_kind="task_id", subject_ref="authored-task")
    assert linked["n"] == 1
    assert linked["linked"][0]["record_kind"] == "semantic_metadata_task"
    assert linked["linked"][0]["attach_permitted"] is False


@pytest.mark.parametrize("field", sorted(bridge.BINDING_FIELDS))
def test_foreign_context_source_or_native_task_rejected_before_database(prepared, field):
    prepared["expected_binding"][field] = 2 if field == "task_revision" else (
        "3" * 64 if field.endswith("sha256") else "foreign:binding")
    with pytest.raises(ValueError):
        bridge.register_semantic_metadata_catalog(**prepared)
    assert not prepared["index"].duckdb_path.exists()


@pytest.mark.parametrize("field", sorted(bridge.DENIAL_FIELDS))
@pytest.mark.parametrize("claim", [True, 0])
def test_authority_cannot_be_inferred_or_widened(prepared, field, claim):
    prepared["owner_receipt"][field] = claim
    with pytest.raises(ValueError, match="authority"):
        bridge.register_semantic_metadata_catalog(**prepared)
    assert not prepared["index"].duckdb_path.exists()


@pytest.mark.parametrize("mutation", ["schema", "unknown_field", "missing_field", "provider_call",
    "bool_provider", "bool_revision", "invalid_hash", "oversize_receipt", "empty_ids", "duplicate_ids",
    "oversize_ids", "control_id", "nonfinite", "cycle"])
def test_bad_owner_receipt_rejected_before_database(prepared, mutation):
    receipt = prepared["owner_receipt"]
    if mutation == "schema": receipt["schema"] = "foreign@1"
    elif mutation == "unknown_field": receipt["sql"] = "SELECT secret"
    elif mutation == "missing_field": receipt.pop("proof_authority")
    elif mutation == "provider_call": receipt["provider_calls"] = 1
    elif mutation == "bool_provider": receipt["provider_calls"] = False
    elif mutation == "bool_revision": receipt["binding"]["task_revision"] = True
    elif mutation == "invalid_hash": receipt["artifact_sha256"] = "F" * 64
    elif mutation == "oversize_receipt": receipt["schema"] = "x" * 65_537
    elif mutation == "empty_ids": receipt["native_evidence_ids"] = []
    elif mutation == "duplicate_ids": receipt["native_evidence_ids"] *= 2
    elif mutation == "oversize_ids": receipt["native_evidence_ids"] = [f"evidence:{i}" for i in range(129)]
    elif mutation == "control_id": receipt["native_evidence_ids"] = ["evidence:\nforeign"]
    elif mutation == "nonfinite": receipt["artifact_bytes"] = float("nan")
    elif mutation == "cycle": receipt["binding"]["task_id"] = receipt
    with pytest.raises(ValueError):
        bridge.register_semantic_metadata_catalog(**prepared)
    assert not prepared["index"].duckdb_path.exists()


@pytest.mark.parametrize("mutation", ["missing", "changed", "writable", "hardlink", "symlink",
    "parent_symlink", "fifo", "directory", "bad_size", "oversize", "unsafe_parent", "relative"])
def test_artifact_exact_hash_and_immutable_file_guards(prepared, tmp_path, mutation):
    artifact = prepared["artifact"]
    if mutation == "missing": artifact.unlink()
    elif mutation == "changed":
        artifact.chmod(0o600); artifact.write_bytes(b"changed"); artifact.chmod(0o444)
    elif mutation == "writable": artifact.chmod(0o644)
    elif mutation == "hardlink": os.link(artifact, artifact.with_suffix(".link"))
    elif mutation == "symlink":
        original = artifact.with_suffix(".original"); artifact.rename(original); artifact.symlink_to(original)
    elif mutation == "parent_symlink":
        alias = tmp_path / "alias"; alias.symlink_to(artifact.parent, target_is_directory=True)
        prepared["artifact"] = alias / artifact.name
    elif mutation == "fifo": artifact.unlink(); os.mkfifo(artifact)
    elif mutation == "directory": artifact.unlink(); artifact.mkdir()
    elif mutation == "bad_size": prepared["owner_receipt"]["artifact_bytes"] += 1
    elif mutation == "oversize":
        artifact.chmod(0o600); artifact.write_bytes(b"x" * 1_000_001); artifact.chmod(0o444)
    elif mutation == "unsafe_parent": artifact.parent.chmod(0o777)
    elif mutation == "relative": prepared["artifact"] = Path("relative.json")
    with pytest.raises((ValueError, OSError)):
        bridge.register_semantic_metadata_catalog(**prepared)
    assert not prepared["index"].duckdb_path.exists()


def test_no_arbitrary_database_program_and_control_plane_refused(prepared):
    prepared["index"] = SupervisorMetaIndex(prepared["artifact"].parent / "control.duckdb")
    with pytest.raises(ValueError, match="separate absolute"):
        bridge.register_semantic_metadata_catalog(**prepared)
    assert not prepared["index"].duckdb_path.exists()
    prepared["index"] = object()
    with pytest.raises(TypeError, match="exact native"):
        bridge.register_semantic_metadata_catalog(**prepared)


def test_single_native_batch_and_one_observational_history_projection(prepared, monkeypatch):
    calls = []
    batch = SupervisorMetaIndex.link_identities
    def track_batch(self, records, *, project):
        calls.append(("batch", len(records), project))
        return batch(self, records, project=project)
    def project(self):
        calls.append(("history",))
        return dict(status="unavailable", reason_code="AuthoredUnavailable", error="PRIVATE_ERROR_BODY",
                    locator_ref="PRIVATE_LOCATOR", authoritative=False, completion_authority=False)
    monkeypatch.setattr(SupervisorMetaIndex, "link_identities", track_batch)
    monkeypatch.setattr(SupervisorMetaIndex, "project_ducklake", project)
    result = bridge.register_semantic_metadata_catalog(**prepared)
    assert calls == [("batch", 7, False), ("history",)]
    assert "PRIVATE" not in json.dumps(result)
    assert result["status"] == "registered_observational"


def test_history_can_be_explicitly_disabled(prepared, monkeypatch):
    monkeypatch.setattr(SupervisorMetaIndex, "project_ducklake", lambda _: pytest.fail("unexpected history"))
    result = bridge.register_semantic_metadata_catalog(**prepared, project_history=False)
    assert result["ducklake"]["status"] == "not_requested"


def test_link_failure_has_no_success_and_leaves_only_inert_native_catalog(prepared, monkeypatch):
    def fail(self, records, *, project):
        raise RuntimeError("authored native batch failure")
    monkeypatch.setattr(SupervisorMetaIndex, "link_identities", fail)
    with pytest.raises(RuntimeError, match="authored native batch"):
        bridge.register_semantic_metadata_catalog(**prepared)
    assert counts(prepared["index"]) == (1, 0, 0)


def test_artifact_swap_during_registration_cannot_yield_success(prepared, monkeypatch):
    original = SupervisorMetaIndex.link_identities
    def swap(self, records, *, project):
        result = original(self, records, project=project)
        artifact = prepared["artifact"]
        replacement = artifact.with_suffix(".replacement")
        replacement.write_bytes(artifact.read_bytes()); replacement.chmod(0o444)
        replacement.replace(artifact)
        return result
    monkeypatch.setattr(SupervisorMetaIndex, "link_identities", swap)
    with pytest.raises(ValueError, match="changed during"):
        bridge.register_semantic_metadata_catalog(**prepared)


def test_history_cannot_grant_authority(prepared, monkeypatch):
    monkeypatch.setattr(SupervisorMetaIndex, "project_ducklake", lambda _: dict(
        status="projected", authoritative=True, completion_authority=False))
    with pytest.raises(ValueError, match="observational DuckLake"):
        bridge.register_semantic_metadata_catalog(**prepared)


@pytest.mark.parametrize("mutation", ["foreign_receipt_id", "foreign_expected_ids", "reordered_ids",
                                      "missing_expected_ids", "string_expected_ids"])
def test_expected_native_evidence_population_bound_independently(prepared, mutation):
    if mutation == "foreign_receipt_id":
        prepared["owner_receipt"]["native_evidence_ids"][0] = "evidence:foreign-task"
    elif mutation == "foreign_expected_ids": prepared["expected_native_evidence_ids"] = ["foreign:evidence"]
    elif mutation == "reordered_ids": prepared["owner_receipt"]["native_evidence_ids"].reverse()
    elif mutation == "missing_expected_ids": prepared["expected_native_evidence_ids"] = []
    elif mutation == "string_expected_ids": prepared["expected_native_evidence_ids"] = "evidence:authored:1"
    with pytest.raises(ValueError, match="population"):
        bridge.register_semantic_metadata_catalog(**prepared)
    assert not prepared["index"].duckdb_path.exists()


def test_artifact_swap_during_history_cannot_yield_success(prepared, monkeypatch):
    def project(self):
        artifact = prepared["artifact"]
        replacement = artifact.with_suffix(".replacement")
        replacement.write_bytes(artifact.read_bytes()); replacement.chmod(0o444)
        replacement.replace(artifact)
        return dict(status="unconfigured", authoritative=False, completion_authority=False)
    monkeypatch.setattr(SupervisorMetaIndex, "project_ducklake", project)
    with pytest.raises(ValueError, match="changed during history"):
        bridge.register_semantic_metadata_catalog(**prepared)
