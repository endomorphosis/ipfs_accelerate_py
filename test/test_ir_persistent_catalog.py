"""Metadata-only IR discovery; private stores, no checkpoint/model execution."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py.model_catalog import LifecycleState, Operation
from ipfs_accelerate_py.model_catalog.catalog import AIServiceCatalog
from ipfs_accelerate_py.model_catalog.sources.ir_persistent import (
    COMPONENT_SCHEMA,
    IRPersistentCatalogError,
    IRPersistentCatalogSource,
)
from ipfs_accelerate_py.model_catalog.sources.static import StaticCatalogSource


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _record(family="legal_ir", dimension=384, dimension_role="input_embedding", *,
            checkpoint="original", role="retained", **declarations):
    checkpoint_sha = hashlib.sha256(checkpoint.encode()).hexdigest()
    identity = {"ir_family_id": family, "dimension": dimension,
                "dimension_role": dimension_role, "role": role, "checkpoint_sha256": checkpoint_sha}
    prefix = "ir-model-asset-binding/v1:"
    if dimension is None:
        identity.update(asset_binding_schema=COMPONENT_SCHEMA, external_lane_binding=None)
        prefix = "ir-model-component-binding/v1:"
    record_id = prefix + hashlib.sha256(_raw(identity)).hexdigest()
    declaration = {"record_id": record_id, "ir_family_id": family, "dimension": dimension,
        "dimension_role": dimension_role, "role": role, "schema_version": None,
        "task_id": None, "profile_id": None, "format_id": None,
        "original_checkpoint_pin": {"path": "/nonexistent/retained-checkpoint.json",
                                    "bytes": 10, "sha256": checkpoint_sha},
        "trained": None, "initialization_only": False, "donor": None,
        "runtime_ready": False, "teacher_qualified": False, "proof_authority": False,
        **declarations}
    if dimension is None:
        declaration.update(asset_binding_schema=COMPONENT_SCHEMA, external_lane_binding=None)
    return {"model_id": record_id, "model_name": family + " retained asset", "model_type": "encoder_decoder",
            "architecture": "declared-fixture-only", "model_revision": checkpoint_sha,
            "revision_id": checkpoint_sha, "huggingface_config": {"ir_checkpoint": declaration}}


def _store(tmp_path, records):
    path = tmp_path / "selected-ir-store.json"
    path.write_bytes(_raw(records))
    return path


def _request(record):
    declaration = record["huggingface_config"]["ir_checkpoint"]
    return {name: declaration[name] for name in (
        "record_id", "ir_family_id", "dimension", "dimension_role", "schema_version",
        "task_id", "profile_id", "format_id", "role")}


def _selection(record):
    return {**_request(record), "checkpoint_sha256": record["model_revision"]}


@pytest.mark.parametrize("family", ["codebase_ir", "security_ir", "legal_ir", "intent_ir", "ui_ux_ir"])
def test_exact_family_is_retained_without_any_invokable_capability(tmp_path, family):
    record = _record(family)
    result = IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()
    selected = result.resolve_ir_binding(_selection(record))
    assert selected["selected_binding"]["ir_family_id"] == family
    assert selected["selected_binding"]["record_id"] == record["model_id"]
    assert selected["selected_binding"]["original_checkpoint_bytes_verified"] is False
    assert result.snapshot.bindings == ()
    assert result.snapshot.deployments == ()
    assert all(item.capabilities == () for item in result.models + result.providers)
    assert all(item.lifecycle is LifecycleState.DECLARED for item in result.models + result.providers)
    assert all(value is None for item in result.models + result.providers for value in item.state.to_dict().values())
    assert not any(selected["authority"].values())
    assert dict(result.models[0].labels)["ir.record-id"] == record["model_id"]
    assert result.models[0].provenance[0].source_record_id == record["model_id"]


@pytest.mark.parametrize("dimension,dimension_role", [(8, "latent"), (384, "input_embedding"), (768, "input_embedding")])
def test_lane_width_does_not_replace_geometry_role_or_donor_declaration(tmp_path, dimension, dimension_role):
    record = _record("codebase_ir", dimension, dimension_role, donor=True,
        auxiliary_dimensions={"input_width": 54, "latent_width": 8}, payload_ir_family="security_ir")
    result = IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()
    item = result.resolve_ir_binding(_selection(record))["selected_binding"]
    assert (item["dimension"], item["dimension_role"]) == (dimension, dimension_role)
    assert item["declaration"]["auxiliary_dimensions"] == {"input_width": 54, "latent_width": 8}
    assert item["declaration"]["payload_ir_family"] == "security_ir"
    assert item["declaration"]["donor"] is True
    assert item["runtime_observation_performed"] is False


@pytest.mark.parametrize("role", ["source_tokens", "unbound_component"])
def test_detached_component_null_lane_and_unknown_contract_are_preserved(tmp_path, role):
    record = _record("intent_ir", None, role, auxiliary_dimensions={"latent_width": 4, "input_width": 147})
    result = IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()
    item = result.ir_bindings[0]
    assert item["dimension"] is None
    assert all(item[key] is None for key in ("task_id", "profile_id", "format_id", "schema_version"))
    assert item["declaration"]["external_lane_binding"] is None
    assert "ir.dimension" not in dict(result.models[0].labels)
    assert result.resolve_ir_binding(_selection(record))["selected_binding"] == item


def test_readiness_declarations_are_not_runtime_observations(tmp_path):
    record = _record(runtime_ready=True, teacher_qualified=True, proof_authority=True, trained=True)
    result = IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()
    assert result.ir_bindings[0]["declaration"]["runtime_ready"] is True
    assert result.ir_bindings[0]["declaration"]["teacher_qualified"] is True
    assert result.ir_bindings[0]["declaration"]["proof_authority"] is True
    assert not any(result.to_dict()["authority"].values())
    assert result.models[0].state.routable is None
    assert result.models[0].lifecycle is LifecycleState.DECLARED


def test_fresh_load_and_refresh_discover_new_persisted_record_without_manager(tmp_path):
    first, second = _record("legal_ir"), _record("intent_ir")
    path = _store(tmp_path, [first])
    source = IRPersistentCatalogSource(path=path)
    old = source.load()
    path.write_bytes(_raw([first, second]))
    fresh = source.refresh()
    assert len(old.ir_bindings) == 1
    assert len(fresh.ir_bindings) == 2
    assert old.binding_snapshot_revision != fresh.binding_snapshot_revision
    assert source.resolve_ir_binding(_selection(second))["selected_binding"]["record_id"] == second["model_id"]


def test_changed_nonselector_declaration_changes_published_catalog_generation(tmp_path):
    record = _record()
    path = _store(tmp_path, [record])
    source = IRPersistentCatalogSource(path=path)
    before = source.load()
    record["huggingface_config"]["ir_checkpoint"]["trained"] = True
    path.write_bytes(_raw([record]))
    after = source.refresh()
    assert before.resolve_ir_binding(_selection(record))["selected_binding"]["declaration"]["trained"] is None
    assert after.resolve_ir_binding(_selection(record))["selected_binding"]["declaration"]["trained"] is True
    assert before.binding_snapshot_revision != after.binding_snapshot_revision
    assert before.snapshot.revision != after.snapshot.revision
    assert dict(before.models[0].labels)["ir.declaration-sha256"] != dict(after.models[0].labels)["ir.declaration-sha256"]
    assert after.models[0].state.routable is None


def test_catalog_refresh_keeps_unrelated_source_and_last_good_generation_on_failure(tmp_path):
    first, second = _record("legal_ir"), _record("intent_ir")
    path = _store(tmp_path, [first])
    source = IRPersistentCatalogSource(path=path)
    other = StaticCatalogSource([{"provider": "job-owner", "model": "existing"}], source="job.existing")
    catalog = AIServiceCatalog({source.source: source, other.source: other})
    old = catalog.snapshot()
    path.write_bytes(_raw([first, second]))
    accepted = catalog.refresh([source.source], raise_on_error=True)
    assert accepted.failed == ()
    assert len(accepted.snapshot.models) == 3
    assert any(item.name == "existing" for item in accepted.snapshot.models)
    path.write_bytes(_raw([first, {"huggingface_config": {"ir_checkpoint": {}}}]))
    refused = catalog.refresh([source.source])
    assert refused.failed == (source.source,)
    assert refused.snapshot is accepted.snapshot
    assert len(old.models) == 2
    assert catalog.resolve(operation=Operation.TEXT_GENERATE).candidates == ()


@pytest.mark.parametrize("field,value", [
    ("ir_family_id", "security_ir"), ("dimension", 768), ("dimension_role", "latent"),
    ("task_id", "source_to_native_ir"), ("schema_version", "legal-ir/v2"),
    ("profile_id", "profile:foreign"), ("format_id", "format:foreign"),
    ("checkpoint_sha256", "0" * 64), ("role", "other"),
    ("record_id", "ir-model-asset-binding/v1:" + "0" * 64),
])
def test_every_explicit_selector_is_required_even_when_record_id_matches(tmp_path, field, value):
    record = _record()
    result = IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()
    request = _selection(record)
    request[field] = value
    with pytest.raises(IRPersistentCatalogError):
        result.resolve_ir_binding(request)


@pytest.mark.parametrize("mutation", ["missing", "extra", "bool_width", "nested_family", "uppercase_sha", "latest"])
def test_malformed_selector_refuses_before_store_access(tmp_path, monkeypatch, mutation):
    record = _record()
    source = IRPersistentCatalogSource(path=tmp_path / "never-opened.duckdb")
    def forbidden():
        pytest.fail("malformed selector must be refused before reading the store")
    monkeypatch.setattr(source, "load", forbidden)
    request = _selection(record)
    if mutation == "missing": request.pop("task_id")
    elif mutation == "extra": request["latest"] = True
    elif mutation == "bool_width": request["dimension"] = True
    elif mutation == "nested_family": request["ir_family_id"] = {}
    elif mutation == "uppercase_sha": request["checkpoint_sha256"] = record["model_revision"].upper()
    else: request["record_id"] = "latest"
    with pytest.raises(IRPersistentCatalogError): source.resolve_ir_binding(request)


@pytest.mark.parametrize("mutation", ["family", "role", "record_id", "revision", "pin_sha", "component", "trained_init", "bad_flag"])
def test_malformed_peer_refuses_entire_store_not_partial_success(tmp_path, mutation):
    good, bad = _record(), _record("intent_ir")
    declaration = bad["huggingface_config"]["ir_checkpoint"]
    if mutation == "family": declaration["ir_family_id"] = "security_ir"
    elif mutation == "role": declaration["dimension_role"] = "latent"
    elif mutation == "record_id": bad["model_id"] = "foreign-model"
    elif mutation == "revision": bad["revision_id"] = "0" * 64
    elif mutation == "pin_sha": declaration["original_checkpoint_pin"]["sha256"] = "0" * 64
    elif mutation == "component": declaration["dimension"] = None; declaration["dimension_role"] = "source_tokens"
    elif mutation == "trained_init": declaration["trained"] = True; declaration["initialization_only"] = True
    else: declaration["runtime_ready"] = "true"
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=_store(tmp_path, [good, bad])).load()


def test_duplicate_ir_records_refuse_even_identical(tmp_path):
    record = _record()
    with pytest.raises(IRPersistentCatalogError, match="duplicate"):
        IRPersistentCatalogSource(path=_store(tmp_path, [record, deepcopy(record)])).load()


def test_non_ir_rows_do_not_gain_typed_bindings_and_wrong_store_refuses(tmp_path):
    record = _record()
    generic = {"model_id": "generic-foundation", "model_type": "language_model", "huggingface_config": {}}
    path = _store(tmp_path, [generic, record])
    assert len(IRPersistentCatalogSource(path=path).load().ir_bindings) == 1
    path.write_bytes(_raw([generic]))
    with pytest.raises(IRPersistentCatalogError, match="no IR"):
        IRPersistentCatalogSource(path=path).load()


def test_lost_ir_declaration_never_falls_back_to_generic_model_type(tmp_path):
    record = _record()
    record["huggingface_config"] = {}
    with pytest.raises(IRPersistentCatalogError):
        IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()


def test_stored_duckdb_json_string_is_parsed_strictly(tmp_path):
    record = _record()
    record["huggingface_config"] = _raw(record["huggingface_config"]).decode()
    result = IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()
    assert result.ir_bindings[0]["checkpoint_sha256"] == record["model_revision"]
    record["huggingface_config"] = '{"ir_checkpoint":null,"ir_checkpoint":{}}'
    with pytest.raises(IRPersistentCatalogError, match="duplicate"):
        IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()


def test_duplicate_outer_json_declaration_field_fails_closed(tmp_path):
    raw = _raw([_record()]).replace(b'"trained":null', b'"trained":false,"trained":null')
    path = tmp_path / "duplicate.json"
    path.write_bytes(raw)
    with pytest.raises(IRPersistentCatalogError, match="duplicate"):
        IRPersistentCatalogSource(path=path).load()


def test_explicit_jsonl_store_preserves_all_bindings(tmp_path):
    first, second = _record(), _record("intent_ir")
    path = tmp_path / "selected.jsonl"
    path.write_bytes(_raw(first) + b"\n" + _raw(second) + b"\n")
    source = IRPersistentCatalogSource(path=path)
    assert len(source.load().ir_bindings) == 2
    with pytest.raises(IRPersistentCatalogError, match="record bound"):
        IRPersistentCatalogSource(path=path, max_records=1).load()


@pytest.mark.parametrize("bad_json", ['{"ir_checkpoint":NaN}', '{"ir_checkpoint":1e999}', 'not JSON'])
def test_nonfinite_or_invalid_stored_config_refuses(tmp_path, bad_json):
    record = _record()
    record["huggingface_config"] = bad_json
    with pytest.raises(IRPersistentCatalogError):
        IRPersistentCatalogSource(path=_store(tmp_path, [record])).load()


def test_binding_results_and_requests_are_detached(tmp_path):
    record = _record()
    source = IRPersistentCatalogSource(path=_store(tmp_path, [record]))
    result = source.load()
    request = _selection(record)
    original = deepcopy(request)
    selected = result.resolve_ir_binding(request)
    selected["selected_binding"]["declaration"]["task_id"] = "foreign"
    selected["authority"]["model_loaded"] = True
    sidecar = result.ir_bindings
    sidecar[0]["declaration"]["original_checkpoint_pin"]["sha256"] = "0" * 64
    assert request == original
    assert result.resolve_ir_binding(request)["selected_binding"]["task_id"] is None
    assert result.ir_bindings[0]["checkpoint_sha256"] == record["model_revision"]
    assert not any(result.to_dict()["authority"].values())


@pytest.mark.parametrize("path", ["relative.duckdb", "models.db", "", None])
def test_store_must_be_explicit_absolute_supported_path(path):
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=path)


def test_missing_empty_symlink_and_oversized_store_refuse(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=missing).load()
    missing.write_bytes(b"")
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=missing).load()
    path = _store(tmp_path, [_record()])
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=alias).load()
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=path, max_store_bytes=1).load()
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=path, max_config_bytes=1).load()


def test_store_replacement_at_read_boundary_refuses(tmp_path, monkeypatch):
    path = _store(tmp_path, [_record()])
    source = IRPersistentCatalogSource(path=path)
    original = source._witness
    calls = 0
    def witness():
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement = path.with_suffix(".new")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return original()
    monkeypatch.setattr(source, "_witness", witness)
    with pytest.raises(IRPersistentCatalogError, match="changed"):
        source.load()


def test_no_model_manager_decoder_or_tensor_module_is_imported(tmp_path):
    before = set(sys.modules)
    IRPersistentCatalogSource(path=_store(tmp_path, [_record()])).load()
    added = set(sys.modules) - before
    assert "ipfs_accelerate_py.model_manager" not in added
    assert not any(name == "torch" or name.startswith(("torch.", "sentence_transformers.", "ipfs_datasets_py.")) for name in added)


def test_real_private_duckdb_read_only_refresh_and_table_preservation(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    first, second = _record(), _record("intent_ir")
    path = tmp_path / "private-ir.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE model_metadata(model_id VARCHAR, model_name VARCHAR, model_type VARCHAR, architecture VARCHAR, model_revision VARCHAR, revision_id VARCHAR, huggingface_config VARCHAR)")
    def insert(record):
        connection.execute("INSERT INTO model_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            [record[name] for name in ("model_id", "model_name", "model_type", "architecture", "model_revision", "revision_id")]
            + [_raw(record["huggingface_config"]).decode()])
    insert(first)
    connection.close()
    source = IRPersistentCatalogSource(path=path)
    genuine = duckdb.connect
    observed = []
    def readonly_only(database, **kwargs):
        assert Path(database) == path
        assert kwargs == {"read_only": True}
        observed.append(database)
        return genuine(database, **kwargs)
    monkeypatch.setattr(duckdb, "connect", readonly_only)
    initial_bytes = path.read_bytes()
    assert len(source.load().ir_bindings) == 1
    assert path.read_bytes() == initial_bytes
    connection = genuine(str(path))
    insert(second)
    connection.close()
    accepted_bytes = path.read_bytes()
    assert len(source.refresh().ir_bindings) == 2
    assert path.read_bytes() == accepted_bytes
    assert len(observed) == 2


def test_real_private_duckdb_wrong_table_fails_closed(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "wrong-store.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE unrelated(value INTEGER)")
    connection.close()
    before = path.read_bytes()
    with pytest.raises(IRPersistentCatalogError): IRPersistentCatalogSource(path=path).load()
    assert path.read_bytes() == before
