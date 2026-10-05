"""Actual read-only catalog nominations remain separate from model qualification."""
from copy import deepcopy
from pathlib import Path
import sys

import duckdb
import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.task_ir_selection as owner
from test.test_ir_persistent_catalog import _raw, _record, _selection


def store(tmp_path, records, *, database=False):
    path = tmp_path / ("selected-models.duckdb" if database else "selected-models.json")
    if not database:
        path.write_bytes(_raw(records))
        return path
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE model_metadata(model_id VARCHAR, model_name VARCHAR, model_type VARCHAR, architecture VARCHAR, model_revision VARCHAR, revision_id VARCHAR, huggingface_config VARCHAR)")
        connection.execute("CREATE TABLE unrelated(value INTEGER)")
        connection.execute("INSERT INTO unrelated VALUES (73)")
        for record in records:
            connection.execute("INSERT INTO model_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                [record[name] for name in ("model_id", "model_name", "model_type", "architecture", "model_revision", "revision_id")]
                + [_raw(record["huggingface_config"]).decode()])
    return path


@pytest.mark.parametrize("database", [False, True])
def test_parallel_family_and_dimension_nominations_preserve_native_declarations(tmp_path, database, monkeypatch):
    records = [_record(family, dimension, "latent" if dimension == 8 else "input_embedding",
                       checkpoint=f"{family}-{dimension}")
               for family in ("codebase_ir", "security_ir", "legal_ir", "intent_ir")
               for dimension in (8, 384, 768)]
    path = store(tmp_path, records, database=database)
    before, requests = path.read_bytes(), [_selection(record) for record in records]
    original_requests = deepcopy(requests)
    actual_connect, opened = duckdb.connect, []

    def observed_connect(database, **kwargs):
        assert Path(database) == path and kwargs == {"read_only": True}
        opened.append(database)
        return actual_connect(database, **kwargs)

    monkeypatch.setattr(duckdb, "connect", observed_connect)
    resolved = owner.resolve_task_ir_selections(catalog_path=path, selections=requests)
    assert len(resolved) == 12 and requests == original_requests
    assert path.read_bytes() == before
    assert len(opened) == (13 if database else 0)
    assert len({row["binding_snapshot_revision"] for row in resolved}) == 1
    assert len({row["catalog_revision"] for row in resolved}) == 1
    for request, result in zip(requests, resolved):
        assert result["schema"] == "ir-persisted-binding-resolution/v1"
        assert result["storage_path"] == str(path)
        assert all(result["selected_binding"][name] == value for name, value in request.items())
        assert result["selected_binding"]["schema_version"] is None
        assert result["selected_binding"]["task_id"] is None
        assert result["selected_binding"]["original_checkpoint_bytes_verified"] is False
        assert result["selected_binding"]["runtime_observation_performed"] is False
        assert not any(result["authority"].values())
        assert not Path(result["selected_binding"]["declaration"]["original_checkpoint_pin"]["path"]).exists()
    if database:
        with actual_connect(str(path), read_only=True) as connection:
            assert connection.execute("SELECT value FROM unrelated").fetchall() == [(73,)]
            assert connection.execute("SELECT count(*) FROM model_metadata").fetchone()[0] == 12


def test_separate_schema_and_decoder_tasks_do_not_collide(tmp_path):
    records = [_record("legal_ir", 384, checkpoint=task + ":" + schema,
        schema_version=schema, task_id=task, profile_id="legal-reviewed", format_id="legal-json")
        for task, schema in (("semantic_ir_reconstruction", "legal-ir/v1"),
                             ("legal_text_reconstruction", "legal-text/v1"),
                             ("semantic_ir_reconstruction", "legal-ir/v2"))]
    result = owner.resolve_task_ir_selections(catalog_path=store(tmp_path, records),
                                            selections=[_selection(row) for row in records])
    assert {(row["selected_binding"]["schema_version"], row["selected_binding"]["task_id"])
            for row in result} == {("legal-ir/v1", "semantic_ir_reconstruction"),
                                  ("legal-text/v1", "legal_text_reconstruction"),
                                  ("legal-ir/v2", "semantic_ir_reconstruction")}


@pytest.mark.parametrize("different_checkpoint", [False, True])
def test_competing_assets_for_one_namespace_refuse_before_store_access(tmp_path, monkeypatch, different_checkpoint):
    first, second = _record(checkpoint="first"), _record(checkpoint="second" if different_checkpoint else "first")
    def unexpected(self):
        pytest.fail("competing namespace must refuse before catalog access")
    monkeypatch.setattr(owner.IRPersistentCatalogSource, "load", unexpected)
    with pytest.raises(ValueError, match="complete namespace"):
        owner.resolve_task_ir_selections(catalog_path=tmp_path / "never-created.json",
                                        selections=[_selection(first), _selection(second)])
    assert not (tmp_path / "never-created.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "bool_dimension", "family", "task_object", "latest", "upper_sha"])
def test_invalid_native_selector_refuses_before_store_access(tmp_path, monkeypatch, mutation):
    request = _selection(_record())
    if mutation == "missing": request.pop("schema_version")
    elif mutation == "extra": request["max_tokens"] = 512
    elif mutation == "bool_dimension": request["dimension"] = True
    elif mutation == "family": request["ir_family_id"] = "arbitrary_ir"
    elif mutation == "task_object": request["task_id"] = {}
    elif mutation == "latest": request["record_id"] = "latest"
    else: request["checkpoint_sha256"] = request["checkpoint_sha256"].upper()
    def unexpected(self):
        pytest.fail("invalid selector must refuse before catalog access")
    monkeypatch.setattr(owner.IRPersistentCatalogSource, "load", unexpected)
    with pytest.raises(ValueError):
        owner.resolve_task_ir_selections(catalog_path=tmp_path / "never-created.json", selections=[request])


@pytest.mark.parametrize("field,value", [("dimension", 768), ("schema_version", "legal-ir/v2"),
    ("task_id", "new-head"), ("profile_id", "new-profile"), ("format_id", "new-format"),
    ("role", "new-role"), ("checkpoint_sha256", "0" * 64)])
def test_matching_record_id_cannot_mask_changed_namespace_or_checkpoint(tmp_path, field, value):
    record = _record()
    request = _selection(record)
    request[field] = value
    path = store(tmp_path, [record])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="exact unique"):
        owner.resolve_task_ir_selections(catalog_path=path, selections=[request])
    assert path.read_bytes() == before


def test_fresh_resolution_refuses_stale_request_after_persisted_identity_change(tmp_path):
    record = _record(task_id="semantic_ir_reconstruction", schema_version="legal-ir/v1")
    path, request = store(tmp_path, [record]), _selection(record)
    first = owner.resolve_task_ir_selections(catalog_path=path, selections=[request])
    record["huggingface_config"]["ir_checkpoint"]["task_id"] = "legal_text_reconstruction"
    path.write_bytes(_raw([record]))
    with pytest.raises(ValueError, match="exact unique"):
        owner.resolve_task_ir_selections(catalog_path=path, selections=[request])
    changed = owner.resolve_task_ir_selections(catalog_path=path, selections=[_selection(record)])
    assert first[0]["binding_snapshot_revision"] != changed[0]["binding_snapshot_revision"]


def test_catalog_change_between_genuine_resolutions_refuses_complete_nomination(tmp_path, monkeypatch):
    first, second = _record("legal_ir"), _record("intent_ir")
    path = store(tmp_path, [first, second])
    actual = owner.IRPersistentCatalogSource.resolve_ir_binding
    calls = []
    def changed(self, request):
        result = actual(self, request)
        calls.append(True)
        if len(calls) == 1:
            second["huggingface_config"]["ir_checkpoint"]["trained"] = True
            path.write_bytes(_raw([first, second]))
        return result
    monkeypatch.setattr(owner.IRPersistentCatalogSource, "resolve_ir_binding", changed)
    with pytest.raises(ValueError, match="generation changed"):
        owner.resolve_task_ir_selections(catalog_path=path, selections=[_selection(first), _selection(second)])
    assert calls == [True, True]


def test_final_catalog_observation_refuses_change_after_last_genuine_resolution(tmp_path, monkeypatch):
    record = _record()
    path = store(tmp_path, [record])
    actual = owner.IRPersistentCatalogSource.resolve_ir_binding
    def changed(self, request):
        result = actual(self, request)
        record["huggingface_config"]["ir_checkpoint"]["trained"] = True
        path.write_bytes(_raw([record]))
        return result
    monkeypatch.setattr(owner.IRPersistentCatalogSource, "resolve_ir_binding", changed)
    with pytest.raises(ValueError, match="generation changed"):
        owner.resolve_task_ir_selections(catalog_path=path, selections=[_selection(record)])


def test_readiness_and_unknown_fields_remain_declarations_not_new_observations(tmp_path):
    record = _record(runtime_ready=True, teacher_qualified=True, proof_authority=True,
                     max_tokens=None, span_policy=None, inventory_id=None, hub_revision=None)
    result = owner.resolve_task_ir_selections(catalog_path=store(tmp_path, [record]), selections=[_selection(record)])[0]
    declaration = result["selected_binding"]["declaration"]
    assert all(declaration[name] is True for name in ("runtime_ready", "teacher_qualified", "proof_authority"))
    assert all(declaration[name] is None for name in ("max_tokens", "span_policy", "inventory_id", "hub_revision"))
    assert not any(result["authority"].values())


def test_detached_component_is_not_relabelled_as_an_external_embedding_lane(tmp_path):
    record = _record("intent_ir", None, "source_tokens", auxiliary_dimensions={"latent_width": 8})
    result = owner.resolve_task_ir_selections(catalog_path=store(tmp_path, [record]), selections=[_selection(record)])[0]
    assert result["selected_binding"]["dimension"] is None
    assert result["selected_binding"]["dimension_role"] == "source_tokens"
    assert result["selected_binding"]["declaration"]["auxiliary_dimensions"] == {"latent_width": 8}
    assert result["authority"]["runtime_admitted"] is False


@pytest.mark.parametrize("selections", [[], (), None, [None], [True]])
def test_selection_population_and_exact_types_are_bounded(tmp_path, selections):
    with pytest.raises(ValueError):
        owner.resolve_task_ir_selections(catalog_path=tmp_path / "absent.json", selections=selections)
    assert not (tmp_path / "absent.json").exists()


@pytest.mark.parametrize("count,accepted", [(16, True), (17, False)])
def test_exact_upper_selection_population(tmp_path, count, accepted):
    records = [_record(checkpoint=str(index), task_id=f"decoder-head-{index}") for index in range(count)]
    path = store(tmp_path, records)
    if accepted:
        assert len(owner.resolve_task_ir_selections(catalog_path=path, selections=[_selection(row) for row in records])) == 16
    else:
        with pytest.raises(ValueError, match="1 to 16"):
            owner.resolve_task_ir_selections(catalog_path=path, selections=[_selection(row) for row in records])


@pytest.mark.parametrize("path_kind", ["missing", "relative", "string", "symlink", "traversal", "wrong_table"])
def test_catalog_path_refuses_without_creating_or_modifying_a_store(tmp_path, path_kind):
    path = tmp_path / "absent.duckdb"
    if path_kind == "relative": path = Path("absent.duckdb")
    elif path_kind == "string": path = str(path)
    elif path_kind == "symlink":
        target = store(tmp_path, [_record()])
        path = tmp_path / "alias.json"
        path.symlink_to(target)
    elif path_kind == "traversal": path = tmp_path / ".." / tmp_path.name / "absent.duckdb"
    elif path_kind == "wrong_table":
        with duckdb.connect(str(path)) as connection: connection.execute("CREATE TABLE other(value INTEGER)")
    before = path.read_bytes() if isinstance(path, Path) and path.is_file() else None
    with pytest.raises(ValueError):
        owner.resolve_task_ir_selections(catalog_path=path, selections=[_selection(_record())])
    if before is not None: assert path.read_bytes() == before
    elif isinstance(path, Path) and path.is_absolute(): assert not path.exists()


def test_selected_results_are_detached_and_no_model_modules_are_loaded(tmp_path):
    record, path = _record(), store(tmp_path, [_record()])
    request = _selection(record)
    modules_before = set(sys.modules)
    result = owner.resolve_task_ir_selections(catalog_path=path, selections=[request])
    added = set(sys.modules) - modules_before
    assert "ipfs_accelerate_py.model_manager" not in added
    assert not any(name == "torch" or name.startswith(("torch.", "sentence_transformers.", "ipfs_datasets_py.")) for name in added)
    result[0]["selected_binding"]["declaration"]["task_id"] = "foreign"
    result[0]["authority"]["model_loaded"] = True
    again = owner.resolve_task_ir_selections(catalog_path=path, selections=[request])
    assert again[0]["selected_binding"]["task_id"] is None
    assert not any(again[0]["authority"].values())


@pytest.mark.parametrize("database", [False, True])
def test_duplicate_persisted_json_identity_field_refuses_without_store_changes(tmp_path, database):
    record = _record()
    request = _selection(record)
    path = store(tmp_path, [record], database=database)
    config = _raw(record["huggingface_config"]).replace(
        b'"dimension":384', b'"dimension":384,"dimension":768').decode()
    if database:
        with duckdb.connect(str(path)) as connection:
            connection.execute("UPDATE model_metadata SET huggingface_config = ?", [config])
    else:
        record["huggingface_config"] = config
        path.write_bytes(_raw([record]))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate JSON field"):
        owner.resolve_task_ir_selections(catalog_path=path, selections=[request])
    assert path.read_bytes() == before
