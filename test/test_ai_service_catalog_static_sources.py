"""Contract tests for persistent and static AI catalog source adapters."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from ipfs_accelerate_py.model_catalog import (
    CatalogSnapshot,
    LifecycleState,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    canonical_json,
    provider_identity,
)
from ipfs_accelerate_py.model_catalog.sources import (
    DEFAULT_PERSISTENT_PRECEDENCE,
    MAX_SOURCE_BYTES,
    PersistentCatalogSource,
    StaticCatalogSource,
    adapt_persistent_source,
    adapt_static_source,
)


def test_static_legacy_fields_partial_rows_and_unknown_runtime_state():
    result = adapt_static_source(
        [
            {
                "provider_name": "OpenAI",
                "model_id": "gpt-4",
                "model_name": "GPT-4",
                "pipeline_types": ["conversational"],
                "context_length": 8192,
                "supports_streaming": True,
                # A static availability claim is descriptive, not a probe.
                "status": "healthy",
            },
            {"provider": "Anthropic"},
        ]
    )

    assert [item.name for item in result.providers] == ["anthropic", "openai"]
    assert [item.name for item in result.models] == ["gpt-4"]
    model = result.models[0]
    assert model.display_name == "GPT-4"
    assert set(model.capabilities[0].operations) == {
        Operation.TEXT_CHAT,
        Operation.STREAM,
    }
    assert model.capabilities[0].max_context_tokens == 8192
    assert model.lifecycle == LifecycleState.DECLARED
    for descriptor in result.providers + result.models:
        assert descriptor.state.known is None
        assert descriptor.state.configured is None
        assert descriptor.state.authorized is None
        assert descriptor.state.reachable is None
        assert descriptor.state.healthy is None
        assert descriptor.state.routable is None


def test_persistent_model_manager_mapping_preserves_revision_and_timestamps():
    result = adapt_persistent_source(
        {
            "source_revision": "manager-rev-7",
            "created_at": "2026-07-01T00:00:00Z",
            "updated_at": "2026-07-26T12:30:00+02:00",
            "precedence": 42,
            "models": {
                "org/example-model": {
                    "model_id": "org/example-model",
                    "model_name": "Example Model",
                    "model_type": "language_model",
                    "architecture": "ExampleForCausalLM",
                    # ModelManager historically persisted naive datetimes.
                    "revision_created_at": "2026-07-20T09:15:00",
                    "model_revision": "weights-rev-3",
                }
            },
        }
    )

    assert result.precedence == 42
    assert result.source_revision == "manager-rev-7"
    assert result.metadata.created_at == "2026-07-01T00:00:00.000000Z"
    assert result.metadata.updated_at == "2026-07-26T10:30:00.000000Z"
    assert result.snapshot.created_at == "2026-07-26T10:30:00.000000Z"
    assert result.models[0].name == "example-model"
    assert result.models[0].display_name == "Example Model"
    assert result.models[0].architecture == "ExampleForCausalLM"
    assert result.models[0].provenance[0].source_record_id == "org/example-model"
    assert result.models[0].provenance[0].observed_at == "2026-07-20T09:15:00.000000Z"
    labels = dict(result.models[0].labels)
    assert labels["source.revision"] == "weights-rev-3"
    assert labels["source.updated-at"] == "2026-07-20T09:15:00.000000Z"


def test_persistent_defaults_have_explicit_higher_precedence():
    result = adapt_persistent_source([])
    assert result.precedence == DEFAULT_PERSISTENT_PRECEDENCE
    assert result.source == "model-manager.persistent"


def test_malformed_rows_are_reported_without_discarding_valid_rows():
    result = adapt_static_source(
        [
            {"provider": "valid", "model": "chat", "operations": ["text.chat"]},
            {"provider": "bad", "model": "unknown-op", "operations": ["telepathy"]},
            {"model": 12},
            None,
            {"provider": "also-valid"},
        ]
    )

    assert [item.name for item in result.providers] == ["also-valid", "valid"]
    assert [item.name for item in result.models] == ["chat"]
    malformed = [item for item in result.diagnostics if item.code == "malformed_row"]
    assert len(malformed) == 3
    assert [item.index for item in malformed] == [1, 2, 3]


def test_secret_shaped_fields_and_values_are_redacted_without_mutating_input():
    source = [
        {
            "provider": "safe",
            "model": "chat",
            "description": "Bearer abcdefghijklmnopqrstuvwxyz",
            "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
            "labels": {
                "owner": "catalog",
                "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            },
        }
    ]
    result = adapt_static_source(source)
    rendered = canonical_json(result.to_dict(), reject_secrets=False)

    assert result.redacted_fields == 3
    assert result.models[0].description == "[REDACTED]"
    assert dict(result.models[0].labels)["owner"] == "catalog"
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "Bearer abcdefghijklmnopqrstuvwxyz" not in rendered
    assert source[0]["api_key"] == "sk-abcdefghijklmnopqrstuvwxyz"
    assert any(item.code == "redacted" for item in result.diagnostics)


def test_duplicate_seeds_have_stable_identity_and_precedence_is_inspectable():
    rows = [
        {
            "provider": "OpenAI",
            "model_id": "openai/gpt-4",
            "description": "low",
            "precedence": 1,
        },
        {
            "provider_name": "openai",
            "model": "gpt-4",
            "description": "high",
            "priority": 9,
        },
    ]
    forward = adapt_static_source(rows, precedence=5, revision="seed-r1")
    reverse = adapt_static_source(list(reversed(rows)), precedence=5, revision="seed-r1")

    assert len(forward.providers) == len(forward.models) == 1
    assert forward.models[0].model_id == reverse.models[0].model_id
    assert forward.models[0].description == reverse.models[0].description == "high"
    assert forward.snapshot.revision == reverse.snapshot.revision
    assert forward.precedence == 5
    assert dict(forward.models[0].labels)["source.precedence"] == "9"
    assert dict(forward.models[0].labels)["source.revision"] == "seed-r1"
    assert forward.providers[0].provider_id == provider_identity("openai")


def test_output_order_is_deterministic_for_unordered_inventories():
    left = adapt_static_source(
        {
            "zeta": ["third", "first"],
            "alpha": ["second"],
        }
    )
    right = adapt_static_source(
        {
            "alpha": ["second"],
            "zeta": ["first", "third"],
        }
    )

    assert left.snapshot.revision == right.snapshot.revision
    assert [item.provider_id for item in left.providers] == sorted(
        item.provider_id for item in left.providers
    )
    assert [item.model_id for item in left.models] == sorted(item.model_id for item in left.models)


def test_explicit_json_and_jsonl_paths_are_the_only_file_inputs(tmp_path):
    document = tmp_path / "models.json"
    document.write_text(
        json.dumps(
            {
                "revision": "file-r1",
                "models": {
                    "legacy-id": {
                        "provider": "local",
                        "model_name": "legacy",
                        "model_type": "embedding_model",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    from_json = PersistentCatalogSource(path=document).load()
    assert from_json.source_revision == "file-r1"
    assert from_json.models[0].capabilities[0].operations == (Operation.EMBEDDING_GENERATE,)

    lines = tmp_path / "models.jsonl"
    lines.write_text(
        '{"provider":"b","model":"two"}\nnot-json\n{"provider":"a","model":"one"}\n',
        encoding="utf-8",
    )
    from_jsonl = StaticCatalogSource(path=lines).read()
    assert {item.name for item in from_jsonl.models} == {"one", "two"}
    assert len([item for item in from_jsonl.diagnostics if item.code == "malformed_row"]) == 1

    with pytest.raises(ValueError, match="exactly one"):
        StaticCatalogSource()
    with pytest.raises(ValueError, match="exactly one"):
        StaticCatalogSource([], path=document)
    with pytest.raises(ValueError, match="local file"):
        StaticCatalogSource(path=tmp_path / "missing.json").load()


def test_explicit_duckdb_model_manager_path_is_read_only_and_bounded(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "manager.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE model_metadata (
            model_id VARCHAR,
            model_name VARCHAR,
            model_type VARCHAR,
            architecture VARCHAR,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        INSERT INTO model_metadata VALUES
        ('org/db-model', 'Database Model', 'language_model',
         'DbArchitecture', '2026-07-01 00:00:00', '2026-07-02 00:00:00')
        """
    )
    connection.close()

    result = PersistentCatalogSource(path=path).load()

    assert result.models[0].name == "db-model"
    assert result.models[0].display_name == "Database Model"
    assert result.models[0].architecture == "DbArchitecture"
    assert result.models[0].provenance[0].observed_at == ("2026-07-02T00:00:00.000000Z")


def test_injected_records_do_not_trigger_path_reads(monkeypatch):
    def fail_read(*args, **kwargs):
        raise AssertionError("unexpected filesystem read")

    monkeypatch.setattr("pathlib.Path.read_text", fail_read)
    result = adapt_static_source([{"provider": "local", "model": "offline"}])
    assert result.models[0].name == "offline"


def test_counts_and_fields_are_bounded():
    with pytest.raises(ValueError, match="record count"):
        adapt_static_source(
            [{"provider": "p", "model": str(index)} for index in range(3)],
            max_records=2,
        )
    result = adapt_static_source(
        [
            {"provider": "p", "model": "valid"},
            {"provider": "p", "model": "x" * 129},
            {"provider": "p", "model": "description", "description": "x" * 4097},
        ]
    )
    assert [item.name for item in result.models] == ["valid"]
    assert result.error_count == 2


def test_oversized_explicit_source_is_rejected_before_read(tmp_path):
    path = tmp_path / "too-large.json"
    with path.open("wb") as stream:
        stream.truncate(MAX_SOURCE_BYTES + 1)

    with pytest.raises(ValueError, match="exceeds"):
        StaticCatalogSource(path=path).load()


def test_empty_sources_return_valid_stable_snapshots():
    values = (
        adapt_static_source([]),
        adapt_static_source({}),
        adapt_static_source({"records": []}),
        adapt_persistent_source({"models": {}}),
    )
    for result in values:
        assert result.providers == ()
        assert result.models == ()
        assert result.diagnostics == ()
    assert values[0].snapshot.revision == values[1].snapshot.revision


def test_datetime_objects_are_preserved_without_using_the_current_clock():
    result = adapt_persistent_source(
        [
            {
                "model_id": "local/model",
                "model_name": "Model",
                "created_at": datetime(2026, 7, 1, 12, 0),
                "updated_at": datetime.fromisoformat("2026-07-02T12:00:00+02:00"),
            }
        ]
    )
    assert result.models[0].provenance[0].observed_at == ("2026-07-02T10:00:00.000000Z")


def test_canonical_snapshot_inputs_retain_identity_but_not_dynamic_state():
    provider = ProviderDescriptor(
        name="canonical",
        state=OperationalState(healthy=True),
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="model",
        state=OperationalState(reachable=True),
    )
    original = CatalogSnapshot(providers=(provider,), models=(model,))

    result = adapt_static_source(original.to_dict())

    assert result.source_revision == original.revision
    assert result.providers[0].provider_id == provider.provider_id
    assert result.models[0].model_id == model.model_id
    assert result.providers[0].state.healthy is None
    assert result.models[0].state.reachable is None
