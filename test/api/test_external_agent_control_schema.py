"""EAAEF-090: versioned mutable external-agent control-plane schema."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.external_agent_control_schema import (
    COLLECTION_AUTHORITY_HISTORY_ONLY,
    COLLECTION_AUTHORITY_MUTABLE,
    DUCKLAKE_GRANTS_CURRENT_AUTHORITY,
    DUCKLAKE_HISTORY_ONLY_MARKER,
    EXTERNAL_AGENT_CONTROL_SCHEMA_INTERFACE,
    EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION,
    EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION_HISTORY,
    HISTORY_ONLY_COLLECTIONS,
    JOIN_CRITICAL_IDENTITIES,
    MUTABLE_COLLECTIONS,
    MUTABLE_COORDINATION_AUTHORITY,
    REQUIRED_COLLECTION_FAMILIES,
    REQUIRED_COLLECTIONS,
    ExternalAgentControlSchema,
    ExternalAgentControlSchemaError,
    admit_schema_version,
    assert_schema_version_monotonic,
    collection_authority,
    default_external_agent_control_schema,
    reject_ducklake_authority,
    required_collections,
    schema_fingerprint,
)


def test_schema_lists_required_collections() -> None:
    schema = default_external_agent_control_schema()
    listed = required_collections(schema)

    assert schema.INTERFACE == EXTERNAL_AGENT_CONTROL_SCHEMA_INTERFACE
    assert listed == REQUIRED_COLLECTIONS
    assert tuple(schema.families) == REQUIRED_COLLECTION_FAMILIES
    assert set(MUTABLE_COLLECTIONS) == set(REQUIRED_COLLECTION_FAMILIES)
    for family in (
        "repositories",
        "handoffs",
        "sessions",
        "runs",
        "goal_revisions",
        "plan_revisions",
        "task_revisions",
        "conflicts",
        "processes",
        "containers",
        "claims",
        "leases",
        "reservations",
        "approvals",
        "events",
        "checkpoints",
        "validations",
        "proofs",
        "merge",
        "artifacts",
        "migrations",
        "cursors",
    ):
        assert family in REQUIRED_COLLECTION_FAMILIES
        assert MUTABLE_COLLECTIONS[family]
        assert all(table in listed for table in MUTABLE_COLLECTIONS[family])

    assert "handoffs" in listed
    assert "runs" in listed
    assert "goal_revisions" in listed
    assert "plan_revisions" in listed
    assert "task_revisions" in listed
    assert "conflicts" in listed
    assert "processes" in listed
    assert "containers" in listed
    assert "leases" in listed
    assert "budget_reservations" in listed
    assert "approvals" in listed
    assert "domain_events" in listed
    assert "checkpoints" in listed
    assert "validation_runs" in listed
    assert "completion_receipts" in listed
    assert "merge_attempts" in listed
    assert "artifacts" in listed
    assert "schema_migrations" in listed
    assert "outbox_cursors" in listed
    assert listed == tuple(dict.fromkeys(listed))
    assert not set(listed) & set(HISTORY_ONLY_COLLECTIONS)
    payload = schema.to_dict()
    assert payload["required_collections"] == list(listed)
    for table, column in JOIN_CRITICAL_IDENTITIES:
        assert table in listed
        assert not column.endswith("_json")


def test_schema_version_is_monotonic() -> None:
    schema = default_external_agent_control_schema()
    history = assert_schema_version_monotonic()

    assert isinstance(schema.schema_version, int)
    assert schema.schema_version >= 1
    assert schema.schema_version == EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION
    assert history == EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION_HISTORY
    assert history[-1] == schema.schema_version
    assert admit_schema_version(schema.schema_version) == schema.schema_version
    assert admit_schema_version(schema.schema_version + 1) == schema.schema_version + 1

    with pytest.raises(ExternalAgentControlSchemaError, match="recede"):
        admit_schema_version(schema.schema_version, previous=schema.schema_version + 1)
    with pytest.raises(ExternalAgentControlSchemaError, match=">= 1"):
        admit_schema_version(0)
    with pytest.raises(ExternalAgentControlSchemaError, match="skip"):
        admit_schema_version(schema.schema_version + 2)
    with pytest.raises(ExternalAgentControlSchemaError, match="monotonic"):
        assert_schema_version_monotonic((1, 1))
    with pytest.raises(ExternalAgentControlSchemaError, match="monotonic"):
        assert_schema_version_monotonic((2, 3))
    with pytest.raises(ExternalAgentControlSchemaError, match="empty"):
        assert_schema_version_monotonic(())
    with pytest.raises(ExternalAgentControlSchemaError, match="latest"):
        assert_schema_version_monotonic((1, 2), current_version=1)


def test_schema_fingerprint_is_stable() -> None:
    first = default_external_agent_control_schema()
    second = ExternalAgentControlSchema()

    assert first.fingerprint() == second.fingerprint()
    assert schema_fingerprint() == first.fingerprint()
    assert schema_fingerprint(first) == schema_fingerprint(second)
    assert first.fingerprint() == first.fingerprint()
    assert first.fingerprint().startswith("b")
    assert first.to_dict()["schema_fingerprint"] == first.fingerprint()
    assert first.to_dict() == second.to_dict()


def test_ducklake_is_history_only_and_never_authority() -> None:
    schema = default_external_agent_control_schema()
    marker = reject_ducklake_authority()

    assert schema.ducklake_role == DUCKLAKE_HISTORY_ONLY_MARKER
    assert schema.ducklake_grants_current_authority is False
    assert DUCKLAKE_GRANTS_CURRENT_AUTHORITY is False
    assert schema.mutable_coordination_authority == MUTABLE_COORDINATION_AUTHORITY
    assert marker["role"] == DUCKLAKE_HISTORY_ONLY_MARKER
    assert marker["grants_current_authority"] is False
    assert marker["authority"] == COLLECTION_AUTHORITY_HISTORY_ONLY
    for table in HISTORY_ONLY_COLLECTIONS:
        assert table.startswith("ducklake_")
        assert collection_authority(table) == COLLECTION_AUTHORITY_HISTORY_ONLY
        assert table not in REQUIRED_COLLECTIONS
    assert collection_authority("leases") == COLLECTION_AUTHORITY_MUTABLE

    with pytest.raises(ExternalAgentControlSchemaError, match="never grants"):
        reject_ducklake_authority(grants_current_authority=True)
    with pytest.raises(ExternalAgentControlSchemaError, match="never grants"):
        reject_ducklake_authority("ducklake_epochs", role="authority")
    with pytest.raises(ExternalAgentControlSchemaError, match="unknown"):
        collection_authority("not_a_control_plane_table")
    with pytest.raises(ExternalAgentControlSchemaError, match="DuckLake never grants"):
        ExternalAgentControlSchema(ducklake_grants_current_authority=True)
    with pytest.raises(ExternalAgentControlSchemaError, match="history-only"):
        ExternalAgentControlSchema(ducklake_role="authority")
