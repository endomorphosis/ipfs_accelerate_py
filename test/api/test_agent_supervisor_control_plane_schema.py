"""Tests for the normalized control-plane schema (ControlPlaneSchema@1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigrationRunner,
    duckdb_available,
    load_default_catalog,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    CONTROL_PLANE_SCHEMA_INTERFACE,
    CONTROL_PLANE_SCHEMA_MIGRATION_ID,
    DIAGNOSTIC_VIEWS,
    JOIN_CRITICAL_COLUMNS,
    LEASE_SEMANTIC_COLUMNS,
    PINNED_DUCKDB_PACKAGE_SPEC,
    PINNED_QUACK_PROFILE_ID,
    SCHEMA_DOMAINS,
    SUPERVISOR_SERVICE_EXTRA_NAME,
    TASK_IDENTITY_COLUMNS,
    assert_domains_installed,
    assert_join_critical_columns_present,
    assert_pyproject_pins_supervisor_duckdb,
    assert_task_and_lease_semantics,
    default_control_plane_schema,
    install_control_plane_schema,
    list_table_columns,
    load_control_plane_sql,
    prove_fresh_and_upgraded_equivalence,
    supervisor_dependency_pin,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    default_compatibility_profile,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane schema hermetic tests",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_control_plane_schema_contract_inventory() -> None:
    schema = default_control_plane_schema()
    assert schema.interface == CONTROL_PLANE_SCHEMA_INTERFACE
    assert schema.migration_id == CONTROL_PLANE_SCHEMA_MIGRATION_ID
    assert set(SCHEMA_DOMAINS) == {
        "meta",
        "intent",
        "schedule",
        "runtime",
        "git",
        "code",
        "evidence",
        "cache",
        "control",
        "improve",
    }
    for domain in SCHEMA_DOMAINS:
        assert schema.domain_tables[domain]
    assert "ready_task_context_v1" in DIAGNOSTIC_VIEWS
    assert "task_cid" in JOIN_CRITICAL_COLUMNS["tasks"]
    assert "fencing_token" in JOIN_CRITICAL_COLUMNS["leases"]
    payload = schema.to_dict()
    assert payload["interface"] == CONTROL_PLANE_SCHEMA_INTERFACE
    assert payload["sql_checksum"].startswith("sha256:")


def test_default_catalog_includes_control_plane_sql() -> None:
    schema = default_control_plane_schema()
    sql_text = load_control_plane_sql()
    assert "CREATE TABLE tasks" in sql_text
    assert "CREATE TABLE leases" in sql_text
    catalog = load_default_catalog()
    migration = schema.ensure_catalog_contains_schema(catalog)
    assert migration.version == 1
    assert migration.migration_id == CONTROL_PLANE_SCHEMA_MIGRATION_ID
    assert catalog.latest_version >= 1


def test_fresh_and_upgraded_share_schema_fingerprint(tmp_path: Path) -> None:
    proof = prove_fresh_and_upgraded_equivalence(
        fresh_database_path=tmp_path / "fresh.duckdb",
        upgraded_database_path=tmp_path / "upgraded.duckdb",
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    assert proof["equivalent"] is True
    assert proof["to_version"] >= 1
    assert proof["schema_fingerprint"]
    assert proof["migration_id"] == CONTROL_PLANE_SCHEMA_MIGRATION_ID

    # Runner-level empty-to-latest proof agrees with the schema helper.
    catalog = load_default_catalog()
    left = tmp_path / "left.duckdb"
    right = tmp_path / "right.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        left,
        catalog=catalog,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    runner_proof = runner.prove_empty_to_latest_equivalence(
        other_database_path=right
    )
    assert runner_proof["equivalent"] is True
    assert runner_proof["schema_fingerprint"] == proof["schema_fingerprint"]


def test_install_preserves_task_cids_and_lease_semantics(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    report = install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    assert report.changed is True
    assert report.to_version >= 1

    with open_duckdb_connection(db) as connection:
        assert_domains_installed(connection)
        assert_task_and_lease_semantics(connection)
        assert_join_critical_columns_present(connection)

        task_columns = list_table_columns(connection, "tasks")
        for column in TASK_IDENTITY_COLUMNS:
            assert column in task_columns
        assert task_columns[0] == "task_cid"

        lease_columns = list_table_columns(connection, "leases")
        for column in LEASE_SEMANTIC_COLUMNS:
            assert column in lease_columns
        assert lease_columns[0] == "task_cid"

        # First-class identity insert path (no JSON-only key).
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_id, objective_id,
                ordinal, status, revision, semantic_fingerprint,
                canonical_task_key, idempotency_key, created_at, updated_at,
                body_json
            ) VALUES (
                'bafy-task-1', 'task-alias-1', 'bafy-goal-1', NULL, NULL,
                1, 'ready', 1, 'sem-fp-1',
                'task/v1/sem-fp-1', 'idem-1', '2020-01-01T00:00:00Z',
                '2020-01-01T00:00:00Z', '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                owner_session_id, logical_epoch, fencing_token, expires_at_ms,
                attempt, state, started_at_ms, release_reason,
                retry_not_before_ms, revision, body_json
            ) VALUES (
                'bafy-task-1', 'bafy-claim-1', 'bafy-resolution-1', 'did:worker:1',
                'session-1', 1, 7, 1700000000000,
                1, 'claimed', 1699999900000, NULL,
                0, 1, '{}'
            )
            """
        )
        row = connection.execute(
            """
            SELECT t.task_cid, l.fencing_token, l.state
            FROM tasks t
            JOIN leases l ON l.task_cid = t.task_cid
            WHERE t.task_cid = 'bafy-task-1'
            """
        ).fetchone()
        assert str(row[0]) == "bafy-task-1"
        assert int(row[1]) == 7
        assert str(row[2]) == "claimed"

        ready = connection.execute(
            """
            SELECT task_cid, lease_fencing_token
            FROM ready_task_context_v1
            WHERE task_cid = 'bafy-task-1'
            """
        ).fetchone()
        assert str(ready[0]) == "bafy-task-1"
        assert int(ready[1]) == 7


def test_no_join_critical_identity_only_in_opaque_json(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    with open_duckdb_connection(db) as connection:
        assert_join_critical_columns_present(connection)
        for table_name, columns in JOIN_CRITICAL_COLUMNS.items():
            present = set(list_table_columns(connection, table_name))
            for column in columns:
                assert column in present
                assert not column.endswith("_json")


def test_diagnostic_views_exist(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    with open_duckdb_connection(db) as connection:
        for view_name in DIAGNOSTIC_VIEWS:
            columns = list_table_columns(connection, view_name)
            assert columns, f"missing diagnostic view {view_name}"


def test_supervisor_duckdb_quack_profile_is_pinned() -> None:
    pin = supervisor_dependency_pin()
    assert pin.extra_name == SUPERVISOR_SERVICE_EXTRA_NAME
    assert pin.duckdb_package_spec == PINNED_DUCKDB_PACKAGE_SPEC
    assert pin.quack_profile_id == PINNED_QUACK_PROFILE_ID
    assert pin.duckdb_major == 1
    assert pin.duckdb_minor == 5

    profile = default_compatibility_profile()
    assert profile.profile_id == PINNED_QUACK_PROFILE_ID
    assert profile.duckdb_version_prefix == "1.5"

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    verified = assert_pyproject_pins_supervisor_duckdb(pyproject)
    assert verified.duckdb_package_spec == "duckdb>=1.5.0,<1.6.0"
    assert "agent-supervisor" in pyproject
    assert "duckdb>=1.5.0,<1.6.0" in pyproject
    assert "*.sql" in pyproject


def test_replay_install_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    first = install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    second = install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    assert first.changed is True
    assert second.changed is False
    assert second.schema_fingerprint == first.schema_fingerprint
    assert second.to_version == first.to_version
