"""Contract tests for ControlPlaneSchema@1 and base control-plane SQL."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigrationRunner,
    duckdb_available,
    load_default_catalog,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    BASE_MIGRATION_ID,
    BASE_MIGRATION_VERSION,
    BOOKKEEPING_TABLES,
    CONTROL_PLANE_SCHEMA_INTERFACE,
    CONTROL_PLANE_SCHEMA_VERSION,
    DIAGNOSTIC_VIEWS,
    DOMAIN_TABLES,
    LEASE_IDENTITY_COLUMNS,
    PINNED_DUCKDB_REQUIREMENT,
    PINNED_OPTIONAL_EXTRA,
    SchemaDomain,
    TASK_IDENTITY_COLUMNS,
    ControlPlaneDependencyProfile,
    ControlPlaneSchema,
    ControlPlaneSchemaError,
    all_domain_tables,
    assert_pyproject_pins_supervisor_duckdb,
    assert_sql_preserves_identity_columns,
    base_migration_sql_path,
    load_base_migration_sql,
    required_tables,
    required_views,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_control_plane_schema_contract_defaults() -> None:
    schema = ControlPlaneSchema.default()
    assert schema.INTERFACE == CONTROL_PLANE_SCHEMA_INTERFACE
    assert schema.version == CONTROL_PLANE_SCHEMA_VERSION
    assert schema.migration_id == BASE_MIGRATION_ID
    assert schema.migration_version == BASE_MIGRATION_VERSION
    assert SchemaDomain.META.value in schema.domains
    assert SchemaDomain.INTENT.value in schema.domains
    assert SchemaDomain.SCHEDULE.value in schema.domains
    assert SchemaDomain.RUNTIME.value in schema.domains
    assert SchemaDomain.GIT.value in schema.domains
    assert SchemaDomain.CODE.value in schema.domains
    assert SchemaDomain.EVIDENCE.value in schema.domains
    assert SchemaDomain.CACHE.value in schema.domains
    assert SchemaDomain.CONTROL.value in schema.domains
    assert SchemaDomain.IMPROVE.value in schema.domains
    assert schema.tables == all_domain_tables()
    assert schema.views == DIAGNOSTIC_VIEWS
    assert schema.bookkeeping_tables == BOOKKEEPING_TABLES
    payload = schema.to_dict()
    assert payload["interface"] == CONTROL_PLANE_SCHEMA_INTERFACE
    assert payload["migration_id"] == BASE_MIGRATION_ID
    assert "tasks" in payload["tables"]
    assert "leases" in payload["tables"]
    assert "ready_task_context_v1" in payload["views"]


def test_dependency_profile_pins_duckdb_1_5_window() -> None:
    profile = ControlPlaneDependencyProfile.default()
    assert profile.optional_extra == PINNED_OPTIONAL_EXTRA
    assert profile.duckdb_requirement == PINNED_DUCKDB_REQUIREMENT
    assert profile.duckdb_major == 1
    assert profile.duckdb_minor == 5
    assert profile.extension_name == "quack"
    with pytest.raises(ControlPlaneSchemaError, match="1.5"):
        ControlPlaneDependencyProfile(duckdb_requirement="duckdb>=0.7.0")
    with pytest.raises(ControlPlaneSchemaError, match="1.5"):
        ControlPlaneDependencyProfile(duckdb_requirement="duckdb>=1.5.0")


def test_pyproject_declares_agent_supervisor_duckdb_pin() -> None:
    report = assert_pyproject_pins_supervisor_duckdb(REPO_ROOT / "pyproject.toml")
    assert report["optional_extra"] == "agent-supervisor"
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "agent-supervisor" in text
    assert "duckdb>=1.5.0,<1.6.0" in text
    assert "*.sql" in text


def test_base_sql_file_exists_and_preserves_identity_columns() -> None:
    path = base_migration_sql_path()
    assert path.is_file()
    sql = load_base_migration_sql()
    assert "CREATE TABLE tasks" in sql
    assert "CREATE TABLE leases" in sql
    assert "CREATE VIEW ready_task_context_v1" in sql
    assert_sql_preserves_identity_columns(sql)
    for column in TASK_IDENTITY_COLUMNS:
        assert column in sql
    for column in LEASE_IDENTITY_COLUMNS:
        assert column in sql
    # Every admitted domain contributes at least one table declaration.
    for domain, tables in DOMAIN_TABLES.items():
        assert tables, f"domain {domain} has no tables"
        for table in tables:
            assert f"CREATE TABLE {table}" in sql, table
    for view in DIAGNOSTIC_VIEWS:
        assert f"CREATE VIEW {view}" in sql, view


def test_default_package_catalog_includes_base_migration() -> None:
    catalog = load_default_catalog()
    assert catalog.latest_version >= BASE_MIGRATION_VERSION
    migration = catalog.get(BASE_MIGRATION_VERSION)
    assert migration.migration_id == BASE_MIGRATION_ID
    assert migration.sql_text.strip()
    assert migration.checksum.startswith("sha256:")
    schema = ControlPlaneSchema.default()
    assert schema.catalog().get(1).migration_id == BASE_MIGRATION_ID


def test_required_table_and_view_catalog_is_complete() -> None:
    tables = required_tables()
    views = required_views()
    assert "control_plane_metadata" in tables
    assert "schema_migrations" in tables
    assert "tasks" in tables
    assert "leases" in tables
    assert "domain_events" in tables
    assert "source_snapshots" in tables
    assert "decision_cache_entries" in tables
    assert "control_surfaces" in tables
    assert "improve_experiments" in tables
    assert "ready_task_context_v1" in views
    assert "active_lease_v1" in views
    # No duplicates in the admitted table list.
    assert len(tables) == len(set(tables))
    assert len(views) == len(set(views))


def test_schema_rejects_version_and_domain_drift() -> None:
    with pytest.raises(ControlPlaneSchemaError, match="version"):
        ControlPlaneSchema(version=2)
    with pytest.raises(ControlPlaneSchemaError, match="migration_id"):
        ControlPlaneSchema(migration_id="0002_other")
    with pytest.raises(ControlPlaneSchemaError, match="domains missing"):
        ControlPlaneSchema(domains=("meta",))


@pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane schema hermetic install tests",
)
def test_install_applies_base_schema_and_verifies_identity(
    tmp_path: Path,
) -> None:
    schema = ControlPlaneSchema.default()
    db = tmp_path / "control.duckdb"
    report = schema.install(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="schema-test-owner",
    )
    assert report.changed is True
    assert report.to_version == BASE_MIGRATION_VERSION
    assert report.schema_fingerprint
    inspection = schema.verify_installed(db)
    assert inspection["schema_fingerprint"] == report.schema_fingerprint
    assert inspection["table_count"] >= len(required_tables())
    assert inspection["view_count"] >= len(required_views())

    # Preserve task CID + lease fencing semantics with a real row round-trip.
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid,
                ordinal, title, status, revision, body_json
            ) VALUES (
                'goal:root', 'root', 'obj:1', '',
                0, 'root', 'active', 1, '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, ordinal, status, revision,
                identity_json, body_json
            ) VALUES (
                'task:demo', 'demo', 'goal:root', 1, 'ready', 1,
                '{"task_cid":"task:demo"}', '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt, state,
                started_at_ms, retry_not_before_ms, revision, fence_epoch,
                owner_session_id, body_json
            ) VALUES (
                'task:demo', 'claim:1', 'resolution:1', 'did:worker:1',
                1, 7, 9999999999999, 1, 'accepted',
                1700000000000, 0, 1, 1,
                'session:1', '{}'
            )
            """
        )
        row = connection.execute(
            """
            SELECT task_cid, task_status, fencing_token, lease_state
            FROM ready_task_context_v1
            WHERE task_cid = 'task:demo'
            """
        ).fetchone()
        assert row is not None
        assert str(row[0]) == "task:demo"
        assert str(row[1]) == "ready"
        assert int(row[2]) == 7
        assert str(row[3]) == "accepted"

        active = connection.execute(
            "SELECT task_cid, fencing_token FROM active_lease_v1"
        ).fetchall()
        assert len(active) == 1
        assert str(active[0][0]) == "task:demo"
        assert int(active[0][1]) == 7


@pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for empty-to-latest fingerprint equivalence",
)
def test_empty_to_latest_fingerprint_equivalence(tmp_path: Path) -> None:
    schema = ControlPlaneSchema.default()
    left = tmp_path / "left.duckdb"
    right = tmp_path / "right.duckdb"
    proof = schema.prove_empty_to_latest_equivalence(left, right)
    assert proof["equivalent"] is True
    assert proof["to_version"] == BASE_MIGRATION_VERSION
    assert proof["schema_fingerprint"]
    assert schema.schema_fingerprint(left) == proof["schema_fingerprint"]
    assert schema.schema_fingerprint(right) == proof["schema_fingerprint"]


@pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for incremental upgrade fingerprint parity",
)
def test_incremental_and_direct_apply_share_fingerprint(tmp_path: Path) -> None:
    schema = ControlPlaneSchema.default()
    catalog = schema.catalog()
    # Only one migration today; still prove runner replay and inspect parity.
    full_db = tmp_path / "full.duckdb"
    replay_db = tmp_path / "replay.duckdb"
    full = ControlPlaneMigrationRunner.for_database(
        full_db,
        catalog=catalog,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    full_report = full.apply()
    replay = ControlPlaneMigrationRunner.for_database(
        replay_db,
        catalog=catalog,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    first = replay.apply()
    second = replay.apply()
    assert first.schema_fingerprint == full_report.schema_fingerprint
    assert second.changed is False
    assert second.schema_fingerprint == full_report.schema_fingerprint
    schema.verify_installed(full_db)
    schema.verify_installed(replay_db)


@pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for default catalog install smoke",
)
def test_default_catalog_install_via_runner(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    report = runner.apply()
    assert report.to_version >= 1
    inspect = runner.inspect()
    assert inspect["current_version"] >= 1
    assert inspect["pending_versions"] == []
    connection = open_duckdb_connection(db)
    try:
        rows = connection.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            """
        ).fetchall()
        tables = {str(row[0]) for row in rows}
    finally:
        connection.close()
    for name in ("tasks", "leases", "schema_contracts", "domain_events"):
        assert name in tables


def test_quack_profile_binding_includes_dependency_pin() -> None:
    schema = ControlPlaneSchema.default()
    profile = schema.quack_profile()
    assert profile["profile_id"] == "agent-supervisor-duckdb-quack-1.5"
    assert profile["dependency_profile"]["duckdb_requirement"] == (
        PINNED_DUCKDB_REQUIREMENT
    )
    assert profile["dependency_profile"]["optional_extra"] == (
        PINNED_OPTIONAL_EXTRA
    )
