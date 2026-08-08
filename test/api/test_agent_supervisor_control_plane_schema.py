"""Tests for ControlPlaneSchema@1 normalized domain installation (DQP-005)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigrationRunner,
    checksum_sql,
    load_default_catalog,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    CONTROL_PLANE_MIGRATION_ID,
    CONTROL_PLANE_MIGRATION_VERSION,
    CONTROL_PLANE_SCHEMA_INTERFACE,
    CONTROL_PLANE_SCHEMA_VERSION,
    LEASE_SEMANTIC_COLUMNS,
    PINNED_DUCKDB_REQUIREMENT,
    PINNED_OPTIONAL_EXTRA,
    REQUIRED_DOMAIN_TABLES,
    REQUIRED_VIEWS,
    SCHEMA_DOMAINS,
    TASK_IDENTITY_COLUMNS,
    ControlPlaneSchema,
    ControlPlaneSchemaDependencyError,
    ControlPlaneSchemaNotInstalledError,
    assert_pyproject_pins_supervisor_duckdb,
    control_plane_sql_path,
    default_schema_catalog,
    domain_table_map,
    duckdb_available,
    duckdb_version_matches_pin,
    load_control_plane_sql,
    parse_duckdb_version,
    required_tables,
    required_views,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane schema hermetic tests",
)


def _schema() -> ControlPlaneSchema:
    return ControlPlaneSchema.default(tool_version="1.5.2")


def test_interface_and_domain_inventory_are_closed() -> None:
    assert CONTROL_PLANE_SCHEMA_INTERFACE == "ControlPlaneSchema@1"
    assert CONTROL_PLANE_SCHEMA_VERSION == 1
    assert SCHEMA_DOMAINS == (
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
    )
    mapping = domain_table_map()
    assert set(mapping) == set(SCHEMA_DOMAINS)
    # Every domain table is accounted for exactly once.
    flattened = [table for members in mapping.values() for table in members]
    assert sorted(flattened) == sorted(REQUIRED_DOMAIN_TABLES)
    assert set(required_tables()) >= set(REQUIRED_DOMAIN_TABLES)
    assert tuple(required_views()) == REQUIRED_VIEWS
    assert "task_cid" in TASK_IDENTITY_COLUMNS
    assert "fencing_token" in LEASE_SEMANTIC_COLUMNS


def test_sql_migration_file_is_present_and_checksum_bound() -> None:
    path = control_plane_sql_path()
    assert path.is_file()
    assert path.name == f"{CONTROL_PLANE_MIGRATION_ID}.sql"
    sql_text = load_control_plane_sql()
    assert "CREATE TABLE tasks" in sql_text
    assert "CREATE TABLE leases" in sql_text
    assert "CREATE VIEW ready_task_context_v1" in sql_text
    # Join-critical identity must be columns, not JSON-only blobs.
    assert "task_cid VARCHAR PRIMARY KEY" in sql_text
    assert "fencing_token BIGINT NOT NULL" in sql_text
    assert "fence_epoch BIGINT NOT NULL" in sql_text
    assert "revision BIGINT NOT NULL" in sql_text
    catalog = default_schema_catalog()
    assert catalog.latest_version >= CONTROL_PLANE_MIGRATION_VERSION
    migration = catalog.get(CONTROL_PLANE_MIGRATION_VERSION)
    assert migration.migration_id == CONTROL_PLANE_MIGRATION_ID
    assert migration.checksum.startswith("sha256:")
    assert len(migration.checksum) == len("sha256:") + 64
    # Catalog SQL is newline-normalized for checksum stability only; content
    # must still match the on-disk migration after checksum recomputation.
    assert migration.checksum == checksum_sql(sql_text)
    assert checksum_sql(migration.sql_text) == checksum_sql(sql_text)


def test_default_package_catalog_exposes_control_plane_migration() -> None:
    catalog = load_default_catalog()
    assert catalog.latest_version >= 1
    migration = catalog.get(1)
    assert migration.migration_id == "0001_control_plane"
    schema = ControlPlaneSchema.default()
    payload = schema.to_dict()
    assert payload["interface"] == CONTROL_PLANE_SCHEMA_INTERFACE
    assert payload["migration_id"] == CONTROL_PLANE_MIGRATION_ID
    assert payload["pinned_duckdb_requirement"] == PINNED_DUCKDB_REQUIREMENT
    assert payload["pinned_optional_extra"] == PINNED_OPTIONAL_EXTRA
    assert set(payload["domains"]) == set(SCHEMA_DOMAINS)


def test_install_creates_required_tables_views_and_metadata(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    schema = _schema()
    report = schema.install(db)
    assert report.changed is True
    assert report.to_version == CONTROL_PLANE_MIGRATION_VERSION
    assert report.schema_fingerprint

    snapshot = schema.verify_installed(db)
    assert snapshot["installed"] is True
    assert snapshot["missing_tables"] == []
    assert snapshot["missing_views"] == []
    assert snapshot["current_version"] == CONTROL_PLANE_MIGRATION_VERSION

    with open_duckdb_connection(db) as connection:
        for table in required_tables():
            row = connection.execute(
                """
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND upper(table_type) IN ('BASE TABLE', 'TABLE')
                  AND table_name = ?
                """,
                [table],
            ).fetchone()
            assert int(row[0]) == 1, f"missing table {table}"
        for view in required_views():
            row = connection.execute(
                """
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND upper(table_type) = 'VIEW'
                  AND table_name = ?
                """,
                [view],
            ).fetchone()
            assert int(row[0]) == 1, f"missing view {view}"
        # Domain seed rows are present for schema metadata.
        seed = connection.execute(
            "SELECT COUNT(*) FROM schema_contracts"
        ).fetchone()
        assert int(seed[0]) >= len(SCHEMA_DOMAINS)
        inventory = connection.execute(
            "SELECT COUNT(*) FROM schema_domain_inventory_v1"
        ).fetchone()
        assert int(inventory[0]) >= len(SCHEMA_DOMAINS)


def test_fresh_and_upgraded_databases_share_fingerprint(tmp_path: Path) -> None:
    schema = _schema()
    left = tmp_path / "left.duckdb"
    right = tmp_path / "right.duckdb"
    proof = schema.prove_empty_to_latest_equivalence(left, right)
    assert proof["equivalent"] is True
    assert proof["to_version"] == CONTROL_PLANE_MIGRATION_VERSION
    assert proof["schema_fingerprint"]
    assert schema.schema_fingerprint(left) == schema.schema_fingerprint(right)

    # Incremental path: bookkeeping-only head then apply 0001 matches fresh install.
    step = tmp_path / "step.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        step,
        catalog=schema.catalog,
        application_version="control-plane-schema",
        tool_version="1.5.2",
    )
    runner.ensure_bookkeeping()
    assert runner.current_version() == 0
    step_report = runner.apply(target_version=1)
    assert step_report.to_version == 1
    assert step_report.schema_fingerprint == proof["schema_fingerprint"]


def test_task_cid_and_lease_semantics_are_preserved(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    schema = _schema()
    schema.install(db)
    probe = schema.verify_task_and_lease_semantics(db)
    assert probe["task_cid"] == "task:cid:schema-probe"
    assert probe["lease_task_cid"] == "task:cid:schema-probe"
    assert probe["lease_fencing_token"] == 1
    assert probe["lease_fence_epoch"] == 1
    assert probe["lease_state"] == "accepted"
    assert probe["ready_view_excludes_accepted_lease"] is True
    assert probe["task_cid_unique"] is True

    with open_duckdb_connection(db) as connection:
        task_columns = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = 'tasks'
                """
            ).fetchall()
        }
        lease_columns = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = 'leases'
                """
            ).fetchall()
        }
    assert set(TASK_IDENTITY_COLUMNS).issubset(task_columns)
    assert set(LEASE_SEMANTIC_COLUMNS).issubset(lease_columns)
    # Opaque JSON companions exist but are not the sole identity surface.
    assert "body_json" in task_columns
    assert "identity_json" in task_columns


def test_existing_task_cids_remain_addressable(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    schema = _schema()
    schema.install(db)
    existing = (
        "task:cid:legacy-001",
        "task:cid:legacy-002",
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
    )
    result = schema.verify_existing_table_compatibility(
        db, existing_task_cids=existing
    )
    assert result["compatible"] is True
    assert result["retained_task_cids"] == list(existing)
    with open_duckdb_connection(db) as connection:
        for task_cid in existing:
            row = connection.execute(
                "SELECT task_cid FROM tasks WHERE task_cid = ?",
                [task_cid],
            ).fetchone()
            assert str(row[0]) == task_cid


def test_verify_installed_fails_closed_on_empty_database(tmp_path: Path) -> None:
    db = tmp_path / "empty.duckdb"
    schema = _schema()
    # Create an empty duckdb file without domain schema.
    with open_duckdb_connection(db) as connection:
        connection.execute("SELECT 1")
    with pytest.raises(ControlPlaneSchemaNotInstalledError):
        schema.verify_installed(db)


def test_replay_install_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    schema = _schema()
    first = schema.install(db)
    second = schema.install(db)
    assert first.changed is True
    assert second.changed is False
    assert first.schema_fingerprint == second.schema_fingerprint
    assert schema.schema_fingerprint(db) == first.schema_fingerprint


def test_pyproject_pins_optional_supervisor_duckdb_profile() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert_pyproject_pins_supervisor_duckdb(pyproject)
    assert f"{PINNED_OPTIONAL_EXTRA} = [" in pyproject or "agent-supervisor = [" in pyproject
    assert "duckdb>=1.5.0,<1.6.0" in pyproject
    # Package data must ship domain SQL with the wheel.
    assert "sql/*.sql" in pyproject
    # Pin helper itself fails closed on weakened text.
    with pytest.raises(ControlPlaneSchemaDependencyError):
        assert_pyproject_pins_supervisor_duckdb("duckdb>=0.7.0\n")
    with pytest.raises(ControlPlaneSchemaDependencyError):
        assert_pyproject_pins_supervisor_duckdb(
            "agent-supervisor = [\"duckdb>=1.5.0\"]\n"
        )


def test_duckdb_version_pin_helpers() -> None:
    assert parse_duckdb_version("1.5.2") == (1, 5, 2)
    assert duckdb_version_matches_pin("1.5.0") is True
    assert duckdb_version_matches_pin("1.5.99") is True
    assert duckdb_version_matches_pin("1.4.0") is False
    assert duckdb_version_matches_pin("2.0.0") is False
    assert pinned_requirement_is_strict()


def pinned_requirement_is_strict() -> bool:
    requirement = PINNED_DUCKDB_REQUIREMENT
    return requirement.startswith("duckdb>=") and "<1.6.0" in requirement


def test_ready_context_view_includes_unleased_ready_tasks(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    schema = _schema()
    schema.install(db)
    now = "2020-01-03T00:00:00Z"
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, parent_goal_cid, objective_id,
                title, status, ordinal, created_at, updated_at, revision
            ) VALUES (
                'goal:cid:ready', 'GOAL-READY', NULL, NULL,
                'ready goal', 'active', 1, ?, ?, 1
            )
            """,
            [now, now],
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_id, status, priority,
                ordinal, track, created_at, updated_at, revision, fence_epoch
            ) VALUES (
                'task:cid:ready', 'TASK-READY', 'goal:cid:ready', NULL,
                'ready', 'P0', 1, 'schema', ?, ?, 1, 0
            )
            """,
            [now, now],
        )
        row = connection.execute(
            """
            SELECT task_cid, goal_cid, status
            FROM ready_task_context_v1
            WHERE task_cid = ?
            """,
            ["task:cid:ready"],
        ).fetchone()
    assert str(row[0]) == "task:cid:ready"
    assert str(row[1]) == "goal:cid:ready"
    assert str(row[2]) == "ready"


def test_schema_module_cold_import_is_side_effect_free() -> None:
    # Importing helpers must not require an open database connection.
    catalog = default_schema_catalog()
    assert catalog.latest_version >= 1
    assert control_plane_sql_path().is_file()
    assert "CREATE TABLE" in load_control_plane_sql()
