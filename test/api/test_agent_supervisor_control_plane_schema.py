"""Tests for the normalized control-plane schema (DQP-005).

Acceptance:

* Fresh and upgraded databases share an identical canonical information-schema
  fingerprint
* Schema preserves existing task CIDs and lease semantics
* DuckDB/Quack profile is explicitly pinned for the optional supervisor service
* No join-critical identity exists only inside opaque JSON
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigrationRunner,
    duckdb_available,
    load_default_catalog,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    CONTROL_PLANE_MIGRATION_ID,
    CONTROL_PLANE_SCHEMA_INTERFACE,
    CONTROL_PLANE_SCHEMA_REVISION,
    DIAGNOSTIC_VIEWS,
    DOMAIN_TABLES,
    JOIN_CRITICAL_IDENTITIES,
    LEASE_IDENTITY_COLUMNS,
    PINNED_DUCKDB_VERSION_SPEC,
    SCHEMA_DOMAINS,
    SUPERVISOR_OPTIONAL_EXTRA,
    TASK_IDENTITY_COLUMNS,
    ControlPlaneSchema,
    assert_dependency_profile_pinned,
    default_control_plane_schema,
    default_dependency_profile,
    install_control_plane_schema,
    load_control_plane_catalog,
    prove_fresh_and_upgraded_equivalence,
    read_pyproject_text,
    verify_installed_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    PINNED_DUCKDB_VERSION_PREFIX,
    default_compatibility_profile,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane schema hermetic tests",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_interface_identity_and_domain_inventory() -> None:
    schema = default_control_plane_schema()
    assert schema.INTERFACE == CONTROL_PLANE_SCHEMA_INTERFACE
    assert CONTROL_PLANE_SCHEMA_INTERFACE == "ControlPlaneSchema@1"
    assert schema.schema_revision == CONTROL_PLANE_SCHEMA_REVISION
    assert schema.migration_id == CONTROL_PLANE_MIGRATION_ID
    assert tuple(schema.domains) == SCHEMA_DOMAINS
    for domain in SCHEMA_DOMAINS:
        assert domain in DOMAIN_TABLES
        assert DOMAIN_TABLES[domain]
    assert "tasks" in schema.all_domain_tables
    assert "leases" in schema.all_domain_tables
    assert schema.sql_path().is_file()
    payload = schema.to_dict()
    assert payload["interface"] == CONTROL_PLANE_SCHEMA_INTERFACE
    assert set(payload["domains"]) == set(SCHEMA_DOMAINS)


def test_default_catalog_includes_control_plane_migration() -> None:
    catalog = load_control_plane_catalog()
    assert catalog.latest_version >= 1
    migration = catalog.get(1)
    assert migration.migration_id == CONTROL_PLANE_MIGRATION_ID
    assert migration.checksum.startswith("sha256:")
    # Package default loader must see the same file.
    default = load_default_catalog()
    assert default.latest_version >= 1
    assert default.get(1).migration_id == CONTROL_PLANE_MIGRATION_ID
    assert default.get(1).checksum == migration.checksum


def test_install_creates_domain_tables_views_and_is_replay_safe(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    report = install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="schema-test",
    )
    assert report.changed is True
    assert report.to_version >= 1
    assert report.schema_fingerprint

    verified = verify_installed_schema(db)
    assert verified["schema_fingerprint"] == report.schema_fingerprint
    for view in DIAGNOSTIC_VIEWS:
        assert view in verified["views_ok"]
    for column in TASK_IDENTITY_COLUMNS:
        assert column in verified["task_columns_ok"]
    for column in LEASE_IDENTITY_COLUMNS:
        assert column in verified["lease_columns_ok"]
    for table, column in JOIN_CRITICAL_IDENTITIES:
        assert f"{table}.{column}" in verified["join_critical_ok"]

    replay = install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="schema-test",
    )
    assert replay.changed is False
    assert replay.schema_fingerprint == report.schema_fingerprint


def test_fresh_and_upgraded_share_canonical_fingerprint(tmp_path: Path) -> None:
    left = tmp_path / "fresh.duckdb"
    right = tmp_path / "upgraded.duckdb"
    proof = prove_fresh_and_upgraded_equivalence(
        left,
        right,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    assert proof["equivalent"] is True
    assert proof["schema_fingerprint"]
    assert proof["to_version"] >= 1

    # Incremental apply of the same catalog must match empty-to-latest.
    catalog = load_control_plane_catalog()
    full_db = tmp_path / "full.duckdb"
    step_db = tmp_path / "step.duckdb"
    full = ControlPlaneMigrationRunner.for_database(
        full_db,
        catalog=catalog,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    full_report = full.apply()
    step = ControlPlaneMigrationRunner.for_database(
        step_db,
        catalog=catalog,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    if catalog.latest_version >= 1:
        step.apply(target_version=1)
    if catalog.latest_version > 1:
        for version in range(2, catalog.latest_version + 1):
            step.apply(target_version=version)
    step_report = step.apply()
    assert step_report.schema_fingerprint == full_report.schema_fingerprint
    assert step_report.schema_fingerprint == proof["schema_fingerprint"]


def test_task_cid_and_lease_semantics_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    task_cid = "baguqeera" + ("a" * 50)
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, parent_goal_cid, ordinal, title, body_json
            ) VALUES (?, ?, '', 1, 'g', '{}')
            """,
            [f"goal:{task_cid}", "goal-alias-1"],
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, ordinal, status, revision,
                identity_json, body_json
            ) VALUES (?, ?, ?, 1, 'ready', 1, '{}', '{}')
            """,
            [task_cid, "task-alias-1", f"goal:{task_cid}"],
        )
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt, state,
                started_at_ms, release_reason, retry_not_before_ms
            ) VALUES (?, ?, '', ?, 1, 7, ?, 1, 'accepted', ?, NULL, 0)
            """,
            [
                task_cid,
                f"claim:{task_cid}",
                "did:worker:1",
                9000000000000,
                1700000000000,
            ],
        )
        connection.execute(
            """
            INSERT INTO token_history (task_cid, fencing_token, recorded_at_ms)
            VALUES (?, 7, ?)
            """,
            [task_cid, 1700000000000],
        )
        row = connection.execute(
            """
            SELECT t.task_cid, t.status, l.state, l.fencing_token, l.expires_at_ms
            FROM tasks AS t
            JOIN leases AS l ON l.task_cid = t.task_cid
            WHERE t.task_cid = ?
            """,
            [task_cid],
        ).fetchone()
        assert str(row[0]) == task_cid
        assert str(row[1]) == "ready"
        assert str(row[2]) == "accepted"
        assert int(row[3]) == 7
        assert int(row[4]) == 9000000000000
        view_row = connection.execute(
            """
            SELECT task_cid, lease_state, fencing_token
            FROM ready_task_context_v1
            WHERE task_cid = ?
            """,
            [task_cid],
        ).fetchone()
        assert str(view_row[0]) == task_cid
        assert str(view_row[1]) == "accepted"
        assert int(view_row[2]) == 7


def test_join_critical_identities_are_first_class_columns(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    verified = verify_installed_schema(db)
    assert verified["opaque_json_only_identities"] == []

    # Every declared join-critical pair is a real non-JSON column.
    with open_duckdb_connection(db) as connection:
        for table, column in JOIN_CRITICAL_IDENTITIES:
            assert not column.endswith("_json"), (table, column)
            row = connection.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = 'main'
                  AND table_name = ?
                  AND column_name = ?
                """,
                [table, column],
            ).fetchone()
            assert row is not None, f"missing {table}.{column}"
            data_type = str(row[0] if not hasattr(row, "keys") else row["data_type"])
            assert "json" not in data_type.lower()


def test_schema_contracts_seed_covers_all_domains(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
    )
    with open_duckdb_connection(db) as connection:
        rows = connection.execute(
            """
            SELECT domain_name
            FROM schema_contracts
            WHERE interface_name = 'ControlPlaneDomain@1'
            ORDER BY domain_name
            """
        ).fetchall()
        domains = {str(row[0]) for row in rows}
        assert domains == set(SCHEMA_DOMAINS)
        root = connection.execute(
            """
            SELECT interface_name FROM schema_contracts
            WHERE contract_id = 'contract:ControlPlaneSchema@1'
            """
        ).fetchone()
        assert str(root[0]) == CONTROL_PLANE_SCHEMA_INTERFACE


def test_duckdb_quack_profile_is_pinned_in_pyproject() -> None:
    profile = default_dependency_profile()
    assert profile.extra_name == SUPERVISOR_OPTIONAL_EXTRA
    assert profile.duckdb_spec == PINNED_DUCKDB_VERSION_SPEC
    assert profile.duckdb_version_prefix == PINNED_DUCKDB_VERSION_PREFIX
    assert profile.extension_name == "quack"
    assert profile.profile_id == default_compatibility_profile().profile_id

    text = read_pyproject_text(REPO_ROOT / "pyproject.toml")
    assert_dependency_profile_pinned(text, profile=profile)
    assert "agent-supervisor" in text
    assert "duckdb>=1.5.0,<1.6.0" in text

    with pytest.raises(Exception, match="pin DuckDB|optional extra"):
        assert_dependency_profile_pinned("[project]\nname='x'\n")


def test_control_plane_schema_dataclass_rejects_domain_drift() -> None:
    with pytest.raises(Exception):
        ControlPlaneSchema(domains=("meta",))  # type: ignore[arg-type]


def test_sql_file_has_no_join_key_only_in_json_comments_contract() -> None:
    """Static guard: SQL declares task_cid/lease keys as columns, not JSON."""

    sql = default_control_plane_schema().sql_text()
    assert "CREATE TABLE tasks" in sql
    assert "task_cid VARCHAR PRIMARY KEY" in sql
    assert "CREATE TABLE leases" in sql
    assert "fencing_token BIGINT NOT NULL" in sql
    assert "expires_at_ms BIGINT NOT NULL" in sql
    assert "CREATE VIEW ready_task_context_v1" in sql
    # Opaque extension payloads are allowed, but not as the only identity.
    assert "extension_json" in sql
    assert "join-critical" in sql.lower() or "Join-critical" in sql
