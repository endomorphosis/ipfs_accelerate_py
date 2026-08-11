"""Tests for the checksum-bound control-plane migration catalog and runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigration,
    ControlPlaneMigrationRunner,
    MigrationAdHocDDLError,
    MigrationCatalog,
    MigrationCatalogError,
    MigrationDowngradeError,
    MigrationDriftError,
    MigrationOwnershipError,
    MigrationPartialError,
    MigrationPostconditionError,
    MigrationPreconditionError,
    OUTCOME_APPLIED,
    checksum_sql,
    duckdb_available,
    load_default_catalog,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane migration hermetic tests",
)


def _migration(
    version: int,
    sql: str,
    *,
    slug: str | None = None,
    depends_on: tuple[int, ...] | None = None,
    preconditions: tuple[str, ...] = (),
    postconditions: tuple[str, ...] = (),
    min_application_version: str | None = None,
    max_application_version: str | None = None,
    min_tool_version: str | None = None,
    max_tool_version: str | None = None,
) -> ControlPlaneMigration:
    name = slug or f"m{version}"
    return ControlPlaneMigration.from_sql(
        version=version,
        migration_id=f"{version:04d}_{name}",
        sql_text=sql,
        description=name,
        depends_on=(
            tuple(range(1, version)) if depends_on is None else depends_on
        ),
        preconditions=preconditions,
        postconditions=postconditions,
        min_application_version=min_application_version,
        max_application_version=max_application_version,
        min_tool_version=min_tool_version,
        max_tool_version=max_tool_version,
    )


def _sample_catalog() -> MigrationCatalog:
    return MigrationCatalog.from_migrations(
        [
            _migration(
                1,
                """
                CREATE TABLE domain_alpha (
                    id VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL
                );
                INSERT INTO domain_alpha VALUES ('seed', 'one');
                """,
                slug="domain_alpha",
                postconditions=(
                    "SELECT COUNT(*) = 1 FROM domain_alpha WHERE id = 'seed'",
                ),
            ),
            _migration(
                2,
                """
                CREATE TABLE domain_beta (
                    id VARCHAR PRIMARY KEY,
                    alpha_id VARCHAR NOT NULL
                );
                INSERT INTO domain_beta VALUES ('b1', 'seed');
                """,
                slug="domain_beta",
                preconditions=(
                    "SELECT COUNT(*) = 1 FROM information_schema.tables "
                    "WHERE table_schema = 'main' AND table_name = 'domain_alpha'",
                ),
                postconditions=(
                    "SELECT COUNT(*) = 1 FROM domain_beta",
                ),
            ),
        ]
    )


def test_checksum_sql_is_stable_and_newline_normalized() -> None:
    left = checksum_sql("CREATE TABLE t (id INT);\n")
    right = checksum_sql("CREATE TABLE t (id INT);\r\n")
    assert left == right
    assert left.startswith("sha256:")


def test_catalog_refuses_duplicate_ids_and_versions() -> None:
    first = _migration(1, "CREATE TABLE a (id INT);", slug="a")
    dup_version = ControlPlaneMigration.from_sql(
        version=1,
        migration_id="0001_other",
        sql_text="CREATE TABLE b (id INT);",
    )
    with pytest.raises(MigrationCatalogError, match="duplicate migration versions"):
        MigrationCatalog.from_migrations([first, dup_version])

    second = ControlPlaneMigration.from_sql(
        version=2,
        migration_id="0001_a",
        sql_text="CREATE TABLE b (id INT);",
    )
    with pytest.raises(MigrationCatalogError, match="duplicate migration_id"):
        MigrationCatalog.from_migrations([first, second])


def test_catalog_refuses_gaps() -> None:
    with pytest.raises(MigrationCatalogError, match="contiguous"):
        MigrationCatalog.from_migrations(
            [
                _migration(1, "CREATE TABLE a (id INT);"),
                _migration(3, "CREATE TABLE c (id INT);", depends_on=(1,)),
            ]
        )


def test_catalog_loads_sql_directory(tmp_path: Path) -> None:
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "0001_control_plane.sql").write_text(
        "CREATE TABLE control_units (id VARCHAR PRIMARY KEY);\n",
        encoding="utf-8",
    )
    (sql_dir / "0002_tasks.sql").write_text(
        "CREATE TABLE tasks (task_cid VARCHAR PRIMARY KEY);\n",
        encoding="utf-8",
    )
    catalog = MigrationCatalog.from_sql_directory(sql_dir)
    assert catalog.latest_version == 2
    assert catalog.get(1).migration_id == "0001_control_plane"
    assert catalog.get(1).checksum == checksum_sql(
        "CREATE TABLE control_units (id VARCHAR PRIMARY KEY);\n"
    )


def test_catalog_refuses_bad_filenames(tmp_path: Path) -> None:
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "1_bad.sql").write_text("CREATE TABLE t (id INT);\n", encoding="utf-8")
    with pytest.raises(MigrationCatalogError, match="NNNN_slug.sql"):
        MigrationCatalog.from_sql_directory(sql_dir)


def test_runner_records_receipt_fields_and_is_replay_safe(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="9.9.9",
        tool_version="1.5.2",
        owner_id="owner-a",
    )
    report = runner.apply()
    assert report.changed is True
    assert report.from_version == 0
    assert report.to_version == 2
    assert len(report.receipts) == 2
    receipt = report.receipts[0]
    assert receipt.version == 1
    assert receipt.checksum == catalog.get(1).checksum
    assert receipt.application_version == "9.9.9"
    assert receipt.tool_version == "1.5.2"
    assert receipt.started_at
    assert receipt.finished_at
    assert receipt.outcome == OUTCOME_APPLIED
    assert receipt.schema_fingerprint
    assert receipt.receipt_cid

    replay = runner.apply()
    assert replay.changed is False
    assert replay.to_version == 2
    assert runner.current_version() == 2
    listed = runner.list_receipts()
    assert [item.version for item in listed] == [1, 2]
    assert listed[0].schema_fingerprint
    inspect = runner.inspect()
    assert inspect["current_version"] == 2
    assert inspect["pending_versions"] == []
    assert inspect["schema_fingerprint"] == report.schema_fingerprint


def test_empty_to_latest_equivalence(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    left = tmp_path / "left.duckdb"
    right = tmp_path / "right.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        left,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    proof = runner.prove_empty_to_latest_equivalence(other_database_path=right)
    assert proof["equivalent"] is True
    assert proof["to_version"] == 2
    assert proof["schema_fingerprint"]


def test_incremental_upgrade_matches_empty_to_latest(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    full_db = tmp_path / "full.duckdb"
    step_db = tmp_path / "step.duckdb"
    full = ControlPlaneMigrationRunner.for_database(
        full_db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    full_report = full.apply()
    step = ControlPlaneMigrationRunner.for_database(
        step_db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    step.apply(target_version=1)
    step_report = step.apply(target_version=2)
    assert step_report.schema_fingerprint == full_report.schema_fingerprint


def test_refuses_checksum_drift_on_altered_sql(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    runner.apply()

    altered = MigrationCatalog.from_migrations(
        [
            _migration(
                1,
                """
                CREATE TABLE domain_alpha (
                    id VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL,
                    extra VARCHAR
                );
                INSERT INTO domain_alpha VALUES ('seed', 'one', 'x');
                """,
                slug="domain_alpha",
            ),
            catalog.get(2),
        ]
    )
    drifted = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=altered,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    with pytest.raises(MigrationDriftError, match="checksum drift"):
        drifted.apply()


def test_refuses_downgrade(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    runner.apply()
    with pytest.raises(MigrationDowngradeError, match="downgrade"):
        runner.apply(target_version=1)


def test_refuses_partial_application_marker(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    runner.ensure_bookkeeping()
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO schema_migration_attempts (
                attempt_id, version, migration_id, checksum,
                application_version, tool_version, started_at, finished_at,
                outcome, schema_fingerprint, error_text, body_json
            ) VALUES (
                'attempt:partial', 1, '0001_domain_alpha', 'sha256:dead',
                '1.0.0', '1.5.2', '2020-01-01T00:00:00Z', '2020-01-01T00:00:01Z',
                'failed', NULL, 'boom', '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO control_plane_metadata (key, value, updated_at)
            VALUES ('partial_application', '1', '2020-01-01T00:00:00Z')
            """
        )
    with pytest.raises(MigrationPartialError, match="partial"):
        runner.apply()


def test_failure_rolls_back_sql(tmp_path: Path) -> None:
    catalog = MigrationCatalog.from_migrations(
        [
            _migration(
                1,
                "CREATE TABLE should_not_exist (id INT);",
                slug="failing",
                postconditions=("SELECT 0",),
            )
        ]
    )
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    with pytest.raises(MigrationPostconditionError):
        runner.apply()
    assert runner.current_version() == 0
    with open_duckdb_connection(db) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'should_not_exist'
            """
        ).fetchone()
    assert int(row[0]) == 0


def test_fault_before_commit_rolls_back(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )

    def explode(phase: str, migration: ControlPlaneMigration) -> None:
        if phase == "before_commit" and migration.version == 1:
            raise RuntimeError("injected fault")

    with pytest.raises(RuntimeError, match="injected fault"):
        runner.apply(fault_injector=explode)
    assert runner.current_version() == 0
    with open_duckdb_connection(db) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'domain_alpha'
            """
        ).fetchone()
    assert int(row[0]) == 0


def test_precondition_failure_refuses_apply(tmp_path: Path) -> None:
    catalog = MigrationCatalog.from_migrations(
        [
            _migration(
                1,
                "CREATE TABLE t (id INT);",
                slug="blocked",
                preconditions=("SELECT 0",),
            )
        ]
    )
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    with pytest.raises(MigrationPreconditionError):
        runner.apply()
    assert runner.current_version() == 0


def test_concurrent_migration_ownership_is_exclusive(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    holder = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
        owner_id="owner-holder",
    )
    holder.ensure_bookkeeping()
    # Simulate a live foreign ownership lease.
    with open_duckdb_connection(db) as connection:
        lease = {
            "owner_id": "owner-holder",
            "acquired_at": "2099-01-01T00:00:00Z",
            "expires_at": "2099-01-01T01:00:00Z",
        }
        connection.execute(
            """
            INSERT INTO control_plane_metadata (key, value, updated_at)
            VALUES ('migration_owner', ?, '2099-01-01T00:00:00Z')
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            [json.dumps(lease, sort_keys=True)],
        )

    contender = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
        owner_id="owner-contender",
    )
    with pytest.raises(MigrationOwnershipError, match="ownership"):
        contender.apply()


def test_runtime_adhoc_ddl_refused_without_compatibility_path(
    tmp_path: Path,
) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    runner.apply()
    guarded = runner.open_guarded_connection()
    try:
        with pytest.raises(MigrationAdHocDDLError, match="ad-hoc DDL"):
            guarded.execute("CREATE TABLE rogue (id INT)")
        # DML against migrated tables remains allowed.
        guarded.execute(
            "INSERT INTO domain_alpha VALUES ('runtime', 'ok')"
        )
        row = guarded.execute(
            "SELECT value FROM domain_alpha WHERE id = 'runtime'"
        ).fetchone()
        assert str(row[0]) == "ok"
    finally:
        guarded.close()


def test_explicit_compatibility_path_allows_bounded_ddl(tmp_path: Path) -> None:
    catalog = _sample_catalog()
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    runner.apply()
    guarded = runner.open_guarded_connection(compatibility_path=True)
    try:
        guarded.execute("CREATE TABLE compat_overlay (id INT)")
        row = guarded.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'compat_overlay'
            """
        ).fetchone()
        assert int(row[0]) == 1
    finally:
        guarded.close()


def test_application_version_bounds_are_enforced(tmp_path: Path) -> None:
    catalog = MigrationCatalog.from_migrations(
        [
            _migration(
                1,
                "CREATE TABLE t (id INT);",
                slug="bounded",
                min_application_version="2.0.0",
            )
        ]
    )
    db = tmp_path / "control.duckdb"
    runner = ControlPlaneMigrationRunner.for_database(
        db,
        catalog=catalog,
        application_version="1.0.0",
        tool_version="1.5.2",
    )
    with pytest.raises(Exception, match="below minimum"):
        runner.apply()


def test_default_package_catalog_loads() -> None:
    catalog = load_default_catalog()
    assert isinstance(catalog, MigrationCatalog)
    # Domain SQL may still be empty in this foundation task; loading must work.
    assert catalog.latest_version >= 0


def test_sql_readme_documents_checksum_runner_contract() -> None:
    readme = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "task_sources"
        / "sql"
        / "README.md"
    )
    text = readme.read_text(encoding="utf-8")
    assert "checksum" in text.lower()
    assert "NNNN_slug.sql" in text
    assert "schema fingerprint" in text.lower()
    assert "ad-hoc DDL" in text or "ad-hoc ddl" in text.lower()
