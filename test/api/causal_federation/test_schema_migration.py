"""CASF-005 additive control-plane schema and migration qualification."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigration,
    ControlPlaneMigrationRunner,
    MigrationCatalog,
    MigrationDriftError,
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    CAUSAL_EVENT_FEDERATION_JOIN_CRITICAL_IDENTITIES,
    CAUSAL_EVENT_FEDERATION_MIGRATION_ID,
    CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION,
    CAUSAL_EVENT_FEDERATION_REFERENCE_TABLES,
    CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_INTERFACE,
    CAUSAL_EVENT_FEDERATION_SCHEMA_REVISION,
    CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS,
    CAUSAL_EVENT_FEDERATION_TABLES,
    CONTROL_PLANE_MIGRATION_ID,
    CONTROL_PLANE_MIGRATION_VERSION,
    CONTROL_PLANE_SCHEMA_INTERFACE,
    DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES,
    CausalEventFederationSchemaExtension,
    datasets_authoritative_operational_schema_sql,
    default_causal_event_federation_schema_extension,
    default_control_plane_schema,
    install_control_plane_schema,
    load_control_plane_catalog,
    load_datasets_authoritative_operational_catalog,
    package_sql_directory,
    verify_causal_event_federation_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for causal-federation schema tests",
)


def _runner(database: Path, catalog: MigrationCatalog) -> ControlPlaneMigrationRunner:
    return ControlPlaneMigrationRunner.for_database(
        database,
        catalog=catalog,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="casf-schema-test",
    )


def _columns(connection: object, table: str) -> set[str]:
    rows = connection.execute(  # type: ignore[attr-defined]
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table],
    ).fetchall()
    return {str(row[0]) for row in rows}


def test_extension_inventory_is_contiguous_and_preserves_base_profile() -> None:
    base = default_control_plane_schema()
    extension = default_causal_event_federation_schema_extension()
    catalog = load_control_plane_catalog()

    assert base.INTERFACE == CONTROL_PLANE_SCHEMA_INTERFACE == "ControlPlaneSchema@1"
    assert base.migration_id == CONTROL_PLANE_MIGRATION_ID == "0001_control_plane"
    assert CONTROL_PLANE_MIGRATION_VERSION == 1
    assert base.sql_path().name == "0001_control_plane.sql"

    assert isinstance(extension, CausalEventFederationSchemaExtension)
    assert extension.INTERFACE == CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_INTERFACE
    assert extension.schema_revision == CAUSAL_EVENT_FEDERATION_SCHEMA_REVISION == 2
    assert extension.migration_id == CAUSAL_EVENT_FEDERATION_MIGRATION_ID
    assert extension.migration_version == CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION == 2
    assert extension.sql_path().name == "0002_causal_event_federation_core.sql"
    assert set(extension.tables) == set(CAUSAL_EVENT_FEDERATION_TABLES)

    assert catalog.latest_version == 2
    assert catalog.get(1).migration_id == CONTROL_PLANE_MIGRATION_ID
    assert catalog.get(2).migration_id == CAUSAL_EVENT_FEDERATION_MIGRATION_ID
    assert catalog.get(2).depends_on == (1,)
    assert catalog.get(2).checksum.startswith("sha256:")

    # The separately qualified datasets-authoritative @1 profile remains a
    # one-migration projection of 0001 and gains no federation relations.
    operational = load_datasets_authoritative_operational_catalog()
    assert operational.latest_version == 1
    assert not set(CAUSAL_EVENT_FEDERATION_TABLES).intersection(
        DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES
    )
    operational_sql = datasets_authoritative_operational_schema_sql()
    assert "CREATE TABLE federations" not in operational_sql
    assert "CREATE TABLE causal_nodes" not in operational_sql


def test_fresh_and_one_to_two_upgrade_are_equivalent_and_replay_safe(
    tmp_path: Path,
) -> None:
    catalog = load_control_plane_catalog()
    fresh_database = tmp_path / "fresh.duckdb"
    upgraded_database = tmp_path / "upgraded.duckdb"

    fresh = _runner(fresh_database, catalog)
    fresh_report = fresh.apply()
    assert fresh_report.from_version == 0
    assert fresh_report.to_version == 2
    assert [receipt.version for receipt in fresh_report.receipts] == [1, 2]

    upgraded = _runner(upgraded_database, catalog)
    foundation_report = upgraded.apply(target_version=1)
    assert foundation_report.to_version == 1
    with open_duckdb_connection(upgraded_database) as connection:
        relation = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'federations'
            """
        ).fetchone()
        assert int(relation[0]) == 0
        assert "event_cid" not in _columns(connection, "domain_events")

    upgrade_report = upgraded.apply(target_version=2)
    assert upgrade_report.from_version == 1
    assert upgrade_report.to_version == 2
    assert [receipt.version for receipt in upgrade_report.receipts] == [2]
    assert upgrade_report.schema_fingerprint == fresh_report.schema_fingerprint

    verified = verify_causal_event_federation_schema(upgraded_database)
    assert verified["valid"] is True
    assert verified["schema_fingerprint"] == fresh_report.schema_fingerprint
    assert set(verified["tables_ok"]) == set(CAUSAL_EVENT_FEDERATION_TABLES)

    replay = upgraded.apply()
    assert replay.changed is False
    assert replay.from_version == replay.to_version == 2
    assert replay.schema_fingerprint == fresh_report.schema_fingerprint


def test_upgrade_preserves_legacy_rows_with_additive_defaults(tmp_path: Path) -> None:
    catalog = load_control_plane_catalog()
    database = tmp_path / "legacy.duckdb"
    runner = _runner(database, catalog)
    runner.apply(target_version=1)

    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            INSERT INTO supervisor_instances (
                supervisor_id, repository_id, process_birth_id, started_at,
                stopped_at, status, revision, extension_schema, extension_json
            ) VALUES ('supervisor:legacy', 'repo:legacy', 'birth:legacy',
                      '2026-01-01T00:00:00Z', NULL, 'running', 1, '', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES ('event:legacy', 'stream:legacy', 1, 1, 'TASK_CREATED',
                      '', '', '', '2026-01-01T00:00:00Z', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO budget_reservations (
                reservation_id, budget_kind, owner_session_id, task_cid,
                amount, reserved_at, expires_at, state
            ) VALUES ('reservation:legacy', 'cpu', 'session:legacy', '', 1,
                      '2026-01-01T00:00:00Z', NULL, 'reserved')
            """
        )

    runner.apply(target_version=2)
    with open_duckdb_connection(database) as connection:
        supervisor = connection.execute(
            """
            SELECT repository_id, tenant_id, federation_id, fencing_epoch
            FROM supervisor_instances WHERE supervisor_id = 'supervisor:legacy'
            """
        ).fetchone()
        assert tuple(supervisor[index] for index in range(4)) == (
            "repo:legacy",
            "",
            "",
            0,
        )
        event = connection.execute(
            """
            SELECT event_id, event_cid, tenant_id, causal_parent_ids_json,
                   changed_fact_refs_json
            FROM domain_events WHERE event_id = 'event:legacy'
            """
        ).fetchone()
        assert tuple(event[index] for index in range(5)) == (
            "event:legacy",
            "",
            "",
            "[]",
            "[]",
        )
        budget = connection.execute(
            """
            SELECT reservation_id, federation_id, parent_reservation_id,
                   fencing_epoch
            FROM budget_reservations
            WHERE reservation_id = 'reservation:legacy'
            """
        ).fetchone()
        assert tuple(budget[index] for index in range(4)) == (
            "reservation:legacy",
            "",
            "",
            0,
        )


def test_extension_join_identities_and_reference_ownership_are_first_class(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control.duckdb"
    install_control_plane_schema(
        database,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="casf-schema-identity-test",
    )
    verified = verify_causal_event_federation_schema(database)

    assert set(verified["join_critical_ok"]) == {
        f"{table}.{column}" for table, column in CAUSAL_EVENT_FEDERATION_JOIN_CRITICAL_IDENTITIES
    }
    with open_duckdb_connection(database) as connection:
        for table, column in CAUSAL_EVENT_FEDERATION_JOIN_CRITICAL_IDENTITIES:
            assert not column.endswith("_json"), (table, column)
            assert column in _columns(connection, table), (table, column)

        for table in CAUSAL_EVENT_FEDERATION_REFERENCE_TABLES:
            columns = _columns(connection, table)
            assert {"owner_id", "source_root", "content_ref"} <= columns

        # Compact arrays coexist with normalized, joinable evidence relations.
        event_columns = _columns(connection, "domain_events")
        assert {"causal_parent_ids_json", "changed_fact_refs_json"} <= event_columns
        assert _columns(connection, "domain_event_causal_parents") >= {
            "event_id",
            "parent_event_id",
        }
        assert _columns(connection, "domain_event_changed_facts") >= {
            "event_id",
            "fact_ref",
        }
        assert _columns(connection, "delivery_attempts") >= {
            "event_id",
            "subscription_id",
            "subscription_revision",
            "consumer_id",
            "attempt_number",
            "fencing_epoch",
        }
        delivery_owner_index = connection.execute(
            """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE schema_name = 'main'
              AND index_name = 'delivery_attempts_owner_attempt_uidx'
            """
        ).fetchone()
        assert delivery_owner_index is not None


def test_section_9_inventory_is_closed_queryable_and_tree_scoped(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    install_control_plane_schema(
        database,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="casf-section-9-test",
    )
    verified = verify_causal_event_federation_schema(database)

    assert len(CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS) >= 100
    assert set(verified["section_9_relations_ok"]) == {
        f"{concept}={relation}"
        for concept, relation in CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS.items()
    }
    assert {
        "intent.programs",
        "intent.program_revisions",
        "scheduling.task_conflicts",
        "scheduling.task_resolutions",
        "scheduling.federation_task_scope",
        "proof.proof_cache_entries",
        "proof.proof_seals",
        "retrieval.bm25_postings",
        "retrieval.vectors",
        "retrieval.knowledge_graph_edges",
    } <= CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS.keys()

    source_bound = {
        "semantic_effect_references",
        "semantic_relationship_references",
        "semantic_capsule_dependencies",
        "semantic_contract_references",
        "environment_binding_references",
        "proof_units",
        "proof_receipts",
        "proof_cache_entries",
        "proof_seals",
        "test_selections",
        "test_receipts",
        "validation_plans",
        "documents",
        "document_chunks",
        "vector_metadata",
        "knowledge_graph_nodes",
        "knowledge_graph_edges",
    }
    with open_duckdb_connection(database) as connection:
        for relation in set(CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS.values()):
            assert _columns(connection, relation), relation
        for relation in source_bound:
            assert {
                "owner_id",
                "source_root",
                "provenance_ref",
                "content_ref",
                "revision",
                "status",
                "freshness_state",
            } <= _columns(connection, relation), relation
        assert {
            "tenant_id",
            "federation_id",
            "task_cid",
            "repository_id",
            "tree_id",
            "goal_cid",
            "plan_revision_id",
            "assignment_revision",
        } <= _columns(connection, "federation_task_bindings")
        assert "vector_ref" in _columns(connection, "vectors")
        assert "body_json" not in _columns(connection, "vectors")


def test_migration_two_failure_rolls_back_all_additive_ddl(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    catalog = load_control_plane_catalog()
    runner = _runner(database, catalog)
    runner.apply(target_version=1)

    def fail_before_commit(phase: str, migration: ControlPlaneMigration) -> None:
        if phase == "before_commit" and migration.version == 2:
            raise RuntimeError("injected migration-2 failure")

    with pytest.raises(RuntimeError, match="migration-2 failure"):
        runner.apply(target_version=2, fault_injector=fail_before_commit)
    assert runner.current_version() == 1
    with open_duckdb_connection(database) as connection:
        relation = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'federations'
            """
        ).fetchone()
        assert int(relation[0]) == 0
        assert "event_cid" not in _columns(connection, "domain_events")


def test_applied_migration_two_checksum_drift_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    catalog = load_control_plane_catalog()
    runner = _runner(database, catalog)
    runner.apply()

    sql_directory = tmp_path / "altered-sql"
    sql_directory.mkdir()
    for source in sorted(package_sql_directory().glob("*.sql")):
        text = source.read_text(encoding="utf-8")
        if source.name == "0002_causal_event_federation_core.sql":
            text += "\n-- unauthorized checksum drift\n"
        (sql_directory / source.name).write_text(text, encoding="utf-8")
    altered = MigrationCatalog.from_sql_directory(sql_directory)
    assert altered.get(2).checksum != catalog.get(2).checksum

    drifted = _runner(database, altered)
    with pytest.raises(MigrationDriftError, match="checksum drift"):
        drifted.apply()
