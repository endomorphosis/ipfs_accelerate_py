"""Published operational version-2 histories must never alias one another."""

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigrationRunner,
    MigrationDriftError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_datasets_authoritative_operational_schema,
    load_datasets_authoritative_operational_base_catalog,
    load_datasets_authoritative_operational_catalog,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    install_eaaef_operational_schema,
    load_eaaef_operational_catalog,
)

pytest.importorskip("duckdb")


def test_versioned_profiles_share_only_the_exact_published_base():
    base = load_datasets_authoritative_operational_base_catalog()
    ordinary = load_datasets_authoritative_operational_catalog()
    eaaef = load_eaaef_operational_catalog()
    assert [item.version for item in base] == [1]
    assert ordinary.get(1) == eaaef.get(1) == base.get(1)
    assert [(item.version, item.migration_id) for item in ordinary] == [
        (1, "0001_datasets_authoritative_operational_control_plane"),
        (2, "0002_operational_hash_observations"),
    ]
    assert [(item.version, item.migration_id) for item in eaaef] == [
        (1, "0001_datasets_authoritative_operational_control_plane"),
        (2, "0002_eaaef_owner_transaction_operational_extension"),
    ]
    assert ordinary.get(2).checksum != eaaef.get(2).checksum


@pytest.mark.parametrize("installed_profile", ["ordinary_hash", "eaaef"])
def test_each_published_history_installs_and_rejects_the_other_without_rewriting(
    tmp_path, installed_profile
):
    path = tmp_path / "control.duckdb"
    install, own_catalog, other_catalog = (
        (
            install_datasets_authoritative_operational_schema,
            load_datasets_authoritative_operational_catalog,
            load_eaaef_operational_catalog,
        )
        if installed_profile == "ordinary_hash"
        else (
            install_eaaef_operational_schema,
            load_eaaef_operational_catalog,
            load_datasets_authoritative_operational_catalog,
        )
    )
    report = install(
        path,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="test:profile",
    )
    assert report.to_version == 2
    with open_duckdb_connection(path) as connection:
        before = [
            tuple(row[i] for i in range(3))
            for row in connection.execute(
                "SELECT version, migration_id, checksum FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        tables_before = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
            ).fetchall()
        ]
    assert before == [
        (item.version, item.migration_id, item.checksum) for item in own_catalog()
    ]
    runner = ControlPlaneMigrationRunner.for_database(
        path,
        catalog=other_catalog(),
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="test:other-profile",
    )
    with pytest.raises(MigrationDriftError):
        runner.apply()
    with open_duckdb_connection(path) as connection:
        assert [
            tuple(row[i] for i in range(3))
            for row in connection.execute(
                "SELECT version, migration_id, checksum FROM schema_migrations ORDER BY version"
            ).fetchall()
        ] == before
        assert [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
            ).fetchall()
        ] == tables_before
