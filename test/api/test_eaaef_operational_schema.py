from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
    EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    EAAEF_OPERATIONAL_MIGRATION_ID,
    EAAEF_OPERATIONAL_MIGRATION_VERSION,
    EAAEFOperationalSchemaError,
    eaaef_operation_vocabulary_cid,
    install_eaaef_operational_schema,
    verify_eaaef_operational_schema,
)

pytest.importorskip("duckdb")


def _vocabulary_cid() -> str:
    return eaaef_operation_vocabulary_cid(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS)


def _install(path: Path):
    return install_eaaef_operational_schema(
        path,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="test:eaaef-offline-materializer",
    )


def test_eaaef_operational_profile_installs_only_through_versioned_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"

    report = _install(path)
    verification = verify_eaaef_operational_schema(
        path, operation_vocabulary_cid=_vocabulary_cid()
    )

    assert report.to_version == EAAEF_OPERATIONAL_MIGRATION_VERSION
    assert report.changed is True
    assert report.receipts[-1].migration_id == EAAEF_OPERATIONAL_MIGRATION_ID
    assert verification["valid"] is True
    assert verification["schema_version"] == EAAEF_OPERATIONAL_MIGRATION_VERSION
    assert verification["runtime_ddl_allowed"] is False
    assert verification["direct_database_open_allowed"] is False
    assert verification["sidecar_writes_allowed"] is False
    assert verification["operation_vocabulary_cid"] == _vocabulary_cid()


def test_eaaef_operational_profile_replays_without_ddl(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    first = _install(path)
    second = _install(path)

    assert first.changed is True
    assert second.changed is False
    assert second.from_version == second.to_version == 2
    assert second.schema_fingerprint == first.schema_fingerprint


def test_eaaef_operational_verifier_rejects_base_profile_only(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
    )

    path = tmp_path / "control.duckdb"
    install_datasets_authoritative_operational_schema(
        path,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="test:base-only",
    )

    with pytest.raises(EAAEFOperationalSchemaError, match="migration history"):
        verify_eaaef_operational_schema(
            path, operation_vocabulary_cid=_vocabulary_cid()
        )


def test_operation_vocabulary_identity_is_order_independent_and_closed() -> None:
    forward = tuple(sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS))
    reverse = tuple(reversed(forward))

    assert eaaef_operation_vocabulary_cid(forward) == (
        eaaef_operation_vocabulary_cid(reverse)
    )
    with pytest.raises(EAAEFOperationalSchemaError, match="duplicate-free"):
        eaaef_operation_vocabulary_cid(("task.get", "task.get"))


def test_profile_rejects_wrong_vocabulary_and_poisoned_index(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _install(path)

    with pytest.raises(EAAEFOperationalSchemaError, match="31-operation"):
        verify_eaaef_operational_schema(
            path,
            operation_vocabulary_cid=eaaef_operation_vocabulary_cid(
                {"task.get", "task.ready"}
            ),
        )

    with open_duckdb_connection(path) as connection:
        connection.execute("DROP INDEX eaaef_task_claim_lease_uidx")
    with pytest.raises(EAAEFOperationalSchemaError, match="index set"):
        verify_eaaef_operational_schema(
            path, operation_vocabulary_cid=_vocabulary_cid()
        )


def test_profile_rejects_forged_migration_checksum_and_column_drift(
    tmp_path: Path,
) -> None:
    checksum_path = tmp_path / "checksum.duckdb"
    _install(checksum_path)
    with open_duckdb_connection(checksum_path) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum=? WHERE version=2",
            ["sha256:" + "0" * 64],
        )
    with pytest.raises(EAAEFOperationalSchemaError, match="migration history"):
        verify_eaaef_operational_schema(
            checksum_path, operation_vocabulary_cid=_vocabulary_cid()
        )

    column_path = tmp_path / "column.duckdb"
    _install(column_path)
    with open_duckdb_connection(column_path) as connection:
        connection.execute(
            "ALTER TABLE eaaef_completion_barriers ADD COLUMN poison VARCHAR"
        )
    with pytest.raises(EAAEFOperationalSchemaError, match="seal"):
        verify_eaaef_operational_schema(
            column_path, operation_vocabulary_cid=_vocabulary_cid()
        )
