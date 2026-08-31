"""Focused execution-store ART repair and restart metadata regressions."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    _DAEMON_EXECUTION_SQL,
    DATABASE_EXECUTION_STORAGE_REPAIR_SCHEMA,
    DatabaseImplementationDaemon,
    DatabaseImplementationExecutionStorageRepairedError,
    DatabaseImplementationExecutionStorageRepairError,
    _database_execution_storage_projection_from_connection,
    _split_sql_statements,
    repair_database_execution_art_index_storage,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for execution-store ART repair tests",
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_execution_store(path: Path) -> dict[str, Any]:
    connection = open_duckdb_connection(path)
    try:
        for statement in _split_sql_statements(_DAEMON_EXECUTION_SQL):
            connection.execute(statement)
        connection.execute(
            "INSERT INTO daemon_execution_metadata(key, value) VALUES (?, ?)",
            ["execution_store_identity", "execution-store:seed"],
        )
        connection.execute(
            """
            INSERT INTO database_task_attempts(
                attempt_id, claim_id, task_cid, task_alias, attempt_number,
                owner_session_id, fencing_token, fence_epoch, lease_id,
                committed_phase, status, started_at_ms, finished_at_ms,
                revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            [
                "attempt:1",
                "claim:1",
                "task:1",
                "PCTDD-TEST",
                1,
                "owner:1",
                7,
                3,
                "lease:1",
                "provider",
                "running",
                100,
                2,
                '{"attempt":1}',
            ],
        )
        connection.execute(
            "INSERT INTO attempt_phases VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["attempt:1", "claimed", 100, 7, 3, 1, '{"phase":"claimed"}'],
        )
        connection.execute(
            "INSERT INTO provider_invocations VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                "provider:1",
                "attempt:1",
                "task:1",
                "provider-key:1",
                "owner:1",
                101,
                '{"status":"ok"}',
            ],
        )
        connection.execute(
            "INSERT INTO effect_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "effect:1",
                "attempt:1",
                "task:1",
                "effect-key:1",
                "effect-idempotency:1",
                "owner:1",
                102,
                '{"status":"applied"}',
            ],
        )
        connection.execute(
            "INSERT INTO attempt_dispatch_journal VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "dispatch:1",
                "attempt:1",
                "task:1",
                "provider",
                "dispatch-key:1",
                "owner:1",
                7,
                3,
                100,
                101,
                "committed",
                '{"outcome":"committed"}',
            ],
        )
        connection.execute(
            "INSERT INTO database_portal_attempt_bindings VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "attempt:1",
                "task:1",
                "claim:1",
                1,
                "owner:1",
                "lease:1",
                7,
                3,
                "binding:1",
                "sha256:projection",
                "prepared",
                '{"binding":"exact"}',
            ],
        )
        connection.execute(
            "INSERT INTO database_portal_terminal_reconciliations VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "attempt:1",
                "task:1",
                "claim:1",
                1,
                "owner:1",
                "lease:1",
                7,
                3,
                "terminalized_for_retry",
                "evidence:1",
                "prepared:1",
                "barrier:1",
                "commit_barrier",
                "receipt:1",
                '{"terminal":"exact"}',
            ],
        )
        connection.execute(
            "INSERT INTO daemon_execution_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                "event:1",
                "attempt:1",
                "task:1",
                "provider_committed",
                103,
                '{"event":"exact"}',
            ],
        )
        connection.execute("CHECKPOINT")
        return _database_execution_storage_projection_from_connection(connection)
    finally:
        connection.close()


def _open_minimal_daemon(path: Path) -> DatabaseImplementationDaemon:
    return DatabaseImplementationDaemon(
        database_path=path.with_name("control.duckdb"),
        coordination_path=path.with_name("coordination.duckdb"),
        execution_path=path,
        owner_session_id="owner:1",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=object(),
        coordinator=object(),
    )


def _leave_wal(path: Path, statements: list[str]) -> Path:
    script = "\n".join(
        [
            "import duckdb, os, sys",
            "connection = duckdb.connect(sys.argv[1])",
            *[f"connection.execute({statement!r})" for statement in statements],
            "os._exit(0)",
        ]
    )
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        timeout=30,
    )
    wal_path = path.with_name(path.name + ".wal")
    assert wal_path.is_file()
    assert wal_path.stat().st_size > 0
    return wal_path


def test_rebuild_preserves_complete_execution_projection_and_catalog(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    path.with_name(path.name + ".wal").touch()

    receipt = repair_database_execution_art_index_storage(path)

    assert receipt["schema"] == DATABASE_EXECUTION_STORAGE_REPAIR_SCHEMA
    assert receipt["logical_projection_equal"] is True
    assert receipt["pre_projection_root"] == before["projection_root"]
    assert receipt["post_projection_root"] == before["projection_root"]
    assert receipt["table_count"] == 9
    assert receipt["index_count"] == 3
    assert set(receipt["row_counts"]) == {item["table"] for item in before["tables"]}
    assert all(receipt["row_counts"].values())
    assert receipt["interrupted_transaction_outcome"] == ("not_inferred_reconcile_exact_operation")
    assert receipt["repair_accepted_interrupted_transaction"] is False
    assert receipt["reconciliation_required"] is True
    assert receipt["same_process_retry_permitted"] is False
    assert Path(receipt["quarantined_source_path"]).is_file()
    assert Path(receipt["quarantined_empty_wal_path"]).is_file()
    assert Path(receipt["prepared_phase_path"]).is_file()
    assert Path(receipt["committed_phase_path"]).is_file()

    connection = open_duckdb_connection(path)
    try:
        after = _database_execution_storage_projection_from_connection(connection)
    finally:
        connection.close()
    assert after == before


def test_repeated_open_updates_process_value_without_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    statements: list[str] = []
    real_open = open_duckdb_connection

    class RecordingConnection:
        def __init__(self, connection: Any) -> None:
            self._connection = connection

        def execute(self, sql: str, parameters: Any = None) -> Any:
            statements.append(" ".join(str(sql).upper().split()))
            if parameters is None:
                return self._connection.execute(sql)
            return self._connection.execute(sql, parameters)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

    def recording_open(database: Path | str, **kwargs: Any) -> Any:
        return RecordingConnection(real_open(database, **kwargs))

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state.open_duckdb_connection",
        recording_open,
    )
    process_ids: list[str] = []
    execution_store_ids: list[str] = []
    for _ in range(8):
        daemon = _open_minimal_daemon(path)
        process_ids.append(daemon.process_instance_id)
        execution_store_ids.append(daemon.execution_store_identity)
        daemon.close()

    assert len(set(process_ids)) == len(process_ids)
    assert len(set(execution_store_ids)) == 1
    assert not any(
        "INSERT OR REPLACE INTO DAEMON_EXECUTION_METADATA" in statement for statement in statements
    )
    assert (
        sum(
            "UPDATE DAEMON_EXECUTION_METADATA SET VALUE = ?" in statement
            for statement in statements
        )
        >= 7
    )


def test_run_once_repairs_exact_post_open_art_fatal_and_never_retries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    daemon = _open_minimal_daemon(path)
    actual = daemon._connection
    calls = 0

    class FatalException(RuntimeError):
        pass

    class FatalAttemptRead:
        def execute(self, sql: str, parameters: Any = None) -> Any:
            nonlocal calls
            calls += 1
            if "FROM database_task_attempts" in str(sql):
                raise FatalException(
                    "FATAL Error: Failed to delete all rows from index. "
                    "Only deleted 0 out of 1 rows."
                )
            if parameters is None:
                return actual.execute(sql)
            return actual.execute(sql, parameters)

        def __getattr__(self, name: str) -> Any:
            return getattr(actual, name)

    daemon._connection = daemon_module._DatabaseExecutionStoreConnection(
        FatalAttemptRead(),
        database_path=path,
    )
    daemon._run_once_database_authoritative = (  # type: ignore[method-assign]
        lambda: {"attempts": daemon.list_running_attempts()}
    )
    with pytest.raises(DatabaseImplementationExecutionStorageRepairedError) as captured:
        daemon.run_once()

    assert calls == 1
    assert daemon._connection is None
    assert captured.value.receipt["logical_projection_equal"] is True
    assert captured.value.receipt["same_process_retry_permitted"] is False
    assert captured.value.receipt["reconciliation_required"] is True

    replacement = _open_minimal_daemon(path)
    try:
        after = _database_execution_storage_projection_from_connection(replacement._connection)
        attempts = replacement.list_running_attempts(
            owner_session_id="owner:1",
            apply_selection=False,
        )
    finally:
        replacement.close()
    # Restart metadata is expected to change, but exact operation tables do not.
    before_tables = {item["table"]: item for item in before["tables"]}
    after_tables = {item["table"]: item for item in after["tables"]}
    for table in before_tables:
        if table != "daemon_execution_metadata":
            assert after_tables[table] == before_tables[table]
    assert [attempt.attempt_id for attempt in attempts] == ["attempt:1"]


def test_non_execution_art_fatal_never_repairs_execution_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    daemon = _open_minimal_daemon(path)
    repair_calls = 0

    class FatalException(RuntimeError):
        pass

    failure = FatalException(
        "FATAL Error: Failed to delete all rows from index. Only deleted 0 out of 1 rows."
    )

    def forbidden_repair(_path: Path) -> dict[str, Any]:
        nonlocal repair_calls
        repair_calls += 1
        return {}

    monkeypatch.setattr(
        daemon_module,
        "_repair_database_execution_art_index_storage_under_writer_lock",
        forbidden_repair,
    )
    daemon._run_once_database_authoritative = (  # type: ignore[method-assign]
        lambda: (_ for _ in ()).throw(failure)
    )
    try:
        with pytest.raises(FatalException) as captured:
            daemon.run_once()
        assert captured.value is failure
        assert repair_calls == 0
        assert daemon._connection is not None
    finally:
        daemon.close()


def test_redundant_nonempty_wal_is_proven_preserved_and_retired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    wal_path = _leave_wal(
        path,
        [
            "BEGIN",
            "UPDATE daemon_execution_metadata SET value='temporary' "
            "WHERE key='execution_store_identity'",
            "UPDATE daemon_execution_metadata SET value='execution-store:seed' "
            "WHERE key='execution_store_identity'",
            "COMMIT",
        ],
    )
    wal_digest = _sha256(wal_path)

    receipt = repair_database_execution_art_index_storage(path)

    proof = receipt["wal_redundancy_proof"]
    assert proof["logical_projection_equal"] is True
    assert proof["main_projection_root"] == proof["recovered_projection_root"]
    assert proof["wal_sha256"] == wal_digest
    assert not wal_path.exists()
    assert _sha256(Path(receipt["quarantined_wal_path"])) == wal_digest
    assert _sha256(Path(receipt["retired_live_wal_path"])) == wal_digest
    connection = open_duckdb_connection(path)
    try:
        after = _database_execution_storage_projection_from_connection(connection)
    finally:
        connection.close()
    assert after == before


def test_distinct_nonempty_wal_is_typed_refusal_and_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = _leave_wal(
        path,
        [
            "UPDATE daemon_execution_metadata SET value='distinct' "
            "WHERE key='execution_store_identity'",
        ],
    )
    before_main = _sha256(path)
    before_wal = _sha256(wal_path)

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError) as captured:
        repair_database_execution_art_index_storage(path)

    assert captured.value.status["reason"] == ("nonempty_wal_changes_logical_projection")
    assert captured.value.status["repair_performed"] is False
    assert captured.value.status["retry_permitted"] is False
    assert _sha256(path) == before_main
    assert _sha256(wal_path) == before_wal


def test_post_install_crash_rolls_back_exact_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    source_digest = _sha256(path)
    real_fsync = daemon_module._fsync_database_execution_storage_directory
    parent_calls = 0

    def fail_first_post_install_fsync(candidate: Path) -> None:
        nonlocal parent_calls
        if candidate == path.parent:
            parent_calls += 1
            if parent_calls == 2:
                raise OSError("injected post-install crash")
        real_fsync(candidate)

    monkeypatch.setattr(
        daemon_module,
        "_fsync_database_execution_storage_directory",
        fail_first_post_install_fsync,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="candidate was not installed",
    ):
        repair_database_execution_art_index_storage(path)

    assert _sha256(path) == source_digest
    connection = open_duckdb_connection(path)
    try:
        after = _database_execution_storage_projection_from_connection(connection)
    finally:
        connection.close()
    assert after == before
    assert tuple(
        (path.parent / ".execution-art-repair-quarantine").glob("*.failed-replacement.duckdb")
    )
