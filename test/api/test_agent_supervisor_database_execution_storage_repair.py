"""Focused execution-store ART repair and restart metadata regressions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
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


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _repair_temporaries(path: Path) -> tuple[Path, ...]:
    return tuple(sorted(path.parent.glob(f".{path.name}.art-repair-*.duckdb*")))


def _execution_art_failure(
    daemon: DatabaseImplementationDaemon,
) -> BaseException:
    cause = RuntimeError(
        "FATAL Error: Failed to delete all rows from index. Only deleted 0 out of 1 rows."
    )
    return daemon_module._DatabaseImplementationExecutionStorageArtFatal(
        database_path=daemon.execution_path,
        connection=daemon._connection,
        cause=cause,
    )


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


def _crash_repair_after_wal_retirement(path: Path) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        [
            "import os, sys",
            "from pathlib import Path",
            "from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon",
            "path = Path(sys.argv[1])",
            "real_exchange = daemon._exchange_database_execution_storage_names",
            "def crash(source_name, target_name, **kwargs):",
            "    if source_name.startswith(f'.{path.name}.art-repair-') and target_name == path.name:",
            "        os._exit(91)",
            "    return real_exchange(source_name, target_name, **kwargs)",
            "daemon._exchange_database_execution_storage_names = crash",
            "daemon.repair_database_execution_art_index_storage(path)",
        ]
    )
    repository_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (
            str(repository_root),
            environment.get("PYTHONPATH", ""),
        )
        if item
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_rebuild_preserves_complete_execution_projection_and_catalog(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    os.chmod(path, 0o664)
    wal_path = path.with_name(path.name + ".wal")
    wal_path.touch()
    os.chmod(wal_path, 0o664)

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
    assert Path(receipt["retired_live_wal_path"]).is_file()
    assert Path(receipt["prepared_phase_path"]).is_file()
    assert Path(receipt["committed_phase_path"]).is_file()
    quarantine = Path(receipt["quarantined_source_path"]).parent
    assert _mode(quarantine) == 0o700
    for evidence_path in (
        Path(receipt["quarantined_source_path"]),
        Path(receipt["quarantined_empty_wal_path"]),
        Path(receipt["retired_live_wal_path"]),
        Path(receipt["prepared_phase_path"]),
        Path(receipt["committed_phase_path"]),
    ):
        assert _mode(evidence_path) == 0o600
    assert not wal_path.exists()
    prepared = json.loads(
        Path(receipt["prepared_phase_path"]).read_text(encoding="utf-8")
    )
    assert prepared["source_sha256"] == prepared["backup_sha256"]
    assert prepared["source_sha256"] == prepared["rollback_sha256"]
    assert prepared["source_size_bytes"] == prepared["backup_size_bytes"]
    assert prepared["source_size_bytes"] == prepared["rollback_size_bytes"]
    assert prepared["source_identity"]["device"] == prepared["rollback_identity"][
        "device"
    ]
    assert prepared["source_identity"]["inode"] == prepared["rollback_identity"][
        "inode"
    ]
    assert prepared["wal_sha256"] == prepared["wal_backup_sha256"]
    assert prepared["wal_size_bytes"] == prepared["wal_backup_size_bytes"] == 0
    assert not _repair_temporaries(path)

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
    execute_calls = 0
    fetch_calls = 0

    class FatalException(RuntimeError):
        pass

    class FatalFetchResult:
        def __init__(self, result: Any) -> None:
            self._result = result

        def fetchall(self) -> Any:
            nonlocal fetch_calls
            fetch_calls += 1
            raise FatalException(
                "FATAL Error: Failed to delete all rows from index. Only deleted 0 out of 1 rows."
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._result, name)

    class FatalAttemptRead:
        def execute(self, sql: str, parameters: Any = None) -> Any:
            nonlocal execute_calls
            execute_calls += 1
            if parameters is None:
                result = actual.execute(sql)
            else:
                result = actual.execute(sql, parameters)
            if "FROM database_task_attempts" in str(sql):
                return FatalFetchResult(result)
            return result

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

    assert execute_calls == 1
    assert fetch_calls == 1
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


def test_resume_path_art_fatal_repairs_without_failure_terminalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    daemon = _open_minimal_daemon(path)
    attempt = daemon.list_running_attempts(
        owner_session_id="owner:1",
        apply_selection=False,
    )[0]
    failure = _execution_art_failure(daemon)
    terminalizations = 0

    def fail_resume(_attempt: Any) -> Any:
        raise failure

    def forbidden_terminalization(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal terminalizations
        terminalizations += 1
        raise AssertionError("execution ART must not terminalize a task")

    daemon.resume_attempt = fail_resume  # type: ignore[method-assign]
    daemon._finalize_failed_attempt = (  # type: ignore[method-assign]
        forbidden_terminalization
    )
    daemon._run_once_database_authoritative = (  # type: ignore[method-assign]
        lambda: daemon._resume_attempt_without_process_crash(attempt)
    )
    with pytest.raises(DatabaseImplementationExecutionStorageRepairedError):
        daemon.run_once()

    assert terminalizations == 0


def test_provider_commit_art_fatal_repairs_without_callback_redispatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    daemon = _open_minimal_daemon(path)
    attempt = daemon.list_running_attempts(
        owner_session_id="owner:1",
        apply_selection=False,
    )[0]
    attempt = replace(attempt, committed_phase="context")
    failure = _execution_art_failure(daemon)
    provider_calls = 0

    def provider(_attempt: Any) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        return {"status": "ok", "accepted": True}

    def fail_commit(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    daemon._provider_fn = provider
    daemon._protect_attempt_write = lambda _attempt: None  # type: ignore[method-assign]
    daemon._protect_attempt_control_binding = (  # type: ignore[method-assign]
        lambda _attempt: None
    )
    daemon.provider_invocation_recorded = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )
    daemon._dispatch_journal_entry = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )
    daemon._begin_callback_dispatch = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )
    daemon._record_callback_dispatch_outcome = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )
    daemon._run_with_attempt_heartbeat = (  # type: ignore[method-assign]
        lambda _attempt, callback: callback()
    )
    daemon.commit_phase = fail_commit  # type: ignore[method-assign]
    daemon._run_once_database_authoritative = (  # type: ignore[method-assign]
        lambda: daemon.run_provider(attempt)
    )
    with pytest.raises(DatabaseImplementationExecutionStorageRepairedError):
        daemon.run_once()

    assert provider_calls == 1


@pytest.mark.parametrize("fatal_site", ["callback", "renewal"])
def test_heartbeat_execution_art_reaches_repair_without_accepting_callback(
    tmp_path: Path,
    fatal_site: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    daemon = _open_minimal_daemon(path)
    attempt = daemon.list_running_attempts(
        owner_session_id="owner:1",
        apply_selection=False,
    )[0]
    attempt = replace(attempt, committed_phase="context")
    failure = _execution_art_failure(daemon)
    renewal_failed = threading.Event()
    callback_calls = 0
    renewal_calls = 0
    accepted_outcomes = 0
    commit_calls = 0

    def renew(*_args: Any, **_kwargs: Any) -> object:
        nonlocal renewal_calls
        renewal_calls += 1
        if fatal_site == "renewal" and renewal_calls >= 2:
            renewal_failed.set()
            raise failure
        return object()

    def provider(_attempt: Any) -> dict[str, Any]:
        nonlocal callback_calls
        callback_calls += 1
        if fatal_site == "callback":
            raise failure
        assert renewal_failed.wait(timeout=2.0)
        return {"status": "ok", "accepted": True}

    def record_outcome(*_args: Any, **_kwargs: Any) -> None:
        nonlocal accepted_outcomes
        accepted_outcomes += 1

    def forbidden_commit(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal commit_calls
        commit_calls += 1
        raise AssertionError("heartbeat ART must stop before phase commit")

    daemon._lease_heartbeat_interval_seconds = 0.001
    daemon._provider_fn = provider
    daemon._attempt_claim = lambda _attempt: object()  # type: ignore[method-assign]
    daemon._renew_attempt_lease = renew  # type: ignore[method-assign]
    daemon._protect_attempt_write = lambda _attempt: None  # type: ignore[method-assign]
    daemon._protect_attempt_control_binding = (  # type: ignore[method-assign]
        lambda _attempt: None
    )
    daemon.provider_invocation_recorded = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )
    daemon._dispatch_journal_entry = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )
    daemon._begin_callback_dispatch = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )
    daemon._record_callback_dispatch_outcome = record_outcome  # type: ignore[method-assign]
    daemon.commit_phase = forbidden_commit  # type: ignore[method-assign]
    daemon._run_once_database_authoritative = (  # type: ignore[method-assign]
        lambda: daemon.run_provider(attempt)
    )

    with pytest.raises(DatabaseImplementationExecutionStorageRepairedError):
        daemon.run_once()

    assert callback_calls == 1
    assert accepted_outcomes == 0
    assert commit_calls == 0
    assert daemon._connection is None
    connection = open_duckdb_connection(path)
    try:
        accepted = connection.execute(
            "SELECT COUNT(*) FROM provider_invocations WHERE idempotency_key = ?",
            [f"provider:{attempt.attempt_id}"],
        ).fetchone()
    finally:
        connection.close()
    assert accepted is not None and int(accepted[0]) == 0


def test_portal_startup_art_fatal_repairs_before_claim_or_continuation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    daemon = _open_minimal_daemon(path)
    failure = _execution_art_failure(daemon)
    claims = 0

    def fail_startup(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    def forbidden_claim(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal claims
        claims += 1
        raise AssertionError("execution ART must stop before claiming")

    daemon._database_portal_bridge = object()
    daemon._database_portal_reconciliation_checked = False
    daemon.reconcile_quiesced_database_portal_attempts = (  # type: ignore[method-assign]
        fail_startup
    )
    daemon.claim_next = forbidden_claim  # type: ignore[method-assign]
    with pytest.raises(DatabaseImplementationExecutionStorageRepairedError):
        daemon.run_once()

    assert claims == 0


@pytest.mark.parametrize(
    "operation",
    ["fetchone", "fetchmany", "fetchall", "iteration"],
)
def test_execution_result_adapter_preserves_art_failure_provenance(
    tmp_path: Path,
    operation: str,
) -> None:
    path = tmp_path / "execution.duckdb"

    class FatalException(RuntimeError):
        pass

    failure = FatalException(
        "FATAL Error: Failed to delete all rows from index. Only deleted 0 out of 1 rows."
    )

    class FatalResult:
        def _outcome(self, candidate: str, default: Any) -> Any:
            if operation == candidate:
                raise failure
            return default

        def fetchone(self) -> Any:
            return self._outcome("fetchone", None)

        def fetchmany(self, size: int = 1) -> Any:
            return self._outcome("fetchmany", [(size,)])

        def fetchall(self) -> Any:
            return self._outcome("fetchall", [])

        def __iter__(self) -> FatalResult:
            return self

        def __next__(self) -> Any:
            if operation == "iteration":
                raise failure
            raise StopIteration

    class RawConnection:
        def execute(self, _sql: str, _parameters: Any = None) -> FatalResult:
            return FatalResult()

    connection = daemon_module._DatabaseExecutionStoreConnection(
        RawConnection(),
        database_path=path,
    )
    result = connection.execute("SELECT 1")
    with pytest.raises(daemon_module._DatabaseImplementationExecutionStorageArtFatal) as captured:
        if operation == "fetchone":
            result.fetchone()
        elif operation == "fetchmany":
            result.fetchmany(7)
        elif operation == "fetchall":
            result.fetchall()
        else:
            next(iter(result))

    assert captured.value.database_path == path
    assert captured.value.connection is connection
    assert captured.value.cause is failure


def test_eager_duckdb_cursor_identity_and_capabilities_are_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cursor.duckdb"
    raw = open_duckdb_connection(path)
    connection = daemon_module._DatabaseExecutionStoreConnection(
        raw,
        database_path=path,
    )
    try:
        result = connection.execute("SELECT 1 AS value")
        assert type(result).__name__ == "DuckDBCursor"
        assert type(result).__module__.endswith("task_sources.duckdb_state")
        assert hasattr(result, "fetchone")
        assert not hasattr(result, "fetchmany")
        assert hasattr(result, "fetchall")
        assert hasattr(result, "__iter__")
        assert not hasattr(result, "__next__")
        assert [row[0] for row in result] == [1]
    finally:
        connection.close()


@pytest.mark.parametrize("transaction_site", ["commit_phase", "reconciliation"])
def test_transaction_art_fatal_is_not_masked_by_invalidated_rollback(
    tmp_path: Path,
    transaction_site: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    daemon = _open_minimal_daemon(path)
    attempt = daemon.list_running_attempts(
        owner_session_id="owner:1",
        apply_selection=False,
    )[0]
    real_connection = daemon._connection
    assert real_connection is not None
    real_connection.close()
    commands: list[str] = []
    failure: BaseException | None = None

    class FatalFetch:
        def fetchone(self) -> Any:
            assert failure is not None
            raise failure

    class InvalidatedTransactionConnection:
        def execute(self, sql: str, _parameters: Any = None) -> Any:
            normalized = " ".join(str(sql).upper().split())
            commands.append(normalized)
            if normalized == "ROLLBACK":
                raise RuntimeError("connection invalidated")
            if normalized.startswith("UPDATE DATABASE_TASK_ATTEMPTS"):
                return FatalFetch()
            return object()

        def close(self) -> None:
            return None

    daemon._connection = daemon_module._DatabaseExecutionStoreConnection(
        InvalidatedTransactionConnection(),
        database_path=path,
    )
    failure = _execution_art_failure(daemon)
    try:
        with pytest.raises(
            daemon_module._DatabaseImplementationExecutionStorageArtFatal
        ) as captured:
            if transaction_site == "commit_phase":
                daemon.commit_phase(
                    attempt,
                    "provider",
                    require_live_claim=False,
                )
            else:
                prepared = {
                    "attempt_id": attempt.attempt_id,
                    "claim_id": attempt.claim_id,
                    "task_cid": attempt.task_cid,
                    "attempt_number": attempt.attempt_number,
                    "owner_session_id": attempt.owner_session_id,
                    "fencing_token": attempt.fencing_token,
                    "fence_epoch": attempt.fence_epoch,
                    "lease_id": attempt.lease_id,
                    "preparation_digest": "sha256:prepared",
                }
                daemon.get_attempt = lambda _attempt_id: attempt  # type: ignore[method-assign]
                daemon._commit_reconciled_attempt_terminal(
                    prepared,
                    succeeded=True,
                    reconciliation={"receipt_id": "receipt:1"},
                )
        assert captured.value is failure
        assert "ROLLBACK" not in commands
    finally:
        daemon.close()


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
    assert proof == receipt["wal_recovery_proof"]
    assert proof["profile"] == (
        daemon_module.DATABASE_EXECUTION_STORAGE_WAL_RECOVERY_PROFILE
    )
    assert proof["disposition"] == "redundant"
    assert proof["logical_projection_equal"] is True
    assert proof["recovered_state_selected"] is False
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


def test_distinct_nonempty_wal_is_recovered_from_private_exact_pair(
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
    main_digest = _sha256(path)
    wal_digest = _sha256(wal_path)

    receipt = repair_database_execution_art_index_storage(path)

    proof = receipt["wal_recovery_proof"]
    assert proof["profile"] == (
        daemon_module.DATABASE_EXECUTION_STORAGE_WAL_RECOVERY_PROFILE
    )
    assert proof["disposition"] == "distinct_recovered"
    assert proof["logical_projection_equal"] is False
    assert proof["recovered_state_selected"] is True
    assert proof["private_replay"] is True
    assert proof["live_authority_opened"] is False
    assert proof["main_projection_root"] != proof["recovered_projection_root"]
    assert proof["selected_projection_root"] == proof["recovered_projection_root"]
    assert receipt["wal_disposition"] == "distinct_recovered"
    assert receipt["wal_redundancy_proof"] == {}
    assert receipt["pre_projection_root"] == proof["recovered_projection_root"]
    assert receipt["post_projection_root"] == proof["recovered_projection_root"]
    assert receipt["wal_evidence_preserved"] is True
    assert _sha256(Path(receipt["quarantined_source_path"])) == main_digest
    assert _sha256(Path(receipt["quarantined_wal_path"])) == wal_digest
    assert _sha256(Path(receipt["retired_live_wal_path"])) == wal_digest
    assert not wal_path.exists()
    connection = open_duckdb_connection(path)
    try:
        row = connection.execute(
            "SELECT value FROM daemon_execution_metadata "
            "WHERE key='execution_store_identity'"
        ).fetchone()
        assert row is not None and row[0] == "distinct"
        after = _database_execution_storage_projection_from_connection(connection)
    finally:
        connection.close()
    assert after["projection_root"] == proof["recovered_projection_root"]


def test_distinct_nonempty_wal_recovers_full_execution_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = _leave_wal(
        path,
        [
            "BEGIN",
            "UPDATE database_task_attempts SET committed_phase='validation', "
            "status='validating', revision=3, body_json='{\"attempt\":2}' "
            "WHERE attempt_id='attempt:1'",
            "INSERT INTO attempt_phases VALUES "
            "('attempt:1','validation',110,7,3,3,'{\"phase\":\"validation\"}')",
            "INSERT INTO daemon_execution_events VALUES "
            "('event:2','attempt:1','task:1','validation_started',111,"
            "'{\"event\":\"validation_started\"}')",
            "UPDATE daemon_execution_metadata SET value='process:recovered' "
            "WHERE key='execution_store_identity'",
            "COMMIT",
        ],
    )
    wal_digest = _sha256(wal_path)

    receipt = repair_database_execution_art_index_storage(path)

    assert receipt["wal_disposition"] == "distinct_recovered"
    assert receipt["wal_recovery_proof"]["wal_sha256"] == wal_digest
    connection = open_duckdb_connection(path)
    try:
        attempt = connection.execute(
            "SELECT committed_phase, status, revision, body_json "
            "FROM database_task_attempts WHERE attempt_id='attempt:1'"
        ).fetchone()
        assert attempt is not None
        assert tuple(attempt[index] for index in range(4)) == (
            "validation",
            "validating",
            3,
            '{"attempt":2}',
        )
        phase = connection.execute(
            "SELECT phase, committed_at_ms, revision, body_json "
            "FROM attempt_phases WHERE attempt_id='attempt:1' "
            "AND phase='validation'"
        ).fetchone()
        assert phase is not None
        assert tuple(phase[index] for index in range(4)) == (
            "validation",
            110,
            3,
            '{"phase":"validation"}',
        )
        event = connection.execute(
            "SELECT event_type, recorded_at_ms, body_json "
            "FROM daemon_execution_events WHERE event_id='event:2'"
        ).fetchone()
        assert event is not None
        assert tuple(event[index] for index in range(3)) == (
            "validation_started",
            111,
            '{"event":"validation_started"}',
        )
        metadata = connection.execute(
            "SELECT value FROM daemon_execution_metadata "
            "WHERE key='execution_store_identity'"
        ).fetchone()
        assert metadata is not None and metadata[0] == "process:recovered"
    finally:
        connection.close()


def test_corrupt_nonempty_wal_is_typed_refusal_with_exact_pair_preserved(
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
    payload = bytearray(wal_path.read_bytes())
    payload[max(0, len(payload) // 2 - 32) : len(payload) // 2 + 32] = b"\xff" * 64
    wal_path.write_bytes(payload)
    main_digest = _sha256(path)
    wal_digest = _sha256(wal_path)

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError) as captured:
        repair_database_execution_art_index_storage(path)

    assert captured.value.status["reason"] == "nonempty_wal_exact_replay_unavailable"
    assert captured.value.status["repair_performed"] is False
    assert captured.value.status["retry_permitted"] is False
    assert _sha256(path) == main_digest
    assert _sha256(wal_path) == wal_digest
    assert not tuple(
        (path.parent / ".execution-art-repair-quarantine").glob("*.committed.json")
    )


def test_nonempty_wal_cannot_introduce_unknown_catalog_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = _leave_wal(
        path,
        ["CREATE TABLE unreviewed_authority(value VARCHAR)"],
    )
    main_digest = _sha256(path)
    wal_digest = _sha256(wal_path)

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError) as captured:
        repair_database_execution_art_index_storage(path)

    assert "unknown or missing table" in str(captured.value)
    assert _sha256(path) == main_digest
    assert _sha256(wal_path) == wal_digest
    assert not tuple(
        (path.parent / ".execution-art-repair-quarantine").glob("*.prepared.json")
    )


@pytest.mark.parametrize("aba_target", ["main", "empty_wal"])
def test_pinned_backup_ignores_source_name_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aba_target: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    source_digest = _sha256(path)
    wal_path = path.with_name(path.name + ".wal")
    if aba_target == "main":
        live_target = path
        decoy = tmp_path / "decoy.duckdb"
        _seed_execution_store(decoy)
        decoy_connection = open_duckdb_connection(decoy)
        try:
            decoy_connection.execute(
                "UPDATE daemon_execution_metadata SET value='decoy' "
                "WHERE key='execution_store_identity'"
            )
            decoy_connection.execute("CHECKPOINT")
        finally:
            decoy_connection.close()
        trigger_call = 1
    else:
        wal_path.touch()
        live_target = wal_path
        decoy = tmp_path / "decoy.wal"
        decoy.write_bytes(b"not-the-captured-empty-wal")
        trigger_call = 2
    live_digest = _sha256(live_target)
    decoy_digest = _sha256(decoy)
    held = tmp_path / f"held-{live_target.name}"
    real_copy = daemon_module._copy_database_execution_storage_descriptor_at
    copy_calls = 0
    aba_calls = 0

    def copy_with_aba(*args: Any, **kwargs: Any) -> None:
        nonlocal copy_calls, aba_calls
        copy_calls += 1
        if copy_calls != trigger_call:
            real_copy(*args, **kwargs)
            return
        aba_calls += 1
        os.replace(live_target, held)
        os.replace(decoy, live_target)
        try:
            real_copy(*args, **kwargs)
        finally:
            os.replace(live_target, decoy)
            os.replace(held, live_target)

    monkeypatch.setattr(
        daemon_module,
        "_copy_database_execution_storage_descriptor_at",
        copy_with_aba,
    )
    receipt = repair_database_execution_art_index_storage(path)

    assert aba_calls == 1
    assert _sha256(Path(receipt["quarantined_source_path"])) == source_digest
    assert _sha256(decoy) == decoy_digest
    if aba_target == "empty_wal":
        assert _sha256(Path(receipt["quarantined_empty_wal_path"])) == live_digest
    connection = open_duckdb_connection(path)
    try:
        after = _database_execution_storage_projection_from_connection(connection)
    finally:
        connection.close()
    assert after == before


def test_rollback_link_aba_is_refused_after_original_backup_is_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    source_digest = _sha256(path)
    decoy = tmp_path / "decoy.duckdb"
    decoy.write_bytes(b"not-the-captured-source")
    decoy_digest = _sha256(decoy)
    held = tmp_path / "held-execution.duckdb"
    real_link = daemon_module.os.link
    aba_calls = 0

    def link_with_aba(source: Any, target: Any, **kwargs: Any) -> None:
        nonlocal aba_calls
        if "art-repair-rollback" not in os.fspath(target):
            real_link(source, target, **kwargs)
            return
        aba_calls += 1
        os.replace(path, held)
        os.replace(decoy, path)
        try:
            real_link(source, target, **kwargs)
        finally:
            os.replace(path, decoy)
            os.replace(held, path)

    monkeypatch.setattr(daemon_module.os, "link", link_with_aba)
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="captured authority",
    ):
        repair_database_execution_art_index_storage(path)

    assert aba_calls == 1
    assert _sha256(path) == source_digest
    assert _sha256(decoy) == decoy_digest
    quarantine = path.parent / ".execution-art-repair-quarantine"
    backups = tuple(quarantine.glob(f"{path.name}.*.duckdb"))
    assert len(backups) == 1
    assert _sha256(backups[0]) == source_digest
    assert not _repair_temporaries(path)


@pytest.mark.parametrize("empty_wal", [False, True])
def test_atomic_install_target_aba_preserves_both_authorities_and_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_wal: bool,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    source_digest = _sha256(path)
    wal_path = path.with_name(path.name + ".wal")
    if empty_wal:
        wal_path.touch()
        wal_digest = _sha256(wal_path)
    else:
        wal_digest = ""
    displaced_source = tmp_path / "writer-displaced-original.duckdb"
    concurrent_authority = tmp_path / "concurrent-authority.duckdb"
    concurrent_authority.write_bytes(b"concurrent-writer-authority")
    concurrent_digest = _sha256(concurrent_authority)
    real_exchange = daemon_module._exchange_database_execution_storage_names
    injected = False

    def exchange_with_target_aba(
        source_name: str,
        target_name: str,
        **kwargs: Any,
    ) -> None:
        nonlocal injected
        if not injected and target_name == path.name:
            injected = True
            os.replace(path, displaced_source)
            os.replace(concurrent_authority, path)
        real_exchange(source_name, target_name, **kwargs)

    monkeypatch.setattr(
        daemon_module,
        "_exchange_database_execution_storage_names",
        exchange_with_target_aba,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match=(
            r"exact main\+WAL rollback failed"
            if empty_wal
            else "candidate was not installed"
        ),
    ):
        repair_database_execution_art_index_storage(path)

    assert injected is True
    assert _sha256(path) == concurrent_digest
    assert _sha256(displaced_source) == source_digest
    quarantine = path.parent / ".execution-art-repair-quarantine"
    backups = tuple(
        candidate
        for candidate in quarantine.glob(f"{path.name}.*.duckdb")
        if "failed-replacement" not in candidate.name
    )
    assert len(backups) == 1
    assert _sha256(backups[0]) == source_digest
    if empty_wal:
        assert not wal_path.exists()
        wal_backups = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
        retired_wals = tuple(quarantine.glob("*.retired-live-wal"))
        assert len(wal_backups) == 1
        assert len(retired_wals) == 1
        assert _sha256(wal_backups[0]) == wal_digest
        assert _sha256(retired_wals[0]) == wal_digest
    assert not _repair_temporaries(path)


def test_empty_wal_retirement_aba_is_refused_without_changing_live_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    source_digest = _sha256(path)
    wal_path = path.with_name(path.name + ".wal")
    wal_path.touch()
    wal_digest = _sha256(wal_path)
    decoy = tmp_path / "decoy.wal"
    decoy.write_bytes(b"not-the-captured-empty-wal")
    held = tmp_path / "held-execution.wal"
    real_noreplace = (
        daemon_module._rename_database_execution_storage_name_noreplace
    )
    aba_calls = 0

    def noreplace_with_aba(
        source: Any,
        target: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal aba_calls
        if os.fspath(source) != wal_path.name or "retired-live-wal" not in os.fspath(target):
            real_noreplace(source, target, **kwargs)
            return
        aba_calls += 1
        os.replace(wal_path, held)
        os.replace(decoy, wal_path)
        try:
            real_noreplace(source, target, **kwargs)
        finally:
            os.replace(held, wal_path)

    monkeypatch.setattr(
        daemon_module,
        "_rename_database_execution_storage_name_noreplace",
        noreplace_with_aba,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match=r"exact main\+WAL rollback failed",
    ):
        repair_database_execution_art_index_storage(path)

    assert aba_calls == 1
    assert _sha256(path) == source_digest
    assert _sha256(wal_path) == wal_digest
    connection = open_duckdb_connection(path)
    try:
        after = _database_execution_storage_projection_from_connection(connection)
    finally:
        connection.close()
    assert after == before
    quarantine = path.parent / ".execution-art-repair-quarantine"
    backup_wals = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(backup_wals) == 1
    assert _sha256(backup_wals[0]) == wal_digest


def test_catalog_mismatch_removes_owned_candidate_without_touching_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = path.with_name(path.name + ".wal")
    wal_path.touch()
    source_digest = _sha256(path)
    wal_digest = _sha256(wal_path)
    real_catalog = daemon_module._database_execution_storage_catalog
    catalog_calls = 0

    def mismatched_catalog(connection: Any) -> dict[str, Any]:
        nonlocal catalog_calls
        catalog_calls += 1
        catalog = real_catalog(connection)
        if catalog_calls == 2:
            catalog = {**catalog, "indexes": []}
        return catalog

    monkeypatch.setattr(
        daemon_module,
        "_database_execution_storage_catalog",
        mismatched_catalog,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="catalog differs",
    ):
        repair_database_execution_art_index_storage(path)

    assert _sha256(path) == source_digest
    assert _sha256(wal_path) == wal_digest
    assert not _repair_temporaries(path)
    quarantine = path.parent / ".execution-art-repair-quarantine"
    backups = tuple(quarantine.glob(f"{path.name}.*.duckdb"))
    backup_wals = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(backups) == 1
    assert len(backup_wals) == 1
    assert _sha256(backups[0]) == source_digest
    assert _sha256(backup_wals[0]) == wal_digest


def test_prepared_receipt_failure_cleans_candidate_and_rollback_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = path.with_name(path.name + ".wal")
    wal_path.touch()
    source_digest = _sha256(path)
    wal_digest = _sha256(wal_path)

    def fail_prepared_receipt(
        phase_path: Path,
        _payload: Any,
        *,
        directory_fd: int | None = None,
    ) -> str:
        assert ".prepared.json.pending-" in str(phase_path)
        assert directory_fd is not None
        descriptor = os.open(
            phase_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, b'{"partial":')
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise OSError("injected prepared receipt failure")

    monkeypatch.setattr(
        daemon_module,
        "_write_database_execution_repair_phase_receipt",
        fail_prepared_receipt,
    )
    for _ in range(2):
        with pytest.raises(
            DatabaseImplementationExecutionStorageRepairError,
            match="prepared phase",
        ):
            repair_database_execution_art_index_storage(path)
        assert not _repair_temporaries(path)
        assert not tuple(
            (path.parent / ".execution-art-repair-quarantine").glob("*.pending-*")
        )

    assert _sha256(path) == source_digest
    assert _sha256(wal_path) == wal_digest
    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert _mode(quarantine) == 0o700
    backups = tuple(quarantine.glob(f"{path.name}.*.duckdb"))
    backup_wals = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(backups) == 2
    assert len(backup_wals) == 2
    assert all(_mode(candidate) == 0o600 for candidate in backups + backup_wals)


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE SCHEMA hidden; CREATE TABLE hidden.secret(value BIGINT)",
        "CREATE MACRO hidden_macro(value) AS value + 1",
        "CREATE TYPE hidden_kind AS ENUM ('secret')",
    ],
)
def test_unknown_catalog_objects_are_typed_refusal_and_untouched(
    tmp_path: Path,
    ddl: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    connection = open_duckdb_connection(path)
    try:
        for statement in ddl.split(";"):
            connection.execute(statement)
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    source_digest = _sha256(path)

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError):
        repair_database_execution_art_index_storage(path)

    assert _sha256(path) == source_digest
    assert not _repair_temporaries(path)
    quarantine = path.parent / ".execution-art-repair-quarantine"
    backups = tuple(quarantine.glob(f"{path.name}.*.duckdb"))
    assert len(backups) == 1
    assert _sha256(backups[0]) == source_digest


@pytest.mark.parametrize("unsafe_kind", ["world_writable", "foreign_owner"])
def test_unsafe_quarantine_is_refused_without_evidence_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    source_digest = _sha256(path)
    quarantine = path.parent / ".execution-art-repair-quarantine"
    quarantine.mkdir(mode=0o700)
    if unsafe_kind == "world_writable":
        os.chmod(quarantine, 0o777)
    else:
        current_euid = os.geteuid()
        monkeypatch.setattr(
            daemon_module.os,
            "geteuid",
            lambda: current_euid + 1,
        )

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError) as captured:
        repair_database_execution_art_index_storage(path)

    assert captured.value.status["reason"] == "unsafe_execution_repair_quarantine"
    assert _sha256(path) == source_digest
    assert not tuple(quarantine.iterdir())
    assert not _repair_temporaries(path)


@pytest.mark.parametrize("symlink_kind", ["database", "wal", "quarantine"])
def test_repair_refuses_symlinked_authority_components(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    real_path = tmp_path / "real.duckdb"
    _seed_execution_store(real_path)
    real_digest = _sha256(real_path)
    path = real_path
    evidence_target = tmp_path / "evidence-target"
    if symlink_kind == "database":
        path = tmp_path / "execution.duckdb"
        path.symlink_to(real_path.name)
    elif symlink_kind == "wal":
        evidence_target.write_bytes(b"not-a-wal")
        path.with_name(path.name + ".wal").symlink_to(evidence_target.name)
    else:
        evidence_target.mkdir()
        (path.parent / ".execution-art-repair-quarantine").symlink_to(evidence_target.name)

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError):
        repair_database_execution_art_index_storage(path)

    assert _sha256(real_path) == real_digest
    assert not _repair_temporaries(path)


def test_pinned_parent_refuses_path_retarget(tmp_path: Path) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    path = parent / "execution.duckdb"
    _seed_execution_store(path)
    cleanup = daemon_module._DatabaseExecutionRepairCleanup(path)
    moved = tmp_path / "moved-authority"
    parent.rename(moved)
    parent.mkdir()
    try:
        with pytest.raises(
            DatabaseImplementationExecutionStorageRepairError,
            match="parent path changed",
        ):
            cleanup.verify_parent_path()
    finally:
        cleanup.close()


def test_relative_quack_execution_path_is_frozen_across_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "lane").mkdir(parents=True)
    (second / "lane").mkdir(parents=True)
    original = first / "lane" / "execution.duckdb"
    decoy = second / "lane" / "execution.duckdb"
    _seed_execution_store(original)
    _seed_execution_store(decoy)
    connection = open_duckdb_connection(decoy)
    try:
        connection.execute(
            "UPDATE daemon_execution_metadata SET value='decoy' "
            "WHERE key='execution_store_identity'"
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    decoy_digest = _sha256(decoy)
    monkeypatch.chdir(first)
    daemon = DatabaseImplementationDaemon(
        database_path="control.duckdb",
        coordination_path="lane/coordination.duckdb",
        execution_path="lane/execution.duckdb",
        authority_mode="quack",
        task_source_kind="duckdb",
        quack_uri="quack:127.0.0.1:45671",
        control_store_id="store:cwd-freeze",
        control_store_generation="generation:cwd-freeze",
        task_source=object(),
        coordinator=object(),
    )
    assert daemon.execution_path == original
    daemon.close()
    monkeypatch.chdir(second)

    receipt = repair_database_execution_art_index_storage(daemon.execution_path)

    assert Path(receipt["database_path"]) == original
    assert _sha256(decoy) == decoy_digest


def test_initially_absent_wal_appearance_blocks_install_and_preserves_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    source_digest = _sha256(path)
    wal_path = path.with_name(path.name + ".wal")
    real_revalidate = daemon_module._revalidate_database_execution_storage_wal
    revalidations = 0
    raced_wal = b"legacy-writer-wal"

    def race_wal(*args: Any, **kwargs: Any) -> None:
        nonlocal revalidations
        real_revalidate(*args, **kwargs)
        revalidations += 1
        if revalidations == 2:
            wal_path.write_bytes(raced_wal)

    monkeypatch.setattr(
        daemon_module,
        "_revalidate_database_execution_storage_wal",
        race_wal,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="candidate was not installed",
    ) as captured:
        repair_database_execution_art_index_storage(path)

    assert "WAL appeared" in str(captured.value.__cause__)
    assert _sha256(path) == source_digest
    assert wal_path.read_bytes() == raced_wal
    assert not _repair_temporaries(path)


@pytest.mark.parametrize("initial_wal", ["absent", "empty", "redundant_nonempty"])
def test_late_wal_after_final_source_read_rolls_back_without_committed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_wal: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = path.with_name(path.name + ".wal")
    if initial_wal == "empty":
        wal_path.touch()
    elif initial_wal == "redundant_nonempty":
        _leave_wal(
            path,
            [
                "BEGIN",
                "UPDATE daemon_execution_metadata SET value='temporary' "
                "WHERE key='execution_store_identity'",
                "UPDATE daemon_execution_metadata "
                "SET value='execution-store:seed' "
                "WHERE key='execution_store_identity'",
                "COMMIT",
            ],
        )
    initial_wal_digest = _sha256(wal_path) if wal_path.exists() else ""
    source_digest = _sha256(path)
    source_inode = path.stat().st_ino
    late_wal = f"late-writer-wal:{initial_wal}".encode()
    real_identity = daemon_module._database_execution_storage_named_identity_at
    installed_identity_reads = 0

    def create_wal_after_final_installed_identity(
        name: str,
        *,
        directory_fd: int,
    ) -> Any:
        nonlocal installed_identity_reads
        identity = real_identity(name, directory_fd=directory_fd)
        if name == path.name and identity[2][1] != source_inode:
            installed_identity_reads += 1
            if installed_identity_reads == 3:
                wal_path.write_bytes(late_wal)
        return identity

    monkeypatch.setattr(
        daemon_module,
        "_database_execution_storage_named_identity_at",
        create_wal_after_final_installed_identity,
    )
    expected_error = (
        "candidate was not installed"
        if initial_wal == "absent"
        else r"exact main\+WAL rollback failed"
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match=expected_error,
    ) as captured:
        repair_database_execution_art_index_storage(path)

    assert "WAL appeared" in str(captured.value.__cause__)
    assert installed_identity_reads >= 3
    assert _sha256(path) == source_digest
    assert path.stat().st_ino == source_inode
    assert wal_path.read_bytes() == late_wal
    assert not _repair_temporaries(path)
    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert len(tuple(quarantine.glob("*.prepared.json"))) == 1
    assert not tuple(quarantine.glob("*.committed.json"))
    wal_backups = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    retired_wals = tuple(quarantine.glob("*.retired-live-wal"))
    if initial_wal == "absent":
        assert not wal_backups
        assert not retired_wals
    else:
        assert len(wal_backups) == 1
        assert len(retired_wals) == 1
        assert _sha256(wal_backups[0]) == initial_wal_digest
        assert _sha256(retired_wals[0]) == initial_wal_digest


@pytest.mark.parametrize("wal_kind", ["redundant", "distinct"])
def test_nonempty_wal_is_restored_exactly_when_install_fails_before_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wal_kind: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    source_digest = _sha256(path)
    statements = [
        "UPDATE daemon_execution_metadata SET value='distinct' "
        "WHERE key='execution_store_identity'",
    ]
    if wal_kind == "redundant":
        statements = [
            "BEGIN",
            "UPDATE daemon_execution_metadata SET value='temporary' "
            "WHERE key='execution_store_identity'",
            "UPDATE daemon_execution_metadata SET value='execution-store:seed' "
            "WHERE key='execution_store_identity'",
            "COMMIT",
        ]
    wal_path = _leave_wal(path, statements)
    wal_digest = _sha256(wal_path)
    wal_inode = wal_path.stat().st_ino
    wal_mode = _mode(wal_path)
    real_exchange = daemon_module._exchange_database_execution_storage_names

    def fail_candidate_install(
        source_name: str,
        target_name: str,
        **kwargs: Any,
    ) -> None:
        if (
            source_name.startswith(f".{path.name}.art-repair-")
            and "rollback" not in source_name
            and target_name == path.name
        ):
            raise OSError("injected crash after WAL retirement")
        real_exchange(source_name, target_name, **kwargs)

    monkeypatch.setattr(
        daemon_module,
        "_exchange_database_execution_storage_names",
        fail_candidate_install,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="candidate was not installed",
    ):
        repair_database_execution_art_index_storage(path)

    assert _sha256(path) == source_digest
    assert _sha256(wal_path) == wal_digest
    assert wal_path.stat().st_ino == wal_inode
    assert _mode(wal_path) == wal_mode
    assert not _repair_temporaries(path)
    quarantine = path.parent / ".execution-art-repair-quarantine"
    prepared_paths = tuple(quarantine.glob("*.prepared.json"))
    assert len(prepared_paths) == 1
    prepared = json.loads(prepared_paths[0].read_text(encoding="utf-8"))
    proof = prepared["wal_recovery_proof"]
    assert proof["proof_id"] == prepared["wal_recovery_proof_id"]
    assert proof["wal_sha256"] == wal_digest
    expected_disposition = (
        "distinct_recovered" if wal_kind == "distinct" else "redundant"
    )
    assert proof["disposition"] == expected_disposition
    assert (proof["main_projection_root"] == proof["recovered_projection_root"]) is (
        wal_kind == "redundant"
    )
    assert proof["catalog_root"]
    retired = tuple(quarantine.glob("*.retired-live-wal"))
    assert not retired
    wal_backups = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(wal_backups) == 1
    assert _sha256(wal_backups[0]) == wal_digest
    assert _mode(wal_backups[0]) == 0o600
    assert not tuple(quarantine.glob("*.committed.json"))


def test_post_install_crash_rolls_back_exact_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    source_digest = _sha256(path)
    source_inode = path.stat().st_ino
    real_identity = daemon_module._database_execution_storage_named_identity_at
    injected = False

    def fail_first_installed_identity(
        name: str,
        *,
        directory_fd: int,
    ) -> Any:
        nonlocal injected
        identity = real_identity(name, directory_fd=directory_fd)
        if name == path.name and identity[2][1] != source_inode and not injected:
            injected = True
            raise OSError("injected post-install crash")
        return identity

    monkeypatch.setattr(
        daemon_module,
        "_database_execution_storage_named_identity_at",
        fail_first_installed_identity,
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
    assert not _repair_temporaries(path)
    assert tuple(
        (path.parent / ".execution-art-repair-quarantine").glob("*.failed-replacement.duckdb")
    )


def test_post_install_crash_restores_exact_main_and_distinct_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    source_digest = _sha256(path)
    source_inode = path.stat().st_ino
    wal_digest = _sha256(wal_path)
    wal_inode = wal_path.stat().st_ino
    wal_mode = _mode(wal_path)
    real_identity = daemon_module._database_execution_storage_named_identity_at
    injected = False

    def fail_first_installed_identity(
        name: str,
        *,
        directory_fd: int,
    ) -> Any:
        nonlocal injected
        identity = real_identity(name, directory_fd=directory_fd)
        if name == path.name and identity[2][1] != source_inode and not injected:
            injected = True
            raise OSError("injected post-install distinct-WAL crash")
        return identity

    monkeypatch.setattr(
        daemon_module,
        "_database_execution_storage_named_identity_at",
        fail_first_installed_identity,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="candidate was not installed",
    ):
        repair_database_execution_art_index_storage(path)

    assert injected is True
    assert _sha256(path) == source_digest
    assert path.stat().st_ino == source_inode
    assert _sha256(wal_path) == wal_digest
    assert wal_path.stat().st_ino == wal_inode
    assert _mode(wal_path) == wal_mode
    assert not _repair_temporaries(path)
    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert len(tuple(quarantine.glob("*.prepared.json"))) == 1
    assert not tuple(quarantine.glob("*.committed.json"))
    assert not tuple(quarantine.glob("*.retired-live-wal"))
    wal_backups = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(wal_backups) == 1
    assert _sha256(wal_backups[0]) == wal_digest


def test_failed_post_exchange_rollback_assets_survive_and_restart_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    source_digest = _sha256(path)
    source_inode = path.stat().st_ino
    wal_digest = _sha256(wal_path)
    wal_inode = wal_path.stat().st_ino
    wal_mode = _mode(wal_path)
    real_identity = daemon_module._database_execution_storage_named_identity_at
    real_exchange = daemon_module._exchange_database_execution_storage_names
    installed_failure_injected = False
    exchange_calls = 0

    def fail_first_installed_identity(
        name: str,
        *,
        directory_fd: int,
    ) -> Any:
        nonlocal installed_failure_injected
        identity = real_identity(name, directory_fd=directory_fd)
        if (
            name == path.name
            and identity[2][1] != source_inode
            and not installed_failure_injected
        ):
            installed_failure_injected = True
            raise OSError("injected post-install verification failure")
        return identity

    def fail_rollback_exchange(*args: Any, **kwargs: Any) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 2:
            raise OSError("injected rollback exchange failure")
        real_exchange(*args, **kwargs)

    monkeypatch.setattr(
        daemon_module,
        "_database_execution_storage_named_identity_at",
        fail_first_installed_identity,
    )
    monkeypatch.setattr(
        daemon_module,
        "_exchange_database_execution_storage_names",
        fail_rollback_exchange,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match=r"exact main\+WAL rollback failed",
    ):
        repair_database_execution_art_index_storage(path)

    assert installed_failure_injected is True
    assert exchange_calls == 2
    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert len(tuple(quarantine.glob("*.prepared.json"))) == 1
    assert not tuple(quarantine.glob("*.committed.json"))
    assert len(tuple(quarantine.glob("*.retired-live-wal"))) == 1
    # Both original-inode recovery names survive uncertain rollback.
    rollback_paths = tuple(
        path.parent.glob(f".{path.name}.art-repair-rollback-*.duckdb")
    )
    displaced_paths = tuple(
        candidate
        for candidate in _repair_temporaries(path)
        if "rollback" not in candidate.name
    )
    assert len(rollback_paths) == 1
    assert _sha256(rollback_paths[0]) == source_digest
    assert rollback_paths[0].stat().st_ino == source_inode
    assert any(
        candidate.stat().st_ino == source_inode
        and _sha256(candidate) == source_digest
        for candidate in displaced_paths
    )

    monkeypatch.undo()
    reconciled = (
        daemon_module._reconcile_database_execution_storage_repair_before_open(
            path
        )
    )
    assert reconciled is not None
    assert reconciled["phase"] == "rolled_back"
    assert reconciled["rollback_mode"] == "post_exchange"
    assert _sha256(path) == source_digest
    assert path.stat().st_ino == source_inode
    assert _sha256(wal_path) == wal_digest
    assert wal_path.stat().st_ino == wal_inode
    assert _mode(wal_path) == wal_mode
    assert len(tuple(quarantine.glob("*.rolled-back.json"))) == 1
    assert not tuple(quarantine.glob("*.committed.json"))


def test_commit_marker_failure_preserves_rollback_assets_for_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    source_digest = _sha256(path)
    source_inode = path.stat().st_ino
    wal_digest = _sha256(wal_path)
    wal_inode = wal_path.stat().st_ino
    wal_mode = _mode(wal_path)
    real_write = daemon_module._write_database_execution_repair_phase_receipt

    def fail_committed_marker(
        phase_path: Path,
        payload: Any,
        *,
        directory_fd: int | None = None,
    ) -> str:
        if ".committed.json.pending-" in str(phase_path):
            assert directory_fd is not None
            descriptor = os.open(
                phase_path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(descriptor, b'{"partial":')
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise OSError("injected committed-marker persistence failure")
        return real_write(
            phase_path,
            payload,
            directory_fd=directory_fd,
        )

    monkeypatch.setattr(
        daemon_module,
        "_write_database_execution_repair_phase_receipt",
        fail_committed_marker,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="commit evidence is unavailable",
    ):
        repair_database_execution_art_index_storage(path)

    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert len(tuple(quarantine.glob("*.prepared.json"))) == 1
    assert not tuple(quarantine.glob("*.committed.json"))
    assert not tuple(quarantine.glob("*.pending-*"))
    assert len(tuple(quarantine.glob("*.retired-live-wal"))) == 1
    rollback_paths = tuple(
        path.parent.glob(f".{path.name}.art-repair-rollback-*.duckdb")
    )
    displaced_paths = tuple(
        candidate
        for candidate in _repair_temporaries(path)
        if "rollback" not in candidate.name
    )
    assert len(rollback_paths) == 1
    assert _sha256(rollback_paths[0]) == source_digest
    assert rollback_paths[0].stat().st_ino == source_inode
    assert any(
        candidate.stat().st_ino == source_inode
        and _sha256(candidate) == source_digest
        for candidate in displaced_paths
    )

    monkeypatch.undo()
    reconciled = (
        daemon_module._reconcile_database_execution_storage_repair_before_open(
            path
        )
    )
    assert reconciled is not None
    assert reconciled["phase"] == "rolled_back"
    assert reconciled["rollback_mode"] == "post_exchange"
    assert _sha256(path) == source_digest
    assert path.stat().st_ino == source_inode
    assert _sha256(wal_path) == wal_digest
    assert wal_path.stat().st_ino == wal_inode
    assert _mode(wal_path) == wal_mode
    assert len(tuple(quarantine.glob("*.rolled-back.json"))) == 1
    assert not tuple(quarantine.glob("*.committed.json"))


def test_sigkill_after_wal_retirement_is_reconciled_before_duckdb_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    source_digest = _sha256(path)
    source_inode = path.stat().st_ino
    wal_digest = _sha256(wal_path)
    wal_inode = wal_path.stat().st_ino
    wal_mode = _mode(wal_path)

    crashed = _crash_repair_after_wal_retirement(path)

    assert crashed.returncode == 91, (crashed.stdout, crashed.stderr)
    assert _sha256(path) == source_digest
    assert path.stat().st_ino == source_inode
    assert not wal_path.exists()
    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert len(tuple(quarantine.glob("*.prepared.json"))) == 1
    assert len(tuple(quarantine.glob("*.retired-live-wal"))) == 1
    assert not tuple(quarantine.glob("*.committed.json"))
    real_open = open_duckdb_connection
    open_calls = 0

    def assert_reconciled_before_open(database: Path | str, **kwargs: Any) -> Any:
        nonlocal open_calls
        if Path(database) == path:
            open_calls += 1
            assert _sha256(path) == source_digest
            assert path.stat().st_ino == source_inode
            assert _sha256(wal_path) == wal_digest
            assert wal_path.stat().st_ino == wal_inode
            assert _mode(wal_path) == wal_mode
        return real_open(database, **kwargs)

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state.open_duckdb_connection",
        assert_reconciled_before_open,
    )
    daemon = DatabaseImplementationDaemon(
        database_path=path.with_name("control.duckdb"),
        coordination_path=path.with_name("coordination.duckdb"),
        execution_path=path,
        owner_session_id="owner:recovery",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=object(),
        coordinator=object(),
        install_schema=False,
    )
    try:
        daemon.open()
    finally:
        daemon.close()

    assert open_calls == 1
    rolled_back = tuple(quarantine.glob("*.rolled-back.json"))
    assert len(rolled_back) == 1
    marker = json.loads(rolled_back[0].read_text(encoding="utf-8"))
    assert marker["phase"] == "rolled_back"
    assert marker["rollback_mode"] == "pre_exchange"
    assert marker["source_restored"] is True
    assert marker["wal_restored"] is True
    assert not tuple(quarantine.glob("*.retired-live-wal"))
    # Marker-backed reconciliation is idempotent and does not touch DuckDB.
    assert daemon_module._reconcile_database_execution_storage_repair_before_open(
        path
    ) is None


def test_orphan_repair_refuses_retired_wal_aba_and_preserves_backup(
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
    wal_digest = _sha256(wal_path)
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    quarantine = path.parent / ".execution-art-repair-quarantine"
    retired = tuple(quarantine.glob("*.retired-live-wal"))
    backups = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(retired) == len(backups) == 1
    held = quarantine / "held-original-retired.wal"
    os.replace(retired[0], held)
    retired[0].write_bytes(b"retired-wal-aba")

    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="retired WAL changed",
    ):
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)

    assert not wal_path.exists()
    assert _sha256(backups[0]) == wal_digest
    assert _sha256(held) == wal_digest
    assert not tuple(quarantine.glob("*.rolled-back.json"))


def test_orphan_repair_refuses_multiple_uncommitted_prepared_phases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    _leave_wal(
        path,
        [
            "UPDATE daemon_execution_metadata SET value='distinct' "
            "WHERE key='execution_store_identity'",
        ],
    )
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    quarantine = path.parent / ".execution-art-repair-quarantine"
    prepared_paths = tuple(quarantine.glob("*.prepared.json"))
    assert len(prepared_paths) == 1
    original = json.loads(prepared_paths[0].read_text(encoding="utf-8"))
    second_repair_id = "a" * 32
    second_backup = quarantine / (
        f"{path.name}.{original['source_sha256'].removeprefix('sha256:')}."
        f"{second_repair_id}.duckdb"
    )
    second_wal_backup = second_backup.with_name(second_backup.name + ".wal")
    second_rollback = path.parent / (
        f".{path.name}.art-repair-rollback-{second_repair_id}.duckdb"
    )
    os.link(Path(original["quarantined_source_path"]), second_backup)
    os.link(Path(original["quarantined_wal_path"]), second_wal_backup)
    os.link(path, second_rollback)
    duplicate_payload = {
        **original,
        "repair_id": second_repair_id,
        "quarantined_source_path": str(second_backup),
        "quarantined_wal_path": str(second_wal_backup),
        "rollback_path": str(second_rollback),
    }
    duplicate = quarantine / f"{second_backup.name}.prepared.json"
    duplicate.write_text(
        daemon_module.canonical_json(duplicate_payload) + "\n",
        encoding="utf-8",
    )
    os.chmod(duplicate, 0o600)

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError) as captured:
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)

    assert captured.value.status["reason"] == "orphan_repair_prepared_ambiguity"
    assert not tuple(quarantine.glob("*.rolled-back.json"))


def test_initially_absent_orphan_repair_refuses_late_wal_before_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    source_digest = _sha256(path)
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    wal_path = path.with_name(path.name + ".wal")
    late_wal = b"late-wal-after-prepared-phase"
    wal_path.write_bytes(late_wal)

    with pytest.raises(DatabaseImplementationExecutionStorageRepairError) as captured:
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)

    assert captured.value.status["reason"] == "orphan_repair_unexpected_wal"
    assert captured.value.status["live_wal_present"] is True
    assert _sha256(path) == source_digest
    assert wal_path.read_bytes() == late_wal
    assert not tuple(
        (path.parent / ".execution-art-repair-quarantine").glob(
            "*.rolled-back.json"
        )
    )


def test_orphan_repair_uses_manual_repair_database_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    wal_digest = _sha256(wal_path)
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    observed_locks: list[Path] = []

    @contextmanager
    def blocked_manual_repair_lock(lock_path: Path, **_kwargs: Any) -> Any:
        observed_locks.append(Path(lock_path))
        raise TimeoutError("simulated concurrent manual ART repair")
        yield

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state.exclusive_file_lock",
        blocked_manual_repair_lock,
    )
    with pytest.raises(TimeoutError, match="concurrent manual ART repair"):
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)

    assert observed_locks == [path.with_name(f".{path.name}.lock")]
    assert not wal_path.exists()
    retired = tuple(
        (path.parent / ".execution-art-repair-quarantine").glob(
            "*.retired-live-wal"
        )
    )
    assert len(retired) == 1
    assert _sha256(retired[0]) == wal_digest


def test_orphan_repair_refuses_coercible_prepared_control_types(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    _leave_wal(
        path,
        [
            "UPDATE daemon_execution_metadata SET value='distinct' "
            "WHERE key='execution_store_identity'",
        ],
    )
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    quarantine = path.parent / ".execution-art-repair-quarantine"
    prepared_paths = tuple(quarantine.glob("*.prepared.json"))
    assert len(prepared_paths) == 1
    prepared = json.loads(prepared_paths[0].read_text(encoding="utf-8"))
    prepared["wal_mode"] = str(prepared["wal_mode"])
    prepared_paths[0].write_text(
        daemon_module.canonical_json(prepared) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="non-integer size or mode",
    ):
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)

    assert not tuple(quarantine.glob("*.rolled-back.json"))


def test_orphan_repair_refuses_unknown_terminal_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    _leave_wal(
        path,
        [
            "UPDATE daemon_execution_metadata SET value='distinct' "
            "WHERE key='execution_store_identity'",
        ],
    )
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    receipt = daemon_module._reconcile_database_execution_storage_repair_before_open(
        path
    )
    assert receipt is not None and receipt["phase"] == "rolled_back"
    quarantine = path.parent / ".execution-art-repair-quarantine"
    marker_paths = tuple(quarantine.glob("*.rolled-back.json"))
    assert len(marker_paths) == 1
    original = json.loads(marker_paths[0].read_text(encoding="utf-8"))
    mutations = (
        ("source_size_bytes", str(original["source_size_bytes"])),
        ("source_restored", 1),
        ("source_sha256", "sha256:not-a-digest"),
        ("quarantined_source_path", "../escaped.duckdb"),
        (
            "source_identity",
            {
                **original["source_identity"],
                "size_bytes": str(original["source_identity"]["size_bytes"]),
            },
        ),
    )
    for field, value in mutations:
        marker = {**original, field: value}
        marker_paths[0].write_text(
            daemon_module.canonical_json(marker) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DatabaseImplementationExecutionStorageRepairError):
            daemon_module._reconcile_database_execution_storage_repair_before_open(
                path
            )
    marker = {**original, "unreviewed": True}
    marker_paths[0].write_text(
        daemon_module.canonical_json(marker) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatabaseImplementationExecutionStorageRepairError) as captured:
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)
    assert captured.value.status["reason"] == "orphan_repair_terminal_schema_invalid"


def test_orphan_scan_refuses_coercible_committed_terminal_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = path.with_name(path.name + ".wal")
    wal_path.touch()
    receipt = repair_database_execution_art_index_storage(path)
    marker_path = Path(receipt["committed_phase_path"])
    original = json.loads(marker_path.read_text(encoding="utf-8"))
    mutations = (
        ("wal_mode", str(original["wal_mode"])),
        ("wal_evidence_preserved", 1),
        ("replacement_sha256", "sha256:not-a-digest"),
        ("quarantined_wal_path", "../escaped.wal"),
        (
            "wal_backup_identity",
            {
                **original["wal_backup_identity"],
                "inode": str(original["wal_backup_identity"]["inode"]),
            },
        ),
        ("wal_disposition", "future_profile"),
    )
    for field, value in mutations:
        marker = {**original, field: value}
        marker_path.write_text(
            daemon_module.canonical_json(marker) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DatabaseImplementationExecutionStorageRepairError):
            daemon_module._reconcile_database_execution_storage_repair_before_open(
                path
            )


@pytest.mark.parametrize(
    "field",
    ("quarantined_source_path", "quarantined_wal_path", "rollback_path"),
)
def test_terminal_scan_refuses_mutually_edited_escaped_prepared_paths(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    path.with_name(path.name + ".wal").touch()
    receipt = repair_database_execution_art_index_storage(path)
    prepared_path = Path(receipt["prepared_phase_path"])
    committed_path = Path(receipt["committed_phase_path"])
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    escaped = str(tmp_path.parent / f"escaped-{field}.duckdb")
    prepared[field] = escaped
    if field in committed:
        committed[field] = escaped
    if field == "quarantined_source_path":
        committed["retired_live_wal_path"] = f"{escaped}.retired-live-wal"
    committed["prepared_phase_id"] = (
        daemon_module._database_execution_storage_payload_root(prepared)
    )
    prepared_path.write_text(
        daemon_module.canonical_json(prepared) + "\n",
        encoding="utf-8",
    )
    committed_path.write_text(
        daemon_module.canonical_json(committed) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="prepared (paths or identities|WAL path or size)",
    ):
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)


def test_same_process_wal_restore_never_overwrites_late_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    source_digest = _sha256(path)
    original_wal_digest = _sha256(wal_path)
    late_wal = b"late-wal-at-same-process-restore"
    real_exchange = daemon_module._exchange_database_execution_storage_names
    real_noreplace = (
        daemon_module._rename_database_execution_storage_name_noreplace
    )
    late_injected = False

    def fail_candidate_exchange(
        source_name: str,
        target_name: str,
        **kwargs: Any,
    ) -> None:
        if (
            source_name.startswith(f".{path.name}.art-repair-")
            and "rollback" not in source_name
            and target_name == path.name
        ):
            raise OSError("injected install failure after WAL retirement")
        real_exchange(source_name, target_name, **kwargs)

    def inject_late_wal_before_restore(
        source_name: str,
        target_name: str,
        *,
        source_directory_fd: int,
        target_directory_fd: int,
    ) -> None:
        nonlocal late_injected
        if "retired-live-wal" in source_name and target_name == wal_path.name:
            descriptor = os.open(
                target_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=target_directory_fd,
            )
            try:
                os.write(descriptor, late_wal)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            late_injected = True
        real_noreplace(
            source_name,
            target_name,
            source_directory_fd=source_directory_fd,
            target_directory_fd=target_directory_fd,
        )

    monkeypatch.setattr(
        daemon_module,
        "_exchange_database_execution_storage_names",
        fail_candidate_exchange,
    )
    monkeypatch.setattr(
        daemon_module,
        "_rename_database_execution_storage_name_noreplace",
        inject_late_wal_before_restore,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match=r"exact main\+WAL rollback failed",
    ):
        repair_database_execution_art_index_storage(path)

    assert late_injected is True
    assert _sha256(path) == source_digest
    assert wal_path.read_bytes() == late_wal
    quarantine = path.parent / ".execution-art-repair-quarantine"
    retired = tuple(quarantine.glob("*.retired-live-wal"))
    backups = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(retired) == len(backups) == 1
    assert _sha256(retired[0]) == original_wal_digest
    assert _sha256(backups[0]) == original_wal_digest
    assert not tuple(quarantine.glob("*.committed.json"))


def test_orphan_wal_restore_never_overwrites_late_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    original_wal_digest = _sha256(wal_path)
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    late_wal = b"late-wal-at-orphan-restore"
    real_noreplace = (
        daemon_module._rename_database_execution_storage_name_noreplace
    )

    def inject_late_wal_before_restore(
        source_name: str,
        target_name: str,
        *,
        source_directory_fd: int,
        target_directory_fd: int,
    ) -> None:
        if "retired-live-wal" in source_name and target_name == wal_path.name:
            descriptor = os.open(
                target_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=target_directory_fd,
            )
            try:
                os.write(descriptor, late_wal)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        real_noreplace(
            source_name,
            target_name,
            source_directory_fd=source_directory_fd,
            target_directory_fd=target_directory_fd,
        )

    monkeypatch.setattr(
        daemon_module,
        "_rename_database_execution_storage_name_noreplace",
        inject_late_wal_before_restore,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="atomic no-replace rename failed",
    ):
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)

    assert wal_path.read_bytes() == late_wal
    quarantine = path.parent / ".execution-art-repair-quarantine"
    retired = tuple(quarantine.glob("*.retired-live-wal"))
    backups = tuple(quarantine.glob(f"{path.name}.*.duckdb.wal"))
    assert len(retired) == len(backups) == 1
    assert _sha256(retired[0]) == original_wal_digest
    assert _sha256(backups[0]) == original_wal_digest
    assert not tuple(quarantine.glob("*.rolled-back.json"))


def test_wal_retirement_never_overwrites_preexisting_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    wal_path = path.with_name(path.name + ".wal")
    wal_path.touch()
    source_digest = _sha256(path)
    wal_digest = _sha256(wal_path)
    foreign = b"foreign-retired-target"
    real_noreplace = (
        daemon_module._rename_database_execution_storage_name_noreplace
    )
    injected_target = ""

    def inject_retirement_target(
        source_name: str,
        target_name: str,
        *,
        source_directory_fd: int,
        target_directory_fd: int,
    ) -> None:
        nonlocal injected_target
        if source_name == wal_path.name and "retired-live-wal" in target_name:
            descriptor = os.open(
                target_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=target_directory_fd,
            )
            try:
                os.write(descriptor, foreign)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            injected_target = target_name
        real_noreplace(
            source_name,
            target_name,
            source_directory_fd=source_directory_fd,
            target_directory_fd=target_directory_fd,
        )

    monkeypatch.setattr(
        daemon_module,
        "_rename_database_execution_storage_name_noreplace",
        inject_retirement_target,
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="candidate was not installed",
    ):
        repair_database_execution_art_index_storage(path)

    assert injected_target
    assert _sha256(path) == source_digest
    assert _sha256(wal_path) == wal_digest
    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert (quarantine / injected_target).read_bytes() == foreign
    assert not tuple(quarantine.glob("*.committed.json"))


def test_direct_retry_reconciles_crash_retired_distinct_wal_first(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    before = _seed_execution_store(path)
    _leave_wal(
        path,
        [
            "UPDATE daemon_execution_metadata SET value='distinct' "
            "WHERE key='execution_store_identity'",
        ],
    )
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91

    receipt = repair_database_execution_art_index_storage(path)

    assert receipt["wal_disposition"] == "distinct_recovered"
    assert receipt["pre_projection_root"] != before["projection_root"]
    connection = open_duckdb_connection(path)
    try:
        value = connection.execute(
            "SELECT value FROM daemon_execution_metadata "
            "WHERE key='execution_store_identity'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert value == "distinct"
    quarantine = path.parent / ".execution-art-repair-quarantine"
    assert len(tuple(quarantine.glob("*.rolled-back.json"))) == 1
    assert len(tuple(quarantine.glob("*.committed.json"))) == 1


def test_prepared_phase_rejects_unreviewed_outcome_and_false_proof_relation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.duckdb"
    _seed_execution_store(path)
    _leave_wal(
        path,
        [
            "UPDATE daemon_execution_metadata SET value='distinct' "
            "WHERE key='execution_store_identity'",
        ],
    )
    crashed = _crash_repair_after_wal_retirement(path)
    assert crashed.returncode == 91
    quarantine = path.parent / ".execution-art-repair-quarantine"
    prepared_paths = tuple(quarantine.glob("*.prepared.json"))
    assert len(prepared_paths) == 1
    original = json.loads(prepared_paths[0].read_text(encoding="utf-8"))

    unreviewed = {
        **original,
        "interrupted_transaction_outcome": "inferred_pass",
    }
    prepared_paths[0].write_text(
        daemon_module.canonical_json(unreviewed) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="prepared phase binding is invalid",
    ):
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)

    false_relation = dict(original)
    proof = dict(false_relation["wal_recovery_proof"])
    proof["main_projection_root"] = proof["recovered_projection_root"]
    proof_without_id = dict(proof)
    proof_without_id.pop("proof_id")
    proof_id = daemon_module._database_execution_storage_payload_root(
        proof_without_id
    )
    proof["proof_id"] = proof_id
    false_relation["wal_recovery_proof"] = proof
    false_relation["wal_recovery_proof_id"] = proof_id
    prepared_paths[0].write_text(
        daemon_module.canonical_json(false_relation) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DatabaseImplementationExecutionStorageRepairError,
        match="does not bind its disposition",
    ):
        daemon_module._reconcile_database_execution_storage_repair_before_open(path)
